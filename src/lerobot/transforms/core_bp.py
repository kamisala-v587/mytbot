from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl import Qwen3VLProcessor

from lerobot.datasets.transforms import ImageTransforms
from lerobot.transforms.core import DataDict, DataTransformFn
from lerobot.transforms.utils import resize_with_pad
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_STR, SAMPLE_ACTION_LOSS_MASK


BP_PREFIX = "behavior_prompt"
DEFAULT_BP_CAMERA_KEYS = [
    f"{OBS_IMAGES}.image0",
    f"{OBS_IMAGES}.image1",
    f"{OBS_IMAGES}.image2",
]
BP_CAMERA_INDEX = {key: idx for idx, key in enumerate(DEFAULT_BP_CAMERA_KEYS)}


@DataTransformFn.register_subclass("bp_pad_or_sample_chunks")
@dataclass
class BPPadOrSampleChunksFn(DataTransformFn):
    """Fix behavior prompt to a batchable number of chunks.

    If a prompt trajectory has more than `num_chunks`, chunks are uniformly
    downsampled. If it has fewer, the last valid chunk is repeated and marked as
    invalid in `behavior_prompt.mask`.
    """

    num_chunks: int = 4

    def __call__(self, data: DataDict) -> DataDict:
        prompt = data[BP_PREFIX]
        mask = prompt["mask"].to(torch.bool)
        source_k = int(mask.shape[0])
        if source_k <= 0:
            raise ValueError("behavior_prompt must contain at least one chunk")

        if source_k >= self.num_chunks:
            select = torch.linspace(0, source_k - 1, steps=self.num_chunks).round().to(torch.long)
            valid_mask = torch.ones(self.num_chunks, dtype=torch.bool)
        else:
            pad = torch.full((self.num_chunks - source_k,), source_k - 1, dtype=torch.long)
            select = torch.cat([torch.arange(source_k, dtype=torch.long), pad], dim=0)
            valid_mask = torch.cat(
                [torch.ones(source_k, dtype=torch.bool), torch.zeros(self.num_chunks - source_k, dtype=torch.bool)],
                dim=0,
            )

        for key in ["state", "action", "action_is_pad", "chunk_indices", "source_time_ratio"]:
            if key in prompt:
                prompt[key] = prompt[key].index_select(0, select.to(prompt[key].device))
        if "images" in prompt:
            prompt["images"] = {
                key: value.index_select(0, select.to(value.device))
                for key, value in prompt["images"].items()
            }
        prompt["mask"] = valid_mask
        prompt["chunk_indices"] = torch.arange(self.num_chunks, dtype=torch.long, device=valid_mask.device)
        data[BP_PREFIX] = prompt
        return data


@DataTransformFn.register_subclass("bp_resize_with_pad")
@dataclass
class BPResizeImagesWithPadFn(DataTransformFn):
    """Resize only behavior_prompt images with padding to a fixed resolution."""

    height: int
    width: int
    mode: str = "bilinear"

    def __call__(self, data: DataDict) -> DataDict:
        prompt = data[BP_PREFIX]
        prompt["images"] = {
            key: resize_with_pad(value, self.height, self.width, self.mode)
            for key, value in prompt["images"].items()
        }
        data[BP_PREFIX] = prompt
        return data


@DataTransformFn.register_subclass("bp_remap_image_key")
@dataclass
class BPRemapImageKeyTransformFn(DataTransformFn):
    """Remap BP image keys to canonical keys and keep only configured BP cameras."""

    mapping: dict[str, str] = field(default_factory=dict)
    bp_camera_keys: list[str] = field(default_factory=lambda: list(DEFAULT_BP_CAMERA_KEYS))

    def __call__(self, data: DataDict) -> DataDict:
        prompt = data[BP_PREFIX]
        images = prompt["images"]
        remapped: dict[str, torch.Tensor] = {}
        for old_key, new_key in self.mapping.items():
            if old_key in images:
                remapped[new_key] = images[old_key]
        if not remapped:
            available = ", ".join(sorted(images.keys()))
            expected = ", ".join(sorted(self.mapping.keys()))
            raise KeyError(f"[BPRemapImageKeyTransformFn] expected [{expected}], got [{available}]")

        unknown = [key for key in self.bp_camera_keys if key not in BP_CAMERA_INDEX]
        if unknown:
            allowed = ", ".join(DEFAULT_BP_CAMERA_KEYS)
            raise KeyError(f"[BPRemapImageKeyTransformFn] unsupported bp_camera_keys={unknown}; allowed [{allowed}]")

        missing = [key for key in self.bp_camera_keys if key not in remapped]
        if missing:
            available = ", ".join(sorted(remapped.keys()))
            raise KeyError(f"[BPRemapImageKeyTransformFn] missing configured BP cameras {missing}; available [{available}]")

        prompt["images"] = {key: remapped[key] for key in self.bp_camera_keys}
        data[BP_PREFIX] = prompt
        return data


@DataTransformFn.register_subclass("bp_normalize")
@dataclass
class BPNormalizeTransformFn(DataTransformFn):
    """Normalize behavior_prompt state/action using the same stats as current samples."""

    selected_keys: list[str] | None = None
    mode: str = "mean_std"
    norm_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __call__(self, data: DataDict) -> DataDict:
        prompt = data[BP_PREFIX]
        eps = 1e-6
        keys = self.selected_keys if self.selected_keys is not None else list(self.norm_stats.keys())
        for key in keys:
            if key not in {OBS_STATE, ACTION}:
                continue
            prompt_key = "state" if key == OBS_STATE else "action"
            if prompt_key not in prompt or key not in self.norm_stats:
                continue
            x = prompt[prompt_key]
            stats = self.norm_stats[key]
            if self.mode == "mean_std":
                mean = torch.from_numpy(stats["mean"]).to(x)
                std = torch.from_numpy(stats["std"]).to(x)
                prompt[prompt_key] = (x - mean) / (std + eps)
            elif self.mode == "min_max":
                min_v = torch.from_numpy(stats["min"]).to(x)
                max_v = torch.from_numpy(stats["max"]).to(x)
                prompt[prompt_key] = (x - min_v) / (max_v - min_v + eps)
            else:
                raise ValueError(f"Unknown normalization mode: {self.mode}")
        data[BP_PREFIX] = prompt
        return data


@DataTransformFn.register_subclass("bp_compose_fields")
@dataclass
class BPComposeFieldsTransform(DataTransformFn):
    """Compose behavior_prompt state/action fields.

    For datasets where state/action are already canonical, this is a no-op. The
    class exists to mirror current-sample ComposeFieldsTransform and keep BP
    processing extensible for non-ALOHA robot schemas.
    """

    mapping: dict[str, list[str]] = field(default_factory=dict)

    def __call__(self, data: DataDict) -> DataDict:
        # BehaviorPromptLeRobotDataset currently stores prompt["state"] and prompt["action"]
        # from already indexed LeRobot samples. No additional source fields remain here.
        return data


@DataTransformFn.register_subclass("bp_delta_action")
@dataclass
class BPDeltaActionTransformFn(DataTransformFn):
    """Convert behavior_prompt action chunks to deltas from their chunk states."""

    mask: Optional[list[bool]] = None

    def __call__(self, data: DataDict) -> DataDict:
        prompt = data[BP_PREFIX]
        if "state" not in prompt or "action" not in prompt:
            return data

        state = prompt["state"]
        action = prompt["action"]
        if state.shape[-1] == 0 or action.shape[-1] == 0:
            return data

        if self.mask is None:
            mask = torch.ones(state.shape[-1], dtype=torch.bool, device=state.device)
        else:
            mask = torch.as_tensor(self.mask, dtype=torch.bool, device=state.device)
        if mask.shape[-1] != state.shape[-1]:
            raise ValueError(
                f"BPDeltaActionTransformFn mask dim {mask.shape[-1]} does not match "
                f"state dim {state.shape[-1]}"
            )

        delta = torch.where(mask, state, torch.zeros_like(state))
        while delta.ndim < action.ndim:
            delta = delta.unsqueeze(-2)
        prompt["action"] = action - delta.to(device=action.device, dtype=action.dtype)
        data[BP_PREFIX] = prompt
        return data


@DataTransformFn.register_subclass("bp_pad_state_and_action")
@dataclass
class BPPadStateAndActionTransformFn(DataTransformFn):
    """Pad behavior_prompt state/action feature dimensions to TBot max dims."""

    max_state_dim: int = 32
    max_action_dim: int = 32

    def __call__(self, data: DataDict) -> DataDict:
        prompt = data[BP_PREFIX]
        prompt["state"] = self._pad_last_dim(prompt["state"], self.max_state_dim)
        prompt["action"] = self._pad_last_dim(prompt["action"], self.max_action_dim)
        data[BP_PREFIX] = prompt
        return data

    @staticmethod
    def _pad_last_dim(value: torch.Tensor, target_dim: int) -> torch.Tensor:
        if value.shape[-1] >= target_dim:
            return value
        return F.pad(value, (0, target_dim - value.shape[-1]))


@DataTransformFn.register_subclass("img_only_qwen3_vl")
@dataclass
class ImgOnlyQwen3VLTransformFn(DataTransformFn):
    """Process current images for Qwen3-VL without appending task text tokens."""

    pretrained_model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    spatial_merge_size: int = 2
    processor: Any = field(default=None, init=False, repr=False)
    _processor_source: str | None = field(default=None, init=False, repr=False)

    def _ensure_processor(self) -> None:
        if self.processor is not None and self._processor_source == self.pretrained_model_name_or_path:
            return
        self.processor = Qwen3VLProcessor.from_pretrained(self.pretrained_model_name_or_path)
        self._processor_source = self.pretrained_model_name_or_path

    def __call__(self, data: DataDict) -> DataDict:
        self._ensure_processor()
        input_ids: list[int] = []
        attention_mask: list[int] = []
        pixel_values = []
        image_grid_thw = []
        for camera_idx in range(3):
            image_key = f"{OBS_IMAGES}.image{camera_idx}"
            img_inputs = self.processor.image_processor(data[image_key][1], do_rescale=False)
            token_count = torch.prod(img_inputs.image_grid_thw) // self.spatial_merge_size**2
            pixel_values.append(img_inputs.pixel_values)
            image_grid_thw.append(img_inputs.image_grid_thw)
            is_valid = bool(data.get(f"{image_key}_mask", torch.tensor(True)))
            input_ids += [self.processor.vision_start_token_id]
            input_ids += [self.processor.image_token_id] * int(token_count.item())
            input_ids += [self.processor.vision_end_token_id]
            attention_mask += [1 if is_valid else 0] * (int(token_count.item()) + 2)
        data[f"{OBS_STR}.pixel_values"] = torch.cat(pixel_values, dim=0)
        data[f"{OBS_STR}.image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
        data[f"{OBS_STR}.input_ids"] = torch.tensor(input_ids, dtype=torch.long)
        data[f"{OBS_STR}.attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
        return data


@DataTransformFn.register_subclass("unify_bp_inputs")
@dataclass
class UnifyBPInputsTransformFn(DataTransformFn):
    """只保留 BPVAPolicy.forward 消费的固定形状字段。

    采样来源等元数据不会进入模型，避免不同样本的可变长度字段导致 batch 拼接失败。
    """

    def __call__(self, data: DataDict) -> DataDict:
        default_action_loss_mask = 0.0 if data.get("robot_type") == "egodex_v" else 1.0
        prompt = data[BP_PREFIX]
        data = {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            SAMPLE_ACTION_LOSS_MASK: data.get(
                SAMPLE_ACTION_LOSS_MASK,
                torch.tensor([default_action_loss_mask], dtype=torch.float32),
            ),
            f"{OBS_IMAGES}.image0": data[f"{OBS_IMAGES}.image0"],
            f"{OBS_IMAGES}.image1": data[f"{OBS_IMAGES}.image1"],
            f"{OBS_IMAGES}.image2": data[f"{OBS_IMAGES}.image2"],
            f"{OBS_IMAGES}.image0_mask": data[f"{OBS_IMAGES}.image0_mask"],
            f"{OBS_IMAGES}.image1_mask": data[f"{OBS_IMAGES}.image1_mask"],
            f"{OBS_IMAGES}.image2_mask": data[f"{OBS_IMAGES}.image2_mask"],
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            f"{OBS_STR}.input_ids": data[f"{OBS_STR}.input_ids"],
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
            BP_PREFIX: {
                "mask": prompt["mask"],
                "chunk_indices": prompt["chunk_indices"],
                "state": prompt["state"],
                "action": prompt["action"],
                "action_is_pad": prompt["action_is_pad"],
                "images": prompt["images"],
            },
        }
        return data
