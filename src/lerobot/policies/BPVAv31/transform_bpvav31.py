from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch

from lerobot.policies.BPVAv31.configuration_bpvav31 import parse_bp_dim_use
from lerobot.policies.q35.transform_q35 import Qwen3_5ProcessorTransformFn
from lerobot.transforms.core import DataDict, DataTransformFn
from lerobot.transforms.core_bp import BP_PREFIX
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_STR, SAMPLE_ACTION_LOSS_MASK

from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK


def load_qwen35_vision(path: str, revision: str | None = None):
    """加载 Qwen3.5 图像处理器和视觉 token id，不编码 instruction。"""

    kwargs = {"revision": revision} if revision else {}
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(path, **kwargs)
    try:
        from transformers import AutoImageProcessor

        image_processor = AutoImageProcessor.from_pretrained(path, **kwargs)
    except Exception:
        from transformers import AutoProcessor

        image_processor = AutoProcessor.from_pretrained(path, **kwargs).image_processor
    merge = int(config.vision_config.spatial_merge_size)
    token_ids = SimpleNamespace(
        vision_start_token_id=int(config.vision_start_token_id),
        vision_end_token_id=int(config.vision_end_token_id),
        image_token_id=int(config.image_token_id),
        spatial_merge_size=merge,
    )
    return image_processor, token_ids


@DataTransformFn.register_subclass("bpvav31_qwen3_5_processor")
@dataclass
class BPVAv31ProcessorTransformFn(Qwen3_5ProcessorTransformFn):
    """当前观测的 Qwen3.5 打包。`use_language=False` 时不调用 tokenizer。"""

    use_language: bool = True
    _vision_source: tuple[str, str | None] | None = field(default=None, init=False, repr=False)

    def _ensure_vision_without_tokenizer(self) -> None:
        source = (self.pretrained_model_name_or_path, self.revision)
        if self.processor is not None and self._vision_source == source and not self.use_language:
            return
        image_processor, token_ids = load_qwen35_vision(
            self.pretrained_model_name_or_path,
            self.revision,
        )
        self.processor = SimpleNamespace(image_processor=image_processor, tokenizer=None)
        self.vision_start_token_id = token_ids.vision_start_token_id
        self.vision_end_token_id = token_ids.vision_end_token_id
        self.image_token_id = token_ids.image_token_id
        self.spatial_merge_size = token_ids.spatial_merge_size
        self._vision_source = source
        self._processor_source = source

    def _image_blocks(self, data: DataDict) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        valid_blocks = []
        invalid_blocks = []
        for index in range(3):
            key = f"{OBS_IMAGES}.image{index}"
            images = data[key]
            if not 0 <= self.current_image_index < images.shape[0]:
                raise ValueError(
                    f"current_image_index={self.current_image_index} is outside {key} "
                    f"with {images.shape[0]} frame(s)"
                )
            image_inputs = self.processor.image_processor(
                images[self.current_image_index],
                do_rescale=False,
            )
            num_image_tokens = int(
                (torch.prod(image_inputs.image_grid_thw) // self.spatial_merge_size**2).item()
            )
            block = {
                "input_ids": [self.vision_start_token_id]
                + [self.image_token_id] * num_image_tokens
                + [self.vision_end_token_id],
                "mm_token_type_ids": [0] + [1] * num_image_tokens + [0],
                "pixel_values": image_inputs.pixel_values,
                "image_grid_thw": image_inputs.image_grid_thw,
                "valid": bool(data.get(f"{key}_mask", True)),
            }
            (valid_blocks if block["valid"] else invalid_blocks).append(block)
        return valid_blocks, invalid_blocks

    def __call__(self, data: DataDict) -> DataDict:
        if self.use_language:
            return super().__call__(data)

        self._ensure_vision_without_tokenizer()
        valid_blocks, invalid_blocks = self._image_blocks(data)
        ordered_blocks = [*valid_blocks, *invalid_blocks]
        data[f"{OBS_STR}.pixel_values"] = torch.cat(
            [block["pixel_values"] for block in ordered_blocks]
        )
        data[f"{OBS_STR}.image_grid_thw"] = torch.cat(
            [block["image_grid_thw"] for block in ordered_blocks]
        )
        input_ids: list[int] = []
        attention_mask: list[int] = []
        mm_token_type_ids: list[int] = []
        for block in valid_blocks:
            input_ids += block["input_ids"]
            mm_token_type_ids += block["mm_token_type_ids"]
            attention_mask += [1] * len(block["input_ids"])
        for block in invalid_blocks:
            input_ids += block["input_ids"]
            mm_token_type_ids += block["mm_token_type_ids"]
            attention_mask += [0] * len(block["input_ids"])
        data[f"{OBS_STR}.input_ids"] = torch.tensor(input_ids)
        data[f"{OBS_STR}.attention_mask"] = torch.tensor(attention_mask)
        data[f"{OBS_STR}.mm_token_type_ids"] = torch.tensor(mm_token_type_ids)
        return data


@DataTransformFn.register_subclass("bpvav31_qwen35_image")
@dataclass
class BPVAv31Qwen35ImageTransformFn(DataTransformFn):
    """把每路 BP 相机打成 Qwen3.5 pixel patches。不调用 tokenizer。"""

    pretrained_model_name_or_path: str = "Qwen/Qwen3.5-0.8B-Base"
    revision: str | None = None
    image_processor: Any = field(default=None, init=False, repr=False)
    spatial_merge_size: int = field(default=2, init=False)
    _source: tuple[str, str | None] | None = field(default=None, init=False, repr=False)

    def _ensure(self) -> None:
        source = (self.pretrained_model_name_or_path, self.revision)
        if self.image_processor is not None and self._source == source:
            return
        image_processor, token_ids = load_qwen35_vision(
            self.pretrained_model_name_or_path,
            self.revision,
        )
        self.image_processor = image_processor
        self.spatial_merge_size = token_ids.spatial_merge_size
        self._source = source

    def __call__(self, data: DataDict) -> DataDict:
        self._ensure()
        prompt = data[BP_PREFIX]
        processed_pixels: dict[str, torch.Tensor] = {}
        processed_grids: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        masks = prompt.get("image_masks") or {}
        for key, images in prompt["images"].items():
            mask = masks.get(key)
            if mask is not None and not bool(torch.as_tensor(mask).any()):
                skipped.append(key)
                continue
            outputs = self.image_processor(list(images), do_rescale=False)
            grids = torch.as_tensor(outputs.image_grid_thw, dtype=torch.long)
            pixels = torch.as_tensor(outputs.pixel_values)
            patch_counts = grids.prod(dim=-1)
            if not torch.all(patch_counts == patch_counts[0]):
                raise ValueError(f"BPVAv31 requires fixed processed image shape within camera {key!r}")
            processed_pixels[key] = pixels.reshape(images.shape[0], int(patch_counts[0]), pixels.shape[-1])
            processed_grids[key] = grids
        if not processed_pixels:
            raise ValueError("BPVAv31 requires at least one unmasked camera to process")
        template_pixels = next(iter(processed_pixels.values()))
        template_grids = next(iter(processed_grids.values()))
        for key in skipped:
            processed_pixels[key] = torch.zeros_like(template_pixels)
            processed_grids[key] = template_grids.clone()
        prompt["bp_pixel_values"] = processed_pixels
        prompt["bp_image_grid_thw"] = processed_grids
        data[BP_PREFIX] = prompt
        return data


def _select_last_dim(tensor: torch.Tensor, spec: Any, *, name: str) -> tuple[torch.Tensor, bool]:
    """按 bp_*_use 清掉最后一维。返回 (tensor, dropped_all)。"""
    parsed = parse_bp_dim_use(spec, name=name)
    keep = parsed.resolve(int(tensor.shape[-1]))
    if keep is None:
        return tensor, False
    if len(keep) == 0:
        return torch.zeros_like(tensor), True
    selected = tensor.clone()
    drop = [index for index in range(selected.shape[-1]) if index not in set(keep)]
    selected[..., drop] = 0
    return selected, False


@DataTransformFn.register_subclass("bpvav31_select_prompt_dims")
@dataclass
class BPVAv31SelectPromptDimsFn(DataTransformFn):
    """只保留示教 state/action 的指定维度。放在 pad 之后，下标和模型输入一致。

    整段关掉 action 时同时把 action_is_pad 置为 true。整段关掉 state 时把 state_is_available 置为 false。
    只丢掉部分维度时，token 仍然有效，未选中的维度置零。
    """

    bp_action_use: Any = True
    bp_state_use: Any = True

    def __call__(self, data: DataDict) -> DataDict:
        prompt = data[BP_PREFIX]
        state, drop_state = _select_last_dim(prompt["state"], self.bp_state_use, name="bp_state_use")
        action, drop_action = _select_last_dim(prompt["action"], self.bp_action_use, name="bp_action_use")
        prompt["state"] = state
        prompt["action"] = action
        if drop_state:
            prompt["state_is_available"] = torch.zeros(
                state.shape[:-1],
                dtype=torch.bool,
                device=state.device,
            )
        if drop_action:
            prompt["action_is_pad"] = torch.ones(
                action.shape[:-1],
                dtype=torch.bool,
                device=action.device,
            )
        data[BP_PREFIX] = prompt
        return data


def apply_future_chunk_filter(prompt: dict) -> dict:
    """丢掉锚点早于当前帧的 chunk。一个都不剩时保留最后一个原本有效的 chunk。

    只改 mask。chunk_indices 仍是 linspace 的原始槽位，方便和示教侧的位置编码对齐。
    """
    if "source_frame_offsets" not in prompt or "current_frame_offset" not in prompt:
        raise KeyError("future_prompt filter requires source_frame_offsets and current_frame_offset")
    offsets = torch.as_tensor(prompt["source_frame_offsets"], dtype=torch.long).reshape(-1)
    current = int(torch.as_tensor(prompt["current_frame_offset"]).reshape(-1)[0])
    mask = torch.as_tensor(prompt["mask"], dtype=torch.bool).reshape(-1)
    if offsets.shape[0] != mask.shape[0]:
        raise ValueError(
            f"source_frame_offsets length {offsets.shape[0]} does not match mask {mask.shape[0]}"
        )
    keep = mask & (offsets >= current)
    if not bool(keep.any()):
        valid = mask.nonzero(as_tuple=False).flatten()
        last = int(valid[-1]) if int(valid.numel()) else int(mask.shape[0] - 1)
        keep = keep.clone()
        keep[last] = True
    prompt["mask"] = keep
    return prompt


def _model_prompt_fields(prompt: dict) -> dict:
    model_prompt = {
        "mask": prompt["mask"],
        "chunk_indices": prompt["chunk_indices"],
        "state": prompt["state"],
        "action": prompt["action"],
        "action_is_pad": prompt["action_is_pad"],
    }
    for optional_key in (
        "state_is_available",
        "image_masks",
        "bp_pixel_values",
        "bp_image_grid_thw",
        "source_time_ratio",
    ):
        if optional_key in prompt:
            model_prompt[optional_key] = prompt[optional_key]
    return model_prompt


class BPVAv31SampleTransform:
    """先变换 behavior_prompt，再把同一条 BP 变换链用到 future / origin_prompt 上。"""

    def __init__(self, bp_transforms: list, current_transforms: list, *, future_filter: bool) -> None:
        self.bp_transforms = list(bp_transforms)
        self.current_transforms = list(current_transforms)
        self.future_filter = bool(future_filter)

    def _run_bp_branch(self, data: DataDict, key: str, *, apply_future_filter: bool) -> DataDict:
        saved = data[BP_PREFIX]
        data[BP_PREFIX] = data[key]
        for transform in self.bp_transforms:
            data = transform(data)
        if apply_future_filter:
            data[BP_PREFIX] = apply_future_chunk_filter(data[BP_PREFIX])
        data[key] = data[BP_PREFIX]
        data[BP_PREFIX] = saved
        return data

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.bp_transforms:
            data = transform(data)
        if "future_prompt" in data:
            data = self._run_bp_branch(data, "future_prompt", apply_future_filter=self.future_filter)
        if "origin_prompt" in data:
            # 当前 episode 的关键帧，只用于检索 / 下一 state 监督，不进 VLM。
            data = self._run_bp_branch(data, "origin_prompt", apply_future_filter=False)
            data["bp_origin_state"] = data["origin_prompt"]["state"]
            data["bp_origin_time_ratio"] = data["origin_prompt"]["source_time_ratio"]
            del data["origin_prompt"]
        for transform in self.current_transforms:
            data = transform(data)
        return data


@DataTransformFn.register_subclass("bpvav31_unify_inputs")
@dataclass
class UnifyBPVAv31InputsTransformFn(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict:
        default_action_loss_mask = 0.0 if data.get("robot_type") == "egodex_v" else 1.0
        model_prompt = _model_prompt_fields(data[BP_PREFIX])
        output = {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            SAMPLE_ACTION_LOSS_MASK: data.get(
                SAMPLE_ACTION_LOSS_MASK,
                torch.tensor([default_action_loss_mask], dtype=torch.float32),
            ),
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            f"{OBS_STR}.input_ids": data[f"{OBS_STR}.input_ids"],
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
            f"{OBS_STR}.mm_token_type_ids": data[f"{OBS_STR}.mm_token_type_ids"],
            BP_PREFIX: model_prompt,
        }
        if "future_prompt" in data:
            output["future_prompt"] = _model_prompt_fields(data["future_prompt"])
        for index in range(3):
            key = f"{OBS_IMAGES}.image{index}"
            output[key] = data[key]
            output[f"{key}_mask"] = data[f"{key}_mask"]
        if ACTION_VALID_MASK in data:
            output[ACTION_VALID_MASK] = data[ACTION_VALID_MASK]
        for key in ("current_progress", "bp_origin_state", "bp_origin_time_ratio"):
            if key in data:
                output[key] = data[key]
        return output
