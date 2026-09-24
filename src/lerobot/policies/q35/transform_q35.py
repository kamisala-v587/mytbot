from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from lerobot.transforms.core import (
    ComposeFieldsTransform,
    DataDict,
    DataTransformFn,
    RemapImageKeyTransformFn,
    DeltaActionTransformFn,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_STR, SAMPLE_ACTION_LOSS_MASK


ACTION_VALID_MASK = "action_is_valid"


@DataTransformFn.register_subclass("q35_delta_action")
@dataclass
class Q35DeltaActionTransformFn(DeltaActionTransformFn):
    """Source-compatible delta transform, including unequal state/action widths."""

    skip: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if self.skip:
            return data
        state_keys = self.mapping[OBS_STATE]
        action_keys = self.mapping[ACTION]
        if not all(key in data for key in state_keys + action_keys):
            return data
        state_list, _ = self._align_for_cat([data[key] for key in state_keys])
        state = torch.cat(state_list, dim=-1)
        action_list, sizes = self._align_for_cat([data[key] for key in action_keys])
        action = torch.cat(action_list, dim=-1)
        mask = self.mask if self.mask is not None else torch.tensor([True] * state.shape[-1])
        state_delta_base = torch.where(torch.as_tensor(mask, device=state.device), state, 0)
        if state_delta_base.shape[-1] > action.shape[-1]:
            state_delta_base = state_delta_base[..., : action.shape[-1]]
        elif state_delta_base.shape[-1] < action.shape[-1]:
            state_delta_base = F.pad(state_delta_base, (0, action.shape[-1] - state_delta_base.shape[-1]))
        action -= state_delta_base[None]
        start = 0
        for key, size in zip(action_keys, sizes, strict=True):
            data[key] = action[..., start : start + size]
            start += size
        return data


@DataTransformFn.register_subclass("q35_qwen3_5_processor")
@dataclass
class Qwen3_5ProcessorTransformFn(DataTransformFn):
    pretrained_model_name_or_path: str = "Qwen/Qwen3.5-0.8B-Base"
    revision: str | None = None
    max_length: int = 48
    task_key: str = "task"
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True
    current_image_index: int = 0
    spatial_merge_size: int = 2

    vision_start_token_id: int | None = field(default=None, init=False)
    vision_end_token_id: int | None = field(default=None, init=False)
    image_token_id: int | None = field(default=None, init=False)
    processor: Any = field(default=None, init=False, repr=False)
    _processor_source: tuple[str, str | None] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.processor = None
        self._processor_source = None

    def _ensure_processor(self) -> None:
        source = (self.pretrained_model_name_or_path, self.revision)
        if self.processor is not None and self._processor_source == source:
            return

        from transformers import AutoProcessor

        kwargs = {"revision": self.revision} if self.revision else {}
        self.processor = AutoProcessor.from_pretrained(self.pretrained_model_name_or_path, **kwargs)
        self._processor_source = source
        tokenizer = self.processor.tokenizer
        self.vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.spatial_merge_size = int(self.processor.image_processor.merge_size)
        if any(
            token_id is None or token_id == tokenizer.unk_token_id
            for token_id in (self.vision_start_token_id, self.vision_end_token_id, self.image_token_id)
        ):
            raise ValueError("Qwen3.5 processor is missing required vision special tokens")

    def __call__(self, data: DataDict) -> DataDict:
        self._ensure_processor()
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

        ordered_blocks = [*valid_blocks, *invalid_blocks]
        data[f"{OBS_STR}.pixel_values"] = torch.cat(
            [block["pixel_values"] for block in ordered_blocks]
        )
        data[f"{OBS_STR}.image_grid_thw"] = torch.cat(
            [block["image_grid_thw"] for block in ordered_blocks]
        )

        lang_inputs = self.processor.tokenizer(
            data[self.task_key],
            max_length=self.max_length,
            padding_side=self.padding_side,
            padding=self.padding,
            truncation=self.truncation,
        )
        input_ids: list[int] = []
        attention_mask: list[int] = []
        mm_token_type_ids: list[int] = []
        for block in valid_blocks:
            input_ids += block["input_ids"]
            mm_token_type_ids += block["mm_token_type_ids"]
            attention_mask += [1] * len(block["input_ids"])
        input_ids += lang_inputs.input_ids
        attention_mask += lang_inputs.attention_mask
        mm_token_type_ids += [0] * len(lang_inputs.input_ids)
        for block in invalid_blocks:
            input_ids += block["input_ids"]
            mm_token_type_ids += block["mm_token_type_ids"]
            attention_mask += [0] * len(block["input_ids"])
        data[f"{OBS_STR}.input_ids"] = torch.tensor(input_ids)
        data[f"{OBS_STR}.attention_mask"] = torch.tensor(attention_mask)
        data[f"{OBS_STR}.mm_token_type_ids"] = torch.tensor(mm_token_type_ids)
        return data


@DataTransformFn.register_subclass("q35_compose_fields")
@dataclass
class Q35ComposeFieldsTransformFn(ComposeFieldsTransform):
    def __call__(self, data: DataDict) -> DataDict:
        action_sources = self.mapping.get(ACTION, [])
        pad_masks = [data.get(f"{key}_is_pad") for key in action_sources]
        present_masks = [mask for mask in pad_masks if mask is not None]
        if present_masks:
            if len(present_masks) != len(action_sources):
                raise ValueError("action padding metadata must be present for every composed action field")
            action_is_pad = torch.stack(
                [torch.as_tensor(mask, dtype=torch.bool).flatten() for mask in present_masks]
            ).any(dim=0)
            data[ACTION_VALID_MASK] = ~action_is_pad
        return super().__call__(data)


@DataTransformFn.register_subclass("q35_remap_image_key")
@dataclass
class RemapQ35ImageKeyTransformFn(RemapImageKeyTransformFn):
    current_image_index: int = 0

    def __call__(self, data: DataDict) -> DataDict:
        for old_key, new_key in self.mapping.items():
            images = data.pop(old_key)
            pad = data.pop(f"{old_key}_is_pad", None)
            if pad is None:
                if images.ndim == 3:
                    images = images.unsqueeze(0)
                temporal_valid = torch.ones(images.shape[0], dtype=torch.bool)
            else:
                temporal_valid = ~torch.as_tensor(pad, dtype=torch.bool).flatten()
                if images.ndim == 3 and temporal_valid.numel() == 1:
                    images = images.unsqueeze(0)
            if images.ndim != 4 or images.shape[0] != temporal_valid.numel():
                raise ValueError(
                    f"{old_key} must resolve to [time, channels, height, width] with one "
                    f"padding flag per frame, got image_shape={tuple(images.shape)} and "
                    f"padding_count={temporal_valid.numel()}"
                )
            if not 0 <= self.current_image_index < temporal_valid.numel():
                raise ValueError(
                    f"current_image_index={self.current_image_index} is outside {new_key} "
                    f"with {temporal_valid.numel()} frame(s)"
                )
            data[new_key] = images
            data[f"{new_key}_mask"] = temporal_valid[self.current_image_index]

        if not self.mapping:
            raise ValueError("Q35 image remapping requires at least one camera mapping")
        template = data[f"{OBS_IMAGES}.image0"]
        for index in range(len(self.mapping), 3):
            key = f"{OBS_IMAGES}.image{index}"
            data[key] = torch.ones_like(template)
            data[f"{key}_mask"] = torch.tensor(False)
        return data


@DataTransformFn.register_subclass("q35_unify_inputs")
@dataclass
class UnifyQ35InputsTransformFn(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict:
        output = {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            f"{OBS_STR}.input_ids": data[f"{OBS_STR}.input_ids"],
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
            f"{OBS_STR}.mm_token_type_ids": data[f"{OBS_STR}.mm_token_type_ids"],
        }
        for index in range(3):
            key = f"{OBS_IMAGES}.image{index}"
            output[key] = data[key]
            output[f"{key}_mask"] = data[f"{key}_mask"]
        if SAMPLE_ACTION_LOSS_MASK in data:
            output[SAMPLE_ACTION_LOSS_MASK] = data[SAMPLE_ACTION_LOSS_MASK]
        if ACTION_VALID_MASK in data:
            output[ACTION_VALID_MASK] = data[ACTION_VALID_MASK]
        return output
