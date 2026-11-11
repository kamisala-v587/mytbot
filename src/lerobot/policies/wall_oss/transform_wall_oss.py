"""Sample transforms that turn a TBot frame into wall-x model inputs.

Mirrors wall-x's LeRobot data path (`wall_x/data/backends/lerobot/{loader,utils}.py`, commit b18112d):
`_vision_preprocess` (longest side to `image_resolution`, then Qwen smart resize), `get_wallx_normal_text` (action
branch, no FAST tokens), `DataCollator._normalize` and `preprocesser_call` (state discretized into `state_bins` bins
and written into the `<|propri|>` slot, image placeholders expanded to the processor's token count). Each sample is
padded to `max_length` tokens so the default collate can stack it; right padding (wall-x RoboDojo `padding_side`)
leaves every real token exactly as without padding, left padding does not (about 1% flow loss, also in float32).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.transforms.core import DataDict, DataTransformFn, NormalizeTransformFn
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_PREFIX, OBS_STATE, SAMPLE_ACTION_LOSS_MASK

PROPRI = "<|propri|>"
ACTION_SLOT = "<|action|>"
IMAGE_PAD = "<|image_pad|>"


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """qwen_vl_utils 0.0.11 `smart_resize` (the version wall-x pins)."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"aspect ratio {max(height, width) / min(height, width)} above 200")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def wall_image(frame: torch.Tensor, resolution: int, factor: int, min_pixels: int, max_pixels: int) -> Image.Image:
    """[C, H, W] float in [0, 1] -> the PIL image wall-x feeds to the processor."""
    img = Image.fromarray((frame.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy())
    width, height = img.size
    if resolution != -1:
        if width > height:
            img = img.resize((resolution, int(resolution * height / width)))
        else:
            img = img.resize((int(resolution * width / height), resolution))
    width, height = img.size
    h, w = smart_resize(height, width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    return img.resize((w, h))


def wall_prompt(camera_names: list[str], instruction: str, num_actions: int, system_prompt: str,
                instruction_label: str) -> str:
    """`get_wallx_normal_text` (action branch) after `replace_action_token` without an action tokenizer."""
    text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nObservation:"
    for name in camera_names:
        text += f" {name}: <|vision_start|>{IMAGE_PAD}<|vision_end|>"
    text += f"\n{instruction_label}:"
    text += (f" {instruction}\nPredict the next action in robot action.\nProprioception: {PROPRI}\n<|im_end|>\n"
             f"<|im_start|>assistant\n{ACTION_SLOT * num_actions}")
    return text


def state_text(norm_state: np.ndarray, bins: int) -> str:
    """Normalized state in [-1, 1] -> space-separated bin indices (`preprocesser_call`)."""
    return " ".join(map(str, np.digitize(norm_state, bins=np.linspace(-1, 1, bins + 1)[:-1]) - 1))


@DataTransformFn.register_subclass("wall_normalize")
@dataclass
class WallNormalizeTransformFn(NormalizeTransformFn):
    """wall-x min-max normalization: (x - min) / (max - min) * 2 - 1, clipped to [-1, 1]; zero range -> 1."""

    mode: str = "wall_min_max"

    def __call__(self, data: DataDict) -> DataDict:
        keys = self.selected_keys if self.selected_keys is not None else list(self.norm_stats.keys())
        for key in keys:
            if key not in data or key not in self.norm_stats:
                continue
            x = data[key]
            lo = torch.as_tensor(np.asarray(self.norm_stats[key]["min"]), dtype=torch.float32).to(x)
            delta = torch.as_tensor(np.asarray(self.norm_stats[key]["max"]), dtype=torch.float32).to(x) - lo
            delta = torch.where(delta == 0, torch.ones_like(delta), delta)
            data[key] = torch.clamp((x - lo) / delta * 2 - 1, -1, 1)
        return data


@DataTransformFn.register_subclass("wall_prompt")
@dataclass
class WallPromptTransformFn(DataTransformFn):
    processor_path: str = "/vla/workspace/models/q35/base_800m"
    camera_names: list[str] = field(default_factory=lambda: ["face_view", "left_wrist_view", "right_wrist_view"])
    image_resolution: int = 256
    image_factor: int = 28
    min_pixels: int = 3136
    max_pixels: int = 1003520
    max_length: int = 448
    padding_side: str = "right"
    system_prompt: str = "You are a helpful assistant."
    instruction_label: str = "Instruction"
    state_bins: int = 256
    action_horizon: int = 32
    task_key: str = "task"
    processor: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.processor = None

    def _ensure_processor(self) -> None:
        if self.processor is None:
            self.processor = load_wall_processor(self.processor_path)

    def __call__(self, data: DataDict) -> DataDict:
        self._ensure_processor()
        tok = self.processor.tokenizer
        names, images = [], []
        for index, name in enumerate(self.camera_names):
            key = f"{OBS_IMAGES}.image{index}"
            if key in data and bool(data.get(f"{key}_mask", True)):
                names.append(name)
                images.append(wall_image(data[key][0], self.image_resolution, self.image_factor, self.min_pixels,
                                         self.max_pixels))
        text = wall_prompt(names, data[self.task_key], self.action_horizon, self.system_prompt,
                           self.instruction_label)
        text = text.replace(PROPRI, state_text(data[OBS_STATE].float().reshape(-1).numpy(), self.state_bins))
        image_inputs = self.processor.image_processor(images=images, return_tensors="pt")
        merge = self.processor.image_processor.merge_size ** 2
        for grid in image_inputs["image_grid_thw"]:
            text = text.replace(IMAGE_PAD, "<|placeholder|>" * int(grid.prod() // merge), 1)
        text = text.replace("<|placeholder|>", IMAGE_PAD)
        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) > self.max_length:
            raise ValueError(f"wall prompt has {len(ids)} tokens > max_length {self.max_length}")
        pad = [tok.pad_token_id] * (self.max_length - len(ids))
        real = [1] * len(ids)
        left = self.padding_side == "left"
        input_ids = torch.tensor(pad + ids if left else ids + pad)
        image_id = tok.convert_tokens_to_ids(IMAGE_PAD)
        data[f"{OBS_PREFIX}input_ids"] = input_ids
        data[f"{OBS_PREFIX}attention_mask"] = torch.tensor([0] * len(pad) + real if left else real + [0] * len(pad))
        data[f"{OBS_PREFIX}mm_token_type_ids"] = (input_ids == image_id).long()
        data[f"{OBS_PREFIX}pixel_values"] = image_inputs["pixel_values"]
        data[f"{OBS_PREFIX}image_grid_thw"] = image_inputs["image_grid_thw"]
        return data


def load_wall_processor(path: str):
    """Qwen3.5 processor with wall-x's two added tokens (`load_wallx_processors`)."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(path)
    processor.tokenizer.add_tokens([PROPRI, ACTION_SLOT])
    return processor


@DataTransformFn.register_subclass("wall_unify_inputs")
@dataclass
class UnifyWallInputsTransformFn(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict:
        keys = [OBS_STATE, ACTION, ACTION_VALID_MASK, SAMPLE_ACTION_LOSS_MASK] + [
            f"{OBS_PREFIX}{k}" for k in ("input_ids", "attention_mask", "mm_token_type_ids", "pixel_values",
                                         "image_grid_thw")]
        return {k: data[k] for k in keys if k in data}
