"""Q35：使用当前帧。客户端若传来多帧，只取最后一帧。"""

from __future__ import annotations

from typing import Any

import torch

from .base import PolicyAdapter, PreparedSample
from .common import (
    CAMERA_ALIASES,
    OBS_STATE,
    hwc_tensor,
    resolve_image_history,
    resolve_images,
    resolve_state,
)
from lerobot.policies.q35.transform_q35 import Qwen3_5ProcessorTransformFn


class Q35Adapter(PolicyAdapter):
    name = "q35"
    family = "vla"
    aliases = ("q35",)

    def _check_type(self) -> None:
        if self.config.type != "q35":
            raise ValueError(
                f"POLICY=q35 但 checkpoint type={self.config.type!r}。请改 POLICY，或把 POLICY 设为 auto。"
            )

    def setup_policy(self) -> None:
        processor_path = (
            getattr(self.config, "qwen3_5_processor_path", None)
            or getattr(self.config, "qwen3_5_pretrained_path", None)
            or self.args.qwen3_5_path
        )
        if not processor_path:
            raise ValueError("缺少 Qwen3.5 路径，请设置 QWEN3_5_PATH。")
        self.processor_fn = Qwen3_5ProcessorTransformFn(
            pretrained_model_name_or_path=str(processor_path),
            max_length=int(getattr(self.config, "tokenizer_max_length", 48)),
            current_image_index=0,
        )

    def prepare(self, obs: dict[str, Any]) -> PreparedSample:
        images = resolve_images(obs)
        state_np = resolve_state(obs)
        sample: dict[str, Any] = {"task": self.resolve_task(obs), OBS_STATE: torch.from_numpy(state_np)}
        masks: dict[str, bool] = {}
        for key in CAMERA_ALIASES:
            history, mask = resolve_image_history(
                images,
                key,
                strict=False,
                frames=1,
                height=self.args.request_image_height,
                width=self.args.request_image_width,
            )
            sample[key] = hwc_tensor(history)
            masks[key] = mask
        sample = self.resize_fn(sample)
        for key, mask in masks.items():
            sample[f"{key}_mask"] = torch.tensor(mask)
        sample = self.processor_fn(sample)
        sample = self.normalize_state_fn(sample)
        kept = {
            OBS_STATE: sample[OBS_STATE],
            "observation.pixel_values": sample["observation.pixel_values"],
            "observation.image_grid_thw": sample["observation.image_grid_thw"],
            "observation.input_ids": sample["observation.input_ids"],
            "observation.attention_mask": sample["observation.attention_mask"],
            "observation.mm_token_type_ids": sample["observation.mm_token_type_ids"],
        }
        return PreparedSample(self.pack_inputs(kept), state_np)


ADAPTER = Q35Adapter
