"""TBot-SA1：两帧历史输入。只给一帧时复制补齐。"""

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
from lerobot.policies.TBot_SA1.transform_tbot_sa1 import Qwen3_VLProcessorTransformFn
from lerobot.policies.names import is_tbot_sa1


class TBotAdapter(PolicyAdapter):
    name = "tbot"
    family = "vla"
    aliases = ("tbot", "TBot_SA1", "tbot_sa1")

    def _check_type(self) -> None:
        if not is_tbot_sa1(self.config.type):
            raise ValueError(
                f"POLICY=tbot 但 checkpoint type={self.config.type!r}。请改 POLICY，或把 POLICY 设为 auto。"
            )

    def setup_policy(self) -> None:
        processor_path = (
            getattr(self.config, "qwen3_vl_processor_path", None)
            or getattr(self.config, "qwen3_vl_pretrained_path", None)
            or self.args.qwen3_vl_path
        )
        if not processor_path:
            raise ValueError("缺少 Qwen3-VL 路径，请设置 QWEN3_VL_PATH。")
        self.processor_fn = Qwen3_VLProcessorTransformFn(
            pretrained_model_name_or_path=str(processor_path),
            max_length=int(getattr(self.config, "tokenizer_max_length", 48)),
        )

    def prepare(self, obs: dict[str, Any]) -> PreparedSample:
        images = resolve_images(obs)
        sample: dict[str, Any] = {"task": self.resolve_task(obs), OBS_STATE: torch.from_numpy(resolve_state(obs))}
        masks: dict[str, bool] = {}
        for key in CAMERA_ALIASES:
            history, mask = resolve_image_history(
                images,
                key,
                strict=self.args.strict_history,
                frames=2,
                height=self.args.request_image_height,
                width=self.args.request_image_width,
            )
            sample[key] = hwc_tensor(history)
            masks[key] = mask
        state = sample[OBS_STATE].numpy()
        sample = self.resize_fn(sample)
        for key, mask in masks.items():
            sample[f"{key}_mask"] = torch.tensor(mask)
        sample = self.processor_fn(sample)
        sample = self.normalize_state_fn(sample)
        return PreparedSample(self.pack_inputs(sample), state)


ADAPTER = TBotAdapter
