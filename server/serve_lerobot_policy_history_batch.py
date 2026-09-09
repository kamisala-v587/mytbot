#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TBot-SA1 动态 batch 服务：严格要求每个已提供相机携带两帧历史。"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

import numpy as np

from serve_lerobot_policy import CAMERA_ALIASES, coerce_history
from serve_lerobot_policy_batch import (
    BatchedTBotSA1PolicyService,
    parse_args,
    run_batch_server,
)


class StrictTwoFrameHistoryMixin:
    """声明并强制两帧推理协议；缺失相机仍由父类生成 blank/mask=False。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        image_delta_indices = [int(value) for value in getattr(self.config, "image_delta_indices", [])]
        if len(image_delta_indices) < 2:
            raise ValueError(
                "checkpoint config.image_delta_indices 至少需要 2 项；当前模型推理使用前两项图像历史。"
            )
        self._metadata.update(
            {
                "image_delta_indices": image_delta_indices,
                "inference_image_delta_indices": image_delta_indices[:2],
                "required_image_history": 2,
                "action_dim": int(self.target_action_dim),
            }
        )

    def _resolve_image_history(
        self, images: dict[str, Any], standardized_key: str
    ) -> tuple[np.ndarray, bool]:
        value = None
        for alias in CAMERA_ALIASES[standardized_key]:
            if alias in images:
                value = images[alias]
                break
        if value is None:
            return super()._resolve_image_history(images, standardized_key)

        raw = np.asarray(value)
        if raw.ndim != 4:
            raise ValueError(
                f"相机 {standardized_key} 必须显式提供两帧 4D history，收到 shape={raw.shape}；"
                "严格服务端不允许把单帧静默复制为 [t,t]。"
            )
        if raw.shape[0] != 2:
            raise ValueError(
                f"相机 {standardized_key} history 的 T 必须为 2，收到 shape={raw.shape}。"
            )
        # 复用生产服务的 HWC/CHW、dtype 转换；先验检查保证不会触发单帧复制/首尾抽样。
        history = coerce_history(raw)
        if history.shape[0] != 2:
            raise RuntimeError(f"相机 {standardized_key} 转换后 T 异常: {history.shape}")
        return history, True


class HistoryBatchedTBotSA1PolicyService(
    StrictTwoFrameHistoryMixin, BatchedTBotSA1PolicyService
):
    """严格两帧协议 + 现有 TBot 动态 batch 推理。"""


def main(args) -> None:
    logging.info("启动参数:\n%s", json.dumps(asdict(args), indent=2, ensure_ascii=False))
    policy = HistoryBatchedTBotSA1PolicyService(args.serve)
    run_batch_server(policy, args.serve, args.batch)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    main(parse_args())
