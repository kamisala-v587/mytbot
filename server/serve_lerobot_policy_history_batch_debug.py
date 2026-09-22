#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TBot-SA1 动态 batch 调试服务：严格两帧历史 + 可选请求字段日志。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from serve_lerobot_policy import CAMERA_ALIASES, coerce_history
from serve_lerobot_policy_batch import (
    BatchOptions,
    BatchedTBotSA1PolicyService,
    parse_args as parse_batch_args,
    run_batch_server,
)


@dataclass(frozen=True)
class DebugBatchServeArgs:
    serve: Any
    batch: BatchOptions
    # 逗号分隔字段名；空元组表示关闭日志。常用: task,prompt,timestep,reset,state,keys
    log_request_fields: tuple[str, ...] = ()


def parse_args() -> DebugBatchServeArgs:
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument(
        "--log_request_fields",
        default="",
        help=(
            "收到请求时打印的字段，逗号分隔；留空则关闭。"
            "示例: task,prompt,timestep 。"
            "特殊值: task/prompt=任务描述（优先 client 的 prompt/task，否则 default_prompt）；"
            "resolved_task=服务端最终采用的任务描述；keys=打印全部顶层 key。"
        ),
    )
    known, rest = extra.parse_known_args()
    fields = tuple(part.strip() for part in known.log_request_fields.split(",") if part.strip())

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *rest]
        batch_args = parse_batch_args()
    finally:
        sys.argv = original_argv

    return DebugBatchServeArgs(
        serve=batch_args.serve,
        batch=batch_args.batch,
        log_request_fields=fields,
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
        history = coerce_history(raw)
        if history.shape[0] != 2:
            raise RuntimeError(f"相机 {standardized_key} 转换后 T 异常: {history.shape}")
        return history, True


def _summarize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={tuple(value.shape)}, dtype={value.dtype})"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"bytes(len={len(value)})"
    if isinstance(value, dict):
        return {k: _summarize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)) and value and not isinstance(value[0], (str, int, float, bool)):
        first = value[0]
        if isinstance(first, np.ndarray):
            return f"{type(value).__name__}[len={len(value)}; item0=ndarray{tuple(first.shape)}]"
        return f"{type(value).__name__}[len={len(value)}; item0={type(first).__name__}]"
    return value


class DebugHistoryBatchedTBotSA1PolicyService(
    StrictTwoFrameHistoryMixin, BatchedTBotSA1PolicyService
):
    """严格两帧协议 + 可选请求字段调试日志。"""

    def __init__(self, args, log_request_fields: tuple[str, ...] = ()) -> None:
        super().__init__(args)
        self._log_request_fields = tuple(log_request_fields)
        self._metadata["debug_log_request_fields"] = list(self._log_request_fields)

    def infer_batch(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._log_request_fields:
            for index, obs in enumerate(observations):
                self._log_selected_fields(obs, batch_index=index, batch_size=len(observations))
        return super().infer_batch(observations)

    def _log_selected_fields(self, obs: dict[str, Any], *, batch_index: int, batch_size: int) -> None:
        payload: dict[str, Any] = {}
        for field in self._log_request_fields:
            if field == "keys":
                payload["keys"] = sorted(obs.keys())
                continue
            if field == "resolved_task":
                payload["resolved_task"] = self._resolve_prompt(obs)
                continue
            if field in {"task", "prompt"}:
                # 与推理一致：prompt 优先，其次 task，否则 default_prompt
                raw_prompt = obs.get("prompt")
                raw_task = obs.get("task")
                payload[field] = {
                    "raw_prompt": raw_prompt if isinstance(raw_prompt, str) else raw_prompt,
                    "raw_task": raw_task if isinstance(raw_task, str) else raw_task,
                    "resolved": self._resolve_prompt(obs),
                }
                continue
            if field not in obs:
                payload[field] = "<missing>"
                continue
            payload[field] = _summarize_value(obs[field])

        logging.info(
            "请求字段 | batch=%d/%d | %s",
            batch_index + 1,
            batch_size,
            json.dumps(payload, ensure_ascii=False, default=str),
        )


def main(args: DebugBatchServeArgs) -> None:
    logging.info(
        "启动参数:\n%s",
        json.dumps(
            {
                "serve": asdict(args.serve),
                "batch": asdict(args.batch),
                "log_request_fields": list(args.log_request_fields),
            },
            indent=2,
            ensure_ascii=False,
        ),
    )
    if args.log_request_fields:
        logging.info("已开启请求字段日志: %s", ",".join(args.log_request_fields))
    else:
        logging.info("请求字段日志关闭（传入 --log_request_fields task 可开启）")

    policy = DebugHistoryBatchedTBotSA1PolicyService(
        args.serve, log_request_fields=args.log_request_fields
    )
    run_batch_server(policy, args.serve, args.batch)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    main(parse_args())
