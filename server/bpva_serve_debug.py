#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BPVA WebSocket server with one-time behavior-prompt video logging."""

from __future__ import annotations

import argparse
import json
import logging
import re
import socket
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bpva_serve import (
    BPVAServeArgs,
    BPVAPolicyService,
    BPTaskSource,
    BehaviorPromptCache,
    WebsocketPolicyServer,
    parse_args as parse_production_args,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


DEFAULT_BPLOGS_DIR = Path(__file__).resolve().parent / "bplogs"


@dataclass
class DebugServeArgs:
    serve: BPVAServeArgs
    bplogs_dir: Path


def parse_args() -> DebugServeArgs:
    debug_parser = argparse.ArgumentParser(add_help=False)
    debug_parser.add_argument("--bplogs_dir", default=str(DEFAULT_BPLOGS_DIR))
    debug_args, production_argv = debug_parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *production_argv]
        serve_args = parse_production_args()
    finally:
        sys.argv = original_argv
    return DebugServeArgs(
        serve=serve_args,
        bplogs_dir=Path(debug_args.bplogs_dir).expanduser().resolve(),
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "task"


def _select_main_camera(camera_keys: list[str]) -> str:
    if not camera_keys:
        raise ValueError("BP 数据集没有相机字段。")
    priorities = ("image0", "cam_high", "head", "main", "front")
    lowered = {key: key.lower() for key in camera_keys}
    for token in priorities:
        for key in camera_keys:
            if token in lowered[key]:
                return key
    return camera_keys[0]


def _tensor_to_rgb_uint8(image: Any) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"图像必须为 3 维，实际 shape={array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(np.nanmax(array)) <= 1.5:
            array = array * 255.0
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"图像通道数必须为 1、3 或 4，实际 shape={array.shape}")
    return np.ascontiguousarray(array)


class LoggingBehaviorPromptCache(BehaviorPromptCache):
    """Production BP cache plus a best-effort MP4 export on cache miss."""

    def __init__(self, *args: Any, run_log_dir: Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.run_log_dir = run_log_dir

    def _build_first_readable(self, source: BPTaskSource) -> tuple[dict[str, Any], int]:
        prompt, episode_idx = super()._build_first_readable(source)
        try:
            output = self._write_episode_video(source, episode_idx)
            logging.info(
                "BP 主视角日志已保存 | task_type=%s | episode=%d | path=%s",
                source.task_type,
                episode_idx,
                output,
            )
        except Exception:
            # 日志属于调试附加功能，失败不能影响 BP 缓存和模型推理。
            logging.exception(
                "BP 主视角日志保存失败，继续推理 | task_type=%s | episode=%d",
                source.task_type,
                episode_idx,
            )
        return prompt, episode_idx

    def _write_episode_video(self, source: BPTaskSource, episode_idx: int) -> Path:
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("导出 MP4 需要 PyAV；请安装 av。") from exc

        meta = LeRobotDatasetMetadata(str(source.dataset_path))
        camera_key = _select_main_camera(list(meta.camera_keys))
        dataset = LeRobotDataset(
            str(source.dataset_path),
            episodes=[episode_idx],
            video_backend=source.video_backend,
        )
        if len(dataset) == 0:
            raise ValueError("待记录的 BP episode 为空。")

        first_frame = _tensor_to_rgb_uint8(dataset[0][camera_key])
        height, width = first_frame.shape[:2]
        output = self.run_log_dir / (
            f"{_safe_name(source.task_type)}_episode_{episode_idx:06d}_{_safe_name(camera_key)}.mp4"
        )
        output.parent.mkdir(parents=True, exist_ok=True)

        with av.open(str(output), mode="w") as container:
            stream = container.add_stream("libx264", rate=max(1, round(float(meta.fps))))
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            for index in range(len(dataset)):
                frame = first_frame if index == 0 else _tensor_to_rgb_uint8(dataset[index][camera_key])
                if frame.shape[:2] != (height, width):
                    raise ValueError(
                        f"episode 内图像尺寸不一致: 首帧={(height, width)}, frame[{index}]={frame.shape[:2]}"
                    )
                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return output


class DebugBPVAPolicyService(BPVAPolicyService):
    def __init__(self, args: BPVAServeArgs, run_log_dir: Path) -> None:
        super().__init__(args)
        production_cache = self.bp_cache
        self.bp_cache = LoggingBehaviorPromptCache(
            production_cache.sources,
            config=production_cache.config,
            state_stats=self.state_stats,
            action_stats=self.action_stats,
            action_mode=self.action_mode,
            run_log_dir=run_log_dir,
        )
        self._metadata["bp_video_logging"] = True
        self._metadata["bplogs_run_dir"] = str(run_log_dir)


def main(args: DebugServeArgs) -> None:
    run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f_%z")
    run_log_dir = args.bplogs_dir / run_id
    run_log_dir.mkdir(parents=True, exist_ok=False)
    logging.info("启动参数:\n%s", json.dumps(asdict(args.serve), indent=2, ensure_ascii=False))
    logging.info("本次启动的 BP 视频目录: %s", run_log_dir)
    policy = DebugBPVAPolicyService(args.serve, run_log_dir)
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "unknown"
    logging.info("BPVA Debug WebSocket 服务 | host=%s ip=%s port=%d", args.serve.host, local_ip, args.serve.port)
    logging.info("Server metadata:\n%s", json.dumps(policy.metadata, indent=2, ensure_ascii=False))
    WebsocketPolicyServer(
        policy=policy,
        host=args.serve.host,
        port=args.serve.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    main(parse_args())
