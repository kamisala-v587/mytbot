#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Concurrent BPVA dynamic-batch server with synthetic no-prompt behavior prompts.

This ablation server keeps the BPVA WebSocket protocol identical to
``bpva_serve_debug_batch.py`` but never loads behavior-prompt episodes from the
mapped datasets. Each task receives a cached synthetic behavior prompt whose
images, state, and action are zeros. By default the prompt mask is all False,
so BP chunks are treated as padding/no prompt by the model attention mask.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from dataclasses import asdict, dataclass
from typing import Any

import torch

from bpva_serve import BPTaskSource, BPVAServeArgs, parse_args as parse_production_args
from bpva_serve_batch import BatchedBPVAPolicyService
from serve_lerobot_policy_batch import BatchOptions, run_batch_server


@dataclass
class MaskBatchBPVAServeArgs:
    serve: BPVAServeArgs
    batch: BatchOptions
    prompt_mask_value: bool
    prompt_pixel_value: float


def parse_args() -> MaskBatchBPVAServeArgs:
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--max_batch_size", type=int, default=8)
    extra_parser.add_argument("--batch_wait_ms", type=float, default=10.0)
    extra_parser.add_argument("--queue_size", type=int, default=64)
    extra_parser.add_argument(
        "--prompt_mask_value",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether synthetic BP chunks are valid. Default False means no prompt tokens.",
    )
    extra_parser.add_argument(
        "--prompt_pixel_value",
        type=float,
        default=0.0,
        help="Synthetic BP image value before model dtype conversion, normally in [0, 1].",
    )
    extra, production_argv = extra_parser.parse_known_args()
    if extra.max_batch_size < 1:
        extra_parser.error("--max_batch_size 必须 >= 1")
    if extra.batch_wait_ms < 0:
        extra_parser.error("--batch_wait_ms 必须 >= 0")
    if extra.queue_size < extra.max_batch_size:
        extra_parser.error("--queue_size 必须 >= --max_batch_size")
    if not 0.0 <= extra.prompt_pixel_value <= 1.0:
        extra_parser.error("--prompt_pixel_value 必须在 [0, 1] 范围内")

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *production_argv]
        serve_args = parse_production_args()
    finally:
        sys.argv = original_argv
    return MaskBatchBPVAServeArgs(
        serve=serve_args,
        batch=BatchOptions(extra.max_batch_size, extra.batch_wait_ms, extra.queue_size),
        prompt_mask_value=bool(extra.prompt_mask_value),
        prompt_pixel_value=float(extra.prompt_pixel_value),
    )


class SyntheticMaskBehaviorPromptCache:
    """Cache synthetic BP tensors with the same schema as production prompts."""

    def __init__(
        self,
        sources: dict[str, BPTaskSource],
        *,
        config: Any,
        prompt_mask_value: bool,
        prompt_pixel_value: float,
    ) -> None:
        self.sources = dict(sources)
        self.config = config
        self.prompt_mask_value = bool(prompt_mask_value)
        self.prompt_pixel_value = float(prompt_pixel_value)
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    @property
    def cached_tasks(self) -> tuple[str, ...]:
        return tuple(sorted(self._cache))

    def get(self, task_type: str) -> dict[str, Any]:
        if task_type not in self.sources:
            configured = ", ".join(sorted(self.sources))
            raise KeyError(f"未知 task_type={task_type!r}；已配置任务: {configured}")
        with self._lock:
            if task_type not in self._cache:
                self._cache[task_type] = self._build(task_type)
                logging.info(
                    "Synthetic BP 已缓存 | task_type=%s | mask_value=%s | pixel_value=%.3f",
                    task_type,
                    self.prompt_mask_value,
                    self.prompt_pixel_value,
                )
            return self._cache[task_type]

    def _build(self, task_type: str) -> dict[str, Any]:
        num_chunks = int(getattr(self.config, "bp_num_chunks"))
        action_steps = int(getattr(self.config, "bp_action_chunk_size"))
        state_dim = int(getattr(self.config, "max_state_dim"))
        action_dim = int(getattr(self.config, "max_action_dim"))
        height, width = tuple(getattr(self.config, "image_resolution", (224, 224)))
        camera_keys = list(getattr(self.config, "bp_camera_keys"))

        prompt = {
            "images": {
                key: torch.full(
                    (num_chunks, 3, int(height), int(width)),
                    fill_value=self.prompt_pixel_value,
                    dtype=torch.float32,
                )
                for key in camera_keys
            },
            "state": torch.zeros(num_chunks, state_dim, dtype=torch.float32),
            "action": torch.zeros(num_chunks, action_steps, action_dim, dtype=torch.float32),
            "action_is_pad": torch.ones(num_chunks, action_steps, dtype=torch.bool),
            "mask": torch.full((num_chunks,), self.prompt_mask_value, dtype=torch.bool),
            "chunk_indices": torch.arange(num_chunks, dtype=torch.long),
        }
        self._validate(prompt, task_type)
        return prompt

    def _validate(self, prompt: dict[str, Any], task_type: str) -> None:
        num_chunks = int(getattr(self.config, "bp_num_chunks"))
        action_steps = int(getattr(self.config, "bp_action_chunk_size"))
        state_dim = int(getattr(self.config, "max_state_dim"))
        action_dim = int(getattr(self.config, "max_action_dim"))
        expected_images = set(getattr(self.config, "bp_camera_keys"))
        if set(prompt.get("images", {})) != expected_images:
            raise ValueError(f"task={task_type!r} BP 相机不匹配: {sorted(prompt.get('images', {}))}")
        for key, image in prompt["images"].items():
            if tuple(image.shape[:2]) != (num_chunks, 3):
                raise ValueError(f"task={task_type!r} BP 图像 {key} shape 非法: {tuple(image.shape)}")
        if tuple(prompt["state"].shape) != (num_chunks, state_dim):
            raise ValueError(f"task={task_type!r} BP state shape 非法: {tuple(prompt['state'].shape)}")
        if tuple(prompt["action"].shape) != (num_chunks, action_steps, action_dim):
            raise ValueError(f"task={task_type!r} BP action shape 非法: {tuple(prompt['action'].shape)}")
        if tuple(prompt["action_is_pad"].shape) != (num_chunks, action_steps):
            raise ValueError(f"task={task_type!r} BP action_is_pad shape 非法")
        if tuple(prompt["mask"].shape) != (num_chunks,):
            raise ValueError(f"task={task_type!r} BP mask shape 非法")


class MaskBatchedBPVAPolicyService(BatchedBPVAPolicyService):
    def __init__(self, args: BPVAServeArgs, *, prompt_mask_value: bool, prompt_pixel_value: float) -> None:
        super().__init__(args)
        production_cache = self.bp_cache
        self.bp_cache = SyntheticMaskBehaviorPromptCache(
            production_cache.sources,
            config=production_cache.config,
            prompt_mask_value=prompt_mask_value,
            prompt_pixel_value=prompt_pixel_value,
        )
        self._metadata["bp_selection"] = "synthetic_mask_ablation"
        self._metadata["bp_prompt_source"] = "zeros_no_dataset_episode"
        self._metadata["bp_prompt_mask_value"] = prompt_mask_value
        self._metadata["bp_prompt_pixel_value"] = prompt_pixel_value


def main(args: MaskBatchBPVAServeArgs) -> None:
    logging.info("启动参数:\n%s", json.dumps(asdict(args), indent=2, ensure_ascii=False))
    policy = MaskBatchedBPVAPolicyService(
        args.serve,
        prompt_mask_value=args.prompt_mask_value,
        prompt_pixel_value=args.prompt_pixel_value,
    )
    logging.info("Server metadata:\n%s", json.dumps(policy.metadata, indent=2, ensure_ascii=False))
    run_batch_server(policy, args.serve, args.batch)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    main(parse_args())
