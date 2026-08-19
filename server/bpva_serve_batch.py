#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Concurrent dynamic-batch WebSocket server for BPVA policies."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from bpva_serve import BPVAServeArgs, BPVAPolicyService, parse_args as parse_production_args
from serve_lerobot_policy_batch import (
    BatchOptions,
    BatchedTBotSA1PolicyService,
    _collate_nested,
    run_batch_server,
)


@dataclass
class BatchBPVAServeArgs:
    serve: BPVAServeArgs
    batch: BatchOptions


def parse_args() -> BatchBPVAServeArgs:
    batch_parser = argparse.ArgumentParser(add_help=False)
    batch_parser.add_argument("--max_batch_size", type=int, default=8)
    batch_parser.add_argument("--batch_wait_ms", type=float, default=10.0)
    batch_parser.add_argument("--queue_size", type=int, default=64)
    batch_args, production_argv = batch_parser.parse_known_args()
    if batch_args.max_batch_size < 1:
        batch_parser.error("--max_batch_size 必须 >= 1")
    if batch_args.batch_wait_ms < 0:
        batch_parser.error("--batch_wait_ms 必须 >= 0")
    if batch_args.queue_size < batch_args.max_batch_size:
        batch_parser.error("--queue_size 必须 >= --max_batch_size")

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *production_argv]
        serve_args = parse_production_args()
    finally:
        sys.argv = original_argv
    return BatchBPVAServeArgs(
        serve=serve_args,
        batch=BatchOptions(
            max_batch_size=batch_args.max_batch_size,
            batch_wait_ms=batch_args.batch_wait_ms,
            queue_size=batch_args.queue_size,
        ),
    )


class BatchedBPVAPolicyService(BPVAPolicyService, BatchedTBotSA1PolicyService):
    """BPVA preprocessing plus shared request-level batch inference."""

    def infer_batch(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not observations:
            return []
        prepared = [self._obs_to_inputs(obs) for obs in observations]
        inputs = _collate_nested([item[0] for item in prepared])
        states = [item[1] for item in prepared]

        with torch.inference_mode():
            action_pred, _ = self.policy.predict_action_chunk(inputs, decode_image=False)
        if action_pred.ndim != 3 or action_pred.shape[0] != len(observations):
            raise RuntimeError(
                f"策略输出 shape 异常: {tuple(action_pred.shape)}，"
                f"期望 ({len(observations)}, T, A)"
            )

        results: list[dict[str, Any]] = []
        for batch_idx, state in enumerate(states):
            model_action = action_pred[
                batch_idx, : self.infer_horizon, : self.target_action_dim
            ]
            action = self.unnormalize_action_fn({"action": model_action})["action"]
            model_action_np = model_action.detach().cpu().numpy().astype(np.float32)
            action_np = action.detach().cpu().numpy().astype(np.float32)
            if self.action_mode == "delta" and self.delta_mask is not None:
                state_pad = np.zeros_like(self.delta_mask, dtype=np.float32)
                usable_dims = min(len(state_pad), len(state))
                state_pad[:usable_dims] = state[:usable_dims]
                action_dims = min(action_np.shape[-1], len(self.delta_mask))
                action_np[:, :action_dims] += (
                    state_pad[None, :action_dims] * self.delta_mask[None, :action_dims]
                )
            results.append(
                {
                    "actions": action_np,
                    "action": action_np[0],
                    "model_actions": model_action_np,
                    "model_action": model_action_np[0],
                }
            )
        return results


def main(args: BatchBPVAServeArgs) -> None:
    logging.info("启动参数:\n%s", json.dumps(asdict(args), indent=2, ensure_ascii=False))
    policy = BatchedBPVAPolicyService(args.serve)
    run_batch_server(policy, args.serve, args.batch)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    main(parse_args())
