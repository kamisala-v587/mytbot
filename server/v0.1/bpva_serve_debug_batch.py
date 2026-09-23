#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Concurrent BPVA dynamic-batch server with behavior-prompt MP4 logging."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from bpva_serve import BPVAServeArgs, parse_args as parse_production_args
from bpva_serve_batch import BatchedBPVAPolicyService
from bpva_serve_debug import DEFAULT_BPLOGS_DIR, LoggingBehaviorPromptCache
from serve_lerobot_policy_batch import BatchOptions, run_batch_server


@dataclass
class DebugBatchBPVAServeArgs:
    serve: BPVAServeArgs
    batch: BatchOptions
    bplogs_dir: Path


def parse_args() -> DebugBatchBPVAServeArgs:
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--max_batch_size", type=int, default=8)
    extra_parser.add_argument("--batch_wait_ms", type=float, default=10.0)
    extra_parser.add_argument("--queue_size", type=int, default=64)
    extra_parser.add_argument("--bplogs_dir", default=str(DEFAULT_BPLOGS_DIR))
    extra, production_argv = extra_parser.parse_known_args()
    if extra.max_batch_size < 1:
        extra_parser.error("--max_batch_size 必须 >= 1")
    if extra.batch_wait_ms < 0:
        extra_parser.error("--batch_wait_ms 必须 >= 0")
    if extra.queue_size < extra.max_batch_size:
        extra_parser.error("--queue_size 必须 >= --max_batch_size")

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *production_argv]
        serve_args = parse_production_args()
    finally:
        sys.argv = original_argv
    return DebugBatchBPVAServeArgs(
        serve=serve_args,
        batch=BatchOptions(extra.max_batch_size, extra.batch_wait_ms, extra.queue_size),
        bplogs_dir=Path(extra.bplogs_dir).expanduser().resolve(),
    )


class DebugBatchedBPVAPolicyService(BatchedBPVAPolicyService):
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


def main(args: DebugBatchBPVAServeArgs) -> None:
    run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f_%z")
    run_log_dir = args.bplogs_dir / run_id
    run_log_dir.mkdir(parents=True, exist_ok=False)
    logging.info("启动参数:\n%s", json.dumps(asdict(args.serve), indent=2, ensure_ascii=False))
    logging.info("本次启动的 BP 视频目录: %s", run_log_dir)
    policy = DebugBatchedBPVAPolicyService(args.serve, run_log_dir)
    run_batch_server(policy, args.serve, args.batch)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    main(parse_args())
