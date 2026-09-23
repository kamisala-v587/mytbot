#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BPVA/BPVAv2 dynamic-batch history server with BP trajectory MP4 logging."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime

from bpva_serve_debug_batch import DebugBatchBPVAServeArgs, parse_args
from bpva_serve_debug import LoggingBehaviorPromptCache
from bpva_serve_history_batch import HistoryBatchedBPVAPolicyService
from serve_lerobot_policy_batch import run_batch_server


class DebugHistoryBatchedBPVAPolicyService(HistoryBatchedBPVAPolicyService):
    """Strict two-frame history protocol plus one-time BP trajectory logging."""

    def __init__(self, args, run_log_dir) -> None:
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
    policy = DebugHistoryBatchedBPVAPolicyService(args.serve, run_log_dir)
    run_batch_server(policy, args.serve, args.batch)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    main(parse_args())
