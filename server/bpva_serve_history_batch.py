#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BPVA 动态 batch 服务：严格要求每个已提供相机携带两帧历史。"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from bpva_serve_batch import BatchedBPVAPolicyService, parse_args
from serve_lerobot_policy_batch import run_batch_server
from serve_lerobot_policy_history_batch import StrictTwoFrameHistoryMixin


class HistoryBatchedBPVAPolicyService(StrictTwoFrameHistoryMixin, BatchedBPVAPolicyService):
    """BPVA 预处理/动态合批与严格两帧输入协议的组合。"""


def main(args) -> None:
    logging.info("启动参数:\n%s", json.dumps(asdict(args), indent=2, ensure_ascii=False))
    policy = HistoryBatchedBPVAPolicyService(args.serve)
    run_batch_server(policy, args.serve, args.batch)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    main(parse_args())
