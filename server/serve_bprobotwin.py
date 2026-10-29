#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BP RoboTwin policy server: one entry for demonstration-conditioned (BP) models and plain VLA models.

Clients (my_robotwin bp_tasks/eval_policy) send the candidate set's language as `task` and its id as
`bp_candidate_set`. BP-family adapters get their demonstration from that candidate set's LeRobot repo (the
"task_type -> dataset" mapping is generated from the BP RoboTwin index, see bprobotwin/routing.py); VLA adapters only
use the language. Every other option is the same as server/serve_policy.py (constants there, or the same CLI flags);
--bp_mapping_path is replaced by the generated candidate-set mapping.

.. code-block:: bash

   cd /vla/workspace/my_tbot
   python server/serve_bprobotwin.py --ckpt_path $CKPT --port 8000          # any adapter (policy auto by ckpt type)
   python server/serve_bprobotwin.py --policy demo_replay --port 8000       # no model: replay the demonstration
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import serve_policy  # noqa: E402  (also prepares the import path)
from bprobotwin.index import DEFAULT_ROBOTWIN_ROOT, load_index, write_mapping  # noqa: E402
from bprobotwin.routing import BPRoboTwinPolicy  # noqa: E402
from policy_runtime.common import serve  # noqa: E402
from policy_runtime.registry import create_policy  # noqa: E402

MAPPING_PATH = SERVER_DIR / "debug_logs" / "bprobotwin" / "candidate_set_mapping.yml"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--robotwin_root", default=str(DEFAULT_ROBOTWIN_ROOT))
    ap.add_argument("--replay_horizon", type=int, default=30, help="demo_replay: actions per response")
    ap.add_argument("--replay_seed", type=int, default=0, help="demo_replay: demonstration choice per episode_uid")
    own, rest = ap.parse_known_args(argv)
    policy_name = serve_policy._build_parser().parse_known_args(rest)[0].policy or serve_policy.POLICY

    index = load_index(own.robotwin_root)
    mapping = write_mapping(index, MAPPING_PATH)
    logging.info("BP RoboTwin index: %d candidate sets of %d tasks | mapping %s", len(index),
                 len({v["task"] for v in index.values()}), mapping)

    if policy_name == "demo_replay":
        from bprobotwin.demo_replay import DemoReplayPolicy

        config = serve_policy.build_config(rest + ["--ckpt_path", "demo_replay", "--no-debug"])
        inner = DemoReplayPolicy(index, horizon=own.replay_horizon, seed=own.replay_seed)
    else:
        config = serve_policy.build_config(rest)
        if config.bp_mapping_path and Path(config.bp_mapping_path).resolve() != mapping.resolve():
            logging.info("bp_mapping_path %s replaced by the candidate-set mapping", config.bp_mapping_path)
        config = replace(config, bp_mapping_path=str(mapping))
        inner = create_policy(config)
    logging.info("生效配置:\n%s", json.dumps(asdict(config), indent=2, ensure_ascii=False))
    serve(BPRoboTwinPolicy(inner, index, str(mapping)), config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    main()
