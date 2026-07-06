#!/usr/bin/env python3
"""CLI for norm stats init / update. Replaces bash wrappers."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
SRC = REPO_ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from norm_stats_pipeline import PipelineConfig, run_init, run_update  # noqa: E402


def _add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p.add_argument("--repo-id-file", type=Path, default=None, help="Default: .config/ds_ids/pretrain_data.txt")
    p.add_argument("--policy-file", type=Path, default=None, help="Default: .config/norm_stats_policy.json")
    p.add_argument("--cache-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--action-mode", choices=["abs", "delta"], default="delta")
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "8")))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")


def _build_cfg(args: argparse.Namespace, *, force: bool) -> PipelineConfig:
    overrides = {
        "repo_id_file": args.repo_id_file,
        "policy_file": args.policy_file,
        "cache_root": args.cache_root,
        "output_root": args.output_root,
        "action_mode": args.action_mode,
        "chunk_size": args.chunk_size,
        "num_workers": args.num_workers,
        "limit": args.limit,
        "dry_run": args.dry_run,
        "force": force,
    }
    return PipelineConfig.defaults(args.repo_root, **{k: v for k, v in overrides.items() if v is not None})


def main() -> None:
    parser = argparse.ArgumentParser(description="TBot norm stats: per-dataset cache + robot_type merge")
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="Full run: compute missing per-dataset stats, then aggregate")
    _add_shared_args(init_p)
    init_p.add_argument("--force", action="store_true", help="Recompute every per-dataset cache entry")

    update_p = sub.add_parser("update", help="Incremental: only new/stale datasets, then re-aggregate")
    _add_shared_args(update_p)

    args = parser.parse_args()
    if args.command == "init":
        run_init(_build_cfg(args, force=args.force))
    else:
        run_update(_build_cfg(args, force=False))


if __name__ == "__main__":
    main()
