#!/usr/bin/env python3
"""Norm stats 唯一公开 Python 入口。"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
for path in (TOOLS_DIR, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from norm_stats.pipeline import PipelineConfig, run_pipeline  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="增量计算并聚合 LeRobot norm stats")
    parser.add_argument("--repo-id-file", "--repo_id_file", type=Path, required=True)
    parser.add_argument("--cache-root", "--cache_root", type=Path, default=REPO_ROOT / "norm_stats_cache")
    parser.add_argument("--output-format", "--output_format", choices=("tbot", "default"), default="tbot")
    parser.add_argument("--output-root", "--output_root", type=Path)
    parser.add_argument("--output-path", "--output_path", type=Path)
    parser.add_argument("--action-mode", "--action_mode", choices=("abs", "delta"), default="delta")
    parser.add_argument("--chunk-size", "--chunk_size", type=int, default=50)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=int(os.getenv("NUM_WORKERS", "8")))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--max-chunks-per-episode", "--max_chunks_per_episode", type=int)
    parser.add_argument("--max-chunks-per-repo", "--max_chunks_per_repo", type=int)
    parser.add_argument("--sample-seed", "--sample_seed", type=int, default=42)
    parser.add_argument("--skip-action-robot-types", "--skip_action_robot_types", nargs="*", default=[])
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    args = parser.parse_args(argv)
    if args.output_format == "default" and args.output_path is None:
        parser.error("--output-format default 必须提供 --output-path")
    if args.output_format == "tbot" and args.output_root is None:
        args.output_root = REPO_ROOT / "norm_stats"
    return args


def main() -> None:
    args = parse_args()
    run_pipeline(PipelineConfig(
        repo_id_file=args.repo_id_file.resolve(), cache_root=args.cache_root.resolve(),
        output_format=args.output_format,
        output_root=args.output_root.resolve() if args.output_root else None,
        output_path=args.output_path.resolve() if args.output_path else None,
        action_mode=args.action_mode, chunk_size=args.chunk_size, num_workers=args.num_workers,
        root=args.root.resolve() if args.root else None,
        max_chunks_per_episode=args.max_chunks_per_episode, max_chunks_per_repo=args.max_chunks_per_repo,
        sample_seed=args.sample_seed, skip_action_robot_types=tuple(args.skip_action_robot_types), dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
