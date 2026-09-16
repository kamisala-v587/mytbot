#!/usr/bin/env python3
"""BP cache 唯一公开 Python 入口。

cd /home/jovyan/workspace/mytbot
PYTHONPATH=src:tools \
/home/jovyan/conda-envs/bptbot/bin/python \
-m tools.bp_cache.generate

"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
for path in (TOOLS_DIR, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bp_cache import BPConfig, run_pipeline  # noqa: E402

# 可直接修改的默认配置；命令行参数优先于这些常量。
REPO_ID_FILE = "/home/jovyan/workspace/mytbot/configs/ds_ids/B200/RoboTwin-LeRobot-v3.0.txt"
BP_CACHE_ROOT = REPO_ROOT / "bp_cache"
MAPPING_OUTPUT = BP_CACHE_ROOT / "mapping.yaml"
SAMPLE_RATIO = 0.1
SAMPLING_SCOPE = "dataset"
SAMPLING_MODE = "random"
SEED = 42
BP_CAMERA_KEYS = ("observation.images.image0",)
TARGET_VIDEO_MB = 5.0
SINGLE_EPISODE_SOFT_MB = 10.0
SINGLE_EPISODE_HARD_MB = 20.0
RESUME = False
OVERWRITE = False
DRY_RUN = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 dataset 抽样并生成 LeRobot v3 BP cache")
    parser.add_argument("--repo-id-file", type=Path, default=REPO_ID_FILE)
    parser.add_argument("--bp-cache-root", type=Path, default=BP_CACHE_ROOT)
    parser.add_argument("--mapping-output", type=Path, default=MAPPING_OUTPUT)
    parser.add_argument("--sample-ratio", type=float, default=SAMPLE_RATIO)
    parser.add_argument("--sampling-scope", choices=("dataset", "task"), default=SAMPLING_SCOPE)
    parser.add_argument("--sampling-mode", choices=("random", "first"), default=SAMPLING_MODE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bp-camera-keys", nargs="+", default=list(BP_CAMERA_KEYS), metavar="CANONICAL_KEY")
    parser.add_argument("--target-video-mb", type=float, default=TARGET_VIDEO_MB)
    parser.add_argument("--single-episode-soft-mb", type=float, default=SINGLE_EPISODE_SOFT_MB)
    parser.add_argument("--single-episode-hard-mb", type=float, default=SINGLE_EPISODE_HARD_MB)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=RESUME)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=DRY_RUN)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        args = parse_args(argv)
        return run_pipeline(BPConfig(
            repo_id_file=args.repo_id_file.expanduser().resolve(),
            bp_cache_root=args.bp_cache_root.expanduser().resolve(),
            mapping_output=args.mapping_output.expanduser().resolve(),
            sample_ratio=args.sample_ratio, sampling_scope=args.sampling_scope, sampling_mode=args.sampling_mode, seed=args.seed,
            bp_camera_keys=tuple(args.bp_camera_keys), target_video_mb=args.target_video_mb,
            single_episode_soft_mb=args.single_episode_soft_mb,
            single_episode_hard_mb=args.single_episode_hard_mb,
            resume=args.resume, overwrite=args.overwrite, dry_run=args.dry_run,
        ))
    except KeyboardInterrupt:
        return 130
    except (ValueError, FileNotFoundError) as exc:
        logging.error("配置错误：%s", exc)
        return 2
    except Exception:
        logging.exception("BP cache 工具启动失败")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
