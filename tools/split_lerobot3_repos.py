#!/usr/bin/env python3
"""Batch split LeRobot v3 datasets with the locally installed ``lerobot3`` package.

Edit the uppercase constants below for normal use, or pass CLI arguments to override them.
The default sampling mode keeps the first ``ceil(total_episodes * SAMPLE_RATIO)`` episodes
for each repo. For example, 50 episodes with ``SAMPLE_RATIO = 0.1`` keeps episodes 0..4.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# =========================
# Editable default settings
# cd /home/jovyan/workspace/mytbot
# /home/jovyan/conda-envs/bptbot/bin/python tools/split_lerobot3_repos.py
# =========================
REPO_ID_FILE = Path("/home/jovyan/workspace/mytbot/configs/ds_ids/B200/RoboTwin-LeRobot-v3.0.txt")
OUTPUT_PREFIX = Path("/home/jovyan/workspace/data/cache")
SAMPLE_RATIO = 0.1
SPLIT_NAME = "keep"
DATA_FILES_SIZE_IN_MB = 5
VIDEO_FILES_SIZE_IN_MB = 5
MAX_WORKERS = 50
SKIP_EXISTING = True
OVERWRITE = False
DRY_RUN = False
HF_HUB_OFFLINE = True
TRANSFORMERS_OFFLINE = True


@dataclass(frozen=True)
class SplitResult:
    repo: str
    status: str
    output_dir: Path
    total_episodes: int | None = None
    kept_episodes: int | None = None
    error: str | None = None


def parse_repo_list(path: Path) -> list[str]:
    repos: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        repo = line.strip()
        if not repo or repo.startswith("#"):
            continue
        repos.append(repo)
    return repos


def repo_output_dir(output_prefix: Path, repo: str) -> Path:
    repo_path = Path(repo)
    if repo_path.is_absolute():
        return output_prefix / repo_path.relative_to("/")
    return output_prefix / repo


def local_repo_id(repo: str) -> str:
    repo_path = Path(repo)
    if not repo_path.is_absolute():
        return repo

    parts = [part for part in repo_path.parts if part != "/"]
    if len(parts) >= 2:
        namespace = sanitize_repo_id_part(parts[-2])
        name = sanitize_repo_id_part(parts[-1])
        return f"{namespace}/{name}"
    return sanitize_repo_id_part(repo_path.name or "local_dataset")


def sanitize_repo_id_part(value: str) -> str:
    # Hugging Face repo ids allow alphanumeric plus '.', '_' and '-'.
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return cleaned or "dataset"


def first_episode_indices(total_episodes: int, sample_ratio: float) -> list[int]:
    if total_episodes <= 0:
        raise ValueError("dataset has no episodes")
    if not 0 < sample_ratio <= 1:
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")
    keep_count = max(1, math.ceil(total_episodes * sample_ratio))
    keep_count = min(keep_count, total_episodes)
    return list(range(keep_count))


def ensure_offline_env(hf_hub_offline: bool, transformers_offline: bool) -> None:
    if hf_hub_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if transformers_offline:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def split_one_repo(
    repo: str,
    output_prefix: Path,
    sample_ratio: float,
    split_name: str,
    data_files_size_in_mb: int,
    video_files_size_in_mb: int,
    skip_existing: bool,
    overwrite: bool,
    dry_run: bool,
) -> SplitResult:
    from lerobot3.datasets.dataset_tools import split_dataset
    from lerobot3.datasets.lerobot_dataset import LeRobotDataset

    output_dir = repo_output_dir(output_prefix, repo)
    split_output_dir = output_dir / split_name

    if split_output_dir.exists():
        if overwrite:
            if dry_run:
                return SplitResult(repo, "would_overwrite", split_output_dir)
            shutil.rmtree(split_output_dir)
        elif skip_existing:
            return SplitResult(repo, "skipped_existing", split_output_dir)
        else:
            raise FileExistsError(f"output already exists: {split_output_dir}")

    repo_path = Path(repo)
    if repo_path.is_absolute() and not (repo_path / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"absolute path is not a LeRobot v3 dataset root, missing meta/info.json: {repo_path}"
        )

    dataset = LeRobotDataset(repo)
    if repo_path.is_absolute():
        # LeRobotDataset supports absolute paths, but split_dataset later uses
        # dataset.repo_id to create the output dataset metadata. Absolute paths
        # are invalid Hub repo ids, so normalize only after local loading.
        dataset.repo_id = local_repo_id(repo)
        dataset.meta.repo_id = dataset.repo_id

    # split_dataset copies these limits from the source metadata, so override them
    # before creating the split dataset instead of relying on lerobot3 defaults.
    dataset.meta.info.data_files_size_in_mb = data_files_size_in_mb
    dataset.meta.info.video_files_size_in_mb = video_files_size_in_mb
    episodes = first_episode_indices(dataset.meta.total_episodes, sample_ratio)

    if dry_run:
        return SplitResult(
            repo=repo,
            status="dry_run",
            output_dir=split_output_dir,
            total_episodes=dataset.meta.total_episodes,
            kept_episodes=len(episodes),
        )

    split_dataset(
        dataset,
        splits={split_name: episodes},
        output_dir=output_dir,
    )
    return SplitResult(
        repo=repo,
        status="generated",
        output_dir=split_output_dir,
        total_episodes=dataset.meta.total_episodes,
        kept_episodes=len(episodes),
    )


def iter_results(
    repos: Iterable[str],
    output_prefix: Path,
    sample_ratio: float,
    split_name: str,
    data_files_size_in_mb: int,
    video_files_size_in_mb: int,
    max_workers: int,
    skip_existing: bool,
    overwrite: bool,
    dry_run: bool,
) -> Iterable[SplitResult]:
    if max_workers <= 1:
        for repo in repos:
            try:
                yield split_one_repo(
                    repo,
                    output_prefix,
                    sample_ratio,
                    split_name,
                    data_files_size_in_mb,
                    video_files_size_in_mb,
                    skip_existing,
                    overwrite,
                    dry_run,
                )
            except Exception as exc:  # noqa: BLE001 - keep batch jobs moving across repos.
                yield SplitResult(repo, "failed", repo_output_dir(output_prefix, repo) / split_name, error=repr(exc))
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_repo = {
            executor.submit(
                split_one_repo,
                repo,
                output_prefix,
                sample_ratio,
                split_name,
                data_files_size_in_mb,
                video_files_size_in_mb,
                skip_existing,
                overwrite,
                dry_run,
            ): repo
            for repo in repos
        }
        for future in as_completed(future_to_repo):
            repo = future_to_repo[future]
            try:
                yield future.result()
            except Exception as exc:  # noqa: BLE001 - report per-repo failure without stopping others.
                yield SplitResult(repo, "failed", repo_output_dir(output_prefix, repo) / split_name, error=repr(exc))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch split LeRobot v3 datasets with lerobot3.")
    parser.add_argument("--repo-id-file", type=Path, default=REPO_ID_FILE)
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    parser.add_argument("--sample-ratio", type=float, default=SAMPLE_RATIO)
    parser.add_argument("--split-name", default=SPLIT_NAME)
    parser.add_argument("--data-files-size-in-mb", type=int, default=DATA_FILES_SIZE_IN_MB)
    parser.add_argument("--video-files-size-in-mb", type=int, default=VIDEO_FILES_SIZE_IN_MB)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=SKIP_EXISTING)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=OVERWRITE)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=DRY_RUN)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_offline_env(HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE)

    if args.overwrite and args.skip_existing:
        logging.info("--overwrite is enabled; existing outputs will be replaced instead of skipped.")
        args.skip_existing = False

    repos = parse_repo_list(args.repo_id_file.expanduser().resolve())
    if not repos:
        logging.error("No repos found in %s", args.repo_id_file)
        return 2

    args.output_prefix.expanduser().mkdir(parents=True, exist_ok=True)
    logging.info("Loaded %d repos from %s", len(repos), args.repo_id_file)
    logging.info("Output prefix: %s", args.output_prefix)
    logging.info("Sample ratio: %.6g, split name: %s, data MB: %d, video MB: %d, workers: %d", args.sample_ratio, args.split_name, args.data_files_size_in_mb, args.video_files_size_in_mb, args.max_workers)

    failed = 0
    for result in iter_results(
        repos=repos,
        output_prefix=args.output_prefix.expanduser().resolve(),
        sample_ratio=args.sample_ratio,
        split_name=args.split_name,
        data_files_size_in_mb=args.data_files_size_in_mb,
        video_files_size_in_mb=args.video_files_size_in_mb,
        max_workers=args.max_workers,
        skip_existing=args.skip_existing,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    ):
        if result.status == "failed":
            failed += 1
            logging.error("[%s] %s -> %s: %s", result.status, result.repo, result.output_dir, result.error)
        else:
            logging.info(
                "[%s] %s -> %s (%s/%s episodes)",
                result.status,
                result.repo,
                result.output_dir,
                result.kept_episodes if result.kept_episodes is not None else "?",
                result.total_episodes if result.total_episodes is not None else "?",
            )

    if failed:
        logging.error("Finished with %d failed repos.", failed)
        return 1
    logging.info("Finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
