#!/usr/bin/env python3
"""Prepare sampled LeRobot v3 caches and their BPVAv2 cache-root mapping.

cd /home/jovyan/workspace/mytbot
/home/jovyan/conda-envs/bptbot/bin/python tools/prepare_bpvav2_cache.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

REPO_ID_FILE = Path("/home/jovyan/workspace/mytbot/configs/ds_ids/B200/RoboTwin-LeRobot-v3.0.txt")
OUTPUT_PREFIX = Path("/home/jovyan/workspace/data/cache_1")
SAMPLE_RATIO = 0.1
SPLIT_NAME = "keep"
DATA_FILES_SIZE_IN_MB = 5
VIDEO_FILES_SIZE_IN_MB = 10  # 视频最大长度，此外，每个轨迹仅一个mp4文件
MAX_WORKERS = 32                           
SKIP_EXISTING = True                   
OVERWRITE = False                 
DRY_RUN = False
HF_HUB_OFFLINE = True
TRANSFORMERS_OFFLINE = True
YAML_OUTPUT = Path("/home/jovyan/workspace/mytbot/configs/B200/bp_cache_roboTwin_1.yaml")


@dataclass(frozen=True)
class SplitResult:
    repo: str
    status: str
    output_dir: Path
    total_episodes: int | None = None
    kept_episodes: int | None = None
    error: str | None = None


def normalize_repo(repo: str) -> str:
    value = repo.strip()
    path = Path(value).expanduser()
    return str(path.resolve(strict=False)) if path.is_absolute() else value


def parse_repo_list(path: Path) -> list[str]:
    repos: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        repo = line.strip()
        if not repo or repo.startswith("#"):
            continue
        repo = normalize_repo(repo)
        if repo in seen:
            raise ValueError(f"duplicate repo at {path}:{line_number}: {repo!r}")
        seen.add(repo)
        repos.append(repo)
    return repos


def repo_output_dir(output_prefix: Path, repo: str) -> Path:
    repo_path = Path(repo)
    if repo_path.is_absolute():
        return output_prefix / repo_path.relative_to("/")
    return output_prefix / repo


def sanitize_repo_id_part(value: str) -> str:
    # Hugging Face repo ids allow alphanumeric plus '.', '_' and '-'.
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return cleaned or "dataset"


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


def first_episode_indices(total_episodes: int, sample_ratio: float) -> list[int]:
    if total_episodes <= 0:
        raise ValueError("dataset has no episodes")
    if not 0 < sample_ratio <= 1:
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")
    return list(range(min(total_episodes, max(1, math.ceil(total_episodes * sample_ratio)))))


def ensure_offline_env(hf_hub_offline: bool, transformers_offline: bool) -> None:
    if hf_hub_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if transformers_offline:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _has_nonempty_file(root: Path, pattern: str) -> bool:
    return any(path.is_file() and path.stat().st_size > 0 for path in root.glob(pattern))


def validate_v3_dataset(root: Path) -> dict:
    info_path = root / "meta/info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid meta/info.json under {root}: {exc}") from exc
    if not isinstance(info, dict):
        raise ValueError(f"meta/info.json must contain an object: {info_path}")
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise ValueError(f"codebase_version is not v3 in {info_path}")
    if int(info.get("total_episodes", 0) or 0) <= 0 or int(info.get("total_frames", 0) or 0) <= 0:
        raise ValueError(f"total_episodes and total_frames must be positive in {info_path}")
    tasks = root / "meta/tasks.parquet"
    if not tasks.is_file() or tasks.stat().st_size == 0:
        raise ValueError(f"missing non-empty {tasks}")
    if not _has_nonempty_file(root / "meta/episodes", "**/*.parquet"):
        raise ValueError(f"no non-empty episode parquet under {root / 'meta/episodes'}")
    if not _has_nonempty_file(root / "data", "**/*.parquet"):
        raise ValueError(f"no non-empty data parquet under {root / 'data'}")
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"features must be a mapping in {info_path}")
    for key, feature in features.items():
        if isinstance(feature, dict) and feature.get("dtype") == "video":
            if not _has_nonempty_file(root / "videos" / key, "**/*"):
                raise ValueError(f"no non-empty video files for feature {key!r} under {root}")
    return info


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
    split_output_dir = repo_output_dir(output_prefix, repo) / split_name
    if split_output_dir.exists():
        if overwrite:
            if dry_run:
                return SplitResult(repo, "would_overwrite", split_output_dir)
            shutil.rmtree(split_output_dir)
        elif skip_existing:
            info = validate_v3_dataset(split_output_dir)
            return SplitResult(
                repo, "skipped_valid_existing", split_output_dir,
                total_episodes=int(info["total_episodes"]), kept_episodes=int(info["total_episodes"]),
            )
        else:
            raise FileExistsError(f"output already exists: {split_output_dir}")

    from lerobot3.datasets.dataset_tools import split_dataset
    from lerobot3.datasets.lerobot_dataset import LeRobotDataset

    repo_path = Path(repo)
    if repo_path.is_absolute() and not (repo_path / "meta/info.json").is_file():
        raise FileNotFoundError(f"absolute dataset root is missing meta/info.json: {repo_path}")
    dataset = LeRobotDataset(repo)
    if repo_path.is_absolute():
        dataset.repo_id = local_repo_id(repo)
        dataset.meta.repo_id = dataset.repo_id

    # split_dataset copies these limits from source metadata.
    dataset.meta.info.data_files_size_in_mb = data_files_size_in_mb
    dataset.meta.info.video_files_size_in_mb = video_files_size_in_mb
    episodes = first_episode_indices(dataset.meta.total_episodes, sample_ratio)
    if dry_run:
        return SplitResult(repo, "dry_run", split_output_dir, dataset.meta.total_episodes, len(episodes))
    split_dataset(dataset, splits={split_name: episodes}, output_dir=repo_output_dir(output_prefix, repo))
    info = validate_v3_dataset(split_output_dir)
    return SplitResult(repo, "generated", split_output_dir, dataset.meta.total_episodes, int(info["total_episodes"]))


def iter_results(
    repos: Iterable[str], output_prefix: Path, sample_ratio: float, split_name: str,
    data_files_size_in_mb: int, video_files_size_in_mb: int,
    max_workers: int, skip_existing: bool, overwrite: bool, dry_run: bool,
) -> Iterable[SplitResult]:
    def run(repo: str) -> SplitResult:
        try:
            return split_one_repo(
                repo, output_prefix, sample_ratio, split_name,
                data_files_size_in_mb, video_files_size_in_mb,
                skip_existing, overwrite, dry_run,
            )
        except Exception as exc:  # noqa: BLE001
            return SplitResult(repo, "failed", repo_output_dir(output_prefix, repo) / split_name, error=repr(exc))

    if max_workers <= 1:
        for repo in repos:
            yield run(repo)
        return
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_repo = {executor.submit(run, repo): repo for repo in repos}
        for future in as_completed(future_to_repo):
            yield future.result()


def atomic_write_mapping(path: Path, mapping: dict[str, str]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yaml.safe_dump(mapping, stream, sort_keys=False, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare sampled LeRobot v3 BPVAv2 caches.")
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
    parser.add_argument("--yaml-output", type=Path, default=YAML_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_offline_env(HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE)
    if args.overwrite and args.skip_existing:
        logging.info("--overwrite enabled; existing outputs will be replaced.")
        args.skip_existing = False
    try:
        repos = parse_repo_list(args.repo_id_file.expanduser().resolve())
    except (OSError, ValueError) as exc:
        logging.error("Cannot load repo list: %s", exc)
        return 2
    if not repos:
        logging.error("No repos found in %s", args.repo_id_file)
        return 2
    output_prefix = args.output_prefix.expanduser().resolve(strict=False)
    if not args.dry_run:
        output_prefix.mkdir(parents=True, exist_ok=True)
    logging.info(
        "Sample ratio: %.6g, split name: %s, data MB: %d, video MB: %d, workers: %d",
        args.sample_ratio, args.split_name, args.data_files_size_in_mb,
        args.video_files_size_in_mb, args.max_workers,
    )
    results = list(iter_results(
        repos, output_prefix, args.sample_ratio, args.split_name,
        args.data_files_size_in_mb, args.video_files_size_in_mb, args.max_workers,
        args.skip_existing, args.overwrite, args.dry_run,
    ))
    by_repo = {result.repo: result for result in results}
    for repo in repos:
        result = by_repo[repo]
        log = logging.error if result.status == "failed" else logging.info
        log("[%s] %s -> %s%s", result.status, repo, result.output_dir,
            f": {result.error}" if result.error else "")
    failed = [result for result in results if result.status == "failed"]
    if failed:
        logging.error("Finished with %d failed repos; YAML was not written.", len(failed))
        return 1
    if args.dry_run:
        logging.info("Dry run successful; YAML was not written.")
        return 0
    mapping = {repo: str(by_repo[repo].output_dir.resolve()) for repo in repos}
    atomic_write_mapping(args.yaml_output, mapping)
    logging.info("Wrote %d cache mappings to %s", len(mapping), args.yaml_output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
