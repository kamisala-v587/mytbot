from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
import sys
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot_github.episode_splitter import ProgressEvent, SplitResult

from .config import BPConfig
from .manifest import (ALGORITHM_VERSION, MANIFEST_NAME, SCHEMA_VERSION, atomic_write_json,
                       build_fingerprint, load_manifest, manifest_status,
                       recognized_for_source, source_signature)
from .mapping import atomic_write_mapping, cache_dir_name, load_mapping, read_repo_ids, source_identity

LOGGER = logging.getLogger("bp_cache")


@dataclass(frozen=True)
class Selection:
    episodes: tuple[int, ...]
    episode_tasks: dict[int, tuple[str, ...]]
    selected_by_task: dict[str, tuple[int, ...]]


def sample_episodes_by_task(
    episode_tasks: dict[int, tuple[str, ...] | list[str]], ratio: float, mode: str, seed: int, repo_key: str,
    sampling_scope: str = "task",
) -> Selection:
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    if mode not in {"random", "first"}:
        raise ValueError("mode must be random or first")
    if sampling_scope not in {"dataset", "task"}:
        raise ValueError("sampling_scope must be dataset or task")
    groups: dict[str, list[int]] = defaultdict(list)
    normalized: dict[int, tuple[str, ...]] = {}
    for episode, tasks in sorted(episode_tasks.items()):
        names = tuple(dict.fromkeys(str(task) for task in tasks if str(task)))
        if not names:
            raise ValueError(f"episode {episode} 缺少 task metadata")
        normalized[episode] = names
        for task in names:
            groups[task].append(episode)
    if not groups:
        raise ValueError("数据集没有可抽样的 task/episode")
    if sampling_scope == "dataset":
        episodes = sorted(normalized)
        count = max(1, math.ceil(len(episodes) * ratio))
        if mode == "first":
            selected = tuple(episodes[:count])
        else:
            import hashlib
            digest = hashlib.sha256(f"{seed}|{repo_key}".encode()).digest()
            selected = tuple(sorted(random.Random(int.from_bytes(digest[:8], "big")).sample(episodes, count)))
        chosen = {task: tuple(ep for ep in selected if task in normalized[ep]) for task in sorted(groups)}
        chosen = {task: values for task, values in chosen.items() if values}
        return Selection(selected, normalized, chosen)
    chosen: dict[str, tuple[int, ...]] = {}
    for task, episodes in sorted(groups.items()):
        count = max(1, math.ceil(len(episodes) * ratio))
        if mode == "first":
            values = episodes[:count]
        else:
            import hashlib
            digest = hashlib.sha256(f"{seed}|{repo_key}|{task}".encode()).digest()
            values = sorted(random.Random(int.from_bytes(digest[:8], "big")).sample(episodes, count))
        chosen[task] = tuple(values)
    # split_episodes 的输入顺序定义 cache episode index；固定按 source index 排序。
    selected = tuple(sorted({episode for values in chosen.values() for episode in values}))
    return Selection(selected, normalized, chosen)


def resolve_cameras(robot_type: str, features: dict[str, Any], canonical_keys: tuple[str, ...]) -> tuple[str, ...]:
    from lerobot.transforms.constants import get_image_mapping

    actual_to_canonical = get_image_mapping(robot_type, features)
    selected: list[str] = []
    for canonical in canonical_keys:
        matches = sorted(actual for actual, target in actual_to_canonical.items()
                         if target == canonical and actual in features)
        if len(matches) != 1:
            raise ValueError(f"canonical camera {canonical!r} 期望唯一实际映射，得到 {matches}")
        selected.append(matches[0])
    if len(set(selected)) != len(selected):
        raise ValueError(f"多个 canonical camera 映射到同一实际 camera：{selected}")
    return tuple(selected)


def _load_dataset(repo_id: str) -> LeRobotDataset:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    path = Path(repo_id).expanduser()
    return LeRobotDataset(str(path.resolve())) if path.is_absolute() else LeRobotDataset(repo_id)


def _episode_tasks(dataset: LeRobotDataset) -> dict[int, tuple[str, ...]]:
    output = {}
    for index in range(dataset.meta.total_episodes):
        episode = dataset.meta.episodes[index]
        output[index] = tuple(str(task) for task in episode.get("tasks", []) if str(task))
    return output


def _validate_output(root: Path, result: SplitResult, cameras: tuple[str, ...]) -> None:
    required = [root / "meta" / name for name in ("info.json", "stats.json", "tasks.parquet")]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    episode_files = list((root / "meta" / "episodes").rglob("*.parquet"))
    data_files = list((root / "data").rglob("*.parquet"))
    if not episode_files:
        missing.append("meta/episodes/**/*.parquet")
    if not data_files:
        missing.append("data/**/*.parquet")
    if result.dataset.meta.info.get("codebase_version") != "v3.0":
        missing.append("codebase_version=v3.0")
    if result.dataset.meta.total_episodes <= 0:
        missing.append("non-empty episodes")
    for camera in cameras:
        if result.dataset.meta.features[camera].get("dtype") == "video":
            paths = {root / result.dataset.meta.get_video_file_path(index, camera)
                     for index in range(result.dataset.meta.total_episodes)}
            if not paths or any(not path.is_file() or path.stat().st_size == 0 for path in paths):
                missing.append(f"required videos for {camera}")
    if missing:
        raise ValueError(f"cache 输出严格验证失败：{missing}")


def _progress_logger(repo_id: str) -> Callable[[ProgressEvent], None]:
    last: dict[str, tuple[int, int | None]] = {}
    def report(event: ProgressEvent) -> None:
        marker = (event.completed, event.total)
        if last.get(event.stage) == marker:
            return
        last[event.stage] = marker
        details = [f"stage={event.stage}", f"repo={repo_id}"]
        if event.total is not None:
            details.append(f"progress={event.completed}/{event.total}")
        for key in ("task", "source_episode", "camera", "file"):
            value = getattr(event, key)
            if value is not None:
                details.append(f"{key}={value}")
        LOGGER.info(" ".join(details))
    return report


def _manifest_payload(source_repo: str, cache_repo_id: str, fingerprint: dict[str, Any], selection: Selection,
                      cameras: tuple[str, ...], result: SplitResult) -> dict[str, Any]:
    source_task_names = {str(old): str(result.dataset.meta.tasks.iloc[new].name)
                         for old, new in result.task_mapping.items()}
    return {
        "schema_version": SCHEMA_VERSION, "algorithm_version": ALGORITHM_VERSION,
        "source_repo": source_repo, "cache_repo_id": cache_repo_id,
        "source_root_signature": fingerprint["source_signature"],
        "config": fingerprint["config"], "fingerprint": fingerprint,
        "selected_source_episodes": [
            {"episode": episode, "tasks": list(selection.episode_tasks[episode])}
            for episode in selection.episodes
        ],
        "selected_by_task": {task: list(values) for task, values in selection.selected_by_task.items()},
        "source_to_cache_episode": {str(k): v for k, v in result.episode_mapping.items()},
        "source_to_cache_task": {str(k): v for k, v in result.task_mapping.items()},
        "source_task_names": source_task_names,
        "actual_camera_keys": list(cameras),
        "video_files": [vars(item) | {"source_episodes": list(item.source_episodes),
                                      "new_episodes": list(item.new_episodes)} for item in result.video_files],
        "warnings": [vars(item) for item in result.warnings],
        "fallbacks": result.encoder_fallbacks,
        "complete": True,
    }


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()



def publish_cache(building: Path, final: Path) -> None:
    """Atomically publish a validated build while preserving an existing cache on failure."""
    backup = Path(f"{final}.backup-{uuid.uuid4().hex}")
    moved_old = False
    try:
        if final.exists():
            os.replace(final, backup)
            moved_old = True
        os.replace(building, final)
    except Exception:
        if moved_old and backup.exists():
            if final.exists():
                shutil.rmtree(final)
            os.replace(backup, final)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)

def run_pipeline(cfg: BPConfig) -> int:
    cfg.validate()
    repos = read_repo_ids(cfg.repo_id_file)
    if cfg.dry_run:
        for repo in repos:
            LOGGER.info("dry-run repo=%s cache=%s", repo, cfg.bp_cache_root / cache_dir_name(repo))
        LOGGER.info("dry-run：共 %d 个 repo；未加载数据集，未写任何文件", len(repos))
        return 0

    cfg.bp_cache_root.mkdir(parents=True, exist_ok=True)
    run_dir = cfg.bp_cache_root / "_bp_cache_runs" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(run_dir / "effective_config.json", cfg.effective_dict())
    mapping = load_mapping(cfg.mapping_output)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()
    progress = tqdm(repos, desc="BP cache repos", unit="repo", disable=not sys.stderr.isatty())
    try:
        for repo in progress:
            progress.set_postfix_str(repo)
            stage = "load"
            final = cfg.bp_cache_root / cache_dir_name(repo)
            building = Path(f"{final}.building")
            source_repo = source_identity(repo)
            cache_repo_id = f"bp-cache/{cache_dir_name(source_repo)}"
            try:
                dataset = _load_dataset(repo)
                signature = source_signature(Path(dataset.root))
                fingerprint = build_fingerprint(source_repo, signature, cfg.effective_dict())
                existing = load_manifest(final / MANIFEST_NAME) if final.exists() else None
                status = manifest_status(existing, fingerprint)
                if final.exists():
                    if cfg.resume and status == "hit":
                        mapping[source_repo] = str(final.resolve())
                        atomic_write_mapping(cfg.mapping_output, mapping)
                        item = {"repo": repo, "status": "resumed", "cache": str(final.resolve())}
                        results.append(item)
                        _append_jsonl(run_dir / "results.jsonl", item)
                        LOGGER.info("resume hit repo=%s cache=%s", repo, final)
                        continue
                    if not cfg.overwrite:
                        raise FileExistsError(f"cache 已存在且状态为 {status}；使用匹配的 --resume 或安全的 --overwrite：{final}")
                    if not recognized_for_source(existing, source_repo):
                        raise PermissionError(f"拒绝覆盖非本工具产物或 source 不匹配目录：{final}")
                if building.exists():
                    shutil.rmtree(building)

                stage = "camera"
                cameras = resolve_cameras(dataset.meta.robot_type, dataset.meta.features, cfg.bp_camera_keys)
                LOGGER.info("stage=camera repo=%s canonical=%s actual=%s", repo, cfg.bp_camera_keys, cameras)
                stage = "sampling"
                selection = sample_episodes_by_task(_episode_tasks(dataset), cfg.sample_ratio,
                                                    cfg.sampling_mode, cfg.seed, source_repo, cfg.sampling_scope)
                LOGGER.info("stage=sampling repo=%s tasks=%d episodes=%d scope=%s", repo,
                            len(selection.selected_by_task), len(selection.episodes), cfg.sampling_scope)
                stage = "split"
                from lerobot_github.episode_splitter import split_episodes

                result = split_episodes(
                    dataset, selection.episodes, building, cache_repo_id, cameras,
                    cfg.target_video_mb, cfg.single_episode_soft_mb, cfg.single_episode_hard_mb,
                    _progress_logger(repo),
                )
                stage = "validate"
                _validate_output(building, result, cameras)
                stage = "manifest"
                manifest = _manifest_payload(source_repo, cache_repo_id, fingerprint, selection, cameras, result)
                atomic_write_json(building / MANIFEST_NAME, manifest)
                publish_cache(building, final)
                mapping[source_repo] = str(final.resolve())
                atomic_write_mapping(cfg.mapping_output, mapping)
                item = {"repo": repo, "status": "generated", "cache": str(final.resolve()),
                        "tasks": len(selection.selected_by_task), "episodes": len(selection.episodes),
                        "cameras": list(cameras), "video_files": len(result.video_files)}
                results.append(item)
                _append_jsonl(run_dir / "results.jsonl", item)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                failure = {"repo": repo, "stage": stage, "error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()}
                failures.append(failure)
                _append_jsonl(run_dir / "failures.jsonl", failure)
                LOGGER.error("repo 失败 stage=%s repo=%s: %s", stage, repo, exc)
    except KeyboardInterrupt:
        summary = {"complete": False, "interrupted": True, "total": len(repos),
                   "succeeded": len(results), "failed": len(failures), "elapsed_s": time.time() - started}
        atomic_write_json(run_dir / "summary.json", summary)
        return 130
    finally:
        progress.close()
    summary = {"complete": not failures, "total": len(repos), "succeeded": len(results),
               "failed": len(failures), "elapsed_s": time.time() - started,
               "results": results, "failures": failures}
    atomic_write_json(run_dir / "summary.json", summary)
    LOGGER.info("完成：成功 %d/%d，失败 %d；记录目录 %s", len(results), len(repos), len(failures), run_dir)
    return 1 if failures else 0
