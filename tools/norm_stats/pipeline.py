"""缓存调度、并行计算与输出聚合。"""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.utils.constants import HF_LEROBOT_HOME

from .cache import atomic_write_json, build_fingerprint, load_cache, write_cache
from .compute import compute_one, enrich_cached_result
from .core import merge_payloads, validate_schema


@dataclass(frozen=True)
class PipelineConfig:
    repo_id_file: Path
    cache_root: Path
    output_format: str
    output_root: Path | None
    output_path: Path | None
    action_mode: str = "delta"
    chunk_size: int = 50
    num_workers: int = 8
    root: Path | None = None
    max_chunks_per_episode: int | None = None
    max_chunks_per_repo: int | None = None
    sample_seed: int = 42
    skip_action_robot_types: tuple[str, ...] = ()
    dry_run: bool = False


def repo_path(repo_id: str, root: Path | None) -> Path:
    path = Path(repo_id)
    return path.resolve() if path.is_absolute() else ((root / repo_id) if root else (HF_LEROBOT_HOME / repo_id)).resolve()


def read_repos(path: Path, root: Path | None) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"repo 列表不存在：{path}")
    repos = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not repos:
        raise ValueError(f"repo 列表为空：{path}")
    seen: dict[Path, str] = {}
    for repo_id in repos:
        canonical = repo_path(repo_id, root)
        if canonical in seen:
            raise ValueError(f"canonical 数据集路径重复：{seen[canonical]!r} 与 {repo_id!r} -> {canonical}")
        seen[canonical] = repo_id
    return repos


def _params(cfg: PipelineConfig) -> dict[str, Any]:
    """只纳入实际影响数值的参数；abs 与 chunk/采样参数无关。"""
    params: dict[str, Any] = {
        "action_mode": cfg.action_mode,
        "skip_action_robot_types": list(cfg.skip_action_robot_types),
    }
    if cfg.action_mode == "delta":
        params.update({
            "chunk_size": cfg.chunk_size,
            "max_chunks_per_episode": cfg.max_chunks_per_episode,
            "max_chunks_per_repo": cfg.max_chunks_per_repo,
            "sample_seed": cfg.sample_seed,
        })
    return params


def _job(repo_id: str, cfg: PipelineConfig) -> tuple:
    # abs worker 也显式清空无效采样参数，防止未来实现误用。
    per_episode = cfg.max_chunks_per_episode if cfg.action_mode == "delta" else None
    per_repo = cfg.max_chunks_per_repo if cfg.action_mode == "delta" else None
    effective_chunk_size = cfg.chunk_size if cfg.action_mode == "delta" else 1
    return (repo_id, cfg.action_mode, effective_chunk_size, str(cfg.root) if cfg.root else None,
            per_episode, per_repo, cfg.sample_seed, cfg.skip_action_robot_types)


def _write_group(results: list[dict[str, Any]], output_path: Path, cfg: PipelineConfig) -> None:
    validate_schema(results)
    output = merge_payloads(results)
    atomic_write_json(output_path, output)
    if cfg.output_format == "tbot":
        manifest = {
            "output_format": cfg.output_format, "action_mode": cfg.action_mode,
            "resolved_robot_type": results[0]["resolved_robot_type"],
            "repo_ids": [item["repo_id"] for item in results],
            "counts": {key: value.get("count") for key, value in output.items() if isinstance(value, dict)},
            "skipped_action_repos": [item["repo_id"] for item in results if item["compute"].get("skip_action")],
        }
        atomic_write_json(output_path.parent / "manifest.json", manifest)


def _write_outputs(results: list[dict[str, Any]], cfg: PipelineConfig) -> list[Path]:
    if cfg.output_format == "default":
        _write_group(results, cfg.output_path, cfg)
        return [cfg.output_path]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["resolved_robot_type"], []).append(result)
    paths = []
    for robot_type, group in sorted(grouped.items()):
        path = cfg.output_root / robot_type / cfg.action_mode / "stats.json"
        _write_group(group, path, cfg)
        paths.append(path)
    return paths


def _eta(done_work: int, total_work: int, elapsed: float) -> str:
    if done_work <= 0 or elapsed <= 0:
        return "未知"
    seconds = max(0.0, (total_work - done_work) / (done_work / elapsed))
    return f"{seconds:.0f}s"


def run_pipeline(cfg: PipelineConfig) -> None:
    if cfg.num_workers <= 0:
        raise ValueError("num-workers 必须为正数")
    if cfg.action_mode == "delta" and cfg.chunk_size <= 0:
        raise ValueError("delta 模式的 chunk-size 必须为正数")
    if cfg.action_mode == "delta":
        for name, value in (("max-chunks-per-episode", cfg.max_chunks_per_episode),
                            ("max-chunks-per-repo", cfg.max_chunks_per_repo)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} 必须为正数")
    repos = read_repos(cfg.repo_id_file, cfg.root)
    params = _params(cfg)
    plans = []
    results: dict[str, dict[str, Any]] = {}
    legacy_hits: set[str] = set()
    for repo_id in repos:
        dataset_id, fingerprint = build_fingerprint(repo_id, repo_path(repo_id, cfg.root), params)
        cached, cache_kind = load_cache(cfg.cache_root, dataset_id, fingerprint)
        plans.append((repo_id, dataset_id, fingerprint, cache_kind))
        if cached is not None:
            results[repo_id] = cached
            if cache_kind == "legacy":
                legacy_hits.add(repo_id)
    misses = [plan for plan in plans if plan[3] is None]
    total_work = sum(plan[2]["total_frames"] for plan in misses)
    print(f"数据集总数：{len(repos)}；缓存 hit：{len(repos) - len(misses)}；miss：{len(misses)}")
    print(f"预计 miss 工作量：{total_work:,} frames")
    for repo_id, _, fingerprint, kind in plans:
        print(f"  {'MISS' if kind is None else 'HIT '} {repo_id} ({fingerprint['total_frames']:,} frames)")
    if cfg.dry_run:
        print("dry-run：未计算，未写缓存或输出。")
        return
    plan_by_repo = {plan[0]: plan for plan in plans}
    # 正常运行时给旧 payload 补齐非视觉 schema，并移除旧视觉元信息后迁移。
    for repo_id in list(results):
        results[repo_id] = enrich_cached_result(results[repo_id], str(cfg.root) if cfg.root else None)
    for repo_id in legacy_hits:
        _, dataset_id, fingerprint, _ = plan_by_repo[repo_id]
        write_cache(cfg.cache_root, dataset_id, fingerprint, results[repo_id], migrated=True)
    start = time.monotonic()
    done_work = 0
    completed = 0
    def accept(result: dict[str, Any]) -> None:
        nonlocal done_work, completed
        repo_id = result["repo_id"]
        _, dataset_id, fingerprint, _ = plan_by_repo[repo_id]
        write_cache(cfg.cache_root, dataset_id, fingerprint, result)
        results[repo_id] = result
        completed += 1
        done_work += fingerprint["total_frames"]
        elapsed = time.monotonic() - start
        print(f"完成 {completed}/{len(misses)}：{repo_id}；耗时 {elapsed:.1f}s；ETA {_eta(done_work, total_work, elapsed)}", flush=True)
    jobs = [_job(plan[0], cfg) for plan in misses]
    if jobs and min(cfg.num_workers, len(jobs)) == 1:
        for job in jobs:
            accept(compute_one(job))
    elif jobs:
        with mp.get_context("spawn").Pool(min(cfg.num_workers, len(jobs))) as pool:
            for result in pool.imap_unordered(compute_one, jobs):
                accept(result)
    ordered = [results[repo_id] for repo_id in repos]
    paths = _write_outputs(ordered, cfg)
    elapsed = time.monotonic() - start
    print(f"全部完成：{len(repos)} 个数据集，耗时 {elapsed:.1f}s")
    for path in paths:
        print(f"输出：{path}")
