"""数据集指纹与兼容缓存。"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

CACHE_VERSION = 2
ALGORITHM_VERSION = "norm-stats-v2"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def atomic_write_json(path: Path, value: Any) -> None:
    """同目录临时文件 + replace，避免中断留下半个 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def build_fingerprint(repo_id: str, repo_path: Path, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    repo_path = repo_path.resolve()
    info_path = repo_path / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"缺少数据集元信息：{info_path}")
    tracked = {info_path}
    tracked.update((repo_path / "meta").glob("episodes*"))
    tracked.update((repo_path / "data").rglob("*.parquet"))
    files = []
    for path in sorted((p for p in tracked if p.is_file()), key=lambda p: p.relative_to(repo_path).as_posix()):
        stat = path.stat()
        files.append({"path": path.relative_to(repo_path).as_posix(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    info_bytes = info_path.read_bytes()
    info = json.loads(info_bytes)
    fingerprint = {
        "cache_version": CACHE_VERSION, "algorithm_version": ALGORITHM_VERSION,
        "repo_id": repo_id, "repo_path": str(repo_path),
        "info_sha256": hashlib.sha256(info_bytes).hexdigest(), "files": files,
        "params": params, "total_frames": int(info.get("total_frames", 0)),
        "total_episodes": int(info.get("total_episodes", 0)),
    }
    dataset_id = hashlib.sha256(canonical_json(fingerprint)).hexdigest()
    return dataset_id, fingerprint


def _valid_result(value: Any, repo_id: str) -> bool:
    return isinstance(value, dict) and value.get("repo_id") == repo_id and isinstance(value.get("payload"), dict)


def _legacy_fingerprint_matches(old: dict[str, Any], current: dict[str, Any]) -> bool:
    """旧 v1 指纹字段较少；逐字段验证其 action/config/data 签名。"""
    if old.get("repo_path") != current.get("repo_path") or old.get("repo_id") != current.get("repo_id"):
        return False
    params = current["params"]
    if old.get("action_mode") != params.get("action_mode"):
        return False
    # abs 的 chunk/采样参数不影响新算法；但旧缓存仅在未启用采样时才是全帧 abs。
    if params.get("action_mode") == "abs":
        if old.get("max_chunks_per_episode") is not None or old.get("max_chunks_per_repo") is not None:
            return False
    else:
        for key in ("chunk_size", "max_chunks_per_episode", "max_chunks_per_repo", "sample_seed"):
            if old.get(key) != params.get(key):
                return False
    if list(old.get("skip_action_robot_types", [])) != list(params.get("skip_action_robot_types", [])):
        return False
    if old.get("info_sha256") != current.get("info_sha256"):
        return False
    by_path = {f["path"]: f for f in current["files"]}
    ordered_paths = ["meta/info.json"]
    ordered_paths += sorted(path for path in by_path if path.startswith("data/") and path.endswith(".parquet"))
    ordered_paths += sorted(path for path in by_path if path.startswith("meta/episodes"))
    manifest = [(path, by_path[path]["size"], by_path[path]["mtime_ns"]) for path in ordered_paths if path in by_path]
    signature = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return old.get("data_signature") == signature


def load_cache(cache_root: Path, dataset_id: str, fingerprint: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    new_dir = cache_root / "datasets" / dataset_id
    payload_path = new_dir / "stats_payload.json"
    metadata_path = new_dir / "metadata.json"
    if payload_path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            result = json.loads(payload_path.read_text(encoding="utf-8"))
            payload_sha256 = hashlib.sha256(canonical_json(result)).hexdigest()
            if (metadata.get("fingerprint") == fingerprint
                    and metadata.get("payload_sha256") == payload_sha256
                    and _valid_result(result, fingerprint["repo_id"])):
                return result, "new"
        except (OSError, json.JSONDecodeError):
            pass
    datasets_dir = cache_root / "datasets"
    if not datasets_dir.is_dir():
        return None, None
    # 目录名不可作为旧缓存身份依据，必须扫描并校验 payload 内 fingerprint。
    for candidate in datasets_dir.glob("*/stats_payload.json"):
        if candidate == payload_path:
            continue
        try:
            wrapper = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        old_fp, result = wrapper.get("fingerprint"), wrapper.get("result")
        if isinstance(old_fp, dict) and _valid_result(result, fingerprint["repo_id"]):
            if old_fp == fingerprint or _legacy_fingerprint_matches(old_fp, fingerprint):
                return result, "legacy"
    return None, None


def write_cache(cache_root: Path, dataset_id: str, fingerprint: dict[str, Any], result: dict[str, Any], *, migrated: bool = False) -> None:
    directory = cache_root / "datasets" / dataset_id
    metadata = {
        "dataset_id": dataset_id, "fingerprint": fingerprint, "repo_id": result["repo_id"],
        "schema": {key: result.get(key) for key in ("keys", "shapes", "mapping", "mask")},
        "compute": result.get("compute", {}), "migrated_from_legacy": migrated,
        # metadata 最后写入，摘要充当两文件事务的提交标记。
        "payload_sha256": hashlib.sha256(canonical_json(result)).hexdigest(),
    }
    atomic_write_json(directory / "stats_payload.json", result)
    atomic_write_json(directory / "metadata.json", metadata)
