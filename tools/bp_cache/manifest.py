from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
ALGORITHM_VERSION = "bp-cache-v2"
MANIFEST_NAME = "bp_cache_manifest.json"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def source_signature(root: Path) -> dict[str, Any]:
    root = root.resolve()
    candidates = [root / "meta" / "info.json", root / "meta" / "tasks.parquet"]
    candidates.extend(sorted((root / "meta" / "episodes").rglob("*.parquet")))
    files = []
    for path in candidates:
        if path.is_file():
            stat = path.stat()
            files.append({"path": path.relative_to(root).as_posix(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    info = root / "meta" / "info.json"
    if not info.is_file():
        raise FileNotFoundError(f"缺少源数据集 meta/info.json：{info}")
    return {"root": str(root), "info_sha256": hashlib.sha256(info.read_bytes()).hexdigest(), "files": files}


def build_fingerprint(source_repo: str, signature: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    keys = ("sample_ratio", "sampling_scope", "sampling_mode", "seed", "bp_camera_keys", "target_video_mb",
            "single_episode_soft_mb", "single_episode_hard_mb")
    payload = {"schema_version": SCHEMA_VERSION, "algorithm_version": ALGORITHM_VERSION,
               "source_repo": source_repo, "source_signature": signature,
               "config": {key: config[key] for key in keys}}
    payload["digest"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    return payload


def load_manifest(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def manifest_status(manifest: dict[str, Any] | None, fingerprint: dict[str, Any]) -> str:
    if not manifest or manifest.get("complete") is not True:
        return "invalid"
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("algorithm_version") != ALGORITHM_VERSION:
        return "invalid"
    return "hit" if manifest.get("fingerprint") == fingerprint else "stale"


def recognized_for_source(manifest: dict[str, Any] | None, source_repo: str) -> bool:
    return bool(manifest and manifest.get("schema_version") == SCHEMA_VERSION
                and manifest.get("algorithm_version") == ALGORITHM_VERSION
                and manifest.get("source_repo") == source_repo)
