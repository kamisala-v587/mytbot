from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

import yaml


from lerobot.datasets.bp_cache import normalize_source_identity, safe_load_yaml_unique


def source_identity(repo_id: str) -> str:
    """Compatibility alias for the shared source identity contract."""
    return normalize_source_identity(repo_id)


def cache_dir_name(repo_id: str) -> str:
    identity = source_identity(repo_id)
    basename = Path(identity).name or "dataset"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", basename).strip(".-") or "dataset"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    return f"{safe}-{digest}"


def read_repo_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"repo 列表不存在：{path}")
    repos = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not repos:
        raise ValueError(f"repo 列表为空：{path}")
    seen: set[str] = set()
    for repo in repos:
        identity = source_identity(repo)
        if identity in seen:
            raise ValueError(f"repo 列表存在重复项：{repo!r} -> {identity}")
        seen.add(identity)
    return [source_identity(repo) for repo in repos]


def load_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    value = safe_load_yaml_unique(path.read_text(encoding="utf-8"))
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ValueError(f"mapping YAML 必须是 string->string：{path}")
    normalized: dict[str, str] = {}
    for key, root in value.items():
        identity = source_identity(key)
        if identity in normalized:
            raise ValueError(f"mapping YAML 规范化后存在重复项：{key!r} -> {identity!r}")
        normalized[identity] = root
    return normalized


def atomic_write_mapping(path: Path, mapping: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yaml.safe_dump(mapping, stream, allow_unicode=True, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)
