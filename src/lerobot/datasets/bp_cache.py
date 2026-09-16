from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub.utils import HFValidationError, validate_repo_id

from lerobot.transforms.constants import get_feature_mapping, get_image_mapping
from lerobot.utils.constants import ACTION, OBS_STATE

SCHEMA_VERSION = 1
ALGORITHM_VERSION = "bp-cache-v2"
_CACHE_COMMAND = (
    "python tools/run_bp_cache.py --repo-id-file ... --bp-cache-root ... "
    "--mapping-output ..."
)


class BPCacheValidationError(ValueError):
    """Raised when a generated behavior-prompt cache cannot be used safely."""


class UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader which rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping", node.start_mark, "found unhashable key", key_node.start_mark
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping", node.start_mark, f"found duplicate key {key!r}", key_node.start_mark
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def safe_load_yaml_unique(text: str) -> Any:
    """Load untrusted YAML and reject duplicate keys at every mapping level."""
    return yaml.load(text, Loader=UniqueKeySafeLoader)


def normalize_source_identity(repo_id: str) -> str:
    """Return the canonical identity shared by cache generation and training.

    Hub IDs are preserved. Absolute paths and explicitly local paths (``~``,
    ``./`` and ``../``) are expanded and resolved without requiring existence.
    """
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("source repo identity must be a non-empty string")
    value = repo_id.strip()
    path = Path(value).expanduser()
    explicitly_local = path.is_absolute() or value == "." or value == ".." or value.startswith(("~/", "./", "../"))
    return str(path.resolve(strict=False)) if explicitly_local else value


@dataclass(frozen=True)
class BPCacheDataset:
    source_repo: str
    root: Path
    repo_id: str
    manifest: dict[str, Any] | None
    info: dict[str, Any]


def _error(message: str) -> BPCacheValidationError:
    return BPCacheValidationError(f"{message}\nGenerate or repair the cache with: {_CACHE_COMMAND}")


def load_bp_cache_root_mapping(mapping_file: str | Path) -> dict[str, Path]:
    """Load the strict canonical-source-repo -> absolute cache-root YAML contract."""
    path = Path(mapping_file).expanduser()
    if not path.is_file():
        raise _error(f"BP cache mapping file does not exist: {path}")
    try:
        raw = safe_load_yaml_unique(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise _error(f"Invalid BP cache mapping YAML {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise _error(f"BP cache mapping must be a top-level mapping, got {type(raw).__name__}")

    result: dict[str, Path] = {}
    seen_roots: set[Path] = set()
    for source_repo, cache_root in raw.items():
        if not isinstance(source_repo, str) or not source_repo.strip():
            raise _error("BP cache mapping source repo keys must be non-empty strings")
        if source_repo != source_repo.strip():
            raise _error(f"BP cache source repo must not contain surrounding whitespace: {source_repo!r}")
        if not isinstance(cache_root, str) or not cache_root.strip():
            raise _error(f"BP cache root for {source_repo!r} must be a non-empty string")
        identity = normalize_source_identity(source_repo)
        root = Path(cache_root).expanduser()
        if not root.is_absolute():
            raise _error(f"BP cache root for {source_repo!r} must be absolute: {cache_root!r}")
        root = root.resolve(strict=False)
        if identity in result:
            raise _error(f"Duplicate normalized BP cache source repo: {identity!r}")
        if root in seen_roots:
            raise _error(f"Duplicate BP cache root: {root}")
        result[identity] = root
        seen_roots.add(root)
    if not result:
        raise _error("BP cache mapping must not be empty")
    return result


def resolve_bp_cache_root(mapping: Mapping[str, Path], source_repo: str) -> Path:
    """Resolve one source repo using its canonical identity."""
    identity = normalize_source_identity(source_repo)
    try:
        return mapping[identity]
    except KeyError as exc:
        raise _error(f"No BP cache mapping for source repo {source_repo!r} (identity {identity!r})") from exc


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise _error(f"BP cache is missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _error(f"Cannot read BP cache {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise _error(f"BP cache {label} must contain a JSON object: {path}")
    return value


def _feature_signature(feature: Any) -> tuple[Any, Any, Any]:
    if not isinstance(feature, Mapping):
        return (None, None, None)
    shape = feature.get("shape")
    return (feature.get("dtype"), tuple(shape) if isinstance(shape, list) else shape, feature.get("names"))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sanitize_repo_id_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    cleaned = re.sub(r"[-.]{2,}", "-", cleaned).strip(".-")
    return (cleaned or "dataset")[:96].rstrip(".-")


def _local_repo_id(source_repo: str) -> str:
    """Return a valid Hub-style ID for loading metadata rooted on local disk."""
    source_path = Path(source_repo)
    if not source_path.is_absolute():
        return source_repo
    parts = [part for part in source_path.parts if part != "/"]
    if len(parts) >= 2:
        return f"{_sanitize_repo_id_part(parts[-2])}/{_sanitize_repo_id_part(parts[-1])}"
    return f"local/{_sanitize_repo_id_part(source_path.name or 'dataset')}"


def _cache_repo_id(info: Mapping[str, Any], source_repo: str) -> str:
    repo_id = info.get("repo_id")
    if isinstance(repo_id, str):
        repo_id = repo_id.strip()
        try:
            validate_repo_id(repo_id)
        except HFValidationError:
            pass
        else:
            return repo_id
    fallback = _local_repo_id(source_repo)
    try:
        validate_repo_id(fallback)
    except HFValidationError as exc:  # Defensive: sanitized local IDs should always pass.
        raise _error(f"Cannot derive a valid local cache repo ID from {source_repo!r}: {exc}") from exc
    return fallback


def _validate_manifest_contract(manifest: dict[str, Any], source_repo: str) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise _error(f"BP cache manifest schema_version must be {SCHEMA_VERSION}")
    if manifest.get("complete") is not True:
        raise _error("BP cache manifest complete must be strictly true")
    if manifest.get("algorithm_version") != ALGORITHM_VERSION:
        raise _error(f"BP cache manifest algorithm_version must be {ALGORITHM_VERSION!r}")
    manifest_source = manifest.get("source_repo")
    if not isinstance(manifest_source, str) or not manifest_source.strip():
        raise _error("BP cache manifest source_repo must be a non-empty string")
    expected = normalize_source_identity(source_repo)
    if normalize_source_identity(manifest_source) != expected or manifest_source != normalize_source_identity(manifest_source):
        raise _error(f"BP cache source repo mismatch: expected canonical {expected!r}, got {manifest_source!r}")
    cache_repo_id = manifest.get("cache_repo_id")
    if not isinstance(cache_repo_id, str) or not cache_repo_id.strip():
        raise _error("BP cache manifest cache_repo_id must be a non-empty string")

    selected = manifest.get("selected_source_episodes")
    if not isinstance(selected, list) or not selected:
        raise _error("BP cache manifest selected_source_episodes must be a non-empty list")
    for item in selected:
        if not isinstance(item, Mapping) or not _is_int(item.get("episode")):
            raise _error("Each selected_source_episodes entry requires an integer episode")
        tasks = item.get("tasks")
        if not isinstance(tasks, list) or not tasks or not all(isinstance(v, str) and v for v in tasks):
            raise _error("Each selected_source_episodes entry requires non-empty string tasks")

    config = manifest.get("config")
    if not isinstance(config, Mapping) or config.get("sampling_scope") not in {"dataset", "task"}:
        raise _error("BP cache manifest config.sampling_scope must be dataset or task")
    selected_by_task = manifest.get("selected_by_task")
    if not isinstance(selected_by_task, Mapping) or not selected_by_task:
        raise _error("BP cache manifest selected_by_task must be a non-empty mapping")
    if not all(isinstance(k, str) and k and isinstance(v, list) and v and all(_is_int(i) for i in v)
               for k, v in selected_by_task.items()):
        raise _error("BP cache manifest selected_by_task has invalid keys or episode lists")

    for field in ("source_to_cache_episode", "source_to_cache_task"):
        value = manifest.get(field)
        if not isinstance(value, Mapping) or not value:
            raise _error(f"BP cache manifest {field} must be a non-empty mapping")
        if not all(isinstance(k, str) and k.isdigit() and _is_int(v) and v >= 0 for k, v in value.items()):
            raise _error(f"BP cache manifest {field} must map decimal string IDs to non-negative integers")
    names = manifest.get("source_task_names")
    if not isinstance(names, Mapping) or not names or not all(
        isinstance(k, str) and k.isdigit() and isinstance(v, str) and v for k, v in names.items()
    ):
        raise _error("BP cache manifest source_task_names must map task IDs to non-empty names")
    cameras = manifest.get("actual_camera_keys")
    if not isinstance(cameras, list) or not cameras or not all(isinstance(v, str) and v for v in cameras):
        raise _error("BP cache manifest actual_camera_keys must be a non-empty string list")
    if len(cameras) != len(set(cameras)):
        raise _error("BP cache manifest actual_camera_keys must not contain duplicates")


def validate_bp_cache_dataset(
    source_repo: str,
    cache_root: str | Path,
    source_meta: Any,
    bp_camera_keys: list[str],
) -> BPCacheDataset:
    """Validate a completed local v3 cache against its original dataset metadata."""
    source_identity = normalize_source_identity(source_repo)
    root = Path(cache_root).expanduser()
    if not root.is_absolute():
        raise _error(f"BP cache root must be absolute: {cache_root!r}")
    if not root.is_dir():
        raise _error(f"BP cache directory does not exist: {root}")

    manifest_path = root / "bp_cache_manifest.json"
    manifest = _load_json(manifest_path, "bp_cache_manifest.json") if manifest_path.exists() else None
    if manifest is not None:
        _validate_manifest_contract(manifest, source_identity)

    info = _load_json(root / "meta/info.json", "meta/info.json")
    _load_json(root / "meta/stats.json", "meta/stats.json")
    tasks_path = root / "meta/tasks.parquet"
    if not tasks_path.is_file() or tasks_path.stat().st_size == 0:
        raise _error(f"BP cache is missing non-empty meta/tasks.parquet: {tasks_path}")
    episode_files = [p for p in (root / "meta/episodes").glob("**/*.parquet") if p.stat().st_size > 0]
    if not episode_files:
        raise _error(f"BP cache contains no non-empty episode metadata parquet under {root / 'meta/episodes'}")
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v3"):
        raise _error(f"BP cache codebase_version must be v3, got {version!r}")
    if int(info.get("total_frames", 0) or 0) <= 0 or int(info.get("total_episodes", 0) or 0) <= 0:
        raise _error("BP cache metadata must contain non-zero total_frames and total_episodes")

    source_info = source_meta.info
    for field in ("fps", "robot_type"):
        if info.get(field) != source_info.get(field):
            raise _error(f"BP cache {field} mismatch for {source_identity!r}: source={source_info.get(field)!r}, cache={info.get(field)!r}")
    source_features = source_info.get("features", {})
    cache_features = info.get("features", {})
    if not isinstance(cache_features, Mapping):
        raise _error("BP cache info.features must be a mapping")

    robot_type = info.get("robot_type")
    try:
        source_feature_mapping = get_feature_mapping(robot_type, source_features)
        cache_feature_mapping = get_feature_mapping(robot_type, cache_features)
        cache_image_mapping = get_image_mapping(robot_type, cache_features)
    except (KeyError, TypeError, ValueError) as exc:
        raise _error(f"BP cache schema cannot be mapped for robot_type={robot_type!r}: {exc}") from exc

    required_features: set[str] = set()
    if robot_type != "egodex_v":
        for canonical in (OBS_STATE, ACTION):
            required = list(cache_feature_mapping.get(canonical, []))
            if not required:
                raise _error(f"BP cache has no mapped {canonical!r} features")
            required_features.update(required)
    missing = required_features.difference(cache_features)
    if missing:
        raise _error(f"BP cache is missing required state/action features: {sorted(missing)}")
    for canonical in (OBS_STATE, ACTION):
        source_keys = source_feature_mapping.get(canonical, [])
        cache_keys = cache_feature_mapping.get(canonical, [])
        if robot_type != "egodex_v" and len(source_keys) != len(cache_keys):
            raise _error(f"BP cache {canonical!r} mapped feature count differs from source")
        for source_key, cache_key in zip(source_keys, cache_keys, strict=True):
            if _feature_signature(source_features.get(source_key)) != _feature_signature(cache_features.get(cache_key)):
                raise _error(f"BP cache feature schema mismatch: {source_key!r} vs {cache_key!r}")

    canonical_to_cache_camera = {canonical: actual for actual, canonical in cache_image_mapping.items()}
    missing_cameras = set(bp_camera_keys).difference(canonical_to_cache_camera)
    if missing_cameras:
        raise _error(f"BP cache cannot map canonical BP cameras: {sorted(missing_cameras)}")
    expected_actual_cameras = [canonical_to_cache_camera[key] for key in bp_camera_keys]
    if manifest is not None and manifest["actual_camera_keys"] != expected_actual_cameras:
        raise _error(
            "BP cache manifest actual_camera_keys mismatch: "
            f"expected {expected_actual_cameras!r}, got {manifest['actual_camera_keys']!r}"
        )

    data_files = [p for p in root.glob("data/**/*.parquet") if p.is_file() and p.stat().st_size > 0]
    if not data_files:
        raise _error(f"BP cache contains no non-empty data parquet files under {root / 'data'}")
    for actual_camera in expected_actual_cameras:
        feature = cache_features[actual_camera]
        if feature.get("dtype") == "video":
            video_files = [p for p in (root / "videos" / actual_camera).glob("**/*") if p.is_file() and p.stat().st_size > 0]
            if not video_files:
                raise _error(f"BP cache contains no videos for required camera {actual_camera!r}")

    return BPCacheDataset(
        source_repo=source_identity,
        root=root.resolve(),
        repo_id=(manifest["cache_repo_id"].strip() if manifest is not None else _cache_repo_id(info, source_identity)),
        manifest=manifest,
        info=info,
    )
