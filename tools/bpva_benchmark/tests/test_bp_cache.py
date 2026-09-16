import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lerobot.datasets.bp_cache import (
    BPCacheValidationError,
    load_bp_cache_root_mapping,
    resolve_bp_cache_root,
    validate_bp_cache_dataset,
)
from lerobot.policies.BPVAv2.configuration_bpva import BPVAv2DatasetConfig


def test_bpvav2_original_defaults_are_backward_compatible():
    cfg = BPVAv2DatasetConfig(repo_id="fake/repo")
    assert cfg.bp_prompt_source == "original"
    assert cfg.bp_cache_root_file is None


def test_bpvav2_cache_fields_normalize_and_validate():
    cfg = BPVAv2DatasetConfig(
        repo_id="fake/repo", bp_prompt_source=" CACHE ", bp_cache_root_file=" /tmp/cache-map.yaml ",
        bp_same_episode_policy="allow",
    )
    assert cfg.bp_prompt_source == "cache"
    assert cfg.bp_cache_root_file == "/tmp/cache-map.yaml"
    with pytest.raises(ValueError, match="bp_cache_root_file is required"):
        BPVAv2DatasetConfig(repo_id="fake/repo", bp_prompt_source="cache", bp_cache_root_file=" ")
    with pytest.raises(ValueError, match="forbid.*unsupported"):
        BPVAv2DatasetConfig(
            repo_id="fake/repo", bp_prompt_source="cache", bp_cache_root_file="/tmp/map.yaml",
            bp_same_episode_policy="forbid",
        )


def test_mapping_exact_hit_missing_and_invalid(tmp_path):
    cache = tmp_path / "cache"
    mapping_file = tmp_path / "map.yaml"
    mapping_file.write_text(yaml.safe_dump({"org/source": str(cache)}))
    mapping = load_bp_cache_root_mapping(mapping_file)
    assert resolve_bp_cache_root(mapping, "org/source") == cache
    with pytest.raises(BPCacheValidationError, match=r"(?s)No BP cache mapping.*run_bp_cache.py"):
        resolve_bp_cache_root(mapping, "org/missing")

    mapping_file.write_text("org/source: relative/cache\n")
    with pytest.raises(BPCacheValidationError, match="must be absolute"):
        load_bp_cache_root_mapping(mapping_file)
    mapping_file.write_text("- not\n- a mapping\n")
    with pytest.raises(BPCacheValidationError, match="top-level mapping"):
        load_bp_cache_root_mapping(mapping_file)


def test_mapping_rejects_yaml_duplicate_key(tmp_path):
    mapping_file = tmp_path / "map.yaml"
    mapping_file.write_text("org/source: /tmp/a\norg/source: /tmp/b\n")
    with pytest.raises(BPCacheValidationError, match="duplicate"):
        load_bp_cache_root_mapping(mapping_file)


def test_cache_rejects_legacy_completed_marker(tmp_path):
    root = tmp_path / "cache"
    info = _write_valid_aloha_cache(root)
    manifest_path = root / "bp_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["completed"] = manifest.pop("complete")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(BPCacheValidationError, match="complete must be strictly true"):
        validate_bp_cache_dataset(
            "org/source", root, SimpleNamespace(info=info), ["observation.images.image0"]
        )

def test_cache_manifest_corruption_has_generation_hint(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    (root / "bp_cache_manifest.json").write_text(json.dumps({"schema_version": 1, "algorithm_version": "bp-cache-v1", "complete": False}))
    with pytest.raises(BPCacheValidationError, match=r"(?s)complete must be strictly true.*run_bp_cache.py"):
        validate_bp_cache_dataset(
            "org/source", root, SimpleNamespace(info={}), ["observation.images.image0"]
        )


def _write_valid_aloha_cache(root: Path):
    features = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": None},
        "action": {"dtype": "float32", "shape": [14], "names": None},
        "observation.images.cam_high": {"dtype": "video", "shape": [3, 224, 224], "names": None},
    }
    info = {
        "codebase_version": "v3.0", "fps": 30, "robot_type": "aloha",
        "total_frames": 1, "total_episodes": 1, "features": features,
    }
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "videos/observation.images.cam_high/chunk-000").mkdir(parents=True)
    (root / "bp_cache_manifest.json").write_text(json.dumps({
        "schema_version": 1, "algorithm_version": "bp-cache-v2", "complete": True,
        "source_repo": "org/source", "cache_repo_id": "local/cache",
        "config": {"sampling_scope": "dataset"},
        "selected_source_episodes": [{"episode": 7, "tasks": ["pick cube"]}],
        "selected_by_task": {"pick cube": [7]},
        "source_to_cache_episode": {"7": 0}, "source_to_cache_task": {"4": 0},
        "source_task_names": {"4": "pick cube"},
        "actual_camera_keys": ["observation.images.cam_high"]
    }))
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/stats.json").write_text("{}")
    (root / "meta/tasks.parquet").write_bytes(b"tasks")
    (root / "meta/episodes/chunk-000/file-000.parquet").write_bytes(b"episodes")
    (root / "data/chunk-000/file-000.parquet").write_bytes(b"data")
    (root / "videos/observation.images.cam_high/chunk-000/file-000.mp4").write_bytes(b"video")
    return info


def test_valid_cache_schema_and_camera_mapping(tmp_path):
    root = tmp_path / "cache"
    info = _write_valid_aloha_cache(root)
    result = validate_bp_cache_dataset(
        "org/source", root, SimpleNamespace(info=info), ["observation.images.image0"]
    )
    assert result.root == root
    assert result.repo_id == "local/cache"


def test_standard_v3_cache_without_manifest_is_accepted(tmp_path):
    root = tmp_path / "cache"
    info = _write_valid_aloha_cache(root)
    (root / "bp_cache_manifest.json").unlink()
    info["repo_id"] = "split/cache"
    (root / "meta/info.json").write_text(json.dumps(info))

    result = validate_bp_cache_dataset(
        "org/source", root, SimpleNamespace(info=info), ["observation.images.image0"]
    )

    assert result.repo_id == "split/cache"
    assert result.manifest is None


def test_standard_v3_cache_uses_valid_id_for_absolute_source(tmp_path):
    root = tmp_path / "cache"
    info = _write_valid_aloha_cache(root)
    (root / "bp_cache_manifest.json").unlink()
    source = tmp_path / "source dataset"

    result = validate_bp_cache_dataset(
        str(source), root, SimpleNamespace(info=info), ["observation.images.image0"]
    )

    assert result.repo_id == f"{tmp_path.name}/source-dataset"
    assert not result.repo_id.startswith("/")


def test_cache_rejects_incompatible_fps(tmp_path):
    root = tmp_path / "cache"
    source_info = _write_valid_aloha_cache(root)
    source_info = {**source_info, "fps": 20}
    with pytest.raises(BPCacheValidationError, match="fps mismatch"):
        validate_bp_cache_dataset(
            "org/source", root, SimpleNamespace(info=source_info), ["observation.images.image0"]
        )


def test_cache_rejects_manifest_camera_mismatch(tmp_path):
    root = tmp_path / "cache"
    info = _write_valid_aloha_cache(root)
    manifest_path = root / "bp_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["actual_camera_keys"] = ["wrong.camera"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(BPCacheValidationError, match="actual_camera_keys mismatch"):
        validate_bp_cache_dataset(
            "org/source", root, SimpleNamespace(info=info), ["observation.images.image0"]
        )


def test_cache_config_warns_for_avoid_and_rejects_forbid(caplog):
    with caplog.at_level("WARNING"):
        BPVAv2DatasetConfig(
            repo_id="fake/repo", bp_prompt_source="cache", bp_cache_root_file="/tmp/map.yaml",
            bp_same_episode_policy="avoid",
        )
    assert "Prefer 'allow'" in caplog.text
    with pytest.raises(ValueError, match="forbid.*unsupported"):
        BPVAv2DatasetConfig(
            repo_id="fake/repo", bp_prompt_source="cache", bp_cache_root_file="/tmp/map.yaml",
            bp_same_episode_policy="forbid",
        )


def test_generator_manifest_payload_is_accepted_by_training_validator(tmp_path):
    from tools.bp_cache.pipeline import Selection, _manifest_payload
    root = tmp_path / "cache"
    info = _write_valid_aloha_cache(root)
    class _Rows:
        def __getitem__(self, index):
            return SimpleNamespace(name="pick cube")
    result = SimpleNamespace(
        episode_mapping={7: 0}, task_mapping={4: 0},
        dataset=SimpleNamespace(meta=SimpleNamespace(tasks=SimpleNamespace(iloc=_Rows()))),
        video_files=[], warnings=[], encoder_fallbacks={},
    )
    selection = Selection(episodes=(7,), episode_tasks={7: ("pick cube",)},
                          selected_by_task={"pick cube": (7,)})
    fingerprint = {"source_signature": {"digest": "source"}, "config": {"sampling_scope": "dataset"}, "digest": "build"}
    manifest = _manifest_payload("org/source", "bp-cache/generated", fingerprint, selection,
                                 ("observation.images.cam_high",), result)
    (root / "bp_cache_manifest.json").write_text(json.dumps(manifest))
    validated = validate_bp_cache_dataset(
        "org/source", root, SimpleNamespace(info=info), ["observation.images.image0"]
    )
    assert validated.repo_id == "bp-cache/generated"
