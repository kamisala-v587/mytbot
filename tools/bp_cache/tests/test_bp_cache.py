from __future__ import annotations

from pathlib import Path

import pytest

from tools.bp_cache.config import BPConfig
from tools.bp_cache.manifest import ALGORITHM_VERSION, SCHEMA_VERSION, manifest_status
from tools.bp_cache.mapping import atomic_write_mapping, cache_dir_name, load_mapping
from tools.bp_cache.pipeline import run_pipeline, sample_episodes_by_task


def test_task_scope_legacy_samples_each_task_and_at_least_one() -> None:
    tasks = {0: ("pick",), 1: ("pick",), 2: ("pick",), 3: ("place",)}
    result = sample_episodes_by_task(tasks, 0.01, "first", 7, "repo")
    assert result.selected_by_task == {"pick": (0,), "place": (3,)}
    assert result.episodes == (0, 3)


def test_dataset_scope_selects_global_ratio_without_task_coverage() -> None:
    tasks = {index: (f"task-{index}",) for index in range(50)}
    result = sample_episodes_by_task(tasks, 0.1, "first", 42, "repo", "dataset")
    assert result.episodes == tuple(range(5))
    assert len(result.episodes) == 5


def test_dataset_scope_random_is_stable_and_repo_keyed() -> None:
    tasks = {index: (f"task-{index}",) for index in range(50)}
    first = sample_episodes_by_task(tasks, 0.1, "random", 42, "repo-a", "dataset")
    assert first == sample_episodes_by_task(tasks, 0.1, "random", 42, "repo-a", "dataset")
    assert first != sample_episodes_by_task(tasks, 0.1, "random", 42, "repo-b", "dataset")


def test_random_is_repo_seeded_and_reproducible() -> None:
    tasks = {index: ("pick",) for index in range(20)}
    first = sample_episodes_by_task(tasks, 0.25, "random", 42, "repo-a")
    assert first == sample_episodes_by_task(tasks, 0.25, "random", 42, "repo-a")
    assert first != sample_episodes_by_task(tasks, 0.25, "random", 42, "repo-b")
    assert len(first.episodes) == 5


def test_task_scope_multitask_episode_is_deduplicated_for_split() -> None:
    result = sample_episodes_by_task({0: ("a", "b"), 1: ("a",), 2: ("b",)}, 0.5, "first", 0, "repo")
    assert result.selected_by_task == {"a": (0,), "b": (0,)}
    assert result.episodes == (0,)


def test_cache_dir_is_deterministic_and_collision_resistant(tmp_path: Path) -> None:
    one = str(tmp_path / "one" / "same")
    two = str(tmp_path / "two" / "same")
    assert cache_dir_name(one) == cache_dir_name(one)
    assert cache_dir_name(one).startswith("same-")
    assert cache_dir_name(one) != cache_dir_name(two)


def test_mapping_atomic_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "mapping.yaml"
    atomic_write_mapping(path, {"source/a": "/cache/a"})
    assert load_mapping(path) == {"source/a": "/cache/a"}
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))
    path.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="string->string"):
        load_mapping(path)


def test_manifest_hit_and_stale() -> None:
    fingerprint = {"digest": "one"}
    manifest = {"complete": True, "schema_version": SCHEMA_VERSION,
                "algorithm_version": ALGORITHM_VERSION, "fingerprint": fingerprint}
    assert manifest_status(manifest, fingerprint) == "hit"
    assert manifest_status(manifest, {"digest": "two"}) == "stale"
    assert manifest_status({**manifest, "complete": False}, fingerprint) == "invalid"
    assert manifest_status({**manifest, "complete": 1}, fingerprint) == "invalid"
    assert manifest_status({**manifest, "complete": "true"}, fingerprint) == "invalid"


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    repo_file = tmp_path / "repos.txt"
    repo_file.write_text("  local/repo  \n", encoding="utf-8")
    root = tmp_path / "cache"
    mapping = tmp_path / "mapping.yaml"
    cfg = BPConfig(repo_id_file=repo_file, bp_cache_root=root, mapping_output=mapping, dry_run=True)
    assert run_pipeline(cfg) == 0
    assert not root.exists()
    assert not mapping.exists()


def test_source_identity_and_repo_list_reject_normalized_duplicates(tmp_path: Path) -> None:
    from lerobot.datasets.bp_cache import normalize_source_identity
    from tools.bp_cache.mapping import read_repo_ids
    local = tmp_path / "dataset"
    assert normalize_source_identity("org/repo") == "org/repo"
    assert normalize_source_identity(str(local)) == str(local.resolve())
    repo_file = tmp_path / "repos.txt"
    repo_file.write_text(f"{local}\n{local}/../dataset\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        read_repo_ids(repo_file)


def test_generator_mapping_rejects_duplicate_yaml_key(tmp_path: Path) -> None:
    path = tmp_path / "mapping.yaml"
    path.write_text("org/repo: /tmp/a\norg/repo: /tmp/b\n", encoding="utf-8")
    with pytest.raises(Exception, match="duplicate key"):
        load_mapping(path)


def test_publish_failure_restores_old_final(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import os
    from tools.bp_cache.pipeline import publish_cache
    final, building = tmp_path / "cache", tmp_path / "cache.building"
    final.mkdir(); building.mkdir()
    (final / "marker").write_text("old"); (building / "marker").write_text("new")
    real_replace = os.replace
    def fail_publish(source, destination):
        if Path(source) == building and Path(destination) == final:
            raise OSError("simulated publish failure")
        return real_replace(source, destination)
    monkeypatch.setattr(os, "replace", fail_publish)
    with pytest.raises(OSError, match="simulated"):
        publish_cache(building, final)
    assert (final / "marker").read_text() == "old"
    assert building.exists()
    assert not list(tmp_path.glob("cache.backup-*"))
