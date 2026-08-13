from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.norm_stats.cache import atomic_write_json, build_fingerprint, load_cache, write_cache
from tools.norm_stats.compute import _delta_values, _matrix, _stack, _validate_delta_layout
from tools.norm_stats.core import RunningStats, merge_payloads, validate_schema
from tools.norm_stats.pipeline import PipelineConfig, _params, _write_group


def test_running_stats_merge_uses_feature_count() -> None:
    left, right = RunningStats(), RunningStats()
    left.update(np.array([[0.0, 2.0], [2.0, 4.0]]))
    right.update(np.array([[10.0, 20.0]]))
    left.merge(right)
    result = left.statistics()
    assert result["count"] == [3]
    assert result["mean"] == pytest.approx([4.0, 26.0 / 3.0])
    assert result["min"] == [0.0, 2.0]
    assert result["max"] == [10.0, 20.0]


def test_default_schema_rejects_mapping_difference() -> None:
    base = {"repo_id": "a", "keys": ["state", "action"], "shapes": {"state": [2], "action": [2]},
            "mapping": {"observation.state": ["state"], "action": ["action"]}, "mask": [True, False]}
    other = {**base, "repo_id": "b", "mapping": {"observation.state": ["state"], "action": ["other"]}}
    with pytest.raises(ValueError, match="mapping/mask"):
        validate_schema([base, other])


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text('{"total_frames": 2, "total_episodes": 1}', encoding="utf-8")
    (root / "meta" / "episodes.jsonl").write_text('{}\n', encoding="utf-8")
    (root / "data" / "chunk-000" / "file.parquet").write_bytes(b"x")
    return root


def test_fingerprint_changes_when_dataset_file_changes(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    first, _ = build_fingerprint(str(root), root, {"action_mode": "delta"})
    parquet = root / "data" / "chunk-000" / "file.parquet"
    parquet.write_bytes(b"longer")
    second, _ = build_fingerprint(str(root), root, {"action_mode": "delta"})
    assert first != second


def test_loads_legacy_cache_by_fingerprint_not_directory(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    params = {"action_mode": "delta", "chunk_size": 50, "max_chunks_per_episode": None,
              "max_chunks_per_repo": None, "sample_seed": 42, "skip_action_robot_types": []}
    dataset_id, current = build_fingerprint(str(root), root, params)
    ordered = [root / "meta" / "info.json", root / "data" / "chunk-000" / "file.parquet", root / "meta" / "episodes.jsonl"]
    manifest = [(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in ordered]
    legacy_fp = {"repo_id": str(root), "repo_path": str(root.resolve()),
                 "info_sha256": current["info_sha256"],
                 "data_signature": hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest(), **params}
    result = {"repo_id": str(root), "payload": {"x": {"count": 1}}}
    path = tmp_path / "cache" / "datasets" / "unrelated-old-name" / "stats_payload.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"fingerprint": legacy_fp, "result": result}), encoding="utf-8")
    loaded, kind = load_cache(tmp_path / "cache", dataset_id, current)
    assert loaded == result
    assert kind == "legacy"


def test_all_skipped_action_has_zero_not_fake_count() -> None:
    result = {
        "repo_id": "a", "keys": ["action"], "shapes": {"action": [2]},
        "mapping": {"observation.state": [], "action": ["action"]}, "mask": [],
        "payload": {"action": None},
    }
    assert merge_payloads([result])["action"]["count"] == [0]


def _result(repo_id: str = "a") -> dict:
    payload = {"state": {"count": 1, "mean": [1.0], "mean_sq": [1.0], "min": [1.0], "max": [1.0]}}
    return {
        "repo_id": repo_id, "resolved_robot_type": "robot", "keys": ["state"],
        "shapes": {"state": [1]}, "mapping": {"observation.state": ["state"], "action": []},
        "mask": [], "payload": payload, "compute": {"skip_action": False},
    }


def _cfg(tmp_path: Path, output_format: str) -> PipelineConfig:
    return PipelineConfig(
        repo_id_file=tmp_path / "repos.txt", cache_root=tmp_path / "cache", output_format=output_format,
        output_root=tmp_path / "out", output_path=tmp_path / "single" / "stats.json",
    )


def test_default_writes_only_requested_stats_file(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "default")
    _write_group([_result()], cfg.output_path, cfg)
    assert cfg.output_path.is_file()
    assert not (cfg.output_path.parent / "manifest.json").exists()


def test_tbot_writes_manifest(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "tbot")
    path = tmp_path / "out" / "robot" / "delta" / "stats.json"
    _write_group([_result()], path, cfg)
    assert (path.parent / "manifest.json").is_file()


def test_abs_effective_params_ignore_chunk_and_sampling(tmp_path: Path) -> None:
    first = _cfg(tmp_path, "default")
    first = PipelineConfig(**{**first.__dict__, "action_mode": "abs", "chunk_size": 10,
                              "max_chunks_per_episode": 2, "max_chunks_per_repo": 3, "sample_seed": 1})
    second = PipelineConfig(**{**first.__dict__, "chunk_size": 99,
                               "max_chunks_per_episode": 20, "max_chunks_per_repo": 30, "sample_seed": 999})
    assert _params(first) == _params(second) == {"action_mode": "abs", "skip_action_robot_types": []}


def test_old_float32_payload_is_promoted_to_float64() -> None:
    stats = RunningStats.from_payload({"count": 2, "mean": [0.1], "mean_sq": [0.02],
                                       "min": [0.0], "max": [0.2]})
    assert stats.mean.dtype == np.float64
    stats.merge(RunningStats.from_payload({"count": 1, "mean": [0.4], "mean_sq": [0.16],
                                           "min": [0.4], "max": [0.4]}))
    assert stats.statistics()["count"] == [3]


def test_half_written_new_cache_is_miss(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    dataset_id, fingerprint = build_fingerprint(str(root), root, {"action_mode": "abs"})
    path = tmp_path / "cache" / "datasets" / dataset_id / "stats_payload.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"repo_id": str(root), "payload": {}}), encoding="utf-8")
    assert load_cache(tmp_path / "cache", dataset_id, fingerprint) == (None, None)


class _Selected:
    def __init__(self, values):
        self.values = values
    def __getitem__(self, key):
        return self.values


class _HF:
    def __init__(self, values):
        self.values = values
    def select(self, indices):
        return _Selected([self.values[index] for index in indices])


class _FakeDataset:
    def __init__(self, values):
        self.hf_dataset = _HF(values)


def test_stack_accepts_python_numpy_and_tensor_values() -> None:
    indices = np.array([0, 1], dtype=np.int64)
    for values in ([1.0, 2.0], [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
                   [__import__("torch").tensor([1.0]), __import__("torch").tensor([2.0])]):
        assert _stack(_FakeDataset(values), "x", indices).shape[0] == 2
    assert _matrix(__import__("torch").tensor([1.0, 2.0])).shape == (2, 1)


def test_delta_mask_broadcast_and_episode_boundary() -> None:
    import torch
    action = torch.tensor([[0., 10., 100.], [1., 11., 101.], [2., 12., 102.], [3., 13., 103.]])
    state = torch.tensor([[1., 20., 1000.], [2., 30., 2000.]])
    delta = _delta_values(action, state, np.array([0, 1]), 3, torch.tensor([True, False, True]))
    assert delta.shape == (2, 3, 3)
    assert delta[0, :, 0].tolist() == [-1., 0., 1.]
    assert delta[0, :, 1].tolist() == [10., 11., 12.]
    assert delta[1, :, 2].tolist() == [-1899., -1898., -1897.]


def test_multi_key_delta_layout_uses_concatenated_dimensions() -> None:
    import torch
    features = {"s1": {"shape": [2]}, "s2": {"shape": [1]},
                "a1": {"shape": [1]}, "a2": {"shape": [2]}}
    mapping = {"observation.state": ["s1", "s2"], "action": ["a1", "a2"]}
    _validate_delta_layout("repo", features, mapping, torch.tensor([True, False, True]))
    with pytest.raises(ValueError, match="维度不兼容"):
        _validate_delta_layout("repo", features, mapping, torch.tensor([True, False]))


def test_mismatched_cache_pair_is_miss(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    dataset_id, fingerprint = build_fingerprint(str(root), root, {"action_mode": "abs"})
    first = {"repo_id": str(root), "payload": {"x": {"count": 1}}}
    write_cache(tmp_path / "cache", dataset_id, fingerprint, first)
    payload_path = tmp_path / "cache" / "datasets" / dataset_id / "stats_payload.json"
    atomic_write_json(payload_path, {"repo_id": str(root), "payload": {"x": {"count": 2}}})
    assert load_cache(tmp_path / "cache", dataset_id, fingerprint) == (None, None)


def test_abs_fingerprint_ignores_ineffective_parameters(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    cfg = _cfg(tmp_path, "default")
    first = PipelineConfig(**{**cfg.__dict__, "action_mode": "abs", "chunk_size": 10, "sample_seed": 1})
    second = PipelineConfig(**{**cfg.__dict__, "action_mode": "abs", "chunk_size": 99, "sample_seed": 999})
    first_id, _ = build_fingerprint(str(root), root, _params(first))
    second_id, _ = build_fingerprint(str(root), root, _params(second))
    assert first_id == second_id




def test_output_ignores_visual_stats(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "default")
    first = _result("a")
    second = _result("b")
    first["visual_stats"] = {"camera_a": {"mean": [[[0.0]]]}}
    second["visual_stats"] = {"different_camera": {"mean": [1.0, 2.0]}}
    _write_group([first, second], cfg.output_path, cfg)
    output = json.loads(cfg.output_path.read_text(encoding="utf-8"))
    assert set(output) == {"state"}


def test_enrich_cached_result_removes_existing_visual_stats() -> None:
    from tools.norm_stats.compute import enrich_cached_result
    result = _result()
    result["visual_stats"] = {"camera": {"mean": [1.0]}}
    cleaned = enrich_cached_result(result, None)
    assert "visual_stats" not in cleaned
