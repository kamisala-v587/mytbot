from datetime import datetime
from types import SimpleNamespace

import pytest

from tools.bpva_benchmark.data_benchmark import (
    build_parser as build_data_parser,
    _effective_num_workers,
    disable_dist_loading_for_single_process,
)
from tools.bpva_benchmark.reporting import resolve_output_dir
from tools.bpva_benchmark.train_benchmark import (
    _apply_overrides,
    build_parser as build_train_parser,
    dataset_inventory,
)


def test_register_bpva_configs_populates_choice_registry():
    """Both BPVA generations must be registered before standalone parsing."""
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.policies import PreTrainedConfig
    from tools.bpva_benchmark.config_utils import register_bpva_configs

    register_bpva_configs()
    assert {"bpva", "bpvav2"} <= set(DatasetConfig.get_known_choices())
    assert {"bpva", "bpvav2"} <= set(PreTrainedConfig.get_known_choices())


def test_load_cfg_helpers_import_register_bpva_configs():
    """`data_benchmark._load_cfg` / `train_benchmark._load` must route through
    `config_utils.register_bpva_configs` rather than only
    `register_third_party_plugins`, otherwise parsing built-in BPVA choices can
    fail outside of `lerobot_train.py`."""
    import inspect

    from tools.bpva_benchmark import data_benchmark, train_benchmark

    assert "register_bpva_configs" in inspect.getsource(data_benchmark._load_cfg)
    assert "register_bpva_configs" in inspect.getsource(train_benchmark._load)


def test_output_timestamp_helper():
    output = resolve_output_dir("base", now=datetime(2026, 8, 28, 12, 34, 56, 123456))
    assert str(output) == "base/2026-08-28/12-34-56-123456"
    assert str(resolve_output_dir("base", exact=True)) == "base"


def test_single_process_dist_override():
    cfg = SimpleNamespace(dataset=SimpleNamespace(dist_loading=True))
    assert disable_dist_loading_for_single_process(cfg, 1)
    assert cfg.dataset.dist_loading is False
    cfg.dataset.dist_loading = True
    assert not disable_dist_loading_for_single_process(cfg, 4)
    assert cfg.dataset.dist_loading is True


def test_train_parser_and_validation():
    parser = build_train_parser()
    assert "gradient" not in parser.format_help().lower()
    with pytest.raises(SystemExit):
        parser.parse_args(["--config-path", "x", "--measure-steps", "0"])


def test_train_batch_size_parser_and_overrides():
    parser = build_train_parser()
    defaults = parser.parse_args(["--config-path", "x"])
    assert defaults.batch_size is None
    assert defaults.num_workers is None

    args = parser.parse_args(
        ["--config-path", "x", "--batch-size", "2", "--num-workers", "0"]
    )
    assert args.batch_size == 2
    cfg = SimpleNamespace(batch_size=16, num_workers=16)
    assert _apply_overrides(cfg, args) == {"batch_size": True, "num_workers": True}
    assert cfg.batch_size == 2
    assert cfg.num_workers == 0

    cfg = SimpleNamespace(batch_size=16, num_workers=16)
    assert _apply_overrides(cfg, defaults) == {
        "batch_size": False,
        "num_workers": False,
    }
    assert (cfg.batch_size, cfg.num_workers) == (16, 16)

    with pytest.raises(SystemExit):
        parser.parse_args(["--config-path", "x", "--batch-size", "0"])


def test_data_parser_sample_rate_validation():
    parser = build_data_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--config-path", "x", "--sample-rate", "1.1"])


def test_effective_num_workers():
    streaming = SimpleNamespace(dataset=SimpleNamespace(streaming=True), num_workers=8)
    regular = SimpleNamespace(dataset=SimpleNamespace(streaming=False), num_workers=8)
    assert _effective_num_workers(streaming) == 1
    assert _effective_num_workers(regular) == 8


def test_data_finalize_parser_defaults_and_positive_validation():
    parser = build_data_parser()
    args = parser.parse_args(["--config-path", "x"])
    assert args.finalize_timeout == 300.0
    assert args.finalize_poll_interval == 0.2

    for option in ("--finalize-timeout", "--finalize-poll-interval"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--config-path", "x", option, "0"])


def test_data_partial_merge_is_collective_free(monkeypatch):
    from tools.bpva_benchmark import data_benchmark

    # The final merge helper consumes JSON payloads only; distributed APIs are
    # deliberately absent from its dependency surface.
    partial = {
        "records": [],
        "collector": {"top_events": [], "stats": {}},
        "monitor": {"samples": [], "errors": []},
    }
    assert data_benchmark.merge_data_partials([partial])["records"] == []
    assert not hasattr(data_benchmark, "gather_object")
    assert not hasattr(data_benchmark, "gather_records")


def test_train_out_of_order_parser_default_and_flag():
    parser = build_train_parser()
    assert parser.parse_args(["--config-path", "x"]).out_of_order is False
    assert parser.parse_args(["--config-path", "x", "--out-of-order"]).out_of_order is True


def test_train_loader_sets_in_order(monkeypatch):
    import torch
    from tools.bpva_benchmark.train_benchmark import _loader

    captured = {}
    class FakeDataLoader:
        def __new__(cls, *, in_order=True, **kwargs):
            captured.update(kwargs)
            captured["in_order"] = in_order
            return captured
    monkeypatch.setattr(torch.utils.data, "DataLoader", FakeDataLoader)
    cfg = SimpleNamespace(
        batch_size=2, num_workers=0,
        dataset=SimpleNamespace(streaming=False),
    )
    result = _loader([1, 2], cfg, SimpleNamespace(type="cpu"), in_order=False)
    assert result["in_order"] is False


def test_microstep_report_ids_are_unique_and_validated():
    from tools.bpva_benchmark.train_benchmark import report_microstep_id
    assert [report_microstep_id(i) for i in range(4)] == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        report_microstep_id(-1)


def test_optimizer_wall_tracker_resets_warmup_and_starts_measure_at_zero():
    from tools.bpva_benchmark.train_benchmark import OptimizerStepWallTracker

    tracker = OptimizerStepWallTracker()
    tracker.begin_microstep(1.0)
    tracker.begin_microstep(2.0)  # accumulation: preserve first microstep boundary
    assert tracker.finish_cycle(3.0, measured=False, optimizer_step=0, warmup_steps=1) is None
    assert tracker.began is None

    tracker.begin_microstep(10.0)
    tracker.begin_microstep(11.0)
    assert tracker.finish_cycle(
        13.5, measured=True, optimizer_step=1, warmup_steps=1
    ) == (0, 3.5)
    assert tracker.began is None


def test_train_loader_preserves_streaming_semantics(monkeypatch):
    import torch
    from torch.utils.data import IterableDataset
    from tools.bpva_benchmark.data_instrumentation import wrap_dataset_for_instrumentation
    from tools.bpva_benchmark.train_benchmark import _loader

    class Stream(IterableDataset):
        def __iter__(self):
            yield {"x": 1}

    captured = {}
    class FakeDataLoader:
        def __new__(cls, *, in_order=True, **kwargs):
            captured.update(kwargs)
            captured["in_order"] = in_order
            return captured
    monkeypatch.setattr(torch.utils.data, "DataLoader", FakeDataLoader)
    wrapped = wrap_dataset_for_instrumentation(Stream())
    cfg = SimpleNamespace(
        batch_size=2, num_workers=16,
        dataset=SimpleNamespace(streaming=True),
    )
    result = _loader(wrapped, cfg, SimpleNamespace(type="cpu"))
    assert isinstance(result["dataset"], IterableDataset)
    assert result["num_workers"] == 1
    assert result["shuffle"] is False
    assert result["sampler"] is None
    assert result["prefetch_factor"] == 4


def test_real_bpvav2_config_parses_without_model_or_dataset_construction():
    from pathlib import Path
    from tools.bpva_benchmark.train_benchmark import _load

    config_path = Path(__file__).parents[3] / "configs" / "bpvav2_pretrain_test.jsonc"
    cfg = _load(str(config_path))
    assert type(cfg.dataset).__name__ == "BPVAv2DatasetConfig"
    assert cfg.dataset._canonical_type == "bpvav2"
    assert type(cfg.policy).__name__ == "BPVAv2Config"
    assert cfg.policy.type == "bpvav2"


def test_train_weighted_loader_uses_raw_dataset_for_sampler(monkeypatch):
    import torch
    from lerobot.datasets import sampler as sampler_module
    from tools.bpva_benchmark.train_benchmark import _loader

    class RawDataset:
        dataset_weights = [0.25, 0.75]

        def __bool__(self):
            raise AssertionError("sampler_dataset truthiness must not be evaluated")

    class WrappedDataset:
        # Deliberately absent on the raw sampler input: weighted detection must
        # not be based on this wrapped DataLoader dataset.
        dataset_weights = None

    raw = RawDataset()
    wrapped = WrappedDataset()
    sampler_calls = []
    sentinel_sampler = object()

    def fake_sampler(*, dataset):
        sampler_calls.append(dataset)
        return sentinel_sampler

    captured = {}

    class FakeDataLoader:
        def __new__(cls, *, in_order=True, **kwargs):
            captured.update(kwargs)
            captured["in_order"] = in_order
            return captured

    monkeypatch.setattr(sampler_module, "MultiLeRobotWeightedSampler", fake_sampler)
    monkeypatch.setattr(torch.utils.data, "DataLoader", FakeDataLoader)
    cfg = SimpleNamespace(
        batch_size=2,
        num_workers=0,
        dataset=SimpleNamespace(streaming=False),
    )

    result = _loader(
        wrapped,
        cfg,
        SimpleNamespace(type="cpu"),
        sampler_dataset=raw,
        in_order=False,
    )

    assert sampler_calls == [raw]
    assert result["dataset"] is wrapped
    assert result["sampler"] is sentinel_sampler
    assert result["shuffle"] is False
    assert result["in_order"] is False


def test_train_weighted_detection_uses_explicit_sampler_dataset(monkeypatch):
    import torch
    from lerobot.datasets import sampler as sampler_module
    from tools.bpva_benchmark.train_benchmark import _loader

    class WrappedDataset:
        dataset_weights = [1.0]

    raw_nonweighted = SimpleNamespace(dataset_weights=None)
    sampler_calls = []
    monkeypatch.setattr(
        sampler_module,
        "MultiLeRobotWeightedSampler",
        lambda *, dataset: sampler_calls.append(dataset) or object(),
    )

    captured = {}

    class FakeDataLoader:
        def __new__(cls, *, in_order=True, **kwargs):
            captured.update(kwargs)
            captured["in_order"] = in_order
            return captured

    monkeypatch.setattr(torch.utils.data, "DataLoader", FakeDataLoader)
    cfg = SimpleNamespace(
        batch_size=2,
        num_workers=0,
        dataset=SimpleNamespace(streaming=False),
    )

    result = _loader(
        WrappedDataset(),
        cfg,
        SimpleNamespace(type="cpu"),
        sampler_dataset=raw_nonweighted,
    )

    assert sampler_calls == []
    assert result["sampler"] is None
    assert result["shuffle"] is True


def test_register_benchmark_configs_includes_tbot_sa1():
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.policies import PreTrainedConfig
    from tools.bpva_benchmark.config_utils import register_benchmark_configs

    register_benchmark_configs()
    assert "TBot_SA1" in set(DatasetConfig.get_known_choices())
    assert "TBot_SA1" in set(PreTrainedConfig.get_known_choices())


def test_dataset_inventory_counts_multi_and_single():
    multi = SimpleNamespace(
        datasets=[SimpleNamespace(repo_id="a"), SimpleNamespace(repo_id="b")],
        num_frames=1234,
    )
    assert dataset_inventory(multi) == {"dataset_count": 2, "total_frames": 1234}

    single = SimpleNamespace(repo_id="only", num_frames=9)
    assert dataset_inventory(single) == {"dataset_count": 1, "total_frames": 9}

    wrapped = SimpleNamespace(current_ds=multi)
    assert dataset_inventory(wrapped) == {"dataset_count": 2, "total_frames": 1234}


def test_tbot_benchmark_config_parses_without_checkpoint():
    from pathlib import Path
    from tools.bpva_benchmark.train_tbot_benchmark import _load_tbot

    config_path = Path(__file__).parents[3] / "configs" / "tbot_sa1_pretrain_test.jsonc"
    cfg = _load_tbot(str(config_path))
    assert type(cfg.dataset).__name__ == "TBotSA1DatasetConfig"
    assert cfg.policy.type == "TBot_SA1"
    assert cfg.policy.pretrained_path is None
    assert cfg.batch_size == 8


def test_tbot_benchmark_rejects_non_tbot_policy():
    from types import SimpleNamespace
    import pytest
    from tools.bpva_benchmark.train_tbot_benchmark import _load_tbot

    cfg = SimpleNamespace(policy=SimpleNamespace(type="bpvav2"))
    with pytest.raises(ValueError, match="TBot_SA1"):
        _load_tbot("unused", base_loader=lambda _path: cfg)


def test_tbot_dataloader_benchmark_config_parses_without_checkpoint():
    from pathlib import Path
    from tools.bpva_benchmark.train_tbot_dataloader_benchmark import _load_tbot_cfg

    config_path = Path(__file__).parents[3] / "configs" / "tbot_sa1_pretrain_test.jsonc"
    cfg = _load_tbot_cfg(str(config_path))
    assert type(cfg.dataset).__name__ == "TBotSA1DatasetConfig"
    assert cfg.policy.type == "TBot_SA1"
    assert cfg.policy.pretrained_path is None


def test_tbot_dataloader_benchmark_rejects_non_tbot_policy():
    from types import SimpleNamespace
    import pytest
    from tools.bpva_benchmark.train_tbot_dataloader_benchmark import _load_tbot_cfg

    cfg = SimpleNamespace(policy=SimpleNamespace(type="bpvav2"))
    with pytest.raises(ValueError, match="TBot_SA1"):
        _load_tbot_cfg("unused", base_loader=lambda _path: cfg)
