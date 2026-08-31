from datetime import datetime
from types import SimpleNamespace

import pytest

from tools.bpva_benchmark.data_benchmark import (
    build_parser as build_data_parser,
    _effective_num_workers,
    disable_dist_loading_for_single_process,
)
from tools.bpva_benchmark.reporting import resolve_output_dir
from tools.bpva_benchmark.train_benchmark import build_parser as build_train_parser


def test_register_bpva_configs_populates_choice_registry():
    """Regression test for `DecodingError: Couldn't find a choice class for 'bpva'`.

    `TrainPipelineConfig.from_pretrained` resolves `dataset.type`/`policy.type`
    via draccus `ChoiceRegistry`, which is only populated once
    `configuration_bpva` has been imported. The benchmark entrypoints must
    trigger that import explicitly since, unlike `lerobot_train.py`, they have
    no other import path that pulls it in as a side effect.
    """
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.policies import PreTrainedConfig
    from tools.bpva_benchmark.config_utils import register_bpva_configs

    register_bpva_configs()
    assert "bpva" in DatasetConfig.get_known_choices()
    assert "bpva" in PreTrainedConfig.get_known_choices()


def test_load_cfg_helpers_import_register_bpva_configs():
    """`data_benchmark._load_cfg` / `train_benchmark._load` must route through
    `config_utils.register_bpva_configs` rather than only
    `register_third_party_plugins`, otherwise parsing a `type: "bpva"` config
    fails outside of `lerobot_train.py`."""
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
