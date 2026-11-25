"""Focused CPU tests for GPU sampling and DataLoader runtime defaults.

AST extraction exercises production statements without importing every policy backend.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker, append_loss_log

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "src/lerobot/scripts/lerobot_train.py"
TREE = ast.parse(TRAIN_PATH.read_text())
TRAIN = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "train")


def load_telemetry(cuda=True):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda), device=object)
    scope = {"AverageMeter": AverageMeter, "MetricsTracker": MetricsTracker, "torch": fake_torch}
    names = {"_GpuStatsMeter", "_maybe_add_gpu_metrics", "_update_gpu_metrics", "_meter_avg_or_val"}
    nodes = [n for n in TREE.body if getattr(n, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAIN_PATH), "exec"), scope)
    return scope


def tracker(scope):
    metrics = {"loss": AverageMeter("loss")}
    scope["_maybe_add_gpu_metrics"](metrics)
    return MetricsTracker(1, 100, 10, metrics)


def test_defaults_and_overrides():
    fields = TrainPipelineConfig.__dataclass_fields__
    assert fields["gpu_stats_freq"].default == 10
    assert fields["dataloader_persistent_workers"].default is True
    # Partial JSON / older saved configs inherit dataclass defaults when unspecified.
    assert TrainPipelineConfig(dataset=None, policy=None).gpu_stats_freq == 10
    for freq in (0, 1, 10):
        assert TrainPipelineConfig(dataset=None, policy=None, gpu_stats_freq=freq).gpu_stats_freq == freq


def test_negative_frequency_validation():
    config_tree = ast.parse((ROOT / "src/lerobot/configs/train.py").read_text())
    check = next(n for n in ast.walk(config_tree) if isinstance(n, ast.If)
                 and "gpu_stats_freq < 0" in ast.unparse(n.test))
    with pytest.raises(ValueError, match="non-negative"):
        exec(compile(ast.Module(body=[check], type_ignores=[]), str(TRAIN_PATH), "exec"),
             {"self": SimpleNamespace(gpu_stats_freq=-1)})


@pytest.mark.parametrize("workers,enabled,expected", [(0, True, False), (0, False, False),
                                                       (2, True, True), (2, False, False)])
def test_worker_guard(workers, enabled, expected):
    statement = next(n for n in ast.walk(TRAIN) if isinstance(n, ast.Assign)
                     and ast.unparse(n.targets[0]) == "dataloader_kwargs['persistent_workers']")
    scope = {"cfg": SimpleNamespace(dataloader_persistent_workers=enabled),
             "num_workers": workers, "dataloader_kwargs": {}}
    exec(compile(ast.Module(body=[statement], type_ignores=[]), str(TRAIN_PATH), "exec"), scope)
    assert scope["dataloader_kwargs"]["persistent_workers"] is expected
    loader = torch.utils.data.DataLoader([1, 2], num_workers=workers,
                                        persistent_workers=expected)
    assert list(loader)[0].item() == 1
    if expected:
        iterator = loader._iterator
        assert list(loader)[0].item() == 1
        assert loader._iterator is iterator
        iterator._shutdown_workers()


@pytest.mark.parametrize("freq,main,cuda,expected", [(10, True, True, [10, 20]),
    (1, True, True, list(range(1, 21))), (0, True, True, []),
    (10, False, True, []), (10, True, False, [])])
def test_sampling_counts_optimizer_updates_only(freq, main, cuda, expected):
    loop = next(n for n in ast.walk(TRAIN) if isinstance(n, ast.While))
    guard = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == "not did_step")
    increment = next(n for n in loop.body if isinstance(n, ast.AugAssign) and ast.unparse(n.target) == "step")
    sample = next(n for n in loop.body if isinstance(n, ast.If) and "cfg.gpu_stats_freq > 0" in ast.unparse(n.test))
    assert loop.body.index(guard) < loop.body.index(increment) < loop.body.index(sample)
    # Reuse the actual production guard, optimizer counter and sampling condition.
    synthetic = ast.parse("for did_step in updates:\n    pass")
    synthetic.body[0].body = [guard, increment, sample]
    scope = {"updates": [False, False, True] * 20, "step": 0,
             "cfg": SimpleNamespace(gpu_stats_freq=freq), "is_main_process": main,
             "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda)),
             "train_tracker": object(), "accelerator": SimpleNamespace(device="cuda", local_process_index=0)}
    seen = []
    scope["_update_gpu_metrics"] = lambda *args: seen.append(scope["step"])
    exec(compile(ast.fix_missing_locations(synthetic), str(TRAIN_PATH), "exec"), scope)
    assert seen == expected
    assert scope["step"] == 20


def test_disabled_and_cpu_do_not_register_gpu_metrics():
    scope = load_telemetry(False)
    assert not any(k.startswith("gpu_") for k in tracker(scope).metrics)
    registration = next(n for n in ast.walk(TRAIN) if isinstance(n, ast.If)
                        and ast.unparse(n.test) == "cfg.gpu_stats_freq > 0")
    callback = Mock()
    exec(compile(ast.Module(body=[registration], type_ignores=[]), str(TRAIN_PATH), "exec"),
         {"cfg": SimpleNamespace(gpu_stats_freq=0), "_maybe_add_gpu_metrics": callback, "train_metrics": {}})
    callback.assert_not_called()


def test_sample_statistics_and_logging_resets(tmp_path):
    scope = load_telemetry()
    t = tracker(scope)
    assert t.to_dict()["gpu_util_pct"] != t.to_dict()["gpu_util_pct"]  # unknown is NaN
    scope["_collect_gpu_stats"] = Mock(return_value={"gpu_util_pct": 60.0})
    scope["_update_gpu_metrics"](t, None, 0)
    scope["_collect_gpu_stats"].return_value = {"gpu_util_pct": 80.0}
    scope["_update_gpu_metrics"](t, None, 0)
    assert t.gpu_util_pct.count == 2
    assert t.to_dict()["gpu_util_pct"] == 70
    t.loss = 2.0
    t.reset_averages()
    assert t.loss.count == 0 and t.loss.val == 0
    assert t.gpu_util_pct.count == 0
    assert t.to_dict()["gpu_util_pct"] == 80
    assert t.to_dict(use_avg=False)["gpu_util_pct"] == 80
    assert t.to_postfix(True)["gpu_util"] == "80"
    assert scope["_meter_avg_or_val"](t.gpu_util_pct) == 80
    assert "gpu_util:80" in str(t)
    append_loss_log(tmp_path / "loss.log", t)
    lines = (tmp_path / "loss.log").read_text().splitlines()
    assert lines[1].split(",")[lines[0].split(",").index("gpu_util_pct")] == "80.000000"
    scope["_collect_gpu_stats"].return_value = {}  # failed sample must not invent zero
    scope["_update_gpu_metrics"](t, None, 0)
    assert t.gpu_util_pct.count == 0 and t.gpu_util_pct.val == 80
    scope["_collect_gpu_stats"].return_value = {"gpu_util_pct": 40.0}
    scope["_update_gpu_metrics"](t, None, 0)
    assert t.gpu_util_pct.count == 1 and t.gpu_util_pct.avg == 40
