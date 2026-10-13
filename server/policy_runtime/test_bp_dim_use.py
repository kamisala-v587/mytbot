from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from policy_runtime.bp_dim_use import apply_bp_dim_use, load_bp_dim_spec, resolve_bp_dim_use


def test_load_jsonc_dataset_fields(tmp_path: Path) -> None:
    path = tmp_path / "bpv3.jsonc"
    path.write_text(
        """
        {
            // 训练配置
            "dataset": {
                "bp_action_use": false,
                "bp_state_use": "0:6,7:13"
            }
        }
        """,
        encoding="utf-8",
    )
    action_use, state_use = load_bp_dim_spec(path)
    assert action_use is False
    assert state_use == "0:6,7:13"


def test_explicit_json_overrides_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "dims.json"
    path.write_text('{"bp_action_use": false, "bp_state_use": true}\n', encoding="utf-8")
    train_cfg = SimpleNamespace(dataset=SimpleNamespace(bp_action_use=True, bp_state_use=False))
    action_use, state_use, source = resolve_bp_dim_use(str(path), train_cfg)
    assert action_use is False
    assert state_use is True
    assert source == str(path)

    action_use, state_use, source = resolve_bp_dim_use("", train_cfg)
    assert action_use is True
    assert state_use is False
    assert source == "checkpoint train_config.json"

    action_use, state_use, source = resolve_bp_dim_use(None, None)
    assert action_use is True and state_use is True and source == "default"


def test_apply_matches_training_and_keeps_cache() -> None:
    action = torch.arange(8 * 50 * 32, dtype=torch.float32).reshape(1, 8, 50, 32)
    state = torch.arange(8 * 32, dtype=torch.float32).reshape(1, 8, 32) + 3
    cached = {
        "action": action,
        "state": state,
        "action_is_pad": torch.zeros(1, 8, 50, dtype=torch.bool),
        "state_is_available": torch.ones(1, 8, dtype=torch.bool),
        "mask": torch.ones(1, 8, dtype=torch.bool),
    }
    selected = apply_bp_dim_use(cached, False, "0:6,7:13")
    assert float(selected["action"].abs().sum()) == 0.0
    assert bool(selected["action_is_pad"].all())
    assert torch.equal(selected["state"][..., :6], state[..., :6])
    assert torch.equal(selected["state"][..., 7:13], state[..., 7:13])
    assert float(selected["state"][..., 6].abs().sum()) == 0.0
    assert float(selected["state"][..., 13].abs().sum()) == 0.0
    assert bool(selected["state_is_available"].all())
    assert bool(selected["mask"].all())
    assert cached["action"] is action
    assert float(action.abs().sum()) > 0
    assert not bool(cached["action_is_pad"].any())


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        test_load_jsonc_dataset_fields(root)
        test_explicit_json_overrides_checkpoint(root)
    test_apply_matches_training_and_keeps_cache()
    print("ok")
