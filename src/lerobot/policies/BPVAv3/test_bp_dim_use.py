import torch

from lerobot.policies.BPVAv3.configuration_bpvav3 import BPVAv3DatasetConfig, parse_bp_dim_use
from lerobot.policies.BPVAv3.transform_bpvav3 import BPVAv3SelectPromptDimsFn
from lerobot.transforms.core_bp import BP_PREFIX


def test_parse_slices_and_bools():
    assert parse_bp_dim_use(True, name="bp_action_use").mode == "all"
    assert parse_bp_dim_use(False, name="bp_action_use").mode == "none"
    assert parse_bp_dim_use("0:6", name="bp_action_use").resolve(14) == [0, 1, 2, 3, 4, 5]
    assert parse_bp_dim_use("0:6,7:13", name="bp_action_use").resolve(14) == list(range(6)) + list(range(7, 13))
    assert parse_bp_dim_use([6, 13], name="bp_action_use").resolve(32) == [6, 13]
    assert parse_bp_dim_use("[0:6]", name="bp_state_use").resolve(32) == list(range(6))


def test_select_prompt_dims_keeps_tokens_for_partial_mask():
    action = torch.arange(2 * 3 * 14, dtype=torch.float32).reshape(2, 3, 14)
    state = torch.arange(2 * 14, dtype=torch.float32).reshape(2, 14) + 100
    data = {
        BP_PREFIX: {
            "action": action.clone(),
            "state": state.clone(),
            "action_is_pad": torch.zeros(2, 3, dtype=torch.bool),
            "state_is_available": torch.ones(2, dtype=torch.bool),
        }
    }
    out = BPVAv3SelectPromptDimsFn(bp_action_use="0:6,7:13", bp_state_use=[0, 1, 2])(data)[BP_PREFIX]
    assert torch.equal(out["action"][..., :6], action[..., :6])
    assert torch.equal(out["action"][..., 7:13], action[..., 7:13])
    assert torch.count_nonzero(out["action"][..., 6]) == 0
    assert torch.count_nonzero(out["action"][..., 13]) == 0
    assert not bool(out["action_is_pad"].any())
    assert torch.equal(out["state"][..., :3], state[..., :3])
    assert torch.count_nonzero(out["state"][..., 3:]) == 0
    assert bool(out["state_is_available"].all())


def test_select_prompt_dims_drops_whole_modality():
    data = {
        BP_PREFIX: {
            "action": torch.ones(2, 4, 32),
            "state": torch.ones(2, 32),
            "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
            "state_is_available": torch.ones(2, dtype=torch.bool),
        }
    }
    out = BPVAv3SelectPromptDimsFn(bp_action_use=False, bp_state_use=False)(data)[BP_PREFIX]
    assert torch.count_nonzero(out["action"]) == 0
    assert torch.count_nonzero(out["state"]) == 0
    assert bool(out["action_is_pad"].all())
    assert not bool(out["state_is_available"].any())


def test_dataset_config_accepts_defaults_and_rejects_bad_index():
    cfg = BPVAv3DatasetConfig(repo_id="test", bp_action_use=True, bp_state_use="0:6")
    assert cfg.bp_action_use is True
    try:
        BPVAv3DatasetConfig(repo_id="test", bp_action_use="0:40")
    except ValueError:
        return
    raise AssertionError("index past max_action_dim should fail")
