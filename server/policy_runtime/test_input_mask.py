from __future__ import annotations

import numpy as np
import torch

from policy_runtime.input_mask import (
    apply_client_masks,
    apply_model_masks,
    parse_mask_list,
    parse_mask_rules,
)


def test_parse_indices_and_stage() -> None:
    client, model = parse_mask_rules(
        ["observation.state[6,13]", "behavior_prompt.action"]
    )
    assert client.stage == "client" and client.indices == (6, 13)
    assert model.stage == "model" and model.indices is None


def test_client_state_and_camera() -> None:
    obs = {
        "observation.state": np.arange(14, dtype=np.float32),
        "observation.images.cam_high": np.ones((4, 4, 3), dtype=np.uint8),
        "observation.images.cam_left_wrist": np.ones((4, 4, 3), dtype=np.uint8),
        "task": "pick the bottle",
    }
    rules = parse_mask_rules(
        [
            "observation.state[6,13]",
            "observation.images.cam_left_wrist",
            "task",
        ]
    )
    masked = apply_client_masks(obs, rules)
    assert masked["observation.state"][6] == 0
    assert masked["observation.state"][13] == 0
    assert masked["observation.state"][0] == 0  # wait, arange 0 is 0 already
    assert masked["observation.state"][1] == 1
    assert "observation.images.cam_left_wrist" not in masked
    assert "observation.images.cam_high" in masked
    assert masked["task"] == ""
    assert obs["task"] == "pick the bottle"


def test_model_action_full_and_gripper() -> None:
    inputs = {
        "behavior_prompt": {
            "action": torch.ones(1, 8, 50, 32),
            "action_is_pad": torch.zeros(1, 8, 50, dtype=torch.bool),
            "state": torch.ones(1, 8, 32),
            "mask": torch.ones(1, 8, dtype=torch.bool),
            "bp_pixel_values": {"observation.images.image0": torch.ones(1, 8, 16, 4)},
            "image_masks": {"observation.images.image0": torch.ones(1, 8, dtype=torch.bool)},
        }
    }
    apply_model_masks(inputs, parse_mask_rules(["behavior_prompt.action"]))
    prompt = inputs["behavior_prompt"]
    assert float(prompt["action"].abs().sum()) == 0.0
    assert bool(prompt["action_is_pad"].all())
    assert bool(prompt["mask"].all())

    inputs["behavior_prompt"]["action"] = torch.ones(1, 8, 50, 32)
    inputs["behavior_prompt"]["action_is_pad"] = torch.zeros(1, 8, 50, dtype=torch.bool)
    apply_model_masks(inputs, parse_mask_rules(["behavior_prompt.action[6,13]"]))
    action = inputs["behavior_prompt"]["action"]
    assert float(action[..., 6].abs().sum()) == 0.0
    assert float(action[..., 13].abs().sum()) == 0.0
    assert float(action[..., 0].abs().sum()) > 0
    assert not bool(inputs["behavior_prompt"]["action_is_pad"].any())


def test_dotted_bp_camera_and_entire_prompt() -> None:
    inputs = {
        "behavior_prompt": {
            "action": torch.ones(1, 2, 4, 8),
            "action_is_pad": torch.zeros(1, 2, 4, dtype=torch.bool),
            "mask": torch.ones(1, 2, dtype=torch.bool),
            "bp_pixel_values": {
                "observation.images.image0": torch.ones(1, 2, 3, 4),
                "observation.images.image1": torch.ones(1, 2, 3, 4),
            },
            "bp_image_grid_thw": {
                "observation.images.image0": torch.tensor([[1, 16, 16], [1, 16, 16]]),
                "observation.images.image1": torch.tensor([[1, 16, 16], [1, 16, 16]]),
            },
            "image_masks": {
                "observation.images.image0": torch.ones(1, 2, dtype=torch.bool),
                "observation.images.image1": torch.ones(1, 2, dtype=torch.bool),
            },
        }
    }
    apply_model_masks(
        inputs,
        parse_mask_rules(["behavior_prompt.bp_pixel_values.observation.images.image0"]),
    )
    pixels = inputs["behavior_prompt"]["bp_pixel_values"]
    assert float(pixels["observation.images.image0"].abs().sum()) == 0.0
    assert float(pixels["observation.images.image1"].abs().sum()) > 0
    assert not bool(inputs["behavior_prompt"]["image_masks"]["observation.images.image0"].any())
    assert bool(inputs["behavior_prompt"]["image_masks"]["observation.images.image1"].all())

    grids = {
        key: value.clone()
        for key, value in inputs["behavior_prompt"]["bp_image_grid_thw"].items()
    }
    apply_model_masks(inputs, parse_mask_rules(["behavior_prompt"]))
    prompt = inputs["behavior_prompt"]
    assert not bool(prompt["mask"].any())
    assert bool(prompt["action_is_pad"].all())
    assert float(prompt["action"].abs().sum()) == 0.0
    assert float(prompt["bp_pixel_values"]["observation.images.image1"].abs().sum()) == 0.0
    for key, value in grids.items():
        assert torch.equal(prompt["bp_image_grid_thw"][key], value)


def test_cli_keeps_indices_inside_brackets() -> None:
    specs = parse_mask_list("behavior_prompt.state[6,13],behavior_prompt.bp_pixel_values")
    rules = parse_mask_rules(specs)
    assert rules[0].path == ("behavior_prompt", "state")
    assert rules[0].indices == (6, 13)
    assert rules[1].path == ("behavior_prompt", "bp_pixel_values")


def test_state_mask_creates_unavailable_flag() -> None:
    inputs = {"behavior_prompt": {"state": torch.ones(2, 4)}}
    apply_model_masks(inputs, parse_mask_rules(["behavior_prompt.state"]))
    prompt = inputs["behavior_prompt"]
    assert torch.count_nonzero(prompt["state"]) == 0
    assert prompt["state_is_available"].shape == (2,)
    assert not bool(prompt["state_is_available"].any())


def test_pixel_mask_creates_image_masks() -> None:
    inputs = {"behavior_prompt": {"bp_pixel_values": {"image0": torch.ones(2, 8, 3)}}}
    apply_model_masks(inputs, parse_mask_rules(["behavior_prompt.bp_pixel_values"]))
    masks = inputs["behavior_prompt"]["image_masks"]
    assert masks["image0"].shape == (2,)
    assert not bool(masks["image0"].any())


if __name__ == "__main__":
    test_parse_indices_and_stage()
    test_client_state_and_camera()
    test_model_action_full_and_gripper()
    test_dotted_bp_camera_and_entire_prompt()
    test_cli_keeps_indices_inside_brackets()
    test_state_mask_creates_unavailable_flag()
    test_pixel_mask_creates_image_masks()
    print("ok")
