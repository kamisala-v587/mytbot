"""Per-repo embodiment description from the q35 mappings (transforms/constants.py), so that the samplers, the cache,
the dataset and the gates do not hard-code the aloha layout (14 dims, grippers 6 / 13, cam_high / cam_*_wrist).

  state / action keys   FEATURE_MAPPING (concatenated in mapping order)
  joint / gripper dims  MASK_MAPPING: delta dims = joints, absolute dims = grippers
  camera slots          IMAGE_MAPPING -> image0 / image1 / image2 (head, left or single wrist, right); a missing
                        camera is None (masked with camera_valid, never copied from another camera)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lerobot.transforms.constants import get_feature_mapping, get_image_mapping, get_mask_mapping, infer_embodiment_variant
from lerobot.utils.constants import ACTION, OBS_STATE

NUM_CAMERAS = 3
SLOT_NAMES = ("image0", "image1", "image2")


@dataclass
class Embodiment:
    robot_type: str
    resolved: str
    state_keys: list[str]
    action_keys: list[str]
    state_sizes: list[int]
    action_sizes: list[int]
    mask: np.ndarray  # [state dim] bool, True = joint (delta), False = gripper (absolute)
    cameras: list[str | None]  # source key per slot (image0..image2)

    @property
    def joint_dims(self) -> np.ndarray:
        return np.flatnonzero(self.mask)

    @property
    def gripper_dims(self) -> np.ndarray:
        return np.flatnonzero(~self.mask)

    @property
    def dual_arm_61(self) -> bool:
        """aloha-style bimanual layout: 6 joints + 1 gripper per arm (RoboTwin aloha, ARX Lift-2, Split Aloha)."""
        return len(self.mask) == 14 and self.gripper_dims.tolist() == [6, 13]

    @property
    def arms(self) -> dict[str, tuple[slice, int]]:
        """Arm -> (joint slice, gripper dim); only for the dual-arm 6+1 layout."""
        return {"L": (slice(0, 6), 6), "R": (slice(7, 13), 13)} if self.dual_arm_61 else {}

    @property
    def camera_valid(self) -> list[bool]:
        return [c is not None for c in self.cameras]


def _sizes(features: dict, keys: list[str]) -> list[int]:
    return [int(np.prod(features[k]["shape"])) for k in keys]


def embodiment_from_meta(robot_type: str, features: dict) -> Embodiment:
    spec = get_feature_mapping(robot_type, features)
    state_keys, action_keys = list(spec[OBS_STATE]), list(spec[ACTION])
    state_sizes, action_sizes = _sizes(features, state_keys), _sizes(features, action_keys)
    mask = np.asarray(get_mask_mapping(robot_type, features), dtype=bool)
    if len(mask) < sum(state_sizes):
        mask = np.concatenate([mask, np.zeros(sum(state_sizes) - len(mask), dtype=bool)])
    mask = mask[: sum(state_sizes)]
    image_map = get_image_mapping(robot_type, features)
    cameras: list[str | None] = [None] * NUM_CAMERAS
    for src, dst in image_map.items():
        if src in features:
            slot = dst.rsplit(".", 1)[-1]
            if slot in SLOT_NAMES and cameras[SLOT_NAMES.index(slot)] is None:
                cameras[SLOT_NAMES.index(slot)] = src
    if cameras[0] is None and not any(cameras):
        raise KeyError(f"no camera of {robot_type} maps to image0..image2")
    return Embodiment(robot_type, infer_embodiment_variant(robot_type, features), state_keys, action_keys,
                      state_sizes, action_sizes, mask, cameras)


# --- gripper events and motion speed --------------------------------------------------------------------------------
def _gripper_range(g: np.ndarray) -> tuple[float, float] | None:
    lo, hi = float(g.min()), float(g.max())
    if hi - lo <= max(1e-4, 0.05 * max(abs(hi), abs(lo))):
        return None  # the gripper does not move in this episode
    return lo, hi


def gripper_events(state: np.ndarray, emb: Embodiment) -> list[int]:
    """Frames where a gripper signal crosses the middle of its own range in this episode (opening or closing; no
    assumption about which value means closed, units differ between robots)."""
    out: set[int] = set()
    for d in emb.gripper_dims:
        g = state[:, d]
        rng = _gripper_range(g)
        if rng is None:
            continue
        above = g > 0.5 * (rng[0] + rng[1])
        out.update(int(t) for t in np.nonzero(above[1:] != above[:-1])[0] + 1)
    return sorted(out)


def motion_speed(state: np.ndarray, emb: Embodiment) -> np.ndarray:
    """Joint speed + gripper speed (each gripper scaled by its range in the episode), 5-frame smoothed. On aloha
    (grippers in [0, 1]) this equals the bpv4 keyframes.motion_speed."""
    d = np.abs(np.diff(state, axis=0))
    v = np.linalg.norm(d[:, emb.joint_dims], axis=1) if len(emb.joint_dims) else np.zeros(len(d))
    for g in emb.gripper_dims:
        rng = _gripper_range(state[:, g])
        if rng is not None:
            v = v + d[:, g] / (rng[1] - rng[0])
    v = np.r_[v[:1], v] if len(v) else np.zeros(len(state))
    return np.convolve(v, np.ones(5) / 5, mode="same")


def first_close(state: np.ndarray, emb: Embodiment) -> tuple[str, int] | None:
    """Dual-arm 6+1 only: the arm whose gripper first closes and the frame. Closed = below half of the gripper value
    at the first frame (all these robots start with open grippers; on aloha this is the AE-2 threshold 0.5)."""
    best = None
    for arm, (_, g) in emb.arms.items():
        open_value = float(state[0, g])
        if open_value <= 0:
            continue
        closed = state[:, g] < 0.5 * open_value
        idx = np.nonzero(closed[1:] & ~closed[:-1])[0]
        if len(idx) and (best is None or idx[0] + 1 < best[1]):
            best = (arm, int(idx[0] + 1))
    return best


def first_reopen(state: np.ndarray, emb: Embodiment, arm: str, after: int) -> int | None:
    _, g = emb.arms[arm]
    opened = state[after:, g] >= 0.5 * float(state[0, g])
    idx = np.nonzero(opened)[0]
    return int(after + idx[0]) if len(idx) else None
