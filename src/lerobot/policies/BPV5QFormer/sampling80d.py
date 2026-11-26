"""S-U on the 80-D state (BPv5): 32 frames uniform over the active span; the active span comes from the valid state
dims (state_mask) of any robot, each dim's frame-to-frame change divided by its group state std, max over dims.
Smoothing, threshold and padding as AE-3 (`BPV4CDX.sampling.active_span`: 5-frame mean, > 0.1 x median, 2 frames).
"""

from __future__ import annotations

import numpy as np

from lerobot.policies.BPV4CDX.sampling import SlotPlan, _fill, active_span

SAMPLER = "su80"
SAMPLER_VERSION = "su80-uniform_active_state80_v1"


def motion_speed_80d(state: np.ndarray, valid: np.ndarray, state_std: np.ndarray) -> np.ndarray:
    """state [T, 80], valid [80] bool, state_std [80] -> per-frame speed [T] (5-frame smoothed)."""
    dims = np.nonzero(valid & (state_std > 1e-6))[0]
    if len(state) < 2 or not len(dims):
        return np.zeros(len(state))
    v = (np.abs(np.diff(state[:, dims], axis=0)) / state_std[dims]).max(axis=1)
    v = np.r_[v[:1], v]
    return np.convolve(v, np.ones(5) / 5, mode="same")


def sample_su80(state: np.ndarray, valid: np.ndarray, state_std: np.ndarray, num_slots: int = 32) -> SlotPlan:
    n = len(state)
    lo, hi = active_span(motion_speed_80d(state, valid, state_std)) if n > 1 else (0, 0)
    frames, repeats = _fill(list(np.round(np.linspace(lo, hi, num_slots)).astype(int)), num_slots, lo, hi)
    # SlotPlan.version is not used for su80 (its registry is BPV4CDX's); records carry SAMPLER_VERSION
    return SlotPlan(frames, repeats, (lo, hi), n, np.zeros(0, dtype=int), SAMPLER)
