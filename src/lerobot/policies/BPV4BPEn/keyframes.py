"""Adaptive keyframe sampling (v2) and gripper-event segmentation for one trajectory (aloha 14-dim state)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEFT_GRIPPER, RIGHT_GRIPPER = 6, 13
JOINTS = [*range(6), *range(7, 13)]


@dataclass
class KeyframePlan:
    frames: np.ndarray  # [K_eff] frame indices, sorted
    forced: np.ndarray  # [K_eff] bool, gripper flip or last active frame
    flips: np.ndarray  # gripper flip frames inside the active span, sorted
    active: tuple[int, int]  # [lo, hi]
    length: int


def gripper_flips(state: np.ndarray) -> list[int]:
    out: set[int] = set()
    for dim in (LEFT_GRIPPER, RIGHT_GRIPPER):
        if state.shape[1] <= dim:
            continue
        closed = state[:, dim] < 0.5
        out.update(int(t) for t in np.nonzero(closed[1:] != closed[:-1])[0] + 1)
    return sorted(out)


def motion_speed(state: np.ndarray) -> np.ndarray:
    """Joint speed plus gripper opening/closing speed (a closing gripper counts as motion), 5-frame smoothed."""
    d = np.abs(np.diff(state, axis=0))
    joints = [j for j in JOINTS if j < state.shape[1]]
    v = np.linalg.norm(d[:, joints], axis=1)
    for dim in (LEFT_GRIPPER, RIGHT_GRIPPER):
        if state.shape[1] > dim:
            v = v + d[:, dim]
    v = np.r_[v[:1], v] if len(v) else np.zeros(len(state))
    return np.convolve(v, np.ones(5) / 5, mode="same")


def plan_keyframes(state: np.ndarray, num_keyframes: int = 32, alpha: float = 0.5) -> KeyframePlan:
    """density = alpha * uniform + (1 - alpha) / speed on the active span; forced gripper flips + last active frame;
    minimum gap max(2, ceil(active / (2K))); short trajectories may yield fewer than K frames."""
    n = len(state)
    if n <= 1:
        return KeyframePlan(np.zeros(1, int), np.ones(1, bool), np.zeros(0, int), (0, 0), n)
    speed = motion_speed(state)
    med = float(np.median(speed))
    active = np.nonzero(speed > 0.1 * med)[0]
    lo, hi = (int(active[0]), int(active[-1])) if len(active) else (0, n - 1)
    lo, hi = max(0, lo - 2), min(n - 1, hi + 2)
    gap = max(2, int(np.ceil((hi - lo + 1) / (2 * num_keyframes))))
    flips = [f for f in gripper_flips(state) if lo <= f <= hi]
    forced = sorted(set(flips) | {hi})
    seg = speed[lo : hi + 1]
    w = 1.0 / (seg + 0.1 * med + 1e-8)
    density = alpha / len(seg) + (1 - alpha) * w / w.sum()
    cdf = np.cumsum(density) / density.sum()
    picks = lo + np.searchsorted(cdf, (np.arange(num_keyframes) + 0.5) / num_keyframes)
    kept = list(forced)
    for p in sorted(set(int(x) for x in picks) - set(forced)):
        if len(kept) >= num_keyframes:
            break
        if all(abs(p - q) >= gap for q in kept):
            kept.append(p)
    kept = sorted(kept)[:num_keyframes]
    while len(kept) < num_keyframes:
        bounds = [lo - gap, *kept, hi + gap]
        gaps = np.diff(bounds)
        i = int(gaps.argmax())
        if gaps[i] < 2 * gap:
            break
        cand = min(max((bounds[i] + bounds[i + 1]) // 2, lo), hi)
        if any(abs(cand - q) < gap for q in kept):
            break
        kept = sorted(kept + [cand])
    frames = np.array(kept, dtype=int)
    forced_mask = np.isin(frames, np.array(forced, dtype=int))
    return KeyframePlan(frames, forced_mask, np.array(flips, dtype=int), (lo, hi), n)


def event_segment(flips: np.ndarray, active: tuple[int, int], frame: int) -> tuple[int, int]:
    """Segment (s, e] containing `frame`, bounded by gripper flips; the first segment is [lo, b1]."""
    lo, hi = active
    bounds = [lo - 1, *[int(b) for b in flips], hi]
    for s, e in zip(bounds[:-1], bounds[1:]):
        if s < frame <= e:
            return s, e
    return (lo - 1, bounds[1]) if frame <= lo else (bounds[-2], hi)
