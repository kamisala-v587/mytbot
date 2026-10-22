"""32-slot demo samplers (task book 4.2 / 4.3).

S-A  bpv4 adaptive sampling v2 (state speed + gripper flips), i.e. `BPV4BPEn.keyframes.plan_keyframes`, with the
     active-span start additionally forced as the first slot (the last active frame is already forced)
S-U  uniform over the active span (same active span as S-A)
S-V  video-only: density from a per-frame visual-change proxy, no state, no forced gripper frames
S-UF (AE-3) S-U, then every gripper event (embodiment.gripper_events) moved onto its nearest free slot; same slot
     count, uniform elsewhere

With an Embodiment (AE-3, any robot) the active span and the gripper events come from the MASK_MAPPING joint /
gripper dims (embodiment.py); without one (AE-1 / AE-2, aloha) the original bpv4 functions are used unchanged.
Every sampler returns exactly `num_slots` frames; if fewer distinct frames are available, the nearest frames are
repeated and the repeat count is recorded (it is never given to the model).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lerobot.policies.BPV4BPEn.keyframes import gripper_flips, motion_speed, plan_keyframes, plan_keyframes_from

from . import embodiment as E

SAMPLERS = ("sa", "su", "sv", "suf")
# bump when the selection rule changes: the cache and the frozen delivery record it
SAMPLER_VERSIONS = {"sa": "sa-keyframes_v2+force_lo", "su": "su-uniform_active_v1", "sv": "sv-vit_diff_v1",
                    "su_emb": "su-uniform_active_v2_embodiment", "suf": "suf-uniform_active+gripper_events_v1",
                    "sa_emb": "sa-keyframes_v2+force_lo_embodiment"}


@dataclass
class SlotPlan:
    frames: np.ndarray  # [num_slots] sorted frame indices (may repeat)
    repeats: int  # number of slots filled by repeating an already chosen frame
    active: tuple[int, int]
    length: int
    flips: np.ndarray  # gripper flips from state, for analysis only (not used by S-U / S-V selection)
    sampler: str
    alpha: float = 0.5
    version_key: str | None = None  # SAMPLER_VERSIONS key when it differs from the sampler name

    @property
    def forced(self) -> np.ndarray:
        """[num_slots] bool: slots on a forced frame (S-A / S-UF: gripper events + lo + hi; S-U / S-V: lo + hi).
        Record only."""
        lo, hi = self.active
        keep = {lo, hi} | (set(self.flips.tolist()) if self.sampler in ("sa", "suf") else set())
        return np.isin(self.frames, np.array(sorted(keep), dtype=int))

    @property
    def version(self) -> dict:
        return {"name": self.sampler, "version": SAMPLER_VERSIONS[self.version_key or self.sampler],
                "num_slots": int(len(self.frames)), "alpha": float(self.alpha) if self.sampler in ("sa", "sv") else None}


def active_span(speed: np.ndarray, rel_threshold: float = 0.1, pad: int = 2) -> tuple[int, int]:
    n = len(speed)
    med = float(np.median(speed))
    idx = np.nonzero(speed > rel_threshold * med)[0]
    lo, hi = (int(idx[0]), int(idx[-1])) if len(idx) else (0, n - 1)
    return max(0, lo - pad), min(n - 1, hi + pad)


def _fill(frames: list[int], num_slots: int, lo: int, hi: int) -> tuple[np.ndarray, int]:
    frames = sorted(set(int(f) for f in frames))
    if not frames:
        frames = [lo]
    repeats = 0
    while len(frames) < num_slots:
        # largest gap between neighbours (including the span ends); new frame at its midpoint, else repeat
        bounds = [lo, *frames, hi]
        gaps = np.diff(bounds)
        i = int(gaps.argmax())
        cand = (bounds[i] + bounds[i + 1]) // 2
        if cand in frames or gaps[i] < 2:
            break
        frames = sorted(frames + [cand])
    # still short (very short trajectory): repeat frames spread over the span; each copy sits next to its source
    missing = num_slots - len(frames)
    out = list(frames)
    if missing > 0:
        out += [frames[int(i)] for i in np.round(np.linspace(0, len(frames) - 1, missing))]
        repeats = missing
    return np.array(sorted(out[:num_slots]), dtype=int), repeats


def _force_start(frames: list[int], lo: int, gap: int, protected: set[int]) -> list[int]:
    """Make `lo` the first slot: move the first slot there if it is within one gap, else insert it and drop the
    unprotected frame whose removal leaves the smallest gap."""
    frames = sorted(frames)
    if not frames or frames[0] == lo:
        return frames
    if frames[0] - lo <= gap and frames[0] not in protected:
        return [lo, *frames[1:]]
    frames = sorted([lo, *frames])
    candidates = [i for i in range(1, len(frames) - 1) if frames[i] not in protected]
    if candidates:
        drop = min(candidates, key=lambda i: frames[i + 1] - frames[i - 1])
        frames.pop(drop)
    return frames


def sample_sa(state: np.ndarray, num_slots: int = 32, alpha: float = 0.5) -> SlotPlan:
    """bpv4 adaptive sampling v2, plus the active-span start forced as the first slot (task book 4.2)."""
    return _sa_from_plan(plan_keyframes(state, num_slots, alpha), num_slots, alpha)


def sample_sa_emb(state: np.ndarray, emb: E.Embodiment, num_slots: int = 32, alpha: float = 0.5) -> SlotPlan:
    """S-A for any embodiment: the same density rule on the MASK_MAPPING joint / gripper speed, gripper events
    (mid-range crossings) forced."""
    if len(state) <= 1:
        return sample_sa(state, num_slots, alpha)
    plan = plan_keyframes_from(E.motion_speed(state, emb), E.gripper_events(state, emb), num_slots, alpha)
    return _sa_from_plan(plan, num_slots, alpha, version_key="sa_emb")


def _sa_from_plan(plan, num_slots: int, alpha: float, version_key: str | None = None) -> SlotPlan:
    lo, hi = plan.active
    gap = max(2, int(np.ceil((hi - lo + 1) / (2 * num_slots))))
    kept = _force_start(list(plan.frames), lo, gap, protected=set(plan.flips.tolist()) | {hi})
    frames, repeats = _fill(kept, num_slots, lo, hi)
    return SlotPlan(frames, repeats, plan.active, plan.length, plan.flips, "sa", alpha, version_key=version_key)


def sample_su(state: np.ndarray, num_slots: int = 32) -> SlotPlan:
    n = len(state)
    lo, hi = active_span(motion_speed(state)) if n > 1 else (0, 0)
    frames = np.round(np.linspace(lo, hi, num_slots)).astype(int)
    frames, repeats = _fill(list(frames), num_slots, lo, hi)
    flips = np.array([f for f in gripper_flips(state) if lo <= f <= hi], dtype=int)
    return SlotPlan(frames, repeats, (lo, hi), n, flips, "su")


def sample_su_emb(state: np.ndarray, emb: E.Embodiment, num_slots: int = 32) -> SlotPlan:
    """S-U for any embodiment: active span from the MASK_MAPPING joint / gripper speed."""
    n = len(state)
    lo, hi = active_span(E.motion_speed(state, emb)) if n > 1 else (0, 0)
    frames, repeats = _fill(list(np.round(np.linspace(lo, hi, num_slots)).astype(int)), num_slots, lo, hi)
    flips = np.array([f for f in E.gripper_events(state, emb) if lo <= f <= hi], dtype=int)
    return SlotPlan(frames, repeats, (lo, hi), n, flips, "su", version_key="su_emb")


def sample_suf(state: np.ndarray, emb: E.Embodiment, num_slots: int = 32) -> SlotPlan:
    """S-UF: S-U, then each gripper event moved onto its nearest free slot (the first and last slots stay on the
    active-span ends). More events than inner slots: an evenly spaced subset of them is kept."""
    n = len(state)
    lo, hi = active_span(E.motion_speed(state, emb)) if n > 1 else (0, 0)
    frames = np.round(np.linspace(lo, hi, num_slots)).astype(int)
    events = [f for f in E.gripper_events(state, emb) if lo < f < hi]
    inner = num_slots - 2
    if len(events) > inner:
        events = [events[int(i)] for i in np.round(np.linspace(0, len(events) - 1, inner))]
    taken = {0, num_slots - 1}
    for e in events:
        free = [j for j in range(num_slots) if j not in taken]
        if not free:
            break
        j = min(free, key=lambda j: abs(int(frames[j]) - e))
        frames[j] = e
        taken.add(j)
    frames, repeats = _fill(list(frames), num_slots, lo, hi)
    return SlotPlan(frames, repeats, (lo, hi), n, np.array(events, dtype=int), "suf")


def video_proxy_speed(frame_features: np.ndarray) -> np.ndarray:
    """[T, D] per-frame pooled visual features -> [T] smoothed feature-change magnitude (task book 4.3)."""
    d = np.linalg.norm(np.diff(frame_features, axis=0), axis=1)
    v = np.r_[d[:1], d] if len(d) else np.zeros(len(frame_features))
    return np.convolve(v, np.ones(5) / 5, mode="same")


def sample_sv(frame_features: np.ndarray, num_slots: int = 32, alpha: float = 0.5,
              state: np.ndarray | None = None) -> SlotPlan:
    """Same density rule as S-A, with the video proxy instead of the state speed and no forced gripper frames."""
    speed = video_proxy_speed(frame_features)
    n = len(speed)
    lo, hi = active_span(speed) if n > 1 else (0, 0)
    gap = max(2, int(np.ceil((hi - lo + 1) / (2 * num_slots))))
    med = float(np.median(speed))
    seg = speed[lo : hi + 1]
    w = 1.0 / (seg + 0.1 * med + 1e-8)
    density = alpha / len(seg) + (1 - alpha) * w / w.sum()
    cdf = np.cumsum(density) / density.sum()
    picks = lo + np.searchsorted(cdf, (np.arange(num_slots) + 0.5) / num_slots)
    kept = [lo, hi]
    for p in sorted(set(int(x) for x in picks)):
        if len(kept) >= num_slots:
            break
        if all(abs(p - q) >= gap for q in kept):
            kept.append(p)
    frames, repeats = _fill(kept, num_slots, lo, hi)
    flips = np.array([f for f in gripper_flips(state) if lo <= f <= hi], dtype=int) if state is not None else np.zeros(0, int)
    return SlotPlan(frames, repeats, (lo, hi), n, flips, "sv", alpha)


def sample(sampler: str, state: np.ndarray, num_slots: int = 32, alpha: float = 0.5,
           frame_features: np.ndarray | None = None, emb: E.Embodiment | None = None) -> SlotPlan:
    """emb = None keeps the AE-1 / AE-2 aloha samplers (bit-identical caches); with an embodiment, S-A / S-U / S-UF
    use its joint / gripper dims."""
    if sampler == "sa":
        return sample_sa_emb(state, emb, num_slots, alpha) if emb is not None else sample_sa(state, num_slots, alpha)
    if sampler == "su":
        return sample_su_emb(state, emb, num_slots) if emb is not None else sample_su(state, num_slots)
    if sampler == "suf":
        if emb is None:
            raise ValueError("sampler suf needs the embodiment (gripper dims)")
        return sample_suf(state, emb, num_slots)
    if sampler == "sv":
        if frame_features is None:
            raise ValueError("sampler sv needs per-frame visual features")
        return sample_sv(frame_features, num_slots, alpha, state=state)
    raise ValueError(f"unknown sampler {sampler!r}, expected one of {SAMPLERS}")


def interpolation_error(state: np.ndarray, frames: np.ndarray) -> float:
    """Coverage metric: RMS error of reconstructing the joint trajectory by linear interpolation between slots."""
    t = np.arange(len(state))
    uniq = np.unique(frames)
    if len(uniq) < 2:
        return float("nan")
    recon = np.stack([np.interp(t, uniq, state[uniq, d]) for d in range(state.shape[1])], axis=1)
    lo, hi = int(uniq[0]), int(uniq[-1])
    return float(np.sqrt(np.mean((recon[lo : hi + 1] - state[lo : hi + 1]) ** 2)))
