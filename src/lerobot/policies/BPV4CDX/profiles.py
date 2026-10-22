"""Time profiles of a per-slot quantity (saliency or attention mass), aligned to an anchor, with episode-level CIs.

Each target segment k of an episode gives a mass over the 32 demo slots (summing to 1 over the slots it covers). A
profile bins the slot time minus an anchor time (the segment centre, the first gripper close, or the place = first
reopen; all normalized by episode length) and reports the mean mass per (segment, slot) pair in each bin x 32, so a
value of 1 means "as much as an average slot". The 95% CI resamples episodes (all segments of an episode together);
bins with fewer than `min_pairs` pairs are left empty (None), never plotted as 0.
"""

from __future__ import annotations

import numpy as np

ANCHORS = ("segment", "grasp", "place")


def episode_bins(times: np.ndarray, masses: np.ndarray, anchors: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """times [32]; masses [S, 32]; anchors [S] (one anchor time per segment) -> (mass sum, pair count) per bin."""
    nb = len(bins) - 1
    mass, count = np.zeros(nb), np.zeros(nb)
    for m, a in zip(masses, anchors):
        d = times - a
        idx = np.digitize(d, bins) - 1
        ok = (idx >= 0) & (idx < nb)
        np.add.at(mass, idx[ok], m[ok])
        np.add.at(count, idx[ok], 1)
    return mass, count


def profile(per_episode: list[tuple[np.ndarray, np.ndarray]], bins: np.ndarray, n_slots: int = 32, n_boot: int = 1000,
            min_pairs: int = 30, seed: int = 0) -> dict:
    """per_episode: (mass sum, pair count) per bin for each episode -> mean / CI / pairs per bin."""
    if not per_episode:
        return {"centres": ((bins[:-1] + bins[1:]) / 2).tolist(), "mean": [], "lo": [], "hi": [], "pairs": [], "episodes": 0}
    mass = np.stack([m for m, _ in per_episode])
    count = np.stack([c for _, c in per_episode])
    tot_c = count.sum(0)
    mean = np.where(tot_c > 0, mass.sum(0) / np.maximum(tot_c, 1), np.nan) * n_slots
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(mass), len(mass))
        c = count[pick].sum(0)
        boots.append(np.where(c > 0, mass[pick].sum(0) / np.maximum(c, 1), np.nan) * n_slots)
    lo, hi = np.nanpercentile(np.stack(boots), [2.5, 97.5], axis=0)
    keep = tot_c >= min_pairs
    f = lambda x: [float(v) if k else None for v, k in zip(x, keep)]  # noqa: E731
    return {"centres": ((bins[:-1] + bins[1:]) / 2).tolist(), "mean": f(mean), "lo": f(lo), "hi": f(hi),
            "pairs": tot_c.astype(int).tolist(), "episodes": int(len(mass))}


def anchor_times(starts: list[int], steps: list[int], length: int, grasp: int | None, place: int | None) -> dict:
    """Per-segment anchor times (normalized) for each anchor kind; None if the event does not exist."""
    L = max(1, length - 1)
    out = {"segment": np.array([(s + (n - 1) / 2) / L for s, n in zip(starts, steps)])}
    out["grasp"] = np.full(len(starts), grasp / L) if grasp is not None else None
    out["place"] = np.full(len(starts), place / L) if place is not None else None
    return out


def plot_profiles(ax, profiles: dict, title: str) -> None:
    """profiles: label -> profile dict; draws mean with CI band, skipping empty bins."""
    for label, p in profiles.items():
        x = np.array(p["centres"], dtype=float)
        y = np.array([np.nan if v is None else v for v in p["mean"]], dtype=float)
        lo = np.array([np.nan if v is None else v for v in p["lo"]], dtype=float)
        hi = np.array([np.nan if v is None else v for v in p["hi"]], dtype=float)
        line, = ax.plot(x, y, marker="o", ms=3, label=f"{label} ({p['episodes']} ep)")
        ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.2)
    ax.axhline(1, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
