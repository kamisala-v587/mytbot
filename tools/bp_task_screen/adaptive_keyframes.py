"""Offline comparison of keyframe sampling: density = alpha * uniform + (1 - alpha) * inverse joint speed.

Forced frames (gripper flips + last frame) are always kept; the remaining slots follow the density by quantiles.
Metrics per alpha (over the BP v2 training tasks, clean_50):
  * event_cov_unforced: share of gripper flips within +-2 frames of a density-sampled keyframe (forcing disabled);
  * idle_share: share of keyframes in near-static frames (speed < 10% of the episode median);
  * fast_share / fast_time: share of keyframes vs share of time in fast frames (speed > 2x median);
  * max_gap: median over episodes of the largest gap between consecutive keyframes (frames).
"""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen_data import load  # noqa: E402

K = 32
TIERS = yaml.safe_load(open("/vla/workspace/my_tbot/configs/ds_ids/pro6k/robo_clean_bp_v2_tiers.yaml"))
TASKS = sorted(t for v in TIERS.values() for t in v)


def flips(s: np.ndarray) -> list[int]:
    out = []
    for dim in (6, 13):
        closed = s[:, dim] < 0.5
        out += [int(t) for t in np.nonzero(closed[1:] != closed[:-1])[0] + 1]
    return sorted(set(out))


def sample(s: np.ndarray, alpha: float, force: bool) -> np.ndarray:
    speed = np.linalg.norm(np.diff(s[:, [*range(6), *range(7, 13)]], axis=0), axis=1)
    speed = np.convolve(np.r_[speed[:1], speed], np.ones(5) / 5, mode="same")
    w = 1.0 / (speed + 0.1 * np.median(speed) + 1e-8)
    density = alpha / len(s) + (1 - alpha) * w / w.sum()
    forced = sorted(set(flips(s) + [len(s) - 1])) if force else []
    n = K - len(forced)
    cdf = np.cumsum(density) / density.sum()
    picks = np.searchsorted(cdf, (np.arange(max(n, 0)) + 0.5) / max(n, 1))
    return np.unique(np.r_[picks, forced].astype(int)), speed


def main() -> None:
    eps = {t: load(t) for t in TASKS}
    print("alpha | event_cov_unforced | idle_share | fast_share (fast_time) | max_gap | unique_frames")
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        cov, idle, fast, fast_t, gaps, uniq = [], [], [], [], [], []
        for task in TASKS:
            for s in eps[task].values():
                kf, speed = sample(s, alpha, force=False)
                med = np.median(speed)
                ev = flips(s)
                if ev:
                    cov.append(np.mean([np.min(np.abs(kf - e)) <= 2 for e in ev]))
                kf_f, _ = sample(s, alpha, force=True)
                idle.append(np.mean(speed[np.clip(kf_f, 0, len(speed) - 1)] < 0.1 * med))
                fast.append(np.mean(speed[np.clip(kf_f, 0, len(speed) - 1)] > 2 * med))
                fast_t.append(np.mean(speed > 2 * med))
                gaps.append(np.diff(kf_f).max() if len(kf_f) > 1 else len(s))
                uniq.append(len(kf_f))
        print(f"{alpha:4.2f} | {np.mean(cov):.2f} | {np.mean(idle):.2f} | {np.mean(fast):.2f} ({np.mean(fast_t):.2f}) | "
              f"{np.median(gaps):.0f} | {np.mean(uniq):.1f}")


if __name__ == "__main__":
    main()
