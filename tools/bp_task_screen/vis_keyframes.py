"""Visualize adaptive keyframe sampling (alpha=0.5, forced gripper flips + last active frame, idle head/tail trimmed)
on one clean episode per task: joint-speed timeline with gripper states / segments, plus the 32 head-camera keyframes."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_keyframes import flips  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path("/vla/workspace/data/robotwin2.0")
OUT = Path(sys.argv[1])
VERSION = sys.argv[2] if len(sys.argv) > 2 else "v1"
K, ALPHA = 32, 0.5
JOINTS = [*range(6), *range(7, 13)]


def joint_speed(s: np.ndarray) -> np.ndarray:
    v = np.linalg.norm(np.diff(s[:, JOINTS], axis=0), axis=1)
    return np.convolve(np.r_[v[:1], v], np.ones(5) / 5, mode="same")


def motion_speed(s: np.ndarray) -> np.ndarray:
    """Joint speed plus gripper opening/closing speed, so a closing gripper counts as motion."""
    d = np.abs(np.diff(s, axis=0))
    v = np.linalg.norm(d[:, JOINTS], axis=1) + d[:, 6] + d[:, 13]
    return np.convolve(np.r_[v[:1], v], np.ones(5) / 5, mode="same")


def keyframes_v2(s: np.ndarray) -> tuple[np.ndarray, list[int], tuple[int, int]]:
    """v1 + gripper motion in the speed + minimum gap max(2, active / (2K)); short episodes may yield < K frames."""
    speed = motion_speed(s)
    active = np.nonzero(speed > 0.1 * np.median(speed))[0]
    lo, hi = (int(active[0]), int(active[-1])) if len(active) else (0, len(s) - 1)
    lo, hi = max(0, lo - 2), min(len(s) - 1, hi + 2)
    gap = max(2, int(np.ceil((hi - lo + 1) / (2 * K))))
    forced = sorted({f for f in flips(s) if lo <= f <= hi} | {hi})
    seg = speed[lo : hi + 1]
    w = 1.0 / (seg + 0.1 * np.median(speed) + 1e-8)
    density = ALPHA / len(seg) + (1 - ALPHA) * w / w.sum()
    cdf = np.cumsum(density) / density.sum()
    picks = lo + np.searchsorted(cdf, (np.arange(K) + 0.5) / K)
    kept = list(forced)
    for p in sorted(set(int(x) for x in picks) - set(forced)):
        if len(kept) >= K:
            break
        if all(abs(p - q) >= gap for q in kept):
            kept.append(p)
    kept = sorted(kept)
    while len(kept) < K:  # fill the largest gaps while respecting the minimum spacing
        bounds = [lo - gap, *kept, hi + gap]
        gaps = np.diff(bounds)
        i = int(gaps.argmax())
        if gaps[i] < 2 * gap:
            break
        cand = min(max((bounds[i] + bounds[i + 1]) // 2, lo), hi)
        if any(abs(cand - q) < gap for q in kept):
            break
        kept = sorted(kept + [cand])
    return np.array(kept), forced, (lo, hi)


def keyframes(s: np.ndarray) -> tuple[np.ndarray, list[int], tuple[int, int]]:
    speed = joint_speed(s)
    active = np.nonzero(speed > 0.1 * np.median(speed))[0]
    lo, hi = (int(active[0]), int(active[-1])) if len(active) else (0, len(s) - 1)
    lo, hi = max(0, lo - 2), min(len(s) - 1, hi + 2)
    forced = sorted({f for f in flips(s) if lo <= f <= hi} | {hi})
    seg = speed[lo : hi + 1]
    w = 1.0 / (seg + 0.1 * np.median(speed) + 1e-8)
    density = ALPHA / len(seg) + (1 - ALPHA) * w / w.sum()
    n = K - len(forced)
    cdf = np.cumsum(density) / density.sum()
    picks = lo + np.searchsorted(cdf, (np.arange(n) + 0.5) / n)
    kf = np.unique(np.r_[picks, forced].astype(int))
    while len(kf) < K:  # fill duplicates collapsed by unique with the largest gaps
        gaps = np.diff(kf)
        i = int(gaps.argmax())
        kf = np.unique(np.r_[kf, (kf[i] + kf[i + 1]) // 2])
    return kf, forced, (lo, hi)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = sorted(p.name for p in ROOT.iterdir() if (p / "aloha-agilex_clean_50").is_dir())
    for task in tasks:
        ds = LeRobotDataset(str(ROOT / task / "aloha-agilex_clean_50"), episodes=[0], video_backend="torchcodec")
        s = np.stack([ds.hf_dataset[i]["observation.state"].numpy() for i in range(len(ds.hf_dataset))])
        kf, forced, (lo, hi) = keyframes_v2(s) if VERSION == "v2" else keyframes(s)
        speed = motion_speed(s) if VERSION == "v2" else joint_speed(s)
        fig = plt.figure(figsize=(16, 9.5))
        ax = fig.add_axes([0.05, 0.74, 0.92, 0.22])
        ax.plot(speed, color="#4C84C4", lw=1, label="joint speed")
        ax.fill_between(range(len(s)), 0, speed.max() * (s[:, 6] < 0.5), color="#5B9B5B", alpha=0.15, label="L gripper closed")
        ax.fill_between(range(len(s)), 0, speed.max() * (s[:, 13] < 0.5), color="#C85A5A", alpha=0.15, label="R gripper closed")
        ax.axvspan(0, lo, color="gray", alpha=0.2)
        ax.axvspan(hi, len(s) - 1, color="gray", alpha=0.2, label="trimmed idle")
        for f in kf:
            ax.axvline(f, color="#D9902B" if f in forced else "black", lw=1.6 if f in forced else 0.6, alpha=0.9)
        ax.set_xlim(0, len(s) - 1)
        ax.set_title(f"{task}  |  episode 0, {len(s)} frames  |  {VERSION}: {len(kf)}/{K} keyframes, alpha={ALPHA}  |  "
                     "orange = forced (gripper flip / last active)")
        ax.legend(loc="upper right", fontsize=8, ncol=4)
        for j, f in enumerate(kf):
            img = ds[int(f)]["observation.images.cam_high"].permute(1, 2, 0).numpy()
            a = fig.add_axes([0.05 + (j % 8) * 0.116, 0.52 - (j // 8) * 0.165, 0.11, 0.155])
            a.imshow(img)
            a.set_xticks([]), a.set_yticks([])
            color = "#D9902B" if f in forced else "black"
            for sp in a.spines.values():
                sp.set_color(color), sp.set_linewidth(2.5 if f in forced else 0.5)
            a.set_title(f"#{j} t={f / (len(s) - 1):.2f}", fontsize=8, color=color)
        fig.savefig(OUT / f"{task}.png", dpi=80)
        plt.close(fig)
        print(task, len(s), "forced", len(forced), "span", (lo, hi), flush=True)


if __name__ == "__main__":
    main()
