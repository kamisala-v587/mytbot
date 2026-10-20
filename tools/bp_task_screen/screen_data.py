"""Per-task cross-demo consistency metrics for RoboTwin clean_50 (state-only, no video decoding).

For each episode:
  * arm label: left / right / both by total joint motion (ratio 3);
  * grasp sequence: arms of gripper-close events (gripper < 0.5 for >= 3 frames), e.g. "LR" for a handover;
  * canonical sequence: the same with arms renamed by first use (A = first arm), so a mirrored episode matches.
Per task:
  * raw_seq_mode / canon_seq_mode: share of episodes with the most common raw / canonical sequence;
  * event_time_std: std of normalized event times inside the modal canonical group (phase alignment by normalized time);
  * kf_align_matched: keyframe nearest-neighbour alignment (8 keyframes, std-normalized state) between episodes
    with the same raw sequence (= same arms in the same order);
  * len_cv: episode length coefficient of variation.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/vla/workspace/data/robotwin2.0")
LEFT_GRIP, RIGHT_GRIP = 6, 13


def load(task: str) -> dict[int, np.ndarray]:
    repo = ROOT / task / "aloha-agilex_clean_50"
    df = pd.concat(pd.read_parquet(p, columns=["episode_index", "observation.state"])
                   for p in sorted((repo / "data").rglob("*.parquet")))
    return {int(ep): np.stack(g["observation.state"].to_numpy()).astype(np.float32) for ep, g in df.groupby("episode_index")}


def arm_label(s: np.ndarray, ratio: float = 3.0) -> str:
    m = np.abs(np.diff(s, axis=0)).sum(0)
    left, right = m[:7].sum(), m[7:14].sum()
    return "L" if left > ratio * right else "R" if right > ratio * left else "B"


def close_events(s: np.ndarray, min_frames: int = 3) -> list[tuple[str, float]]:
    events = []
    for arm, dim in (("L", LEFT_GRIP), ("R", RIGHT_GRIP)):
        closed = s[:, dim] < 0.5
        t = 1
        while t < len(closed):
            if closed[t] and not closed[t - 1] and closed[t : t + min_frames].all():
                events.append((arm, t / max(1, len(s) - 1)))
            t += 1
    return sorted(events, key=lambda e: e[1])


def canonical(seq: str) -> str:
    names, out = {}, []
    for a in seq:
        if a not in names:
            names[a] = "AB"[len(names)]
        out.append(names[a])
    return "".join(out)


def keyframes(s: np.ndarray, k: int = 8) -> np.ndarray:
    return s[np.round(np.linspace(0, len(s) - 1, k)).astype(int)]


def kf_align(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = ((a[:, None] - b[None]) ** 2).sum(-1)
    j, i = d.argmin(1), np.arange(len(a))
    return float((j == i).mean()), float((np.abs(j - i) <= 1).mean())


def analyze(task: str) -> dict:
    eps = load(task)
    std = np.concatenate(list(eps.values()))[:, :14].std(0) + 1e-6
    info = {}
    for ep, s in eps.items():
        ev = close_events(s)
        seq = "".join(a for a, _ in ev)
        info[ep] = {"arm": arm_label(s), "seq": seq, "canon": canonical(seq), "times": [t for _, t in ev], "len": len(s)}
    raw = Counter(v["seq"] for v in info.values())
    can = Counter(v["canon"] for v in info.values())
    mode_canon = can.most_common(1)[0][0]
    group = [v for v in info.values() if v["canon"] == mode_canon and v["times"]]
    times = np.array([v["times"] for v in group]) if group else np.zeros((0, 0))
    by_raw: dict[str, list[int]] = {}
    for ep, v in info.items():
        by_raw.setdefault(v["seq"], []).append(ep)
    kf = []
    for members in by_raw.values():
        for a, b in list(combinations(members, 2))[:200]:
            kf.append(kf_align(keyframes(eps[a][:, :14] / std), keyframes(eps[b][:, :14] / std)))
    lens = np.array([v["len"] for v in info.values()])
    arms = Counter(v["arm"] for v in info.values())
    return {
        "task": task,
        "n": len(info),
        "arm_counts": {k: arms.get(k, 0) for k in "LRB"},
        "n_events_mode": Counter(len(v["seq"]) for v in info.values()).most_common(1)[0],
        "raw_seq_top": raw.most_common(3),
        "raw_seq_mode": raw.most_common(1)[0][1] / len(info),
        "canon_seq_top": can.most_common(3),
        "canon_seq_mode": can.most_common(1)[0][1] / len(info),
        "event_time_std": float(times.std(0).mean()) if times.size else None,
        "kf_align_matched": {"top1": float(np.mean([x[0] for x in kf])), "within1": float(np.mean([x[1] for x in kf])),
                             "pairs": len(kf)} if kf else None,
        "len_cv": float(lens.std() / lens.mean()),
        "len_mean": float(lens.mean()),
    }


if __name__ == "__main__":
    tasks = sorted(p.name for p in ROOT.iterdir() if (p / "aloha-agilex_clean_50").is_dir())
    out = [analyze(t) for t in tasks]
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    for r in out:
        kf = r["kf_align_matched"]
        print(f"{r['task']:26s} arms L/R/B={r['arm_counts']['L']:2d}/{r['arm_counts']['R']:2d}/{r['arm_counts']['B']:2d} "
              f"raw={r['raw_seq_mode']:.2f} canon={r['canon_seq_mode']:.2f} top={r['canon_seq_top'][0][0] or '-':6s} "
              f"evt_std={(r['event_time_std'] or 0):.3f} kf={kf['top1'] if kf else float('nan'):.2f}/{kf['within1'] if kf else float('nan'):.2f} "
              f"len={r['len_mean']:.0f}±{r['len_cv']:.2f}")
