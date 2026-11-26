#!/usr/bin/env python3
"""Compute RobotWin 80D mixed norm stats: joint dims delta, gripper dims abs.

delta dims: left joints 0-5, right joints 40-45
abs dims:   left gripper 16, right gripper 56
Other slots are empty for RoboTwin and are skipped in the active masks.

For general cleaned 80D roots (AgiBot / InternData / …) prefer
`tools/bpv5/compute_80d_mixed_norm_stats.py`, which uses the full BPv5
delta/abs table and discovers leaves from cleaning-result.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TOOLS = Path(__file__).resolve().parents[1]
REPO = TOOLS.parent
sys.path[:0] = [str(TOOLS), str(REPO / "src")]

from norm_stats.core import RunningStats  # noqa: E402

DELTA_DIMS = list(range(0, 6)) + list(range(40, 46))
ABS_DIMS = [16, 56]
ACTIVE = sorted(DELTA_DIMS + ABS_DIMS)
CHUNK = 50
ACTION_DIM = 80


def episode_table(repo: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted((repo / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {repo}/data")
    eps, acts, sts = [], [], []
    for f in files:
        t = pq.read_table(f, columns=["episode_index", "action_80D", "state_80D"])
        eps.append(t.column("episode_index").to_numpy())
        acts.append(np.stack(t.column("action_80D").to_numpy(zero_copy_only=False)).astype(np.float64))
        sts.append(np.stack(t.column("state_80D").to_numpy(zero_copy_only=False)).astype(np.float64))
    return np.concatenate(eps), np.concatenate(acts), np.concatenate(sts)


def targets_for_episode(action: np.ndarray, state: np.ndarray, chunk: int) -> np.ndarray:
    length = len(action)
    if length < chunk:
        return np.zeros((0, ACTION_DIM), dtype=np.float64)
    starts = np.arange(0, length - chunk + 1, dtype=np.int64)
    out = np.zeros((len(starts) * chunk, ACTION_DIM), dtype=np.float64)
    row = 0
    for t0 in starts:
        base = state[t0]
        chunk_a = action[t0 : t0 + chunk]
        tgt = chunk_a.copy()
        tgt[:, DELTA_DIMS] = chunk_a[:, DELTA_DIMS] - base[DELTA_DIMS]
        out[row : row + chunk] = tgt
        row += chunk
    return out


def compute_repo(repo: Path, chunk: int) -> dict:
    ep, action, state = episode_table(repo)
    action_stats = RunningStats()
    state_stats = RunningStats()
    n_ep = 0
    n_win = 0
    for e in np.unique(ep):
        idx = np.flatnonzero(ep == e)
        a = action[idx]
        s = state[idx]
        state_stats.update(s)
        tgt = targets_for_episode(a, s, chunk)
        if len(tgt):
            action_stats.update(tgt)
            n_win += len(tgt) // chunk
        n_ep += 1
    return {
        "repo": str(repo),
        "episodes": n_ep,
        "frames": int(len(ep)),
        "delta_windows": n_win,
        "action_payload": action_stats.to_payload(),
        "state_payload": state_stats.to_payload(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--repo-id-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--chunk-size", type=int, default=CHUNK)
    args = ap.parse_args()

    repos = [ln.strip() for ln in args.repo_id_file.read_text().splitlines() if ln.strip()]
    action_merged = RunningStats()
    state_merged = RunningStats()
    parts_meta = []
    for i, rel in enumerate(repos, 1):
        repo = args.root / rel
        print(f"[{i}/{len(repos)}] {rel}", flush=True)
        part = compute_repo(repo, args.chunk_size)
        action_merged.merge(RunningStats.from_payload(part["action_payload"]))
        state_merged.merge(RunningStats.from_payload(part["state_payload"]))
        parts_meta.append({k: part[k] for k in ("repo", "episodes", "frames", "delta_windows")})

    delta_mask = np.zeros(ACTION_DIM, dtype=np.float64)
    abs_mask = np.zeros(ACTION_DIM, dtype=np.float64)
    action_mask = np.zeros(ACTION_DIM, dtype=np.float64)
    state_mask = np.zeros(ACTION_DIM, dtype=np.float64)
    delta_mask[DELTA_DIMS] = 1.0
    abs_mask[ABS_DIMS] = 1.0
    action_mask[ACTIVE] = 1.0
    state_mask[ACTIVE] = 1.0

    out = {
        "action_80D": action_merged.statistics(),
        "state_80D": state_merged.statistics(),
        "metadata": {
            "robot_type": "aloha",
            "group_name": "aloha_rt_clean_mixed",
            "action_mode": "mixed",
            "chunk_size": args.chunk_size,
            "action_definition": (
                f"delta: action[t+k]-state[t] on dims {DELTA_DIMS}; "
                f"abs: action[t+k] on dims {ABS_DIMS}"
            ),
            "delta_mask": delta_mask.tolist(),
            "abs_mask": abs_mask.tolist(),
            "action_mask": action_mask.tolist(),
            "state_mask": state_mask.tolist(),
            "datasets": [p["repo"] for p in parts_meta],
            "episodes": int(sum(p["episodes"] for p in parts_meta)),
            "frames": int(sum(p["frames"] for p in parts_meta)),
            "delta_windows": int(sum(p["delta_windows"] for p in parts_meta)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print("wrote", args.output)
    a = out["action_80D"]
    print("active action mean:")
    for d in ACTIVE:
        kind = "delta" if d in DELTA_DIMS else "abs"
        print(f"  d{d:02d} {kind:5} mean={a['mean'][d]: .4f} std={a['std'][d]:.4f}")


if __name__ == "__main__":
    main()
