"""Input-level reference: cross-demo keyframe alignment using raw (std-normalized) keyframe states only."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

shard = Path(sys.argv[1])
run = torch.load(shard, map_location="cpu", weights_only=False)
cache: dict[str, dict[int, np.ndarray]] = {}


def episodes(repo: str) -> dict[int, np.ndarray]:
    if repo not in cache:
        df = pd.concat(pd.read_parquet(p, columns=["episode_index", "observation.state"])
                       for p in sorted(Path(repo, "data").rglob("*.parquet")))
        states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
        std = states.std(0) + 1e-6
        cache[repo] = {int(ep): np.stack(g["observation.state"].to_numpy()).astype(np.float32) / std
                       for ep, g in df.groupby("episode_index")}
    return cache[repo]


def keyframes(s: np.ndarray, k: int = 8) -> np.ndarray:
    return s[np.round(np.linspace(0, len(s) - 1, k)).astype(int)]


def align(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = ((a[:, None] - b[None]) ** 2).sum(-1)
    j = d.argmin(1)
    i = np.arange(len(j))
    return float((j == i).mean()), float((np.abs(j - i) <= 1).mean())


seen, out = set(), {"same_arm_all": [], "same_arm_single": [], "cross_arm_single": []}
for r in run["results"]:
    key = (r["task"], r["episode"])
    if key in seen:
        continue
    seen.add(key)
    eps = episodes(r["repo"])
    cur = keyframes(eps[r["episode"]])
    if r["other_arm"] == r["arm"]:
        v = align(cur, keyframes(eps[r["other_episode"]]))
        out["same_arm_all"].append(v)
        if r["arm"] in ("left", "right"):
            out["same_arm_single"].append(v)
    if r["arm"] in ("left", "right") and r["other_x_episode"] is not None:
        out["cross_arm_single"].append(align(cur, keyframes(eps[r["other_x_episode"]])))
print(json.dumps({k: {"top1": float(np.mean([x[0] for x in v])), "within1": float(np.mean([x[1] for x in v])), "n": len(v)}
                  for k, v in out.items() if v}, indent=1))
