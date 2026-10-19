"""Offline check: pick a demo for an episode by comparing frame-0 head images at low resolution.

For single-arm tasks, report how often the best / worst match uses the same arm as the query episode,
against the same-arm fraction of the pool (random pick).
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_v31 import arm_labels  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

ROOT = "/vla/workspace/data/robotwin2.0/{}/aloha-agilex_clean_50"
TASKS = sys.argv[1].split(",")
POOL = int(sys.argv[2]) if len(sys.argv) > 2 else 10
HEAD = "observation.images.cam_high"


def thumb(img: torch.Tensor) -> torch.Tensor:
    img = img.float()
    if img.max() > 1.5:
        img = img / 255.0
    return F.adaptive_avg_pool2d(img[None], (24, 32))[0].flatten()


rng = random.Random(0)
summary = {}
for task in TASKS:
    repo = ROOT.format(task)
    arms = arm_labels(repo, 3.0)
    ds = LeRobotDataset(repo, video_backend="torchcodec", skip_video_file_validation=True)
    starts = {}
    for i, ep in enumerate(ds.hf_dataset["episode_index"]):
        starts.setdefault(int(ep), i)
    feats = {ep: thumb(ds[idx][HEAD]) for ep, idx in starts.items()}
    single = [e for e in feats if arms[e] in ("left", "right")]
    best_hit, worst_hit, rand_frac = [], [], []
    for q in single:
        pool = rng.sample([e for e in feats if e != q], POOL)
        d = torch.stack([(feats[q] - feats[e]).square().mean() for e in pool])
        best_hit.append(arms[pool[int(d.argmin())]] == arms[q])
        worst_hit.append(arms[pool[int(d.argmax())]] == arms[q])
        rand_frac.append(np.mean([arms[e] == arms[q] for e in pool]))
    summary[task] = {"best_same_arm": float(np.mean(best_hit)), "worst_same_arm": float(np.mean(worst_hit)),
                     "random_same_arm": float(np.mean(rand_frac)), "n": len(single)}
    print(task, summary[task], flush=True)
print(json.dumps(summary, indent=1))
