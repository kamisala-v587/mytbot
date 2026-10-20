"""Offline check: does picking the demo whose first head-camera frame looks most similar give a demo with the
same grasp sequence (same arm for single-arm tasks, same first arm for handovers)?

Each dataset episode plays the role of a test scene; candidates are other episodes of the same task.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen_data import close_events, load  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset

TASKS = ["place_cans_plasticbox", "scan_object", "place_shoe", "rotate_qrcode", "place_mouse_pad", "place_empty_cup",
         "move_can_pot", "place_phone_stand", "handover_mic", "place_can_basket"]
ROOT = Path("/vla/workspace/data/robotwin2.0")
HEAD = "observation.images.cam_high"


def thumb(image: torch.Tensor) -> torch.Tensor:
    image = image.float()
    if image.shape[0] != 3 and image.shape[-1] == 3:
        image = image.permute(2, 0, 1)
    if image.max() > 1.5:
        image = image / 255.0
    return F.adaptive_avg_pool2d(image[None], (28, 28))[0].flatten()


def first_frames(task: str) -> dict[int, torch.Tensor]:
    ds = LeRobotDataset(str(ROOT / task / "aloha-agilex_clean_50"), video_backend="torchcodec")
    out = {}
    for ep in range(ds.meta.total_episodes):
        start = int(ds.meta.episodes[ep]["dataset_from_index"])
        out[ep] = thumb(ds[start][HEAD])
    return out


def pick(query: torch.Tensor, pool: list[int], thumbs: dict[int, torch.Tensor]) -> int:
    d = torch.stack([(thumbs[e] - query).square().mean() for e in pool])
    return pool[int(d.argmin())]


def main() -> None:
    report = {}
    for task in TASKS:
        seqs = {ep: "".join(a for a, _ in close_events(s)) for ep, s in load(task).items()}
        thumbs = first_frames(task)
        eps = sorted(thumbs)
        runtime_pool = random.Random(f"0:{task}").sample(range(len(eps)), 10)
        groups: dict[str, list[int]] = {}
        for e in eps:
            groups.setdefault(seqs[e], []).append(e)
        rng = random.Random(0)
        balanced = [e for g in groups.values() for e in rng.sample(g, min(len(g), 10 // max(1, len(groups))))]
        res = {}
        for name, pool in (("runtime_pool10", runtime_pool), ("balanced_pool10", balanced), ("all_others", eps)):
            hits, worst_hits, n = 0, 0, 0
            for q in eps:
                cand = [e for e in pool if e != q]
                if not cand:
                    continue
                best = pick(thumbs[q], cand, thumbs)
                d = torch.stack([(thumbs[e] - thumbs[q]).square().mean() for e in cand])
                worst = cand[int(d.argmax())]
                hits += seqs[best] == seqs[q]
                worst_hits += seqs[worst] == seqs[q]
                n += 1
            res[name] = {"best_match": hits / n, "worst_match": worst_hits / n, "n": n,
                         "pool_seqs": {k: sum(seqs[e] == k for e in pool) for k in groups}}
        res["random_match"] = float(sum((len(g) / len(eps)) ** 2 for g in groups.values()))
        report[task] = res
        print(f"{task:22s} seqs={ {k: len(v) for k, v in groups.items()} } random={res['random_match']:.2f} "
              + " ".join(f"{k}: best={v['best_match']:.2f} worst={v['worst_match']:.2f}" for k, v in res.items() if isinstance(v, dict)),
              flush=True)
    Path(sys.argv[1]).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
