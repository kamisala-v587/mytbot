"""Current-observation ViT cache for the main-path dataset (`dataset.obs_cache_dir`).

For every episode of the index: all frames x 3 camera slots through the same preprocessing as training
(`BPV4CDXBMainDataset.current_inputs`: decode, resize_with_pad, Qwen processor; a missing camera is a black image)
and the frozen Qwen3.5 ViT (batched attention as in training) ->
  <out>/<rel>/epXXXXX.npy   [T, 3, tokens, D] bf16 bits stored as int16
Only valid for models whose ViT is the frozen base ViT (BP B-4: identical to Qwen base) and for unaugmented images.
Resumable (existing files are skipped); one process per rank, ranks split the repos.

  python -m lerobot.policies.BPV4CDXB.build_obs_cache --train_config <ckpt>/train_config.json --index <index.json> \
      --out /vla/workspace/cache/bpv4cdxb_obs_vit --rank 0 --world 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

CHUNK = 16  # frames per ViT batch (x 3 slots)


class Chunks(torch.utils.data.Dataset):
    def __init__(self, ds, jobs) -> None:
        self.ds, self.jobs = ds, jobs

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, i):
        eid, start, length = self.jobs[i]
        frames = list(range(start, min(start + CHUNK, length)))
        out = self.ds.current_inputs(eid, frames)
        return {"i": i, "pixels": out["obs.pixel_values"].to(torch.bfloat16), "grid": out["obs.image_grid_thw"],
                "n": len(frames)}


def main() -> None:
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.BPV4BPEn.build_feature_cache import load_vit, visual_tokens

    from .dataset_main import BPV4CDXBMainDataset
    from .vit_batched import patch_vision_attention

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_config", required=True, help="train_config.json of a main-path run (dataset settings)")
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qwen_path", default="/vla/workspace/models/q35/base_800m")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(a.out)

    index = json.loads(Path(a.index).read_text())
    rels = sorted({v[0] for v in index["episodes"].values()})
    mine = set(rels[a.rank::a.world])
    todo = {eid: v for eid, v in index["episodes"].items()
            if v[0] in mine and not (out / v[0] / f"ep{int(v[1]):05d}.npy").is_file()}
    logging.info("rank %d/%d: %d repos, %d episodes to build", a.rank, a.world, len(mine), len(todo))
    if not todo:
        return
    keep = set(todo)
    sub = dict(index, buckets=[dict(b, train=[e for e in b["train"] if e in keep], val=[]) for b in index["buckets"]],
               episodes={e: index["episodes"][e] for e in keep}, load_procs=8)
    sub["buckets"] = [b for b in sub["buckets"] if b["train"]]
    cfg = TrainPipelineConfig.from_pretrained(str(Path(a.train_config).parent)) if Path(a.train_config).is_file() else None
    if cfg is None:
        raise FileNotFoundError(a.train_config)
    cfg.dataset.split = "train"
    cfg.dataset.obs_cache_dir = ""
    cfg.dataset.pixel_transfer_dtype = "bfloat16"
    torch.multiprocessing.set_start_method("spawn", force=True)
    ds = BPV4CDXBMainDataset(cfg.dataset, cfg, cfg.policy, sub)

    jobs = [(eid, s, int(v[2])) for eid, v in sorted(todo.items()) for s in range(0, int(v[2]), CHUNK)]
    loader = torch.utils.data.DataLoader(Chunks(ds, jobs), batch_size=None, num_workers=a.workers,
                                         prefetch_factor=2, persistent_workers=False)
    device = torch.device("cuda")
    vlm = load_vit(None, a.qwen_path, device)
    # the policy casts the whole model to bf16, buffers included: its ViT rotary inv_freq is bf16 (transformers keeps
    # it float32); without this the tokens differ from the training path by ~4% (mean |diff| / mean |x|)
    rot = vlm.model.visual.rotary_pos_emb
    rot.inv_freq = rot.inv_freq.to(torch.bfloat16)
    patch_vision_attention(vlm.model.visual)
    pending: dict[str, list] = {}
    done, t0, frames_done = 0, time.time(), 0
    with torch.no_grad():
        for b in loader:
            eid, start, length = jobs[int(b["i"])]
            feats = visual_tokens(vlm, b["pixels"].to(device), b["grid"].to(device))  # [n x 3, tokens, D]
            feats = feats.reshape(int(b["n"]), -1, *feats.shape[1:]).to(torch.bfloat16).cpu()
            pending.setdefault(eid, []).append((start, feats))
            frames_done += int(b["n"])
            got = sum(f.shape[0] for _, f in pending[eid])
            if got == length:
                arr = torch.cat([f for _, f in sorted(pending.pop(eid), key=lambda x: x[0])])
                rel, ep = todo[eid][0], int(todo[eid][1])
                path = out / rel / f"ep{ep:05d}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".tmp{os.getpid()}.npy")
                np.save(tmp, arr.view(torch.int16).numpy())
                os.replace(tmp, path)
                done += 1
                if done % 50 == 0:
                    rate = frames_done / (time.time() - t0)
                    logging.info("rank %d: %d / %d episodes, %.0f frames/s", a.rank, done, len(todo), rate)
    logging.info("rank %d finished: %d episodes in %.0f s", a.rank, done, time.time() - t0)


if __name__ == "__main__":
    main()
