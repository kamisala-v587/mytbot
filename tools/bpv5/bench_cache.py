"""BPv5 cache build throughput: wall ViT images/s (per-image vs batched attention, outputs compared) and the CPU time
of one worker per episode (state read + sampling + decode + resize + processor).

usage: python tools/bpv5/bench_cache.py [--images 768] [--episodes 6]
"""

from __future__ import annotations

import argparse
import time

import torch

from lerobot.policies.BPV5QFormer.build_cache import EpisodeFrames, VLM, load_vision, vision_tokens

ap = argparse.ArgumentParser()
ap.add_argument("--images", type=int, default=768)
ap.add_argument("--episodes", type=int, default=6)
a = ap.parse_args()

items = [("robotwin/adjust_bottle/aloha-agilex_clean_50", e) for e in range(2)] + \
        [("abc/train__arrange_the_flowers_into_the_vase__zedx", 0), ("interndata/articulation_tasks/franka/open_the_microwave_part2", 1023),
         ("dom", 5), ("hifi_umi/chunk-0001/part-0000", 3)]
ds = EpisodeFrames(items[: a.episodes], 32, 224, VLM)
t0 = time.time()
got = [ds[i] for i in range(len(ds))]
cpu = (time.time() - t0) / len(ds)
errs = [g["error"] for g in got if "error" in g]
print(f"worker CPU time per episode: {cpu:.2f} s ({len(ds)} episodes, errors {errs})")

ok = [g for g in got if "error" not in g]
pix = torch.cat([g["pixels"] for g in ok])
grid = torch.cat([g["grids"] for g in ok])
reps = max(1, a.images // len(grid))
pix, grid = pix.repeat(reps, 1), grid.repeat(reps, 1)
pix = pix.reshape(len(grid), -1, pix.shape[-1]).reshape(-1, pix.shape[-1])
dev = torch.device("cuda")
pix, grid = pix.to(dev, torch.bfloat16), grid.to(dev)
for batched in (False, True):
    vision = load_vision(VLM, dev, batched_attention=batched)
    with torch.no_grad():
        out = vision_tokens(vision, pix, grid)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            out = vision_tokens(vision, pix, grid)
        torch.cuda.synchronize()
    dt = (time.time() - t0) / 3
    print(f"batched_attention={batched}: {len(grid)} images in {dt:.2f} s -> {len(grid) / dt:.0f} images/s "
          f"({len(grid) / dt / 96:.1f} episodes/s); peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    if not batched:
        ref = out.float()
    else:
        d = (out.float() - ref).abs()
        print(f"batched vs per-image: max abs diff {d.max():.3g}, mean {d.mean():.3g}, scale {ref.abs().mean():.3g}")
    del vision
    torch.cuda.empty_cache()
