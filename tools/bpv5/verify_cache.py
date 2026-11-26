"""Check BPv5 demo cache records: fields and, for one episode, the features against an independent path
(LeRobotDataset frames + the full Qwen3.5 model's get_image_features, as AE-3's build_cache).

usage: python tools/bpv5/verify_cache.py --cache_dir <dir> [--rel robotwin/adjust_bottle/aloha-agilex_clean_50 --ep 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--cache_dir", required=True)
ap.add_argument("--rel", default="robotwin/adjust_bottle/aloha-agilex_clean_50")
ap.add_argument("--ep", type=int, default=0)
ap.add_argument("--vlm", default="/vla/workspace/models/wall-oss/wall-oss-1.0-lite-vlm")
a = ap.parse_args()

for p in sorted(Path(a.cache_dir).rglob("*.pt")):
    r = torch.load(p, weights_only=False)
    print(f"{r['rel']:55s} feats {tuple(r['feats'].shape)} {r['feats'].dtype} {p.stat().st_size / 1e6:.2f} MB | "
          f"len {r['length']} active {r['active']} frames {r['frames'][:3].tolist()}..{r['frames'][-2:].tolist()} "
          f"repeats {r['repeats']} cams {[c.split('.')[-1] if c else None for c in r['cameras']]} "
          f"valid {r['camera_valid'][0].tolist()} fps {r['fps']}")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.BPV4BPEn.build_feature_cache import load_vit, visual_tokens  # noqa: E402
from lerobot.policies.BPV4CDX.dataset import pool_spatial  # noqa: E402
from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision  # noqa: E402
from lerobot.transforms.utils import resize_with_pad  # noqa: E402

rec = torch.load(Path(a.cache_dir) / a.rel / f"ep{a.ep:06d}.pt", weights_only=False)
ds = LeRobotDataset(f"/physical-ai-data/dataset_80D/{a.rel}", video_backend="torchcodec")
start = int(ds.meta.episodes[a.ep]["dataset_from_index"])
cams = [c for c in rec["cameras"] if c]
images = []
for f in rec["frames"].tolist():
    item = ds[start + f]
    images += [resize_with_pad(item[c][None], 224, 224)[0] for c in cams]
processor, _ = load_qwen35_vision(a.vlm)
out = processor(images, do_rescale=False)
vlm = load_vit(None, a.vlm, torch.device("cuda"))
with torch.no_grad():
    t = visual_tokens(vlm, torch.as_tensor(out.pixel_values).to("cuda", torch.bfloat16),
                      torch.as_tensor(out.image_grid_thw, dtype=torch.long).to("cuda"))
ref = pool_spatial(t.reshape(len(rec["frames"]), len(cams), t.shape[-2], t.shape[-1]).float(), 5)
got = rec["feats"][:, [j for j, c in enumerate(rec["cameras"]) if c]].float().to(ref.device)
d = (ref - got).abs()
cos = torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
print(f"independent path: max abs diff {d.max():.4g}, mean {d.mean():.4g}, scale {ref.abs().mean():.4g}, cosine {cos:.6f}")
per = torch.nn.functional.cosine_similarity(ref.flatten(2), got.flatten(2), dim=-1)  # [32, cams]
low = (per < 0.999).nonzero().tolist()
print(f"per (slot, camera) cosine: min {per.min():.5f}; below 0.999: {len(low)} of {per.numel()} -> "
      f"{[(s, cams[c].split('.')[-1], round(float(per[s, c]), 4)) for s, c in low[:12]]}")
