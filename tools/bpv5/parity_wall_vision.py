"""Layer-by-layer comparison of the ViT: transformers Qwen3_5VisionModel vs wall-x's own Qwen3_5VisionModel, same weights
(model.visual.* of the exported checkpoint), float32. Finds where the two first diverge.

usage: python tools/bpv5/parity_wall_vision.py [--device cuda:0]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--vlm", default="/vla/workspace/models/wall-oss/wall-oss-1.0-lite-vlm")
ap.add_argument("--wall", default="/vla/workspace/models/wall-oss/wall-oss-1.0-lite-robodojo-base")
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()

from lerobot.policies.wall_oss.transform_wall_oss import load_wall_processor  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel  # noqa: E402
from wall_x.model.qact.qwen3_5 import Qwen3_5Config  # noqa: E402
from wall_x.model.qact.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel as WallVisionModel  # noqa: E402

processor = load_wall_processor(a.vlm)
img = Image.fromarray((np.random.default_rng(0).random((256, 256, 3)) * 255).astype(np.uint8))
px = processor.image_processor(images=[img], return_tensors="pt")
pv, grid = px["pixel_values"].to(a.device, torch.float32), px["image_grid_thw"].to(a.device)

state = {}
with safe_open(f"{a.vlm}/model.safetensors", "pt") as f:
    for k in f.keys():
        if k.startswith("model.visual."):
            state[k[len("model.visual."):]] = f.get_tensor(k).float()

hf_cfg = AutoConfig.from_pretrained(a.vlm).vision_config
hf_cfg._attn_implementation = "sdpa"
hf = Qwen3_5VisionModel(hf_cfg)
print("hf load", hf.load_state_dict(state, strict=False))
wcfg = Qwen3_5Config.from_pretrained(a.wall).vision_config
wcfg._attn_implementation = "sdpa"
wall = WallVisionModel(wcfg)  # what Qwen3_5_VLAModel builds for model_type qwen3_5
print("wall load", wall.load_state_dict(state, strict=False))
hf, wall = hf.float().to(a.device).eval(), wall.float().to(a.device).eval()


def taps(model):
    out = {}
    hooks = [model.patch_embed.register_forward_hook(lambda m, i, o: out.__setitem__("patch_embed", o))]
    for j, blk in enumerate(model.blocks):
        hooks.append(blk.register_forward_hook(lambda m, i, o, j=j: out.__setitem__(f"block{j:02d}", o[0] if isinstance(o, tuple) else o)))
    hooks.append(model.merger.register_forward_hook(lambda m, i, o: out.__setitem__("merger", o)))
    return out, hooks


with torch.no_grad():
    th, hh = taps(hf)
    hf(pv, grid_thw=grid)
    tw, hw = taps(wall)
    wall(pv, grid_thw=grid)
    pe_hf = hf.fast_pos_embed_interpolate(grid)
    pe_w = wall.fast_pos_embed_interpolate(grid)
    print(f"pos_embed interp: max abs diff {(pe_hf - pe_w).abs().max():.3g}")
    try:
        r_hf, r_w = hf.rot_pos_emb(grid), wall.rot_pos_emb(grid)
        r_hf = r_hf[0] if isinstance(r_hf, tuple) else r_hf
        r_w = r_w[0] if isinstance(r_w, tuple) else r_w
        print(f"rot_pos_emb: shapes {tuple(r_hf.shape)} {tuple(r_w.shape)}; max abs diff "
              f"{(r_hf.float() - r_w.float()).abs().max():.3g}" if r_hf.shape == r_w.shape else "rot_pos_emb shapes differ")
    except Exception as exc:  # noqa: BLE001
        print("rot_pos_emb compare:", type(exc).__name__, exc)
for k in th:
    x, y = th[k].float(), tw.get(k)
    if y is None:
        print(k, "missing in wall")
        continue
    y = y.float()
    if x.shape != y.shape:
        print(k, "shape", tuple(x.shape), tuple(y.shape))
        continue
    d = (x - y).abs()
    print(f"{k:12s} max {d.max():.3g} mean {d.mean():.3g} (scale {y.abs().mean():.3g})")
