"""Numerical parity of the exported wall VLM (transformers Qwen3.5) against wall-x's own model on the original
checkpoint: one image + text, no action tokens, logits compared on every token.

usage: python tools/bpv5/parity_wall_vlm.py [--device cuda:0]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--vlm", default="/vla/workspace/models/wall-oss/wall-oss-1.0-lite-vlm")
ap.add_argument("--wall", default="/vla/workspace/models/wall-oss/wall-oss-1.0-lite-robodojo-base")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
a = ap.parse_args()
dtype = getattr(torch, a.dtype)

from lerobot.policies.wall_oss.transform_wall_oss import load_wall_processor  # noqa: E402

processor = load_wall_processor(a.vlm)
img = Image.fromarray((np.random.default_rng(0).random((256, 256, 3)) * 255).astype(np.uint8))
text = ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>pick up the red block and put it in the "
        "bowl<|im_end|>\n<|im_start|>assistant\n")
inputs = processor(text=[text], images=[img], return_tensors="pt")
inputs = {k: v.to(a.device) for k, v in inputs.items()}
inputs["pixel_values"] = inputs["pixel_values"].to(dtype)


def to_dtype(model):
    for p in model.parameters():  # buffers (rope inv_freq) keep their dtype
        p.data = p.data.to(dtype)
    return model.to(a.device).eval()


from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration  # noqa: E402

hf = to_dtype(Qwen3_5ForConditionalGeneration.from_pretrained(a.vlm, dtype=dtype))
with torch.no_grad():
    logits_hf = hf(**inputs).logits.float()
    vis_hf = hf.model.get_image_features(pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"])
    vis_hf = (vis_hf.pooler_output if hasattr(vis_hf, "pooler_output") else vis_hf)
    vis_hf = torch.cat(list(vis_hf)) if isinstance(vis_hf, (list, tuple)) else vis_hf
    pos_hf = hf.model.get_rope_index(input_ids=inputs["input_ids"], mm_token_type_ids=inputs["mm_token_type_ids"],
                                     image_grid_thw=inputs["image_grid_thw"], attention_mask=inputs["attention_mask"])[0]
del hf
torch.cuda.empty_cache()

from safetensors.torch import load_file  # noqa: E402
from wall_x.model.qact.qwen3_5 import Qwen3_5_VLAForConditionalGeneration, Qwen3_5Config  # noqa: E402

cfg = Qwen3_5Config.from_pretrained(a.wall)
cfg.update_model_config({"data": {"use_state_string_representation": True, "action_horizon_flow": 32},
                         "dof_config": {"a": 26}, "agent_pos_config": {"s": 20}, "attn_implementation": "sdpa"})
wall = Qwen3_5_VLAForConditionalGeneration(cfg, processor, None)
res = wall.load_state_dict(load_file(f"{a.wall}/model.safetensors"), strict=False)
print("wall load missing", [k for k in res.missing_keys if k != "lm_head.weight"][:5], "unexpected", res.unexpected_keys[:5])
wall = to_dtype(wall)
# wall-x's text-only path calls a transformers API changed after 5.2; its VLA path (the one training uses) works.
# Append 32 action slots: the prefix is causal, so its logits do not depend on what follows.
wall_inputs = processor(text=[text + "<|action|>" * 32], images=[img], return_tensors="pt")
wall_inputs = {k: v.to(a.device) for k, v in wall_inputs.items()}
wall_inputs["pixel_values"] = wall_inputs["pixel_values"].to(dtype)
n_prefix = inputs["input_ids"].shape[1]
assert torch.equal(wall_inputs["input_ids"][:, :n_prefix], inputs["input_ids"])
act = torch.zeros(1, 32, 26, device=a.device)
with torch.no_grad():
    out = wall(**wall_inputs, action_chunk=act, dof_mask=torch.ones_like(act),
               proprioception=torch.zeros(1, 1, 20, device=a.device), agent_pos_mask=torch.ones(1, 1, 20, device=a.device),
               dataset_names=["x"], sample_time=torch.tensor([0.5], device=a.device), mode="train")
    logits_wall = out.logits.float()[:, :n_prefix]
    vis_wall = wall.model.get_image_features(pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"])
    vis_wall = (vis_wall.pooler_output if hasattr(vis_wall, "pooler_output") else vis_wall)
    vis_wall = torch.cat(list(vis_wall)) if isinstance(vis_wall, (list, tuple)) else vis_wall
    try:
        pos_wall = wall.model.compute_3d_position_ids(
            input_ids=inputs["input_ids"], image_grid_thw=inputs["image_grid_thw"], video_grid_thw=None,
            inputs_embeds=None, attention_mask=inputs["attention_mask"], past_key_values=None,
            mm_token_type_ids=inputs["mm_token_type_ids"])
    except Exception as exc:  # noqa: BLE001
        pos_wall = None
        print("wall position ids:", type(exc).__name__, exc)
vd = (vis_hf.float() - vis_wall.float()).abs()
print(f"vision features: shape {tuple(vis_hf.shape)} vs {tuple(vis_wall.shape)}; max abs diff {vd.max():.4g}, "
      f"mean {vd.mean():.4g}, scale {vis_wall.float().abs().mean():.4g}")
if pos_wall is not None:
    print("position ids equal:", torch.equal(pos_hf.cpu(), pos_wall.cpu()),
          "first diff at", (pos_hf.cpu() != pos_wall.cpu()).any(0).nonzero()[:3].tolist())

diff = (logits_hf - logits_wall).abs()
img_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
is_img = (inputs["input_ids"][0] == img_id).cpu()
per_pos = diff[0].mean(-1).cpu()
first_img = int(is_img.nonzero()[0]) if is_img.any() else len(is_img)
print(f"mean diff: before image {per_pos[:first_img].mean():.4g}, image tokens {per_pos[is_img].mean():.4g}, "
      f"text after image {per_pos[~is_img][first_img:].mean():.4g}")
print("per-position mean diff:", [round(float(x), 3) for x in per_pos])
cos = torch.nn.functional.cosine_similarity(logits_hf.flatten(1), logits_wall.flatten(1)).item()
agree = (logits_hf.argmax(-1) == logits_wall.argmax(-1)).float().mean().item()
print(f"dtype {a.dtype}: logits max abs diff {diff.max():.4g}, mean {diff.mean():.4g}, scale {logits_wall.abs().mean():.4g}; "
      f"cosine {cos:.6f}; argmax agreement {agree:.4f} over {logits_hf.shape[1]} tokens")
