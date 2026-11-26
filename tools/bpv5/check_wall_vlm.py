"""Check the exported wall VLM: transformers loading info, a weight against the wall original, and the Q35 backbone
(fingerprint, Action Expert size for a width ratio).

usage: python tools/bpv5/check_wall_vlm.py [--vlm ...wall-oss-1.0-lite-vlm] [--ratio 0.25]
"""

from __future__ import annotations

import argparse

import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--vlm", default="/vla/workspace/models/wall-oss/wall-oss-1.0-lite-vlm")
ap.add_argument("--wall", default="/vla/workspace/models/wall-oss/wall-oss-1.0-lite-robodojo-base")
ap.add_argument("--ratio", type=float, default=0.25)
a = ap.parse_args()

from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration  # noqa: E402

model, info = Qwen3_5ForConditionalGeneration.from_pretrained(a.vlm, dtype=torch.bfloat16, output_loading_info=True)
print({k: (v if len(v) < 8 else f"{len(v)}: {sorted(v)[:5]}") for k, v in info.items()})
own = dict(model.named_parameters())
with safe_open(f"{a.wall}/model.safetensors", "pt") as f:
    for k in ("model.language_model.layers.3.self_attn.q_proj.weight", "model.visual.merger.linear_fc2.weight",
              "model.language_model.embed_tokens.weight"):
        ref = f.get_tensor(k).to(torch.bfloat16)
        print(k, "equal to wall (bf16):", torch.equal(own[k].data, ref))
del model

from lerobot.policies.q35.qwen3_5_backbone import validate_qwen35_model_config  # noqa: E402
from transformers import AutoConfig  # noqa: E402

print("fingerprint:", validate_qwen35_model_config(AutoConfig.from_pretrained(a.vlm)).name)

from lerobot.policies.q35.configuration_q35 import Q35Config  # noqa: E402
from lerobot.policies.q35.modeling_q35 import Q35Model  # noqa: E402

cfg = Q35Config(qwen3_5_pretrained_path=a.vlm, expert_width_ratio=a.ratio)
q = Q35Model(cfg)
bb = q.core.backbone
count = lambda m: sum(p.numel() for p in m.parameters()) / 1e6  # noqa: E731
print(f"Q35 fingerprint {bb.fingerprint.name}; VLM {count(bb.vlm):.1f}M; expert {count(bb.expert):.1f}M "
      f"(ratio {a.ratio}); expert hidden {bb.expert.config.hidden_size}, layers {len(bb.expert.layers)}")
