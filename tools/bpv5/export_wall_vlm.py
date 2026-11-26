"""Export the VLM + ViT of a wall-oss checkpoint as a plain Qwen3.5 checkpoint (what Q35 / BPV5QFormer load).

Drops the action expert (`*_action`, `mlp_action_expert`, `norm_action`, `action_processor`), splits wall's fused
`mlp.gate_up_proj` back into `gate_proj` / `up_proj`, casts to --dtype (default bfloat16; the wall file is float32), removes
`vla_config` from config.json and adds the Qwen3.5 tokenizer / processor files (the wall checkpoint ships none; its
model card names Qwen3.5-4B's tokenizer).

usage: python tools/bpv5/export_wall_vlm.py [--src ...wall-oss-1.0-lite-robodojo-base] [--dst ...wall-oss-1.0-lite-vlm]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ACTION = re.compile(r"_action\b|_action\.|mlp_action_expert|norm_action|action_processor\.")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
                   "preprocessor_config.json", "video_preprocessor_config.json")

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/vla/workspace/models/wall-oss/wall-oss-1.0-lite-robodojo-base")
ap.add_argument("--dst", default="/vla/workspace/models/wall-oss/wall-oss-1.0-lite-vlm")
ap.add_argument("--tokenizer", default="/vla/workspace/models/q35/base_800m")
ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                help="the wall file is float32; bfloat16 halves the size (training runs in bfloat16 anyway)")
a = ap.parse_args()
DTYPE = getattr(torch, a.dtype)
src, dst = Path(a.src), Path(a.dst)
dst.mkdir(parents=True, exist_ok=True)

state, dropped = {}, 0
with safe_open(str(src / "model.safetensors"), "pt") as f:
    for k in f.keys():
        if ACTION.search(k):
            dropped += 1
            continue
        t = f.get_tensor(k).to(DTYPE)
        if k.endswith(".mlp.gate_up_proj.weight"):
            # wall fuses the MLP input projections as cat([gate, up], dim=0); transformers' Qwen3.5 keeps them split
            gate, up = t.chunk(2, dim=0)
            state[k.replace("gate_up_proj", "gate_proj")] = gate.contiguous()
            state[k.replace("gate_up_proj", "up_proj")] = up.contiguous()
            continue
        state[k] = t.contiguous()
save_file(state, str(dst / "model.safetensors"), metadata={"format": "pt"})

cfg = json.loads((src / "config.json").read_text())
cfg.pop("vla_config", None)
cfg.pop("_attn_implementation", None)
cfg["text_config"]["dtype"] = a.dtype
cfg["dtype"] = a.dtype
(dst / "config.json").write_text(json.dumps(cfg, indent=1))
for name in TOKENIZER_FILES:
    if (Path(a.tokenizer) / name).is_file():
        shutil.copy(Path(a.tokenizer) / name, dst / name)
# the wall processor config replaces the copied one (identical today; the checkpoint's own is authoritative)
shutil.copy(src / "preprocessor_config.json", dst / "preprocessor_config.json")
(dst / "EXPORT.md").write_text(
    f"Exported by my_tbot/tools/bpv5/export_wall_vlm.py from {src}\n"
    f"kept {len(state)} tensors ({a.dtype}), dropped {dropped} action-expert tensors, config without vla_config;\n"
    f"tokenizer / processor files from {a.tokenizer} (Qwen3.5 family tokenizer).\n")
n = sum(t.numel() for t in state.values())
print(f"kept {len(state)} tensors ({n / 1e9:.3f}B params), dropped {dropped}; -> {dst}")
