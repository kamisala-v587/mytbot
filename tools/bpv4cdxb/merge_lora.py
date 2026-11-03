"""Merge the Qwen LoRA of a Stage B checkpoint (B-3 / B-4) into the base weights and save a LoRA-free checkpoint.

The result is the same model without adapters (config: qwen_lora_layers = 0, qwen_lora_layer_ids = [], init_from = ""),
usable as `init_from` of a run that trains the whole Qwen language model (policy.vlm_trainable = true).

usage (my_tbot/, env.local.sh):
  python tools/bpv4cdxb/merge_lora.py --ckpt <run>/checkpoints/020000/pretrained_model --out <dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REMAP = (("/home/jovyan/workspace/mytbot", "/vla/workspace/my_tbot"), ("/home/jovyan/workspace", "/vla/workspace"))

def main() -> None:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.BPV4CDXB.lora_b3 import merge_qwen_lora
    from lerobot.policies.BPV4CDXB.modeling_bpv4cdxb import BPV4CDXBPolicy

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="<run>/checkpoints/<step>/pretrained_model with Qwen LoRA")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ckpt, out = Path(args.ckpt), Path(args.out)
    cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    lora = {k: getattr(cfg, k) for k in ("qwen_lora_layers", "qwen_lora_layer_ids", "qwen_lora_rank", "qwen_lora_alpha")}
    if not lora["qwen_lora_layers"] and not lora["qwen_lora_layer_ids"]:
        raise SystemExit(f"{ckpt} has no Qwen LoRA")
    cfg.init_from = ""  # every weight is in the checkpoint
    cfg.device = "cpu"
    # checkpoints trained on another machine: map their workspace root to this one when the path is missing here
    for key in ("qwen3_5_pretrained_path", "qwen3_5_processor_path", "frozen_encoder"):
        path = getattr(cfg, key, None)
        if path and not Path(path).exists():
            for src, dst in REMAP:
                if path.startswith(src) and Path(dst + path[len(src):]).exists():
                    setattr(cfg, key, dst + path[len(src):])
                    print(f"{key}: {path} -> {getattr(cfg, key)}")
    policy = BPV4CDXBPolicy.from_pretrained(str(ckpt), config=cfg)
    n = merge_qwen_lora(policy.model.core.backbone.vlm)
    policy.config.qwen_lora_layers = 0
    policy.config.qwen_lora_layer_ids = []
    policy.save_pretrained(out)
    if (ckpt / "stats.json").is_file():
        shutil.copy2(ckpt / "stats.json", out / "stats.json")
    info = {"source_checkpoint": str(ckpt.resolve()), "merged_lora_layers": n, "lora": lora,
            "demo_encoder": getattr(policy.model, "demo_encoder", None) is not None}
    (out / "merge_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), "->", out)


if __name__ == "__main__":
    main()
