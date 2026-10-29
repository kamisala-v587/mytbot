"""Stage B configs for the B200 main path (StageB实施.md), derived from configs/Pro6k/bpv4cdxb_plan/b_main.jsonc.

All variants: same-bucket demos, query episodes length-weighted (dataset train sampler, MultiPassSampler), constant
5e-5, 16 queries x 4 steps per GPU, 24 loader workers, batched ViT attention + bf16 pixel transfer (both
bit-identical to the original path, StageB_ViT与像素传输提速开发.md V1 ~ V5).

  b2_main   B-2: only the Action Expert (+ [obs_start] / [obs_end]) trains
  b3_main   B-3 (--b2_ckpt): + Qwen LoRA on layers 0-3 and 20-23 (layer 23 k / v only), rank 16, alpha 32;
            new optimizer, 500-step warmup
  b4_main   B-4 (--b3_ckpt): + trainable demo encoder (online Z_demo from the AE-3 P5 cache, lr 5e-6); the LoRA
            keeps training; ViT stays frozen; new optimizer, 500-step warmup

usage: python tools/bpv4cdxb/make_b200_configs.py [--b2_ckpt <pretrained_model>] [--b3_ckpt <pretrained_model>]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bpv4cdx"))
from make_gate_configs import load_jsonc  # noqa: E402

W = "/home/jovyan/workspace"
PATHS = {
    "qwen": f"{W}/models/q35/base_800m",
    "frozen": f"{W}/models/BPVA/BPv4CDX/StageA_AE_frozen/0750000",
    "zdemo": f"{W}/cache/bpv4cdxb_zdemo/cefa4afba111",
    "p5": f"{W}/cache/bpv4cdx_ae3_su",
    "index": f"{ROOT}/outputs/BPv4CDX/stage_b/stage_b_main_index.json",
    "stats": f"{ROOT}/norm_stats",
    "out": f"{ROOT}/outputs/BPv4CDX/stage_b",
}
# first two and last two Qwen3.5 blocks (block = 3 linear-attention layers + 1 full-attention layer); layer 23 k / v only
LORA = {"qwen_lora_layers": 0, "qwen_lora_layer_ids": [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23], "qwen_lora_rank": 16,
        "qwen_lora_alpha": 32.0}
TOP = {"batch_size": 16, "num_workers": 24, "steps": 1_000_000, "save_freq": 5000, "log_freq": 100}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT / "configs/Pro6k/bpv4cdxb_plan/generated_b200"))
    ap.add_argument("--b2_ckpt", default="", help="B-2 <pretrained_model> dir the B-3 variant starts from (b3_main)")
    ap.add_argument("--b3_ckpt", default="", help="B-3 <pretrained_model> dir the B-4 variant starts from (b4_main)")
    args = ap.parse_args()
    base = load_jsonc(ROOT / "configs/Pro6k/bpv4cdxb_plan/b_main.jsonc")
    base["dataset"].update({"external_stats_root": PATHS["stats"], "qwen3_5_processor_path": PATHS["qwen"],
                            "index": PATHS["index"], "data_root": "/share", "zdemo_dir": PATHS["zdemo"],
                            "split": "train", "contrast_prob": 0.0, "pixel_transfer_dtype": "bfloat16"})
    base["policy"].update({"qwen3_5_pretrained_path": PATHS["qwen"], "frozen_encoder": PATHS["frozen"],
                           "new_modules_lr": 5e-5, "act_expert_lr": 5e-5, "optimizer_lr": 5e-5,
                           "scheduler_warmup_steps": 0, "scheduler_decay_lr": 5e-5, "vit_batched_attention": True})
    variants = {"b2_main": {"top": TOP}}
    if args.b2_ckpt:
        variants["b3_main"] = {"policy": {"init_from": str(Path(args.b2_ckpt).resolve()), **LORA,
                                          "scheduler_warmup_steps": 500}, "top": TOP}
    if args.b3_ckpt:
        variants["b4_main"] = {"policy": {"init_from": str(Path(args.b3_ckpt).resolve()), **LORA,
                                          "qwen_lora_trainable": True, "demo_encoder_trainable": True,
                                          "demo_encoder_lr": 5e-6, "scheduler_warmup_steps": 500},
                               # new seed: B-2 / B-3 used 42, and the sampler order depends only on the seed
                               "dataset": {"demo_features": "p5", "demo_cache_dir": PATHS["p5"]},
                               "top": {**TOP, "seed": 2024}}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, change in variants.items():
        cfg = copy.deepcopy(base)
        cfg["policy"].update(change.get("policy", {}))
        cfg["dataset"].update(change.get("dataset", {}))
        cfg.update(change.get("top", {}))
        cfg["output_dir"] = f"{PATHS['out']}/{name}"
        (out / f"{name}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        print(out / f"{name}.json")


if __name__ == "__main__":
    main()
