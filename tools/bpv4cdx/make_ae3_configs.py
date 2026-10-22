"""AE-3 configs (bpv4cdx AE-3 plan, section 5), derived from configs/Pro6k/bpv4cdx_plan/ae_main.jsonc.

Only the AE-3 differences are set here:
  data       split manifest + data roots (machine-independent repo ids, held-out lists), memory-mapped arrays,
             cache layout `rel`, P5-only cache, cache index (train on a partial cache)
  sampler    from the G0 audit (suf or su)
  batch      128 trajectories in total (16 per GPU on 8 GPUs) x 4 segments
  schedule   warmup 2000, cosine to the floor (1e-5 for the 1e-4 groups) at 300k steps, then constant; 1M steps cap
             (stop by hand, see HOM advice); save every 10k
  HOM        1000 held-out episodes per save (task-stratified), all held-out episodes every 5th save
Variants: ae3_main, ae3_g3_overfit (G3 on RoboTwin place_shoe clean_50 #0), ae3_smoke (dev machine smoke test).

Usage: python tools/bpv4cdx/make_ae3_configs.py --manifest configs/ds_ids/B200/ae3_split.json --data_roots /share \
         --cache_root <dir> --array_store <dir> --qwen_path <base_800m> --stats_root <norm_stats> --out_root <dir> \
         --sampler suf --num_gpus 8 [--out_dir configs/Pro6k/bpv4cdx_plan/generated_ae3]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bpv4cdx"))
from make_gate_configs import MAIN, load_jsonc  # noqa: E402

TOTAL_TRAJECTORIES = 128
G3_EPISODE = "RoboTwin-LeRobot-v3.0/place_shoe/aloha-agilex_clean_50|0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data_roots", required=True, help="':'-separated local dataset roots")
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--array_store", required=True)
    ap.add_argument("--qwen_path", required=True)
    ap.add_argument("--stats_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--sampler", choices=["suf", "su"], required=True, help="from `ae3_data audit` (decision)")
    ap.add_argument("--num_gpus", type=int, default=8)
    ap.add_argument("--out_dir", default=str(ROOT / "configs/Pro6k/bpv4cdx_plan/generated_ae3"))
    args = ap.parse_args()
    if TOTAL_TRAJECTORIES % args.num_gpus:
        raise SystemExit(f"num_gpus must divide {TOTAL_TRAJECTORIES}")
    base = load_jsonc(MAIN)
    cache_dir = str(Path(args.cache_root) / f"bpv4cdx_ae3_{args.sampler}")
    base["dataset"].update({
        "repo_id_file": None, "data_root": None,
        "split_manifest": str(Path(args.manifest).resolve()), "data_roots": [r for r in args.data_roots.split(":") if r],
        "manifest_sources": [], "array_store": args.array_store, "cache_layout": "rel",
        "cache_index": str(Path(cache_dir) / "_index.txt"), "load_threads": 32,
        "sampler": args.sampler, "cache_dir": cache_dir, "spatial_tokens": 5,
        "qwen3_5_processor_path": args.qwen_path, "external_stats_root": args.stats_root,
        "heldout_monitor": True, "heldout_monitor_max_episodes": 1000, "heldout_monitor_full_every": 5,
        "heldout_monitor_batch": 16, "heldout_monitor_urdf": "auto",
    })
    base["policy"].update({"qwen3_5_pretrained_path": args.qwen_path, "scheduler_warmup_steps": 2000,
                           "scheduler_decay_steps": 300_000})
    base.update({"steps": 1_000_000, "save_freq": 10_000, "log_freq": 100, "seed": 42, "num_workers": 16,
                 "batch_size": TOTAL_TRAJECTORIES // args.num_gpus})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    variants = {
        "ae3_main": {},
        "ae3_g3_overfit": {
            "dataset": {"episode_whitelist": [G3_EPISODE], "segment_mode": "fixed", "generic_text_prob": 1.0,
                        "length_weighted_sampling": False, "heldout_monitor": False},
            "policy": {"scheduler_warmup_steps": 50, "scheduler_decay_steps": 1000, "preflight_memory_check": False},
            "top": {"batch_size": 1, "num_workers": 0, "steps": 1000, "save_freq": 1000, "log_freq": 50},
        },
        "ae3_smoke": {
            "dataset": {"heldout_monitor_max_episodes": 24, "heldout_monitor_full_every": 2, "load_threads": 16},
            "policy": {"scheduler_warmup_steps": 20, "scheduler_decay_steps": 200, "preflight_memory_check": False},
            "top": {"batch_size": 4, "num_workers": 4, "steps": 300, "save_freq": 100, "log_freq": 20},
        },
    }
    for name, change in variants.items():
        cfg = copy.deepcopy(base)
        cfg["dataset"].update(change.get("dataset", {}))
        cfg["policy"].update(change.get("policy", {}))
        cfg.update(change.get("top", {}))
        cfg["output_dir"] = str(Path(args.out_root) / "BPv4CDX/stage_a3" / name)
        (out / f"{name}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        print(out / f"{name}.json")


if __name__ == "__main__":
    main()
