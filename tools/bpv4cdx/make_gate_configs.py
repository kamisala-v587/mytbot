"""Derive the bpv4cdx gate / branch configs from configs/Pro6k/bpv4cdx_plan/ae_main.jsonc.

Writes <out_dir>/<name>.json for every variant below. Paths can be rewritten for another machine:
  python tools/bpv4cdx/make_gate_configs.py --data_root /data/robotwin2.0 --cache_root /cache \
      --qwen_path /models/q35/base_800m --out_root /outputs --stats_root /mytbot/norm_stats \
      --repo_root /mytbot/configs/ds_ids/pro6k
Cache dirs: <cache_root>/<cache_prefix><sampler> (P64 uses the same cache as its sampler); without --cache_root the
main config's cache dir is kept and only its sampler suffix is switched.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "configs/Pro6k/bpv4cdx_plan/ae_main.jsonc"

G3_EPISODE = "place_shoe/aloha-agilex_clean_50|0"
# bpv4_bpen main run: 4 GPUs x 16 trajectories x 4 queries; keep 64 trajectories (256 segments) per step
TOTAL_TRAJECTORIES = 64

G3 = {
    "dataset": {"episode_whitelist": [G3_EPISODE], "segment_mode": "fixed", "generic_text_prob": 1.0,
                "length_weighted_sampling": False},
    # single trajectory: one step per epoch, so no DataLoader workers
    "top": {"batch_size": 1, "num_workers": 0, "steps": 1000, "save_freq": 1000, "log_freq": 50},
    "policy": {"scheduler_warmup_steps": 50, "scheduler_decay_steps": 1000, "preflight_memory_check": False},
}
SUBSET = {"dataset": {"repo_id_file": "@repo_root/bpv4cdx_g45_subset_rel.txt"},
          "top": {"steps": 2000, "save_freq": 2000}, "policy": {"scheduler_decay_steps": 2000}}


def subset(dataset: dict | None = None, policy: dict | None = None) -> dict:
    """AE-2 side branch: the subset run with one dimension changed."""
    return {"dataset": {**SUBSET["dataset"], **(dataset or {})}, "top": dict(SUBSET["top"]),
            "policy": {**SUBSET["policy"], **(policy or {})}}


VARIANTS = {
    # main branch: S-A / P5 / no state / N128 / text 50% generic / expert reads segment_query only
    "ae_main": {},
    # G7 second training seed
    "ae_main_seed43": {"top": {"seed": 43}},
    # independent text-only control (G4): no Z_demo tokens, same data and steps
    "ae_textonly": {"policy": {"demo_input": "none"}},
    # sampler controls (S-U mandatory, S-V optional)
    "ae_su": {"dataset": {"sampler": "su"}},
    "ae_sv": {"dataset": {"sampler": "sv"}},
    # full-size side branches (only after they pass on the subset, design doc 4.0)
    "ae_p64": {"dataset": {"spatial_tokens": 64}, "policy": {"spatial_tokens": 64}},
    "ae_s0": {"policy": {"use_segment_start_state": True}},
    "ae_fullprefix": {"policy": {"expert_context": "full_prefix"}},
    # 3.5.4 conditional LoRA on top of a selected AE-3 checkpoint: pass --policy.pretrained_path=<ckpt>
    "ae_main_lora": {"policy": {"qwen_lora_layers": 4}},
    # G3: one episode, 4 fixed segments, generic text, no state; full-prefix diagnostic when G3 fails
    "ae_g3_overfit": G3,
    "ae_g3_fullprefix": {**G3, "policy": {**G3["policy"], "expert_context": "full_prefix"}},
    # AE-2: the 10 test-set tasks (clean_50 + randomized_100), 2000 steps; main, text-only, S-U and side branches
    "ae_g45_subset": subset(),
    "ae_g45_subset_textonly": subset(policy={"demo_input": "none"}),
    "ae_g45_subset_su": subset({"sampler": "su"}),
    "ae_g45_subset_s0": subset(policy={"use_segment_start_state": True}),
    "ae_g45_subset_textdrop": subset({"text_drop_prob": 0.2}),
    "ae_g45_subset_generic08": subset({"generic_text_prob": 0.8}),
    "ae_g45_subset_fullprefix": subset(policy={"expert_context": "full_prefix"}),
    "ae_g45_subset_p64": subset({"spatial_tokens": 64}, {"spatial_tokens": 64}),
    "ae_g45_subset_n64": subset(policy={"num_demo_tokens": 64}),
    "ae_g45_subset_n32": subset(policy={"num_demo_tokens": 32}),
    # AE-2 at larger scale: all 50 RoboTwin tasks (clean_50 + randomized_100, 6,750 training trajectories), 20k steps;
    # not AE-3 (AE-3 "full" means the ~6000 h dataset)
    **{f"ae2l_{k}": {"dataset": dict(d), "top": {"steps": 20000, "save_freq": 2500},
                     "policy": {"scheduler_decay_steps": 20000, **p}}
       for k, d, p in (("main", {}, {}), ("textonly", {}, {"demo_input": "none"}), ("su", {"sampler": "su"}, {}))},
}


def sampler_cache_dir(cache_dir: str, sampler: str) -> str:
    """<parent>/<prefix>_<sa|su|sv> -> same prefix for the variant's sampler."""
    p = Path(cache_dir)
    name = re.sub(r"_(sa|su|sv)$", f"_{sampler}", p.name)
    return str(p.with_name(name if name != p.name or p.name.endswith(f"_{sampler}") else f"{p.name}_{sampler}"))


def load_jsonc(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^\s*//.*$", "", text, flags=re.M)
    text = re.sub(r"(?<![:\"/])//[^\n\"]*$", "", text, flags=re.M)
    return json.loads(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT / "configs/Pro6k/bpv4cdx_plan/generated"))
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--cache_root", default=None, help="cache dir = <cache_root>/<cache_prefix><sampler>")
    ap.add_argument("--cache_prefix", default="bpv4cdx_")
    ap.add_argument("--qwen_path", default=None)
    ap.add_argument("--out_root", default=None, help="output_dir = <out_root>/BPv4CDX/stage_a/<name>")
    ap.add_argument("--stats_root", default=None)
    ap.add_argument("--repo_root", default=None, help="directory of the repo list files")
    ap.add_argument("--num_gpus", type=int, default=4,
                    help="per-rank batch = TOTAL_TRAJECTORIES / num_gpus (same total batch as bpv4_bpen)")
    args = ap.parse_args()
    base = load_jsonc(MAIN)
    repo_root = args.repo_root or str(ROOT / "configs/ds_ids/pro6k")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, change in VARIANTS.items():
        cfg = copy.deepcopy(base)
        for section in ("dataset", "policy"):
            cfg[section].update(change.get(section, {}))
        if name != "ae_g3_overfit":
            if TOTAL_TRAJECTORIES % args.num_gpus:
                raise ValueError(f"num_gpus must divide {TOTAL_TRAJECTORIES}")
            cfg["batch_size"] = TOTAL_TRAJECTORIES // args.num_gpus
        cfg.update(change.get("top", {}))
        d, p = cfg["dataset"], cfg["policy"]
        if args.data_root:
            d["data_root"] = args.data_root
        if args.qwen_path:
            d["qwen3_5_processor_path"] = args.qwen_path
            p["qwen3_5_pretrained_path"] = args.qwen_path
        if args.stats_root:
            d["external_stats_root"] = args.stats_root
        if args.cache_root:
            d["cache_dir"] = str(Path(args.cache_root) / f"{args.cache_prefix}{d['sampler']}")
        else:
            d["cache_dir"] = sampler_cache_dir(d["cache_dir"], d["sampler"])
        d["repo_id_file"] = d["repo_id_file"].replace("@repo_root", repo_root)
        if args.repo_root and not d["repo_id_file"].startswith(repo_root):
            d["repo_id_file"] = str(Path(repo_root) / Path(d["repo_id_file"]).name)
        if args.out_root:
            cfg["output_dir"] = str(Path(args.out_root) / "BPv4CDX/stage_a" / name)
        else:
            cfg["output_dir"] = str(Path(cfg["output_dir"]).parent / name)
        (out / f"{name}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        print(out / f"{name}.json")


if __name__ == "__main__":
    main()
