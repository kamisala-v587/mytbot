"""Local evaluation copies of the AE-3 checkpoints from the B200 (weights symlinked, train_config rewritten):
  ae3    the AE-3 data setting on this machine: split manifest (same held-out rule), local roots / caches / arrays
  rt     RoboTwin only, the AE-2 data setting (same 750 held-out episodes and S-U cache as the AE-2 report)
  probe  as rt, with the bp_probe cache built by the AE-3 (embodiment) S-U sampler
Each variant is a run dir <out>/<variant>/checkpoints/<step>/pretrained_model (heldout_monitor --run works on it).

usage: python tools/bpv4cdx/make_ae3_eval_ckpts.py --src /vla/workspace/models/BPVA/BPv4CDX/A --out <eval dir> \
         --manifest <ae3_split.json> --ae2_config <ae2l_su .../pretrained_model/train_config.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ae2_config", required=True)
    ap.add_argument("--data_roots", default="/public-data/dataset:/vla/workspace/data")
    ap.add_argument("--cache_root", default="/vla/workspace/cache")
    ap.add_argument("--array_store", default="/vla/workspace/cache/ae3_arrays_dev")
    ap.add_argument("--qwen_path", default="/vla/workspace/models/q35/base_800m")
    ap.add_argument("--stats_root", default=str(ROOT / "norm_stats"))
    args = ap.parse_args()
    ae2 = json.loads(Path(args.ae2_config).read_text())
    for step_dir in sorted(p for p in Path(args.src).iterdir() if p.is_dir() and p.name.isdigit()):
        base = json.loads((step_dir / "train_config.json").read_text())
        for variant in ("ae3", "rt", "probe"):
            cfg = json.loads(json.dumps(base))
            d, p = cfg["dataset"], cfg["policy"]
            p["qwen3_5_pretrained_path"] = args.qwen_path
            if variant == "ae3":
                d.update({"split_manifest": str(Path(args.manifest).resolve()), "data_roots": args.data_roots.split(":"),
                          "cache_dir": f"{args.cache_root}/bpv4cdx_ae3_su", "cache_index": "",
                          "array_store": args.array_store})
            else:
                d = cfg["dataset"] = json.loads(json.dumps(ae2["dataset"]))
                d["cache_dir"] = f"{args.cache_root}/" + ("bpv4cdx_su" if variant == "rt" else "bpv4cdx_ae3su_probe")
                if variant == "probe":
                    d["repo_id_file"] = str(ROOT / "configs/ds_ids/pro6k/robo_bpv4_bpen_probe.txt")
                    d["data_root"] = None
            d["qwen3_5_processor_path"] = args.qwen_path
            d["external_stats_root"] = args.stats_root
            cfg["output_dir"] = str(Path(args.out) / variant)
            dst = Path(args.out) / variant / "checkpoints" / step_dir.name / "pretrained_model"
            dst.mkdir(parents=True, exist_ok=True)
            for name in ("model.safetensors", "config.json", "stats.json"):
                link = dst / name
                if not link.exists():
                    link.symlink_to((step_dir / name).resolve())
            config = json.loads((step_dir / "config.json").read_text())
            config["qwen3_5_pretrained_path"] = args.qwen_path
            config["qwen3_5_processor_path"] = args.qwen_path
            (dst / "config.json").unlink()
            (dst / "config.json").write_text(json.dumps(config, indent=2))
            (dst / "train_config.json").write_text(json.dumps(cfg, indent=2))
            print(dst)


if __name__ == "__main__":
    main()
