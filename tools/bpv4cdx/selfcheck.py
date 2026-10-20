"""Environment self-check for the bpv4cdx training machine. Writes a json report and exits non-zero on failure.

Checks: python / torch / CUDA, GPU count and memory, key packages, data root and every repo in the list,
external norm stats, Qwen3.5-0.8B base files, and (optionally) cache completeness for each sampler cache dir.

Usage: python tools/bpv4cdx/selfcheck.py --data_root <dir> --qwen_path <base_800m> --stats_root <norm_stats> \
         [--repo_file configs/ds_ids/pro6k/bpv4cdx_train_rel.txt] [--cache_dir <cache>/bpv4cdx_sa ...] \
         [--min_gpus 8] [--out outputs/BPv4CDX/selfcheck.json]
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def check_packages() -> dict:
    out = {}
    for name in ("torch", "transformers", "accelerate", "torchcodec", "draccus", "pandas", "safetensors", "matplotlib",
                 "sklearn"):
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", "ok")
        except Exception as e:  # noqa: BLE001
            out[name] = f"MISSING: {e}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--qwen_path", required=True)
    ap.add_argument("--stats_root", default=str(ROOT / "norm_stats"))
    ap.add_argument("--repo_file", default=str(ROOT / "configs/ds_ids/pro6k/bpv4cdx_train_rel.txt"))
    ap.add_argument("--cache_dir", action="append", default=[])
    ap.add_argument("--min_gpus", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "outputs/BPv4CDX/selfcheck.json"))
    args = ap.parse_args()
    report: dict = {"python": platform.python_version(), "packages": check_packages()}
    problems: list[str] = []
    problems += [f"package {k}: {v}" for k, v in report["packages"].items() if str(v).startswith("MISSING")]

    import torch

    report["cuda_available"] = torch.cuda.is_available()
    report["gpus"] = [{"name": torch.cuda.get_device_name(i),
                       "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
                      for i in range(torch.cuda.device_count())]
    if len(report["gpus"]) < args.min_gpus:
        problems.append(f"only {len(report['gpus'])} GPUs visible (expected >= {args.min_gpus})")

    from lerobot.policies.BPV4CDX.paths import read_repo_list

    repos = read_repo_list(args.repo_file, args.data_root)
    missing = [r for r in repos if not (Path(r) / "meta" / "info.json").is_file()]
    report["repos"] = {"listed": len(repos), "missing": missing}
    problems += [f"repo missing: {r}" for r in missing]
    episodes = {}
    for r in repos:
        if r not in missing:
            episodes[r] = int(json.loads((Path(r) / "meta" / "info.json").read_text())["total_episodes"])
    report["episodes_total"] = sum(episodes.values())
    robot_types = {json.loads((Path(r) / "meta" / "info.json").read_text()).get("robot_type") for r in episodes}
    report["robot_types"] = sorted(str(x) for x in robot_types)

    stats = [Path(args.stats_root) / str(rt) / "delta" / "stats.json" for rt in robot_types]
    report["norm_stats"] = {str(p): p.is_file() for p in stats}
    problems += [f"norm stats missing: {p}" for p in stats if not p.is_file()]

    qwen = Path(args.qwen_path)
    need = ["config.json", "tokenizer_config.json"]
    report["qwen_files"] = {f: (qwen / f).is_file() for f in need}
    report["qwen_has_weights"] = bool(list(qwen.glob("*.safetensors")))
    problems += [f"qwen file missing: {qwen / f}" for f in need if not (qwen / f).is_file()]
    if not report["qwen_has_weights"]:
        problems.append(f"no *.safetensors under {qwen}")

    from lerobot.policies.BPV4BPEn.build_feature_cache import cache_path

    report["caches"] = {}
    for cdir in args.cache_dir:
        absent = sum(1 for r, n in episodes.items() for ep in range(n) if not cache_path(cdir, r, ep).is_file())
        report["caches"][cdir] = {"expected": report["episodes_total"], "missing": absent}
        if absent:
            problems.append(f"cache {cdir}: {absent} episodes missing")

    report["problems"] = problems
    report["pass"] = not problems
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("python", "cuda_available", "episodes_total", "problems", "pass")}, indent=1))
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
