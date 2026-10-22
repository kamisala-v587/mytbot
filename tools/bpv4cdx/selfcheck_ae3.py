"""AE-3 self-check for the training machine (B200, no internet). Writes a json report; exits non-zero on problems.

Checks: python / torch / CUDA / GPUs; the packages every AE-3 step imports; the repo list (egodex excluded) and that
every repo has meta/info.json; per robot type the external norm stats exist AND contain every FEATURE_MAPPING key the
robot uses; Qwen3.5 base files; the URDF shipped in the repo (HOM end-effector error); free disk at the cache root and
the array store; CPU count and memory.

Usage: python tools/bpv4cdx/selfcheck_ae3.py --repo_file configs/ds_ids/B200/pretrain-data.txt \
         --qwen_path <base_800m> --stats_root <norm_stats> --cache_root <dir> --array_store <dir> [--min_gpus 8]
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
PACKAGES = ("torch", "transformers", "accelerate", "torchcodec", "draccus", "pandas", "pyarrow", "safetensors",
            "matplotlib", "sklearn", "numpy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_file", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--qwen_path", required=True)
    ap.add_argument("--stats_root", default=str(ROOT / "norm_stats"))
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--array_store", required=True)
    ap.add_argument("--min_gpus", type=int, default=8)
    ap.add_argument("--min_free_tb", type=float, default=1.0, help="free space needed at the cache root")
    ap.add_argument("--out", default=str(ROOT / "outputs/BPv4CDX/selfcheck_ae3.json"))
    args = ap.parse_args()
    report: dict = {"python": platform.python_version()}
    problems: list[str] = []

    pk = {}
    for name in PACKAGES:
        try:
            pk[name] = getattr(importlib.import_module(name), "__version__", "ok")
        except Exception as e:  # noqa: BLE001
            pk[name] = f"MISSING: {e}"
            problems.append(f"package {name} missing (no internet on this machine: evaluations needing it move to the dev machine)")
    report["packages"] = pk

    import torch

    report["gpus"] = [{"name": torch.cuda.get_device_name(i),
                       "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
                      for i in range(torch.cuda.device_count())]
    if len(report["gpus"]) < args.min_gpus:
        problems.append(f"only {len(report['gpus'])} GPUs visible (expected >= {args.min_gpus})")
    report["cpu_count"] = os.cpu_count()
    try:
        mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
        report["memory_gb"] = round(int(mem["MemTotal"].split()[0]) / 2**20, 1)
    except Exception:  # noqa: BLE001
        report["memory_gb"] = None
    for name, path in (("cache_root", args.cache_root), ("array_store", args.array_store)):
        Path(path).mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(path).free / 2**40
        report[f"{name}_free_tb"] = round(free, 2)
        if name == "cache_root" and free < args.min_free_tb:
            problems.append(f"only {free:.2f} TB free at {path} (P5 cache ~0.7 TB + array store)")

    from lerobot.policies.BPV4CDX.ae3_data import EXCLUDE_DEFAULT, _manifest_entry, source_of
    from lerobot.policies.BPV4CDX.paths import read_repo_list
    from lerobot.transforms.constants import get_feature_mapping

    repos = [r for r in read_repo_list(args.repo_file, args.data_root) if not any(x in r for x in EXCLUDE_DEFAULT)]
    with ThreadPoolExecutor(32) as pool:
        entries = list(pool.map(_manifest_entry, repos))
    bad = [e for e in entries if "error" in e]
    ok = [e for e in entries if "error" not in e]
    report["repos"] = {"listed_without_egodex": len(repos), "ok": len(ok), "errors": bad[:50],
                       "by_source": dict(Counter(source_of(e["rel"]) for e in ok)),
                       "episodes": sum(e["total_episodes"] for e in ok),
                       "hours": round(sum(e["total_frames"] / max(1, e["fps"]) for e in ok) / 3600, 1)}
    problems += [f"repo {e['rel']}: {e['error']}" for e in bad]

    stats_report = {}
    for (rt, resolved), example in {(e["robot_type"], e["resolved"]): e for e in ok}.items():
        cands = [Path(args.stats_root) / resolved / "delta" / "stats.json", Path(args.stats_root) / rt / "delta" / "stats.json"]
        path = next((p for p in cands if p.is_file()), None)
        if path is None:
            stats_report[f"{rt} ({resolved})"] = "MISSING"
            problems.append(f"norm stats missing for {rt} ({resolved}): tried {cands[0]}")
            continue
        stats = json.loads(path.read_text())
        repo = next(r for r in repos if r.rstrip("/").endswith(example["rel"].split("/", 1)[-1]))
        info = json.loads(Path(repo, "meta", "info.json").read_text())
        spec = get_feature_mapping(rt, info["features"])
        keys = list(spec["observation.state"]) + list(spec["action"])
        absent = [k for k in keys if k not in stats]
        stats_report[f"{rt} ({resolved})"] = {"file": str(path), "missing_keys": absent}
        if absent:
            problems.append(f"norm stats {path} lack {absent}")
    report["norm_stats"] = stats_report

    qwen = Path(args.qwen_path)
    for f in ("config.json", "tokenizer_config.json"):
        if not (qwen / f).is_file():
            problems.append(f"qwen file missing: {qwen / f}")
    if not list(qwen.glob("*.safetensors")):
        problems.append(f"no *.safetensors under {qwen}")
    urdf = ROOT / "assets/urdf/aloha-agilex/arx5_description_isaac.urdf"
    report["urdf"] = urdf.is_file()
    if not urdf.is_file():
        problems.append(f"URDF missing: {urdf} (HOM end-effector error would be skipped)")

    report["problems"] = problems
    report["pass"] = not problems
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("python", "cpu_count", "memory_gb", "cache_root_free_tb", "pass")}, indent=1))
    print(json.dumps(report["repos"]["by_source"], indent=1), f"episodes {report['repos']['episodes']}, hours {report['repos']['hours']}")
    for p in problems[:40]:
        print("PROBLEM:", p)
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
