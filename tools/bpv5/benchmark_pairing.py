"""Repeatable synthetic CPU benchmark; each (size, mode, repeat) gets a fresh process.

Run: python tools/bpv5/benchmark_pairing.py --sizes 1000 3000 --repeats 2
JSONL reports build wall time, tracemalloc build peak, process peak RSS (including
serialization), pickle bytes and deterministic sampled candidate checksum. This
measures one L0 leaf only, not dataset startup, training or real metadata I/O.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time
import tracemalloc
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def pairing_module():
    packages = {}
    for name in ("lerobot", "lerobot.policies", "lerobot.policies.BPV5QFormer"):
        package = types.ModuleType(name)
        package.__path__ = []
        packages[name] = package
    data = types.ModuleType("lerobot.policies.BPV5QFormer.data80d")
    packages[data.__name__] = data
    packages["lerobot.policies.BPV5QFormer"].data80d = data
    name = "_bpv5_pairing_benchmark"
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/lerobot/policies/BPV5StageB/pairing.py")
    module = importlib.util.module_from_spec(spec)
    packages[name] = module
    with patch.dict(sys.modules, packages):
        spec.loader.exec_module(module)
        yield module


def measure(size, mode):
    with pairing_module() as pairing:
        items = [("agibot/synthetic", ep, 1) for ep in range(size)]
        pools = {"agibot/synthetic": list(range(size))}
        tracemalloc.start()
        started = time.perf_counter()
        partners, stats = pairing.build_partner_map(items, pools, storage=mode)
        wall = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        checksum = sum(pairing.choose_partner(value, ep, ep * 104729 - 31)
                       for (_, ep), value in partners.items())
        started = time.perf_counter()
        payload = pickle.dumps(partners, protocol=5)
        pickle_wall = time.perf_counter() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux KiB.
        rss_bytes = rss if sys.platform == "darwin" else rss * 1024
        return {"mode": mode, "n": size, "build_wall_s": wall,
                "build_tracemalloc_peak_bytes": peak, "process_peak_rss_bytes": rss_bytes,
                "pickle_bytes": len(payload), "pickle_wall_s": pickle_wall,
                "sample_checksum": checksum, "stats": stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 3000])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--worker", choices=["list", "shared_bucket"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 4 for size in args.sizes):
        parser.error("repeats must be >= 1 and sizes >= 4")
    if args.worker:
        print(json.dumps(measure(args.sizes[0], args.worker), sort_keys=True))
        return
    for size in args.sizes:
        for repeat in range(args.repeats):
            pair = []
            for mode in ("list", "shared_bucket"):
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    "--worker", mode, "--sizes", str(size)], check=True,
                    text=True, capture_output=True, timeout=120)
                row = json.loads(result.stdout)
                row["repeat"] = repeat + 1
                pair.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
            if pair[0]["sample_checksum"] != pair[1]["sample_checksum"] or pair[0]["stats"] != pair[1]["stats"]:
                raise RuntimeError("mode results differ")


if __name__ == "__main__":
    main()
