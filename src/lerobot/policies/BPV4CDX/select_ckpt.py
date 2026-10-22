"""AE-3 checkpoint selection on the held-out set (design doc 4.4): for every saved checkpoint of a run, the matched
error and E_alt / E_match (same partners, same noise, protocol of gate_g4_g7). Never uses the independent test.

The priority between the ratio and the error is not decided in the design doc; the provisional rule here: among the
checkpoints whose alternate gap passes (>= 1.10, CI low > 1), the lowest matched error; else the highest ratio.

Usage: python -m lerobot.policies.BPV4CDX.select_ckpt --run <output_dir>/<timestamp> --out <dir> --name <run> \
         [--seeds 1] [--dataset.x=...]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from .common import load_cfg, make_policy, paired_bootstrap_ratio, reproducibility, write_result
from .gate_g4_g7 import THRESH, Evaluator


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training output dir containing checkpoints/<step>/pretrained_model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--batch", type=int, default=24)
    args, overrides = ap.parse_known_args()
    ckpts = sorted(p for p in Path(args.run, "checkpoints").glob("*/pretrained_model") if p.parent.name.isdigit())
    if not ckpts:
        raise SystemExit(f"no checkpoints under {args.run}/checkpoints")
    cfg = load_cfg(ckpts[-1], overrides)
    ev = Evaluator(cfg, ckpts[-1], None, args.seeds, args.batch, split="val")
    ev.encode_all(ev.policy)
    part = ev.partners(np.random.default_rng(0))
    alt = [a[0] if a else None for a in part["alternate"]]
    rows = []
    for ck in ckpts:
        policy = ev.policy if ck == ckpts[-1] else make_policy(load_cfg(ck, overrides), ck)
        z = ev.encode_all(policy)
        matched = ev.errors(policy, z, generic=False)["err"]
        alternate = ev.errors(policy, [z[j] if j is not None else z[i] for i, j in enumerate(alt)], generic=False)["err"]
        keys = [k for k in matched if alt[k[0]] is not None]
        r = paired_bootstrap_ratio(np.array([alternate[k] for k in keys]), np.array([matched[k] for k in keys]),
                                   np.array([k[0] for k in keys]))
        rows.append({"checkpoint": str(ck), "step": int(ck.parent.name), "matched_error": float(np.mean(list(matched.values()))),
                     "alternate": r, "alternate_gap": bool(r["ratio"] >= THRESH and r["ci95"][0] > 1.0)})
        logging.info("%s matched %.4f alt %.3f", ck.parent.name, rows[-1]["matched_error"], r["ratio"])
        if policy is not ev.policy:
            del policy
    passing = [r for r in rows if r["alternate_gap"]]
    pick = min(passing, key=lambda r: r["matched_error"]) if passing else max(rows, key=lambda r: r["alternate"]["ratio"])
    res = {"run": args.run, "rows": rows, "selected": pick["checkpoint"],
           "rule": "provisional: alternate gap passes -> lowest matched error; else highest ratio",
           "reproducibility": reproducibility(cfg, ckpts[-1], ev.policy, ev.ds)}
    print(write_result(args.out, args.name, res))
    for r in rows:
        print(r["step"], round(r["matched_error"], 4), round(r["alternate"]["ratio"], 3), r["alternate_gap"])
    print("selected:", pick["checkpoint"])


if __name__ == "__main__":
    main()
