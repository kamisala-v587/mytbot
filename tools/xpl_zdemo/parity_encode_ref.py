"""PARITY B, upstream side (F1 T6, q35va env) — encoder reference over MY records.

Loads N records from the freshly built F1 feature cache and runs them through the
frozen Stage-A demo encoder exactly as upstream ``BPV4CDXB.zdemo_cache.encode``
does (load_frozen(...).encoder, float32 inputs, cuda autocast bf16), saving
``z_ref.pt`` = {z: float32 [N, 128, 1024], keys, frozen_sha256} for the mibot-side
XR1 comparison (bp/tools/parity_zdemo_xr1.py, throwaway).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lerobot.policies.BPV4CDXB.frozen import load_frozen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, help="the F1 real feature cache root")
    ap.add_argument("--out", required=True, help="output dir for z_ref.pt")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--keys", default=None, help="comma-separated keys (default: first N of index.json)")
    ap.add_argument("--frozen", default=None, help="StageA_AE_frozen dir (default $BPV4CDXB_FROZEN)")
    args = ap.parse_args()

    import os

    frozen_dir = args.frozen or os.environ.get("BPV4CDXB_FROZEN")
    if not frozen_dir:
        raise SystemExit("--frozen or $BPV4CDXB_FROZEN is required")
    cache = Path(args.cache)
    if (cache / "index.json").is_file():
        index = json.loads((cache / "index.json").read_text())
    else:  # pre-finalize subset (e.g. the 8-episode smoke): glob the records directly
        index = {f"{p.parent.name}/{p.stem}": f"{p.parent.name}/{p.stem}.pt"
                 for p in cache.glob("*/episode_*.pt")}
        if not index:
            raise SystemExit(f"no records and no index.json under {cache}")
    keys = args.keys.split(",") if args.keys else sorted(index)[: args.count]
    device = torch.device("cuda")
    frozen = load_frozen(frozen_dir)
    encoder = frozen.encoder.to(device)

    feats, times, valids = [], [], []
    for key in keys:
        rec = torch.load(cache / index[key], map_location="cpu", weights_only=True)
        feats.append(rec["feats"].float())
        times.append(rec["time"].float())
        valids.append(rec["camera_valid"].bool())
    feats = torch.stack(feats).to(device)
    times = torch.stack(times).to(device)
    valids = torch.stack(valids).to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = encoder(feats, times, valids).float().cpu()  # zdemo_cache.encode, verbatim
    if tuple(z.shape) != (len(keys), 128, 1024):
        raise SystemExit(f"encoder output {tuple(z.shape)} != [N, 128, 1024]")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"z": z, "keys": keys, "frozen_sha256": frozen.ckpt_sha256}, out / "z_ref.pt")
    print(f"z_ref.pt: {tuple(z.shape)} over {len(keys)} records; mean abs z {float(z.abs().mean()):.4f}; "
          f"frozen sha {frozen.ckpt_sha256[:12]}; keys: {keys}")


if __name__ == "__main__":
    main()
