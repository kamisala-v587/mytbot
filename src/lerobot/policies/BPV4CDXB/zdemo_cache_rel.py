"""Z_demo cache for the Stage B main path on the AE-3 cache (`rel` layout), multi-GPU, resumable.

Every episode of the repos in --repo_file that has an AE-3 P5 record (`<demo_cache>/<rel>/epXXXXX.pt`) ->
frozen encoder -> `<out>/<frozen tag>/<rel>/epXXXXX.pt` = {z_demo [128, hidden] bf16, frozen_sha256,
sampler_version, spatial}. The sampler version of every record must equal the frozen delivery's.

  build     one process per GPU: --rank r --world N (DataLoader workers read the records)
  progress  done / total, rate since the previous call, ETA (reads `<out>/<tag>/_progress/rank*.json`)
  verify    N random cached episodes re-encoded through the full Stage A model; writes `<out>/<tag>/verify.json`

usage: python -m lerobot.policies.BPV4CDXB.zdemo_cache_rel build --frozen <StageA_AE_frozen/<step>> \
         --demo_cache <cache>/bpv4cdx_ae3_su --repo_file configs/et3.txt --out <cache>/bpv4cdxb_zdemo --rank 0 --world 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.BPV4CDX.paths import rel_id


def episode_list(repo_file: str, demo_cache: Path) -> list[tuple[str, int]]:
    items = []
    for line in Path(repo_file).read_text().splitlines():
        if not line.strip():
            continue
        rel = rel_id(line.strip())
        d = demo_cache / rel
        if d.is_dir():
            eps = sorted(int(n[2:7]) for n in os.listdir(d) if n.startswith("ep") and n.endswith(".pt") and ".tmp" not in n)
            items += [(rel, e) for e in eps]
    return items


class Records(torch.utils.data.Dataset):
    def __init__(self, items, demo_cache: Path) -> None:
        self.items, self.demo_cache = items, demo_cache

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        rel, ep = self.items[i]
        r = torch.load(self.demo_cache / rel / f"ep{ep:05d}.pt", map_location="cpu", weights_only=False)
        return {"rel": rel, "ep": ep, "feats": r["feats"], "time": r["time_ratio"].float(),
                "valid": r["camera_valid"].bool(), "version": (r.get("sampler_version") or {}).get("version")}


def collate(batch):
    return {"rel": [b["rel"] for b in batch], "ep": [b["ep"] for b in batch], "version": [b["version"] for b in batch],
            "feats": torch.stack([b["feats"] for b in batch]), "time": torch.stack([b["time"] for b in batch]),
            "valid": torch.stack([b["valid"] for b in batch])}


def cmd_build(args) -> None:
    from .frozen import load_frozen

    device = "cuda"
    frozen = load_frozen(args.frozen, device)
    want = frozen.manifest["sampler"]["version"]
    demo_cache, root = Path(args.demo_cache), Path(args.out) / frozen.tag
    items = episode_list(args.repo_file, demo_cache)[args.rank :: args.world]
    listed: dict[str, set[str]] = {}
    todo = []
    for rel, ep in items:
        if rel not in listed:
            d = root / rel
            listed[rel] = set(os.listdir(d)) if d.is_dir() else set()
        if f"ep{ep:05d}.pt" not in listed[rel]:
            todo.append((rel, ep))
    prog = root / "_progress"
    prog.mkdir(parents=True, exist_ok=True)
    pfile = prog / f"rank{args.rank}.json"
    total, done0 = len(items), len(items) - len(todo)
    loader = torch.utils.data.DataLoader(Records(todo, demo_cache), batch_size=args.batch, num_workers=args.workers,
                                         collate_fn=collate, prefetch_factor=4 if args.workers else None)
    t0, done, last = time.time(), 0, 0.0
    for b in loader:
        bad = [f"{r}|{e}" for r, e, v in zip(b["rel"], b["ep"], b["version"]) if v != want]
        if bad:
            raise SystemExit(f"sampler version != frozen {want}: {bad[:3]}")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z = frozen.encoder(b["feats"].to(device).float(), b["time"].to(device), b["valid"].to(device)).float()
        z = z.to(torch.bfloat16).cpu()
        for rel, ep, zi in zip(b["rel"], b["ep"], z):
            p = root / rel / f"ep{ep:05d}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            torch.save({"z_demo": zi.clone(), "frozen_sha256": frozen.ckpt_sha256, "sampler_version": want,
                        "spatial": int(frozen.manifest["vit"]["spatial_tokens_used"])}, tmp)
            os.replace(tmp, p)
        done += len(b["ep"])
        if time.time() - last > 30:
            last = time.time()
            pfile.write_text(json.dumps({"rank": args.rank, "total": total, "done": done0 + done, "time": last,
                                         "rate": done / max(1e-6, last - t0)}))
    pfile.write_text(json.dumps({"rank": args.rank, "total": total, "done": done0 + done, "time": time.time(),
                                 "rate": done / max(1e-6, time.time() - t0), "finished": True}))
    print(f"rank {args.rank}: {done} written, {done0} existed, {total} total", flush=True)


def cmd_progress(args) -> None:
    from .frozen import sha256_file

    tag = sha256_file(Path(args.frozen) / "StageA_AE_frozen.ckpt")[:12]
    prog = Path(args.out) / tag / "_progress"
    rows = [json.loads(p.read_text()) for p in sorted(prog.glob("rank*.json"))]
    if not rows:
        print("no progress yet:", prog)
        return
    total, done = sum(r["total"] for r in rows), sum(r["done"] for r in rows)
    rate = sum(r["rate"] for r in rows if not r.get("finished"))
    fin = sum(bool(r.get("finished")) for r in rows)
    eta = (total - done) / rate / 60 if rate > 0 else 0.0
    print(f"{time.strftime('%F %T')}  Z_demo {done} / {total} ({100 * done / max(1, total):.1f}%), "
          f"{rate:.0f} episodes/s, ETA {eta:.0f} min, ranks finished {fin} / {len(rows)}")


def cmd_verify(args) -> None:
    from lerobot.policies.BPV4CDX.common import load_cfg, make_policy

    from .frozen import load_frozen

    device = "cuda"
    frozen = load_frozen(args.frozen, device)
    demo_cache, root = Path(args.demo_cache), Path(args.out) / frozen.tag
    items = episode_list(args.repo_file, demo_cache)
    rng = np.random.default_rng(0)
    pick = [items[i] for i in sorted(rng.choice(len(items), size=min(args.n, len(items)), replace=False))]
    pick = [it for it in pick if (root / it[0] / f"ep{it[1]:05d}.pt").is_file()]
    ck = Path(args.frozen) / "pretrained_model"
    policy = make_policy(load_cfg(ck), ck).to(device).eval()
    ds = Records(pick, demo_cache)
    diffs, scale = [], []
    for s in range(0, len(pick), 8):
        b = collate([ds[i] for i in range(s, min(len(pick), s + 8))])
        batch = {"demo.feats": b["feats"].to(device).float(), "demo.time": b["time"].to(device),
                 "demo.camera_valid": b["valid"].to(device)}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z_full = policy.model.encode_demo(batch).float().cpu()
        for rel, ep, zf in zip(b["rel"], b["ep"], z_full):
            zc = torch.load(root / rel / f"ep{ep:05d}.pt", weights_only=False)["z_demo"].float()
            diffs.append(float((zc - zf).abs().max()))
            scale.append(float(zf.abs().mean()))
    res = {"episodes": len(diffs), "max_abs_diff": max(diffs) if diffs else None, "mean_abs_z": float(np.mean(scale)) if scale else None}
    res["pass"] = bool(diffs) and res["max_abs_diff"] < 0.02 * max(1.0, res["mean_abs_z"] * 10)
    (root / "verify.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "progress", "verify"])
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--demo_cache", default="")
    ap.add_argument("--repo_file", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n", type=int, default=32, help="verify: episodes")
    args = ap.parse_args()
    {"build": cmd_build, "progress": cmd_progress, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
