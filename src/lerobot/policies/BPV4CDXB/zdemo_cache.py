"""Z_demo cache for Stage B (development doc 2.2 / 3.4, B-0 step 3).

For every episode of the pairing index: the demo ViT cache record (`BPV4CDX.build_cache`, task_variant layout) ->
P-pooled features -> frozen encoder -> Z_demo [128, hidden] bf16 at `<out>/<frozen tag>/<repo>/epXXXXX.pt`.
Checks (written to `<out>/<frozen tag>/manifest.json`):
  - every demo cache record uses the sampler version recorded in the frozen delivery;
  - --verify N episodes are re-encoded through the full Stage A model (`encode_demo` of the delivered checkpoint) and
    must match the cached Z_demo (max abs difference reported).

usage: python -m lerobot.policies.BPV4CDXB.zdemo_cache --frozen <StageA_AE_frozen/<step>> --demo_cache <dir> \
         --index outputs/BPv4CDX/stage_b/stage_b_index.json --out /vla/workspace/cache/bpv4cdxb_zdemo [--verify 32]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.BPV4CDX.dataset import pool_spatial
from lerobot.policies.BPV4CDX.sampling import SAMPLER_VERSIONS

from .frozen import describe, load_frozen, sha256_file


def zdemo_path(root: str | Path, tag: str, repo_rel: str, episode: int) -> Path:
    return Path(root) / tag / repo_rel / f"ep{int(episode):05d}.pt"


def load_demo(demo_cache: Path, repo_rel: str, episode: int, spatial: int) -> dict:
    rec = torch.load(demo_cache / repo_rel / f"ep{int(episode):05d}.pt", map_location="cpu", weights_only=False)
    feats = rec["feats"]
    if feats.shape[2] != spatial:
        feats = pool_spatial(feats, spatial)
    valid = rec.get("camera_valid")
    return {"feats": feats.float(), "time": rec["time_ratio"].float(),
            "valid": valid.bool() if valid is not None else torch.ones(feats.shape[:2], dtype=torch.bool),
            "version": (rec.get("sampler_version") or {}).get("version")}


@torch.no_grad()
def encode(encoder, items: list[dict], device: str) -> torch.Tensor:
    feats = torch.stack([d["feats"] for d in items]).to(device)
    t = torch.stack([d["time"] for d in items]).to(device)
    v = torch.stack([d["valid"] for d in items]).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return encoder(feats, t, v).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--demo_cache", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--verify", type=int, default=32, help="episodes re-encoded through the full Stage A model")
    args = ap.parse_args()
    device = "cuda"
    frozen = load_frozen(args.frozen, device)
    spatial = int(frozen.manifest["vit"]["spatial_tokens_used"])
    want_version = frozen.manifest["sampler"]["version"]
    index = json.loads(Path(args.index).read_text())
    eps = sorted(index["episodes"].items())
    demo_cache, out_root = Path(args.demo_cache), Path(args.out)
    t0, done, versions, skipped = time.time(), 0, {}, 0
    for s in range(0, len(eps), args.batch):
        chunk = eps[s : s + args.batch]
        todo = [(k, e) for k, e in chunk if not zdemo_path(out_root, frozen.tag, e["repo"], e["episode"]).is_file()]
        skipped += len(chunk) - len(todo)
        if not todo:
            continue
        items = [load_demo(demo_cache, e["repo"], e["episode"], spatial) for _, e in todo]
        for d in items:
            versions[d["version"]] = versions.get(d["version"], 0) + 1
        bad = [k for (k, _), d in zip(todo, items) if d["version"] != want_version]
        if bad:
            raise SystemExit(f"demo cache sampler version {items[0]['version']} != frozen {want_version}: {bad[:3]}")
        z = encode(frozen.encoder, items, device)
        for (k, e), zi in zip(todo, z):
            p = zdemo_path(out_root, frozen.tag, e["repo"], e["episode"])
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"z_demo": zi.to(torch.bfloat16).cpu(), "frozen_sha256": frozen.ckpt_sha256,
                        "sampler_version": want_version, "spatial": spatial}, p)
        done += len(todo)
        if (s // args.batch) % 10 == 0:
            print(f"{s + len(chunk)} / {len(eps)} ({time.time() - t0:.0f} s)", flush=True)

    verify = {}
    if args.verify:
        from lerobot.policies.BPV4CDX.common import load_cfg, make_policy

        ck = Path(args.frozen) / "pretrained_model"
        policy = make_policy(load_cfg(ck), ck).to(device).eval()
        rng = np.random.default_rng(0)
        pick = [eps[i] for i in sorted(rng.choice(len(eps), size=min(args.verify, len(eps)), replace=False))]
        diffs = []
        for s in range(0, len(pick), 8):
            chunk = pick[s : s + 8]
            items = [load_demo(demo_cache, e["repo"], e["episode"], spatial) for _, e in chunk]
            batch = {"demo.feats": torch.stack([d["feats"] for d in items]).to(device),
                     "demo.time": torch.stack([d["time"] for d in items]).to(device),
                     "demo.camera_valid": torch.stack([d["valid"] for d in items]).to(device)}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                z_full = policy.model.encode_demo(batch).float().cpu()
            for (k, e), zf in zip(chunk, z_full):
                zc = torch.load(zdemo_path(out_root, frozen.tag, e["repo"], e["episode"]), weights_only=False)["z_demo"].float()
                diffs.append(float((zc - zf).abs().max()))
        scale = float(z_full.abs().mean())
        verify = {"episodes": len(diffs), "max_abs_diff": max(diffs), "mean_abs_z": scale,
                  "pass": max(diffs) < 0.02 * max(1.0, scale * 10)}
    manifest = {"frozen": describe(frozen), "frozen_dir": str(Path(args.frozen).resolve()),
                "demo_cache": str(demo_cache), "index": str(Path(args.index).resolve()),
                "index_sha256": sha256_file(Path(args.index)), "episodes": len(eps), "written": done,
                "skipped_existing": skipped, "sampler_versions_seen": versions, "spatial": spatial, "verify": verify}
    (out_root / frozen.tag).mkdir(parents=True, exist_ok=True)
    (out_root / frozen.tag / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ("episodes", "written", "skipped_existing", "sampler_versions_seen", "verify")},
                     indent=1))
    print(out_root / frozen.tag)


if __name__ == "__main__":
    main()
