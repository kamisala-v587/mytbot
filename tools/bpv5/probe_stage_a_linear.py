#!/usr/bin/env python3
"""Stage A offline linear probe on Z_demo (Q-Former only; no VLM/AE).

Loads demo_encoder weights from a Stage A checkpoint, encodes the frozen
feasibility episode pool, fits ridge (dual) for labels A/B/C, and writes R²
vs three controls: random Q-Former, within-leaf Z shuffle, leaf-mean.

usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python tools/bpv5/probe_stage_a_linear.py \\
    --step 40000 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file

from lerobot.policies.BPV4CDX.modeling_bpv4cdx import TemporalVideoQFormer

CACHE = Path("/vla/workspace/cache/bpv5_cleaned_p5")
RUN = Path("/vla/workspace/my_tbot/outputs/BPv5/stage_a/main/2026-11-17_23-32")
FEAS = Path("/vla/workspace/code/conclusion/bpv5/assets/stage_a_probe_label_feasibility.json")
OUT_DIR = Path("/vla/workspace/code/conclusion/bpv5/assets/stage_a_accept_probe")
ALPHAS = (1e-1, 1.0, 10.0, 1e2, 1e3, 1e4)


def qf_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        visual_dim=2560,
        num_cameras=3,
        spatial_tokens=5,
        num_slots=32,
        num_demo_tokens=128,
        qformer_dim=1024,
        qformer_num_heads=16,
        qformer_num_layers=2,
        qformer_ff_mult=4,
    )


def load_encoder(ckpt_dir: Path | None, device: str, *, random: bool = False) -> TemporalVideoQFormer:
    enc = TemporalVideoQFormer(qf_cfg(), 2560)
    if not random:
        assert ckpt_dir is not None
        sd = load_file(str(ckpt_dir / "model.safetensors"))
        prefix = "model.demo_encoder."
        sub = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
        incompatible = enc.load_state_dict(sub, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"demo_encoder load bad: {incompatible}")
    enc.eval().requires_grad_(False)
    return enc.to(device=device, dtype=torch.bfloat16)


@torch.no_grad()
def encode_pool(
    enc: TemporalVideoQFormer,
    rows: list[dict],
    device: str,
    cache: Path,
    batch_size: int = 8,
) -> np.ndarray:
    zs = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        feats, times, cams = [], [], []
        for r in chunk:
            rec = torch.load(cache / r["rel"] / f"ep{r['ep']:06d}.pt", map_location="cpu", weights_only=True)
            feats.append(rec["feats"])
            times.append(rec["time_ratio"].float())
            cams.append(rec["camera_valid"].bool())
        z = enc(
            torch.stack(feats).to(device=device, dtype=torch.bfloat16),
            torch.stack(times).to(device=device, dtype=torch.bfloat16),
            torch.stack(cams).to(device),
        )
        zs.append(z.float().cpu().numpy())
    return np.concatenate(zs, axis=0)  # [N, 128, 2560]


def flatten_z(z: np.ndarray) -> np.ndarray:
    return z.reshape(z.shape[0], -1).astype(np.float64)


def ridge_predict(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, alpha: float) -> np.ndarray:
    """Dual ridge: α = (XXᵀ + λI)⁻¹ y ; ŷ = X_te X_trᵀ α."""
    n = x_tr.shape[0]
    k = x_tr @ x_tr.T
    k.flat[:: n + 1] += float(alpha)
    # y_tr: [n, d]
    try:
        dual = np.linalg.solve(k, y_tr)
    except np.linalg.LinAlgError:
        dual = np.linalg.lstsq(k, y_tr, rcond=None)[0]
    return x_te @ (x_tr.T @ dual)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-dim R²; empty → nan."""
    if y_true.size == 0:
        return float("nan")
    vals = []
    for j in range(y_true.shape[1]):
        yt, yp = y_true[:, j], y_pred[:, j]
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        ss_res = float(np.sum((yt - yp) ** 2))
        vals.append(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan"))
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def leaf_split_indices(rows: list[dict], mask: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tr, te = [], []
    by = defaultdict(list)
    for i, ok in enumerate(mask):
        if ok:
            by[rows[i]["rel"]].append(i)
    for idxs in by.values():
        idxs = list(idxs)
        rng.shuffle(idxs)
        if len(idxs) < 2:
            # single ep in leaf for this label: put in test only if ≥1 train exists elsewhere;
            # skip unstable singles by sending to train fold marker then drop if alone.
            te.extend(idxs)
            continue
        mid = len(idxs) // 2
        tr.extend(idxs[:mid])
        te.extend(idxs[mid:])
    return np.asarray(tr, dtype=int), np.asarray(te, dtype=int)


def pick_alpha(x_tr, y_tr, seed: int) -> float:
    """Leave-one-out style: half of train as inner val."""
    if len(x_tr) < 4:
        return 100.0
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x_tr))
    mid = max(1, len(perm) // 2)
    a_i, b_i = perm[:mid], perm[mid:]
    best_a, best_r = ALPHAS[0], -1e9
    for a in ALPHAS:
        pred = ridge_predict(x_tr[a_i], y_tr[a_i], x_tr[b_i], a)
        r = r2_score(y_tr[b_i], pred)
        if np.isfinite(r) and r > best_r:
            best_r, best_a = r, a
    return float(best_a)


def eval_label(
    name: str,
    rows: list[dict],
    z_bp: np.ndarray,
    z_rand: np.ndarray,
    seed: int,
) -> dict:
    y = np.array([r[name] if r.get(name) is not None else [np.nan] * 3 for r in rows], dtype=np.float64)
    mask = np.array([r.get(name) is not None and np.isfinite(r[name]).all() for r in rows])
    # A already filtered ≥1cm in feasibility; keep as-is
    tr_i, te_i = leaf_split_indices(rows, mask, seed)
    if len(tr_i) < 2 or len(te_i) < 1:
        return {
            "label": name,
            "n_pool": int(mask.sum()),
            "n_train": int(len(tr_i)),
            "n_test": int(len(te_i)),
            "skip": "too_small",
        }

    x_bp = flatten_z(z_bp)
    x_rand = flatten_z(z_rand)
    alpha = pick_alpha(x_bp[tr_i], y[tr_i], seed + 1)

    pred_bp = ridge_predict(x_bp[tr_i], y[tr_i], x_bp[te_i], alpha)
    pred_rand = ridge_predict(x_rand[tr_i], y[tr_i], x_rand[te_i], alpha)

    # within-leaf shuffle of Z on train+test pool, then same split indices
    z_shuf = z_bp.copy()
    by = defaultdict(list)
    for i in np.where(mask)[0]:
        by[rows[i]["rel"]].append(int(i))
    rng = np.random.default_rng(seed + 2)
    for idxs in by.values():
        if len(idxs) < 2:
            continue
        perm = rng.permutation(len(idxs))
        src = [idxs[j] for j in perm]
        block = z_bp[src].copy()
        for k, i in enumerate(idxs):
            z_shuf[i] = block[k]
    x_shuf = flatten_z(z_shuf)
    pred_shuf = ridge_predict(x_shuf[tr_i], y[tr_i], x_shuf[te_i], alpha)

    # leaf-mean baseline: for each test ep, predict mean y of train eps in same leaf
    pred_mean = np.zeros_like(y[te_i])
    leaf_tr = defaultdict(list)
    for i in tr_i:
        leaf_tr[rows[i]["rel"]].append(y[i])
    global_mean = y[tr_i].mean(axis=0)
    for j, i in enumerate(te_i):
        vecs = leaf_tr.get(rows[i]["rel"])
        pred_mean[j] = np.mean(vecs, axis=0) if vecs else global_mean

    def by_source(pred: np.ndarray) -> dict[str, float]:
        out = {}
        for src in ("agibot", "interndata"):
            sel = [k for k, i in enumerate(te_i) if rows[i]["rel"].startswith(src + "/")]
            if not sel:
                out[src] = float("nan")
                continue
            out[src] = r2_score(y[te_i][sel], pred[sel])
        return out

    return {
        "label": name,
        "n_pool": int(mask.sum()),
        "n_train": int(len(tr_i)),
        "n_test": int(len(te_i)),
        "alpha": alpha,
        "r2": {
            "bp": r2_score(y[te_i], pred_bp),
            "random_qf": r2_score(y[te_i], pred_rand),
            "shuffle_z": r2_score(y[te_i], pred_shuf),
            "leaf_mean": r2_score(y[te_i], pred_mean),
        },
        "r2_by_source": {
            "bp": by_source(pred_bp),
            "random_qf": by_source(pred_rand),
            "shuffle_z": by_source(pred_shuf),
            "leaf_mean": by_source(pred_mean),
        },
        "rmse_cm": {
            "bp": float(np.sqrt(np.mean((y[te_i] - pred_bp) ** 2)) * 100.0),
            "leaf_mean": float(np.sqrt(np.mean((y[te_i] - pred_mean) ** 2)) * 100.0),
        },
    }


def build_rows(feas: dict) -> list[dict]:
    rows = []
    for src, s in feas["sources"].items():
        for leaf in s.get("leaf_detail", []):
            rel = leaf["rel"]
            for e in leaf["episodes"]:
                if not e.get("has_A"):
                    continue
                rows.append(
                    {
                        "rel": rel,
                        "ep": int(e["ep"]),
                        "source": src,
                        "arm": e.get("arm"),
                        "A": e.get("A"),
                        "B": e.get("B") if e.get("has_B") else None,
                        "C": e.get("C") if e.get("has_C") else None,
                    }
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--feas", type=Path, default=FEAS)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_random", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    feas = json.loads(args.feas.read_text())
    rows = build_rows(feas)
    ckpt = args.run / "checkpoints" / f"{args.step:07d}" / "pretrained_model"
    if not (ckpt / "model.safetensors").is_file():
        raise SystemExit(f"missing ckpt: {ckpt}")

    print(f"step={args.step} n_eps={len(rows)} ckpt={ckpt}", flush=True)
    enc = load_encoder(ckpt, args.device, random=False)
    z_bp = encode_pool(enc, rows, args.device, args.cache, args.batch_size)
    del enc
    torch.cuda.empty_cache()

    if args.skip_random:
        z_rand = np.zeros_like(z_bp)
    else:
        torch.manual_seed(args.seed + 123)
        enc_r = load_encoder(None, args.device, random=True)
        z_rand = encode_pool(enc_r, rows, args.device, args.cache, args.batch_size)
        del enc_r
        torch.cuda.empty_cache()

    results = {
        "step": args.step,
        "n_eps": len(rows),
        "ckpt": str(ckpt),
        "seconds_encode": round(time.time() - t0, 1),
        "labels": {},
    }
    for lab in ("A", "B", "C"):
        results["labels"][lab] = eval_label(lab, rows, z_bp, z_rand, seed=args.seed + ord(lab))

    results["seconds_total"] = round(time.time() - t0, 1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"probe_step_{args.step:07d}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps({"out": str(out), "seconds": results["seconds_total"], "r2": {k: v.get("r2") for k, v in results["labels"].items()}}, indent=2))


if __name__ == "__main__":
    main()
