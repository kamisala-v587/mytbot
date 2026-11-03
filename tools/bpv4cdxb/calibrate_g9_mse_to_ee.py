"""Empirically calibrate normalized G9 action-chunk MSE to ALOHA link6 position error.

Uses held-out queries/time points from a fixed G9 rows.jsonl, the real per-repo delta+normalization transforms,
and URDF FK. Adds controlled errors to the ground-truth normalized 50-step chunk, then reports both-arm and
moving-arm mean link6 distance. Different temporal / dimension patterns show why MSE has no unique cm conversion.

Example:
  source tools/bpv4cdx/env.local.sh
  python tools/bpv4cdxb/calibrate_g9_mse_to_ee.py --rows outputs/BPv4CDX/stage_b/g9_bp/2026-11-02_12-04_100k_rows.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.BPV4CDX.common import ActionInverter
from lerobot.policies.BPV4CDX.ee_fk import ArmFK
from lerobot.policies.BPV4CDXB.dataset_main import make_main_dataset
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/vla/workspace/my_tbot/outputs/BPv4CDX/stage_b/b4_350k_sft/2026-11-02_12-04/2026-11-02_12-05/checkpoints/100000/pretrained_model")
    ap.add_argument("--index", default="/vla/workspace/my_tbot/outputs/BPv4CDX/stage_b/g9_bp/b4_sft_g9_index.json")
    ap.add_argument("--rows", default="/vla/workspace/my_tbot/outputs/BPv4CDX/stage_b/g9_bp/2026-11-02_12-04_100k_rows.jsonl")
    ap.add_argument("--out", default="/vla/workspace/my_tbot/outputs/BPv4CDX/stage_b/g9_bp/mse_to_ee_calibration.json")
    ap.add_argument("--queries", type=int, default=160)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261103)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines()]
    # Only held-out sets: same RoboTwin task family and embodiment as the shipped FK.
    qs = {}
    for r in rows:
        if r.get("group") == "heldout_sets":
            qs[(r["q"], int(r["frame"]))] = r
    rng = np.random.default_rng(args.seed)
    keys = sorted(qs)
    if len(keys) > args.queries:
        keys = [keys[i] for i in sorted(rng.choice(len(keys), args.queries, replace=False))]

    ckpt = Path(args.ckpt)
    cfg = TrainPipelineConfig.from_pretrained(str(ckpt / "train_config.json"))
    cfg.dataset.split = "val"
    idx = json.loads(Path(args.index).read_text())
    idx["load_procs"] = 12
    ds, _ = make_main_dataset(cfg, idx)
    fk = ArmFK()
    levels = (0.01, 0.1, 0.2, 0.3)
    patterns = ("iid_all_dims", "chunk_bias_all_dims", "iid_joints_only", "iid_grippers_only")
    vals: dict[str, dict[str, list[float]]] = {p: {str(m): [] for m in levels} for p in patterns}
    rms_check = defaultdict(list)

    for qi, (eid, frame) in enumerate(keys):
        rel, ep = eid.rsplit("|", 1)
        ep = int(ep)
        repo = ds.repos[rel]
        emb = repo.emb
        if not emb.dual_arm_61:
            continue
        H = int(cfg.policy.chunk_size)
        sample = repo.query(ep, frame, H)
        gt = sample[ACTION].float().numpy()[:, :14]
        valid = sample[ACTION_VALID_MASK].bool().numpy()
        if valid.sum() < 2:
            continue
        start_state = np.asarray(repo.states[ep][frame], dtype=np.float64)[:14]
        inverter = ActionInverter(repo)
        gt_abs = inverter(gt, start_state)
        gt_fk = fk.positions(torch.as_tensor(gt_abs, dtype=torch.float64)).numpy()  # [H, 2, 3]
        joints = np.r_[0:6, 7:13]
        grippers = np.array([6, 13])
        moving = np.sum(np.abs(np.diff(gt_abs[:, joints], axis=0)), axis=0).reshape(2, 6).sum(1)
        moving_arm = int(np.argmax(moving))
        for mi, mse in enumerate(levels):
            for draw in range(args.draws):
                seed = args.seed + qi * 1009 + mi * 97 + draw
                for pattern in patterns:
                    gen = np.random.default_rng(seed)
                    if pattern == "chunk_bias_all_dims":
                        noise = np.broadcast_to(gen.normal(size=(1, 14)), (H, 14)).copy()
                    elif pattern == "iid_joints_only":
                        noise = np.zeros((H, 14)); noise[:, joints] = gen.normal(size=(H, len(joints)))
                    elif pattern == "iid_grippers_only":
                        noise = np.zeros((H, 14)); noise[:, grippers] = gen.normal(size=(H, len(grippers)))
                    else:
                        noise = gen.normal(size=(H, 14))
                    mask = np.broadcast_to(valid[:, None], noise.shape).copy()
                    if pattern == "iid_joints_only": mask[:, grippers] = False
                    if pattern == "iid_grippers_only": mask[:, joints] = False
                    rms = float(np.sqrt(np.mean(noise[mask] ** 2)))
                    if rms == 0:
                        continue
                    # Exact per-query normalized MSE on the dimensions active in this pattern.
                    noise *= np.sqrt(mse) / rms
                    pred_norm = gt.copy()
                    pred_norm[mask] += noise[mask]
                    pred_abs = inverter(pred_norm, start_state)
                    pred_fk = fk.positions(torch.as_tensor(pred_abs, dtype=torch.float64)).numpy()
                    dist = np.linalg.norm(pred_fk - gt_fk, axis=-1) * 100.0
                    w = valid.astype(float)[:, None]
                    both_cm = float((dist * w).sum() / (w.sum() * 2))
                    moving_cm = float(dist[:, moving_arm][valid].mean())
                    vals[pattern][str(mse)].append((both_cm, moving_cm))
                    rms_check[(pattern, str(mse))].append(float(np.mean((noise[mask]) ** 2)))
        if (qi + 1) % 40 == 0:
            print(f"calibrated {qi + 1}/{len(keys)} queries", flush=True)

    out = {"metric": "masked mean squared normalized delta action error, valid steps and 14 real dimensions",
           "queries": len(keys), "draws": args.draws, "levels": list(levels),
           "noise_models": {}, "notes": [
               "iid_all_dims: independent Gaussian normalized error at every timestep and action dimension",
               "chunk_bias_all_dims: one Gaussian normalized error per action dimension, constant over the 50-step chunk",
               "iid_joints_only / iid_grippers_only: same numeric MSE computed only over that dimension subset; this demonstrates dimensional ambiguity",
               "FK: ALOHA link6 position, absolute joint targets reconstructed with the actual dataset transform and the query start state",
               "This is a controlled perturbation calibration, not an identity mapping for arbitrary model residuals."]}
    for p in patterns:
        out["noise_models"][p] = {}
        for mse in map(str, levels):
            a = np.asarray(vals[p][mse], dtype=float)
            if not len(a):
                continue
            out["noise_models"][p][mse] = {
                "n": len(a), "normalized_mse_mean": float(np.mean(rms_check[(p, mse)])),
                "ee_cm_both_arms_mean": float(a[:, 0].mean()), "ee_cm_both_arms_p50": float(np.median(a[:, 0])),
                "ee_cm_both_arms_p90": float(np.percentile(a[:, 0], 90)),
                "ee_cm_moving_arm_mean": float(a[:, 1].mean()), "ee_cm_moving_arm_p50": float(np.median(a[:, 1])),
                "ee_cm_moving_arm_p90": float(np.percentile(a[:, 1], 90)),
            }
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print("MSE | noise model | EE cm both (mean/p50/p90) | EE cm moving arm (mean/p50/p90)")
    for mse in map(str, levels):
        for p in patterns:
            r = out["noise_models"].get(p, {}).get(mse)
            if r:
                print(f"{mse} | {p} | {r['ee_cm_both_arms_mean']:.3f}/{r['ee_cm_both_arms_p50']:.3f}/{r['ee_cm_both_arms_p90']:.3f} | "
                      f"{r['ee_cm_moving_arm_mean']:.3f}/{r['ee_cm_moving_arm_p50']:.3f}/{r['ee_cm_moving_arm_p90']:.3f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
