"""G3: single-trajectory overfit check (task book 9.4).

On the trained `ae_g3_overfit` checkpoint (one episode, 4 fixed segments, generic text, no state):
  - masked action error of sampled segments (fixed noise, several seeds) vs. the baseline that outputs the
    episode's mean action for every segment: reduction must be >= 80%;
  - distinctness: mean pairwise distance between the predicted segments must reach >= 50% of the pairwise distance
    between the true segments;
  - all outputs finite with the right shape.
Blank-video and slot-shuffle errors are recorded (not a gate: one episode can be memorised from segment_query alone).

Usage: python -m lerobot.policies.BPV4CDX.gate_g3 --ckpt <pretrained_model> --out <dir> [--seeds 5] [draccus overrides]
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import torch

from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION

from .common import collate, load_cfg, make_dataset, make_policy, masked_error, reproducibility, to_device, write_result


def sample(policy, batch, seed: int, n: int, horizon: int, dim: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn((n, horizon, dim), generator=g).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return policy.model.sample_segments(batch, noise=noise).float()


def pairwise(x: torch.Tensor, valid: torch.Tensor) -> float:
    d = []
    for i, j in itertools.combinations(range(x.shape[0]), 2):
        m = (valid[i] & valid[j])[:, None].float()
        d.append(float(((x[i] - x[j]) ** 2 * m).sum() / (m.sum() * x.shape[-1]).clamp_min(1)))
    return float(np.mean(d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--name", default="g3", help="result file name (e.g. g3_fullprefix for the diagnostic run)")
    args, overrides = ap.parse_known_args()
    cfg = load_cfg(args.ckpt, overrides)
    ds = make_dataset(cfg)
    if len(ds) != 1:
        raise ValueError(f"G3 expects the single-episode dataset of ae_g3_overfit, got {len(ds)} trajectories")
    policy = make_policy(cfg, args.ckpt)
    item = ds[0]
    batch = to_device(collate([item]))
    target = batch[ACTION][0]  # [S, H, A]
    valid = batch[ACTION_VALID_MASK][0].bool()
    s, h, a = target.shape
    dim = int(policy.config.max_action_dim)

    preds = [sample(policy, batch, seed, s, h, dim)[..., :a] for seed in range(args.seeds)]
    err = torch.stack([masked_error(p, target, valid) for p in preds])  # [seeds, S]
    w = valid[..., None].float()
    mean_action = (target * w).sum((0, 1)) / w.sum((0, 1)).clamp_min(1)
    base = masked_error(mean_action.expand_as(target), target, valid)  # [S]
    reduction = 1.0 - float(err.mean() / base.mean())
    distinct = float(np.mean([pairwise(p, valid) for p in preds]) / max(1e-12, pairwise(target, valid)))

    blank = dict(batch)
    blank["demo.feats"] = torch.zeros_like(batch["demo.feats"])
    shuffled = dict(batch)
    perm = torch.randperm(batch["demo.feats"].shape[1], generator=torch.Generator().manual_seed(0)).cuda()
    shuffled["demo.feats"] = batch["demo.feats"][:, perm]
    e_blank = float(masked_error(sample(policy, blank, 0, s, h, dim)[..., :a], target, valid).mean())
    e_shuffle = float(masked_error(sample(policy, shuffled, 0, s, h, dim)[..., :a], target, valid).mean())

    finite = all(bool(torch.isfinite(p).all()) for p in preds)
    result = {
        "episode": ds.cfg.episode_whitelist,
        "segments": item["seg.start"].tolist(),
        "seeds": args.seeds,
        "error_per_seed": err.mean(1).tolist(),
        "error_mean": float(err.mean()),
        "error_seed_std": float(err.mean(1).std()),
        "baseline_mean_action_error": float(base.mean()),
        "reduction_vs_mean_action": reduction,
        "distinctness_ratio": distinct,
        "finite_and_shape_ok": bool(finite and preds[0].shape == target.shape),
        "recorded_not_gating": {"blank_video_error": e_blank, "slot_shuffle_error": e_shuffle},
        "thresholds": {"reduction": 0.8, "distinctness": 0.5},
    }
    result["pass"] = bool(reduction >= 0.8 and distinct >= 0.5 and result["finite_and_shape_ok"])
    result["reproducibility"] = reproducibility(cfg, args.ckpt, policy, ds)
    path = write_result(args.out, args.name, result)
    print(path)
    print({k: result[k] for k in ("error_mean", "baseline_mean_action_error", "reduction_vs_mean_action",
                                  "distinctness_ratio", "pass")})


if __name__ == "__main__":
    main()
