"""B-1 formal overfit: single pair (1000 steps) + contrast pair with swap follow (2000 steps).

Pass criteria (`bpv4cdx Stage B 开发文档 v1.md` §5.2):
  - single: reduction vs mean-action baseline ≥ 80%; distinctness ≥ 50% of GT pairwise distance
  - contrast: both queries meet the same fit bar; at t < t_div, swap-demo follow rate ≥ 0.9

usage: source tools/bpv4cdxb/env.local.sh; CUDA_VISIBLE_DEVICES=2 python tools/bpv4cdxb/overfit_b1.py
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.BPV4CDX.common import collate, masked_error, to_device
from lerobot.policies.BPV4CDXB.dataset import BPV4CDXBDataset
from lerobot.policies.BPV4CDXB.modeling_bpv4cdxb import BPV4CDXBPolicy
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION
from lerobot.utils.jsonc import load_jsonc

OUT = ROOT / "outputs/BPv4CDX/stage_b/b1_overfit"


def pick_contrast_pairs(index: dict) -> tuple[list[list[str]], str, int]:
    train_eps = [
        e
        for e, m in index["episodes"].items()
        if index["split"][m["anchor"]] == "train" and m["task"] == "place_shoe_bp"
    ]
    for anchor in sorted({index["episodes"][e]["anchor"] for e in train_eps}):
        by_set: dict[str, list[str]] = {}
        for e in train_eps:
            if index["episodes"][e]["anchor"] == anchor:
                by_set.setdefault(index["episodes"][e]["set"], []).append(e)
        sets = [s for s in index["contrast"].get(anchor, []) if by_set.get(s)]
        if len(sets) < 2:
            continue
        q1, q2 = by_set[sets[0]][0], by_set[sets[1]][0]
        pairs = [[q1, index["matched"][q1][0]["demo"]], [q2, index["matched"][q2][0]["demo"]]]
        t_div = int(index["divergence"][anchor]["t_div"])
        return pairs, anchor, t_div
    raise RuntimeError("no place_shoe_bp train contrast anchor")


def make_cfg(fixed: list[list[str]], *, steps: int, queries: int, name: str) -> TrainPipelineConfig:
    frozen = Path(os.environ["BPV4CDXB_FROZEN"])
    ztag = hashlib.sha256((frozen / "StageA_AE_frozen.ckpt").read_bytes()).hexdigest()[:12]
    cfg_dict = load_jsonc(ROOT / "configs/Pro6k/bpv4cdxb_plan/b_main.jsonc")
    cfg_dict["dataset"].update(
        {
            "fixed_pairs": fixed,
            "all_frames": False,
            "queries_per_item": queries,
            "contrast_prob": 0.0,
            "zdemo_dir": f"{os.environ['BPV4CDXB_ZDEMO']}/{ztag}",
            "index": os.environ["BPV4CDXB_INDEX"],
            "data_root": os.environ["BPV4CDXB_DATA_ROOT"],
            "qwen3_5_processor_path": os.environ["BPV4CDXB_QWEN_PATH"],
        }
    )
    cfg_dict["policy"].update(
        {
            "frozen_encoder": str(frozen),
            "qwen3_5_pretrained_path": os.environ["BPV4CDXB_QWEN_PATH"],
            "preflight_memory_check": False,
        }
    )
    cfg_dict.update(
        {
            "batch_size": max(1, len(fixed)),
            "num_workers": 0,
            "steps": steps,
            "save_checkpoint": False,
            "log_freq": 100,
            "output_dir": str(OUT / name),
            "seed": 42,
        }
    )
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg_dict, f)
        path = f.name
    return TrainPipelineConfig.from_pretrained(path)


def train(policy: BPV4CDXBPolicy, ds: BPV4CDXBDataset, steps: int, lr: float = 1e-4) -> list[float]:
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr)
    losses = []
    for step in range(steps):
        batch = to_device(collate([ds[i] for i in range(len(ds))]))
        loss, logs = policy(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step % 100 == 0 or step == steps - 1:
            print(f"  step {step}: loss={logs['loss_action']:.4f}")
    return losses


@torch.no_grad()
def predict(policy: BPV4CDXBPolicy, batch: dict, seed: int = 0) -> torch.Tensor:
    n = batch[ACTION].shape[0] * batch[ACTION].shape[1]
    h, dim = int(policy.config.chunk_size), int(policy.config.max_action_dim)
    g = torch.Generator(device="cuda").manual_seed(seed)
    noise = torch.randn((n, h, dim), generator=g, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return policy.predict_action_chunk(batch, noise=noise).float()


def pairwise(x: torch.Tensor, valid: torch.Tensor) -> float:
    d = []
    for i, j in itertools.combinations(range(x.shape[0]), 2):
        m = (valid[i] & valid[j])[:, None].float()
        d.append(float(((x[i] - x[j]) ** 2 * m).sum() / (m.sum() * x.shape[-1]).clamp_min(1)))
    return float(np.mean(d)) if d else 0.0


def eval_fit(policy: BPV4CDXBPolicy, ds: BPV4CDXBDataset, *, frames: list[int], seeds: int = 3) -> dict:
    """Evaluate one or more fixed pairs on the given query frames (same frames for every pair)."""
    items = []
    for i in range(len(ds)):
        q, d = ds.items[i]
        items.append(ds.build(q, d, frames))
    batch = to_device(collate(items))
    target = batch[ACTION]  # [B, S, H, A]
    valid = batch[ACTION_VALID_MASK].bool()
    b, s, h, a = target.shape
    preds = [predict(policy, batch, seed).reshape(b, s, h, -1)[..., :a] for seed in range(seeds)]
    err = torch.stack(
        [masked_error(p.reshape(b * s, h, a), target.reshape(b * s, h, a), valid.reshape(b * s, h)) for p in preds]
    )
    w = valid.reshape(b * s, h, 1).float()
    tgt = target.reshape(b * s, h, a)
    mean_action = (tgt * w).sum((0, 1)) / w.sum((0, 1)).clamp_min(1)
    base = masked_error(mean_action.expand_as(tgt), tgt, valid.reshape(b * s, h))
    reduction = 1.0 - float(err.mean() / base.mean().clamp_min(1e-8))
    # distinctness over time steps of the first pair
    distinct = float(
        np.mean([pairwise(p[0], valid[0]) for p in preds]) / max(1e-12, pairwise(target[0], valid[0]))
    )
    return {
        "error_mean": float(err.mean()),
        "baseline_mean_action_error": float(base.mean()),
        "reduction_vs_mean_action": reduction,
        "distinctness_ratio": distinct,
        "pass_fit": bool(reduction >= 0.8 and distinct >= 0.5),
        "per_pair_error": [
            float(
                masked_error(
                    preds[0][i].reshape(s, h, a), target[i], valid[i]
                ).mean()
            )
            for i in range(b)
        ],
    }


def eval_follow(
    policy: BPV4CDXBPolicy,
    ds: BPV4CDXBDataset,
    *,
    frames: list[int],
    seed: int = 0,
) -> dict:
    """Swap demos on the two contrast queries; follow if swapped pred is closer to the other GT."""
    assert len(ds) == 2
    (q1, d1), (q2, d2) = ds.items
    b11 = to_device(collate([ds.build(q1, d1, frames)]))
    b12 = to_device(collate([ds.build(q1, d2, frames)]))
    b21 = to_device(collate([ds.build(q2, d1, frames)]))
    b22 = to_device(collate([ds.build(q2, d2, frames)]))
    gt1, v1 = b11[ACTION][0], b11[ACTION_VALID_MASK][0].bool()
    gt2, v2 = b22[ACTION][0], b22[ACTION_VALID_MASK][0].bool()
    s, h, a = gt1.shape
    p11 = predict(policy, b11, seed).reshape(s, h, -1)[..., :a]
    p12 = predict(policy, b12, seed).reshape(s, h, -1)[..., :a]
    p21 = predict(policy, b21, seed).reshape(s, h, -1)[..., :a]
    p22 = predict(policy, b22, seed).reshape(s, h, -1)[..., :a]

    def closer_to_other(pred, own_gt, other_gt, valid):
        e_own = masked_error(pred, own_gt, valid)
        e_other = masked_error(pred, other_gt, valid)
        return e_other < e_own

    # for q1 with d2 (swapped): should move toward gt2; for q2 with d1 toward gt1
    # align by frame index; use intersection of valid steps
    follow_q1 = closer_to_other(p12, gt1, gt2, v1 & v2)
    follow_q2 = closer_to_other(p21, gt2, gt1, v1 & v2)
    rate = float(torch.cat([follow_q1, follow_q2]).float().mean())
    return {
        "follow_rate": rate,
        "follow_q1_frac": float(follow_q1.float().mean()),
        "follow_q2_frac": float(follow_q2.float().mean()),
        "pass_follow": bool(rate >= 0.9),
        "matched_err": {
            "q1": float(masked_error(p11, gt1, v1).mean()),
            "q2": float(masked_error(p22, gt2, v2).mean()),
        },
        "swapped_err_to_own": {
            "q1": float(masked_error(p12, gt1, v1).mean()),
            "q2": float(masked_error(p21, gt2, v2).mean()),
        },
        "swapped_err_to_other": {
            "q1": float(masked_error(p12, gt2, v1 & v2).mean()),
            "q2": float(masked_error(p21, gt1, v1 & v2).mean()),
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = json.loads(Path(os.environ["BPV4CDXB_INDEX"]).read_text())
    pairs, anchor, t_div = pick_contrast_pairs(index)
    print("anchor", anchor, "t_div", t_div, "pairs", pairs)

    # ---- single pair ----
    print("=== B-1 single pair ===")
    cfg1 = make_cfg([pairs[0]], steps=1000, queries=8, name="single")
    ds1 = BPV4CDXBDataset(cfg1.dataset, cfg1, cfg1.policy)
    policy = BPV4CDXBPolicy(cfg1.policy).cuda()
    train(policy, ds1, 1000)
    q0 = pairs[0][0]
    length = int(index["episodes"][q0]["length"])
    eval_frames = [int(x) for x in np.linspace(0, max(0, length - 2), 16).round()]
    single = eval_fit(policy, ds1, frames=eval_frames)
    print("single", {k: single[k] for k in ("reduction_vs_mean_action", "distinctness_ratio", "pass_fit")})

    # ---- contrast ----
    print("=== B-1 contrast ===")
    cfg2 = make_cfg(pairs, steps=2000, queries=8, name="contrast")
    ds2 = BPV4CDXBDataset(cfg2.dataset, cfg2, cfg2.policy)
    # fresh Action Expert for contrast (doc: both experiments are interface checks)
    policy2 = BPV4CDXBPolicy(cfg2.policy).cuda()
    train(policy2, ds2, 2000)
    # eval frames: dense before t_div + a few after
    pre = list(range(0, max(1, t_div), max(1, t_div // 12)))[:12]
    if not pre:
        pre = [0]
    post = [int(x) for x in np.linspace(t_div, max(t_div, min(length, index["episodes"][pairs[1][0]]["length"]) - 2), 4).round()]
    contrast_fit = eval_fit(policy2, ds2, frames=pre + post)
    follow = eval_follow(policy2, ds2, frames=pre)
    print("contrast fit", {k: contrast_fit[k] for k in ("reduction_vs_mean_action", "distinctness_ratio", "pass_fit", "per_pair_error")})
    print("follow", {k: follow[k] for k in ("follow_rate", "pass_follow", "matched_err", "swapped_err_to_other")})

    result = {
        "anchor": anchor,
        "t_div": t_div,
        "pairs": pairs,
        "single": single,
        "contrast_fit": contrast_fit,
        "follow": follow,
        "pass": bool(single["pass_fit"] and contrast_fit["pass_fit"] and follow["pass_follow"]),
        "thresholds": {"reduction": 0.8, "distinctness": 0.5, "follow_rate": 0.9},
    }
    out = OUT / "b1_overfit_result.json"
    out.write_text(json.dumps(result, indent=2))
    print("wrote", out)
    print("B-1 overfit", "PASS" if result["pass"] else "FAIL")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
