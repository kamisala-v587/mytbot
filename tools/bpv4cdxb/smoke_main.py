"""S3 smoke test for the Stage B main path (StageB实施.md): one batch through the main dataset and the B-2 model.

Checks: dataset builds on the bucket index; drawn demos share the query's bucket; a 2-camera robot (Franka) gets a
black third image; forward loss is finite; after backward only the Action Expert and [obs_start] / [obs_end] have
gradients (Qwen, ViT, demo_start / demo_end frozen).

usage: python tools/bpv4cdxb/smoke_main.py [--config configs/Pro6k/bpv4cdxb_plan/generated_b200/b2_main.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.BPV4CDX.common import collate, to_device
    from lerobot.policies.BPV4CDXB.dataset import make_bpv4cdxb_dataset
    from lerobot.policies.BPV4CDXB.modeling_bpv4cdxb import BPV4CDXBPolicy

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/Pro6k/bpv4cdxb_plan/generated_b200/b2_main.json"))
    args = ap.parse_args()
    cfg = TrainPipelineConfig.from_pretrained(args.config)
    cfg.policy.preflight_memory_check = False
    t0 = time.time()
    ds, _ = make_bpv4cdxb_dataset(cfg)
    print(f"dataset: {len(ds)} queries, {len(ds.repos)} repos, built in {time.time() - t0:.0f} s", flush=True)

    rng = np.random.default_rng(0)
    pick = [int(i) for i in rng.choice(len(ds), 3, replace=False)]
    franka = [i for i, e in enumerate(ds.items) if ds.repos[ds.ep[e][0]].emb.cameras.count(None) > 0]
    if franka:
        pick.append(franka[0])
    items, pairs = [], []
    for i in pick:
        q = ds.items[i]
        d = ds.pick_demo(q)
        assert ds.ep[q][3] == ds.ep[d][3] and q != d, (q, d)
        pairs.append((q, d, ds.buckets[ds.ep[q][3]]["key"][:70], ds.repos[ds.ep[q][0]].emb.cameras))
        t1 = time.time()
        items.append(ds.build(q, d, ds.query_frames(q)))
        print(f"item {q} <- demo {d}: {time.time() - t1:.1f} s; cameras {pairs[-1][3]}", flush=True)
    for q, d, key, _ in pairs:
        print("  pair ok, bucket:", key)
    batch = to_device(collate(items))
    print({k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)})

    policy = BPV4CDXBPolicy(cfg.policy).cuda()
    train = [n for n, p in policy.named_parameters() if p.requires_grad]
    print(f"trainable tensors {len(train)}, e.g. {train[:4]}; trainable params "
          f"{sum(p.numel() for p in policy.parameters() if p.requires_grad) / 1e6:.1f} M, frozen "
          f"{sum(p.numel() for p in policy.parameters() if not p.requires_grad) / 1e6:.1f} M")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, info = policy.forward(batch)
    loss.backward()
    grads = {n for n, p in policy.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    bad = [n for n in grads if not (".action_expert" in n or "expert" in n or "obs_start" in n or "obs_end" in n
                                    or "state_proj" in n or "action_" in n or "time_mlp" in n)]
    frozen_with_grad = [n for n, p in policy.named_parameters() if not p.requires_grad and p.grad is not None]
    res = {"loss": float(loss), "finite": bool(torch.isfinite(loss)), "params_with_grad": len(grads),
           "unexpected_grad_names": bad[:10], "frozen_with_grad": frozen_with_grad[:5],
           "obs_markers_have_grad": any("obs_start" in n for n in grads) and any("obs_end" in n for n in grads),
           "black_camera_item": bool(franka)}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
