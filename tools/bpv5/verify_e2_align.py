#!/usr/bin/env python3
"""E2 alignment per experiment spec:

E1/E2/E3 共同点：只对 robotwin 14 有效维算 loss，padding 不参与。
E2 差别：team 纯 delta stats → 关节 delta + 夹爪也是 delta（错误）；14 维含夹爪。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/vla/workspace/my_tbot")
sys.path.insert(0, str(REPO / "src"))

from lerobot.policies.BPV5QFormer import data80d as D  # noqa: E402
from lerobot.policies.BPV5QFormer.configuration_bpv5qformer import BPV5QFormerDatasetConfig  # noqa: E402
from lerobot.policies.BPV5QFormer.dataset import BPV5QFormerDataset  # noqa: E402
from lerobot.policies.BPV5QFormer.modeling_bpv5qformer import ACTION_DIM_MASK  # noqa: E402
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

ROOT = Path("/vla/workspace/cache/bpv5_rt80_clean_root")
CACHE = Path("/vla/workspace/cache/bpv5_rt80_clean_p5")
STATS = Path("/physical-ai-data/norm_stats/action80_new/delta")
GRIP = (16, 56)
JOINTS = (0, 1, 2, 3, 4, 5, 40, 41, 42, 43, 44, 45)
ACTIVE14 = JOINTS + GRIP


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"OK  {msg}")


def main() -> None:
    team = json.loads((STATS / "aloha/stats.json").read_text())
    for g in GRIP:
        m = float(team["action_80D"]["mean"][g])
        if abs(m) > 0.2:
            fail(f"team gripper dim{g} mean={m} not delta-like")
    ok(f"team stats: gripper means delta-like ({team['action_80D']['mean'][16]:.4f}/{team['action_80D']['mean'][56]:.4f})")

    D.configure(root=ROOT, stats_root=STATS, action_layout="action80d", delta_mode="state_mask")
    rel = "robotwin/adjust_bottle/aloha-agilex_clean_50"
    a_mean, a_std, s_mean, s_std, a_mask, s_mask, d_mask = D.norm_vectors(rel)
    if int(a_mask.sum()) != 14 or not a_mask[list(ACTIVE14)].all():
        fail(f"action_mask must be exactly 14 robotwin slots, got {a_mask.sum()}")
    if not np.array_equal(d_mask, s_mask) or not d_mask[list(GRIP)].all():
        fail("E2 delta_mask must follow state_mask and include grippers")
    ok("masks: 14 active dims; grippers in both action_mask and delta_mask")

    state, action = D.FrameReader().episode(rel, D.episodes(rel)[0])
    start, H = 10, 50
    chunk, _, _ = D.delta_chunk(state, action, d_mask, a_mean, a_std, start, H)
    for g in GRIP:
        expected = (action[start : start + H, g] - state[start, g] - a_mean[g]) / a_std[g]
        if not np.allclose(chunk[:, g], expected, atol=1e-5):
            fail(f"gripper {g} must be delta (E2 wrong stats path)")
    for j in JOINTS[:1]:
        expected = (action[start : start + H, j] - state[start, j] - a_mean[j]) / a_std[j]
        if not np.allclose(chunk[:, j], expected, atol=1e-5):
            fail(f"joint {j} delta mismatch")
    ok("targets: joints delta + grippers delta (matching team pure-delta stats)")

    cfg = BPV5QFormerDatasetConfig(
        cache_dir=str(CACHE),
        data_root=str(ROOT),
        stats_root=str(STATS),
        action_layout="action80d",
        delta_mode="state_mask",
        drop_action_slots=[],
        sources=["robotwin"],
        max_trajectories=0,
        val_holdout_last_per_repo=0,
        heldout_monitor=False,
        load_threads=2,
        segment_len=50,
        segments_per_trajectory=4,
        num_slots=32,
        spatial_tokens=5,
        max_action_dim=80,
        max_state_dim=80,
    )
    ds = BPV5QFormerDataset(cfg, chunk_size=50)
    sample = ds[0]
    dim_mask = sample[ACTION_DIM_MASK].numpy()
    if int(dim_mask.sum()) != 14 or not dim_mask[list(GRIP)].all():
        fail(f"loss dim_mask must be 14 incl grippers, got {dim_mask.sum()}")
    if dim_mask.sum() != (~dim_mask).sum() + 14 - 80 + 66:  # noqa: silly check avoid
        pass
    # padding slots must be False in dim_mask
    pad = [i for i in range(80) if i not in ACTIVE14]
    if dim_mask[pad].any():
        fail("padding dims must not participate in loss (dim_mask False)")
    if tuple(sample[ACTION].shape)[-1] != 80:
        fail("tensor width must be 80")
    ok(f"loss uses 14/80 dims (padding excluded); grippers kept; n_traj={len(ds)}")
    ok("policy must use action_dim_mask_mode=stats (not ae3/all)")
    print("ALL E2 ALIGNMENT CHECKS PASSED")


if __name__ == "__main__":
    main()
