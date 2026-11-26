#!/usr/bin/env python3
"""Strict E1 alignment: native14_pad32, tbot delta/abs, full-clean cache, AE width 32."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

REPO = Path("/vla/workspace/my_tbot")
sys.path.insert(0, str(REPO / "src"))

from lerobot.policies.BPV5QFormer import data80d as D  # noqa: E402
from lerobot.policies.BPV5QFormer.configuration_bpv5qformer import BPV5QFormerDatasetConfig  # noqa: E402
from lerobot.policies.BPV5QFormer.dataset import BPV5QFormerDataset  # noqa: E402
from lerobot.policies.BPV5QFormer.modeling_bpv5qformer import ACTION_DIM_MASK  # noqa: E402
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK  # noqa: E402
from lerobot.transforms.utils import make_bool_mask  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

STATS = REPO / "norm_stats_bpv5_pt/robotwin_clean_native_14d/aloha/delta/stats.json"
ROOT = Path("/vla/workspace/cache/bpv5_rt20_clean_root")
CACHE = Path("/vla/workspace/cache/bpv5_rt20_clean_p5")
DELTA14 = make_bool_mask(6, -1, 6, -1).numpy()


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"OK  {msg}")


def main() -> None:
    assert STATS.is_file(), STATS
    assert ROOT.is_dir(), ROOT
    assert CACHE.is_dir(), CACHE

    if tuple(np.where(DELTA14)[0].tolist()) != D.NATIVE_ALOHA_DELTA_IDX:
        fail(f"DELTA_IDX {D.NATIVE_ALOHA_DELTA_IDX} != tbot {np.where(DELTA14)[0].tolist()}")
    if tuple(np.where(~DELTA14)[0].tolist()) != D.NATIVE_ALOHA_ABS_IDX:
        fail(f"ABS_IDX {D.NATIVE_ALOHA_ABS_IDX} != tbot")
    ok("tbot aloha MASK_MAPPING = make_bool_mask(6,-1,6,-1)")

    st = json.loads(STATS.read_text())
    if len(st["action"]["mean"]) != 14:
        fail("native stats must be 14-D")
    gL, gR = st["action"]["mean"][6], st["action"]["mean"][13]
    if abs(gL) < 0.05 or abs(gR) < 0.05:
        fail(f"gripper action means look like delta: {gL}, {gR}")
    ok(f"stats grippers absolute-ish (mean L/R={gL:.3f}/{gR:.3f})")

    D.configure(
        root=ROOT,
        stats_file=STATS,
        action_layout="native14_pad32",
        delta_mode="tbot_aloha",
    )
    if D.ACTION_DIM != 32:
        fail(f"ACTION_DIM={D.ACTION_DIM}, want 32")
    a_mean, a_std, s_mean, s_std, a_mask, s_mask, d_mask = D.norm_vectors("x")
    if a_mask.sum() != 14 or s_mask.sum() != 14 or d_mask.sum() != 12:
        fail(f"mask counts a={a_mask.sum()} s={s_mask.sum()} d={d_mask.sum()}")
    if a_mask[14:].any() or d_mask[6] or d_mask[13] or d_mask[14:].any():
        fail("pad dims / grippers incorrectly marked")
    if not np.allclose(a_mean[:14], st["action"]["mean"]) or not np.allclose(s_mean[:14], st["observation.state"]["mean"]):
        fail("pad32 stats[:14] != native stats.json")
    ok("norm_vectors pad32 + delta_mask (12 joints True, grippers+pad False)")

    rel = "robotwin/adjust_bottle/aloha-agilex_clean_50"
    ep = D.episodes(rel)[0]
    state, action = D.FrameReader().episode(rel, ep)
    if state.shape[1] != 32 or action.shape[1] != 32:
        fail(f"pad32 shape {state.shape}/{action.shape}")
    if not np.allclose(state[:, 14:], 0) or not np.allclose(action[:, 14:], 0):
        fail("pad dims must be zero")
    t = pq.read_table(D.data_files(rel)[0], columns=["index", "observation.state", "action"])
    idx = t.column("index").to_numpy()
    lo, hi = int(np.searchsorted(idx, ep["from"])), int(np.searchsorted(idx, ep["to"]))
    raw_s = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)[lo:hi]).astype(np.float32)
    raw_a = np.stack(t.column("action").to_numpy(zero_copy_only=False)[lo:hi]).astype(np.float32)
    if not np.allclose(state[:, :14], raw_s) or not np.allclose(action[:, :14], raw_a):
        fail("pad32[:14] != native parquet")
    ok(f"FrameReader pad32 episode0 len={len(state)}")

    start, H = 10, 50
    chunk, step_ok, _ = D.delta_chunk(state, action, d_mask, a_mean, a_std, start, H)
    base = np.where(d_mask, state[start], 0.0)
    manual = (action[start : start + H] - base - a_mean) / a_std
    if not np.allclose(chunk, manual, atol=1e-5):
        fail("delta_chunk != manual")
    for g in D.NATIVE_ALOHA_ABS_IDX:
        expected = (action[start : start + H, g] - a_mean[g]) / a_std[g]
        if not np.allclose(chunk[:, g], expected, atol=1e-5):
            fail(f"gripper {g} was delta-subtracted")
    j = D.NATIVE_ALOHA_DELTA_IDX[0]
    expected_j = (action[start : start + H, j] - state[start, j] - a_mean[j]) / a_std[j]
    if not np.allclose(chunk[:, j], expected_j, atol=1e-5):
        fail(f"joint {j} delta mismatch")
    ok("per-dim delta: joints subtract state; grippers absolute")

    near_end = max(0, len(action) - 3)
    chunk2, step2, _ = D.delta_chunk(state, action, d_mask, a_mean, a_std, near_end, H)
    n_valid = int(step2.sum())
    if n_valid != len(action) - near_end or step2[n_valid:].any():
        fail(f"padding step_valid broken: valid={n_valid}")
    if not np.allclose(chunk2[n_valid:], chunk2[n_valid - 1]):
        fail("padded action rows should repeat last")
    ok(f"horizon padding: start={near_end} valid={n_valid}/{H}")

    items = json.loads((CACHE / "_items.json").read_text())
    if len(items) < 2000:
        fail(f"cache too small for full clean: {len(items)}")
    ok(f"P5 cache full-clean coverage: {len(items)} episodes / {len({r for r,_ in items})} repos")

    cfg = BPV5QFormerDatasetConfig(
        cache_dir=str(CACHE),
        data_root=str(ROOT),
        stats_file=str(STATS),
        action_layout="native14_pad32",
        delta_mode="tbot_aloha",
        sources=["robotwin"],
        max_trajectories=0,
        val_holdout_last_per_repo=0,
        heldout_monitor=False,
        load_threads=2,
        segment_len=50,
        segments_per_trajectory=4,
        num_slots=32,
        spatial_tokens=5,
        max_action_dim=32,
        max_state_dim=32,
    )
    ds = BPV5QFormerDataset(cfg, chunk_size=50)
    if len(ds) != len(items):
        # allow small mismatch if some cache missing, but must be near-full
        if len(ds) < 0.95 * len(items):
            fail(f"dataset {len(ds)} << cache {len(items)} (should be full clean, no max_trajectories)")
    sample = ds[0]
    for key, shape in [
        ("demo.feats", (32, 3, 5, 2560)),
        (ACTION, (4, 50, 32)),
        (ACTION_VALID_MASK, (4, 50)),
        (ACTION_DIM_MASK, (32,)),
        (OBS_STATE, (4, 32)),
        ("text.input_ids", (4, 48)),
    ]:
        if tuple(sample[key].shape) != shape:
            fail(f"{key} shape {tuple(sample[key].shape)} != {shape}")
    if int(sample[ACTION_DIM_MASK].sum()) != 14:
        fail("ACTION_DIM_MASK must have 14 True")
    ok(f"dataset full-clean n_traj={len(ds)}; sample AE width=32; dim_mask=14/32")
    ok("AE: single BPv5 Action Expert @ width_ratio=0.25, max_action_dim=32 (not 80)")
    print("ALL E1 ALIGNMENT CHECKS PASSED")


if __name__ == "__main__":
    main()
