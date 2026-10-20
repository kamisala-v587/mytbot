"""Is the grasp point in the BP tokens read from the scene or from the arm pose at visible keyframes?

Visibility conditions (over the cached keyframes of each trajectory, (s, e] = event segment ending at the grasp):
  all     every keyframe
  kf0     only the first keyframe (home pose, scene only)
  event   hide (s, e] -- what the event mask does when the query is in the approach phase
  event2  hide (s, e] and the following carry segment (e, e2] -- the grasp arm pose is no longer visible
Per condition:
  bp      BP tokens (resampler output) -> grasp point
  arm     grasp-arm end-effector position at the nearest visible keyframes before / after the grasp -> grasp point
  bp|arm  BP tokens -> what the arm feature cannot explain (grasp point minus the arm-feature prediction)
  bp->ee  BP tokens -> grasp-arm end-effector position at the first visible keyframe after the grasp
Targets are residuals after removing the per (task, grasp arm) train mean, as in probe_eval.

Usage: python -m lerobot.policies.BPV4BPEn.probe_shortcut --labels labels.json --out out.json \
         --ckpt main10k=<pretrained_model> --ckpt nomask10k=<...>
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.BPV4BPEn.build_feature_cache import cache_path
from lerobot.policies.BPV4BPEn.keyframes import event_segment, plan_keyframes
from lerobot.policies.BPV4BPEn.modeling_bpv4_bpen import sinusoidal_encoding
from lerobot.policies.BPV4BPEn.probe_eval import episodes, gram, load_modules, ridge_r2

URDF = "/vla/workspace/my_robotwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
ARMS = {"L": ([f"fl_joint{i}" for i in range(1, 7)], "fl_link6", slice(0, 6)),
        "R": ([f"fr_joint{i}" for i in range(1, 7)], "fr_link6", slice(7, 13))}
CONDS = ("all", "kf0", "event", "event2")
K = 32


def _repo_ee(args):
    """Per episode of one repo: planned keyframes and the grasp-arm end-effector position at each of them."""
    repo, arms, num_kf, alpha = args
    sys.path.append("/vla/.conda/miniconda3/envs/myrobotwin/lib/python3.10/site-packages")
    import yourdfpy

    robot = yourdfpy.URDF.load(URDF, load_meshes=False, build_scene_graph=True)
    df = pd.concat(pd.read_parquet(f, columns=["episode_index", "observation.state"])
                   for f in sorted(Path(repo, "data").rglob("*.parquet")))
    out = {}
    for ep, g in df.groupby("episode_index"):
        arm = arms.get(int(ep))
        if arm is None:
            continue
        s = np.stack(g["observation.state"].to_numpy()).astype(np.float64)
        frames = plan_keyframes(s, num_kf, alpha).frames
        joints, link, sl = ARMS[arm]
        ee = []
        for f in frames:
            robot.update_cfg(dict(zip(joints, s[f, sl])))
            ee.append(robot.get_transform(link, robot.base_link)[:3, 3])
        out[int(ep)] = (frames, np.array(ee))
    return repo, out


class Trajectories(torch.utils.data.Dataset):
    def __init__(self, items, cache_dir, grasp_frames):
        self.items, self.cache_dir, self.grasp_frames = items, cache_dir, grasp_frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        repo, ep, _ = self.items[i]
        rec = torch.load(cache_path(self.cache_dir, repo, ep), map_location="cpu", weights_only=False)
        k_eff = rec["feats"].shape[0]
        feats = torch.zeros((K, *rec["feats"].shape[1:]), dtype=rec["feats"].dtype)
        feats[:k_eff] = rec["feats"]
        time = torch.zeros(K)
        time[:k_eff] = rec["time_ratio"]
        frames = np.full(K, -10**9, dtype=np.int64)
        frames[:k_eff] = rec["frames"].numpy()
        valid = np.arange(K) < k_eff
        flips, active = rec["flips"].numpy(), tuple(int(v) for v in rec["active"])
        gf = self.grasp_frames[i]
        s, e = event_segment(flips, active, gf)
        _, e2 = event_segment(flips, active, e + 1)
        vis = {
            "all": valid,
            "kf0": np.arange(K) == 0,
            "event": valid & ~((frames > s) & (frames <= e)),
            "event2": valid & ~((frames > s) & (frames <= max(e, e2))),
        }
        return {"feats": feats, "time": time, "frames": torch.from_numpy(frames),
                "vis": torch.from_numpy(np.stack([vis[c] for c in CONDS])), "length": int(rec["length"]), "index": i}


def arm_feature(frames, ee, vis, gf, length):
    """[before EE xyz, before gap, has_before, after EE xyz, after gap, has_after] from visible keyframes."""
    out = np.zeros(10)
    idx = np.flatnonzero(vis)
    before, after = idx[frames[idx] < gf], idx[frames[idx] >= gf]
    if len(before):
        j = before[-1]
        out[0:3], out[3], out[4] = ee[j], (gf - frames[j]) / length, 1.0
    if len(after):
        j = after[0]
        out[5:8], out[8], out[9] = ee[j], (frames[j] - gf) / length, 1.0
    return out


def ridge_small(x, y, tr, alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Low-dimensional ridge with intercept; alpha by 5-fold CV on train; returns predictions for all rows."""
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-8
    z = np.c_[(x - mu) / sd, np.ones(len(x))]

    def fit(rows, a):
        reg = a * len(rows) * np.eye(z.shape[1])
        reg[-1, -1] = 0
        return np.linalg.solve(z[rows].T @ z[rows] + reg, z[rows].T @ y[rows])

    tri = np.flatnonzero(tr)
    folds = np.array_split(np.random.default_rng(0).permutation(tri), 5)
    errs = []
    for a in alphas:
        err = 0.0
        for f in folds:
            rows = np.setdiff1d(tri, f)
            err += float(((z[f] @ fit(rows, a) - y[f]) ** 2).sum())
        errs.append(err)
    return z @ fit(tri, alphas[int(np.argmin(errs))])


def r2(pred, y):
    return [float(v) for v in 1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo_file", default="/vla/workspace/my_tbot/configs/ds_ids/pro6k/robo_bpv4_bpen_clean50_rand100.txt")
    ap.add_argument("--cache_dir", default="/vla/workspace/cache/bpv4_bpen_vit_base800m_k32_v2")
    ap.add_argument("--ckpt", action="append", default=[], help="name=path to pretrained_model")
    args = ap.parse_args()

    labels = json.loads(Path(args.labels).read_text())
    items = [it for it in episodes(args.repo_file) if (labels.get(f"{it[0]}|{it[1]}") or {}).get("grasp")]
    lab = [labels[f"{r}|{e}"] for r, e, _ in items]

    arms_by_repo: dict[str, dict[int, str]] = {}
    for (r, e, _), rec in zip(items, lab):
        arms_by_repo.setdefault(r, {})[e] = rec["arm"]
    with get_context("spawn").Pool(32) as pool:
        ee_by_repo = dict(pool.map(_repo_ee, [(r, a, K, 0.5) for r, a in arms_by_repo.items()]))
    print("FK done", flush=True)

    ckpts = dict(c.split("=", 1) for c in args.ckpt)
    cfg = PreTrainedConfig.from_pretrained(next(iter(ckpts.values())))
    models = {"random": load_modules(None, cfg), **{n: load_modules(p, cfg) for n, p in ckpts.items()}}
    grasp_frames = [int(rec["grasp_frame"]) for rec in lab]
    loader = torch.utils.data.DataLoader(Trajectories(items, args.cache_dir, grasp_frames), batch_size=32, num_workers=16)

    n = len(items)
    bp = {(m, c, kind): [] for m in models for c in CONDS for kind in ("mean", "tok")}
    arm_x = {c: np.zeros((n, 10)) for c in CONDS}
    vis_count = {c: np.zeros(n) for c in CONDS}
    frame_mismatch = 0
    with torch.no_grad():
        for batch in loader:
            x, t = batch["feats"].cuda(), batch["time"].cuda()
            vis = batch["vis"].cuda()  # [B, C, K]
            b = x.shape[0]
            for name, (enc, res) in models.items():
                content = enc(x) + sinusoidal_encoding(t, enc.output_proj[-1].out_features)[:, :, None]  # [B, K, Q, D]
                q = content.shape[2]
                tokens = content.reshape(b, K * q, -1)
                for ci, c in enumerate(CONDS):
                    key_vis = vis[:, ci, :, None].expand(-1, -1, q).reshape(b, K * q)
                    r = res(tokens, key_vis)
                    bp[(name, c, "mean")].append(r.mean(1).cpu())
                    bp[(name, c, "tok")].append(r.reshape(b, -1).half().cpu())
            for j, i in enumerate(batch["index"].tolist()):
                repo, ep, _ = items[i]
                planned, ee = ee_by_repo[repo][ep]
                frames = batch["frames"][j].numpy()
                k_eff = int((frames > -10**9).sum())
                if not np.array_equal(planned, frames[:k_eff]):
                    frame_mismatch += 1
                    continue
                ee_pad = np.zeros((K, 3))
                ee_pad[:k_eff] = ee
                length = max(1, int(batch["length"][j]))
                for ci, c in enumerate(CONDS):
                    vis_count[c][i] = int(batch["vis"][j, ci].sum())
                    arm_x[c][i] = arm_feature(frames, ee_pad, batch["vis"][j, ci].numpy(), grasp_frames[i], length)
    bp = {k: torch.cat(v).numpy() for k, v in bp.items()}
    print("features done, frame mismatches:", frame_mismatch, flush=True)

    task = np.array([Path(r).parent.name for r, _, _ in items])
    is_val = np.array([v for _, _, v in items])
    group = np.array([f"{tk}|{rec['arm']}" for tk, rec in zip(task, lab)])
    y = np.array([rec["grasp"] for rec in lab])

    def residualize(v, ok):
        means = {g: v[ok & ~is_val & (group == g)].mean(0) for g in np.unique(group[ok & ~is_val])}
        keep = ok & np.array([g in means for g in group])
        return v - np.stack([means.get(g, np.zeros(v.shape[1])) for g in group]), keep

    resid, ok = residualize(y, np.array([v > 0 for v in vis_count["all"]]))
    result = {"n_train": int((ok & ~is_val).sum()), "n_val": int((ok & is_val).sum()), "frame_mismatch": frame_mismatch,
              "within_group_std_m": [float(v) for v in resid[ok & is_val].std(0)], "conds": {}}
    for c in CONDS:
        cr = result["conds"][c] = {}
        ax = arm_x[c].copy()
        has_after = ax[:, 9] > 0
        for lo, hi, flag in ((0, 3, 4), (5, 8, 9)):  # same (task, arm) mean removal as the target
            has = ax[:, flag] > 0
            if (has & ~is_val).any():
                res_ee, keep_ee = residualize(ax[:, lo:hi], ok & has)
                ax[:, lo:hi] = np.where(keep_ee[:, None], res_ee, 0.0)
                ax[:, flag] = keep_ee
        cr["visible_kf_mean"] = float(vis_count[c][ok].mean())
        cr["frac_with_visible_after_grasp"] = float(has_after[ok].mean())
        tr, va = ok & ~is_val, ok & is_val
        arm_pred = ridge_small(ax, resid, tr)
        cr["arm"] = r2(arm_pred[va], resid[va])
        cr["arm_pred"] = {"val_index": np.flatnonzero(va).tolist(), "pred": arm_pred[va].round(5).tolist(),
                          "true": resid[va].round(5).tolist()}
        beyond = resid - arm_pred
        ee_after, ok_ee = residualize(arm_x[c][:, 5:8], ok & has_after)
        for m in models:
            for kind in ("mean", "tok"):
                g = gram(bp[(m, c, kind)], ~is_val)
                key = f"{m}.{kind}"
                cr[f"bp:{key}"] = ridge_r2(g, tr, va, resid)
                cr[f"bp|arm:{key}"] = ridge_r2(g, tr, va, beyond)
                cr[f"bp->ee:{key}"] = ridge_r2(g, ok_ee & ~is_val, ok_ee & is_val, ee_after)
                del g
                print(c, key, {k: v["r2_mean"] for k, v in cr.items() if k.endswith(key)}, flush=True)
        print(c, "arm", cr["arm"], flush=True)
    result["items"] = [f"{r}|{e}" for r, e, _ in items]
    Path(args.out).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
