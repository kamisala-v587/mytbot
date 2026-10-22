"""G7-D (not a gate, design doc 7.9): what does the frozen Z_demo keep?

Features per episode (all 32 slots): trained Z_demo (token mean), Z_demo of a randomly initialised encoder (lower
bound) and raw ViT features (upper reference: slot-0 head-camera mean token; mean over all slots / cameras).
Probes (episode-level 5-fold CV, reusing the bpv4 probe code):
  --data probe    bp_probe repos (--probe_repo_file, --labels): grasped-object id and placement-target id within
                  task, grasp arm (balanced accuracy); grasped-object initial / final xy (R^2 after removing the
                  (task, arm) mean); poses and ids from scene_info.json
  --data heldout  held-out episodes of the checkpoint's training repos (no object metadata): grasp arm, and the
                  joint configuration at the first gripper close as a target-position proxy (R^2 after removing the
                  (task, arm) mean)
Gradient saliency (--saliency N episodes): |d loss / d demo feature * feature| per slot, camera and spatial token for
every fixed target segment; reports how much of the slot saliency falls inside the segment's own time window
relative to the window's share of slots (> 1: the segment reads its own part of the demo).

Usage: python -m lerobot.policies.BPV4CDX.probe_zdemo --ckpt <pretrained_model> --data probe \
         --probe_repo_file <txt> --labels <labels_probe.json> --out <dir> --name <run> [--saliency 40]
       python -m lerobot.policies.BPV4CDX.probe_zdemo --ckpt <pretrained_model> --data heldout --out <dir> --name <run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.BPV4BPEn.probe_identity import classify, episode_meta, position_r2
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION

from .common import collate, load_cfg, make_dataset, make_policy, reproducibility, to_device, write_result
from .gate_g0 import ARM_JOINTS, first_close
from .paths import task_name


@torch.no_grad()
def features(policy, ds) -> dict[str, np.ndarray]:
    out = {"z": [], "vit_kf0_head": [], "vit_mean": []}
    for s in range(0, len(ds), 32):
        demo = [ds.demo(i) for i in range(s, min(len(ds), s + 32))]
        batch = to_device(collate(demo))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out["z"].append(policy.model.encode_demo(batch).float().mean(1).cpu())
        f = batch["demo.feats"].float()  # [B, 32, 3, P, D]
        out["vit_kf0_head"].append(f[:, 0, 0].mean(1).cpu())
        out["vit_mean"].append(f.mean((1, 2, 3)).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def demeaned(y: np.ndarray, group: np.ndarray) -> np.ndarray:
    y = y.astype(np.float64).copy()
    for g in np.unique(group):
        s = group == g
        y[s] -= y[s].mean(0)
    return y


def probe_data(ds, labels: dict) -> tuple[list[dict], list[int]]:
    metas, keep = [], []
    for i, (ri, ep) in enumerate(ds.items):
        repo = ds.repos[ri].repo
        lab = labels.get(f"{repo}|{ep}") or {}
        if not lab.get("grasp"):
            continue
        scene = json.loads(Path(repo, "meta", "scene_info.json").read_text())
        metas.append({**episode_meta(repo, ep, scene), "arm": lab["arm"]})
        keep.append(i)
    return metas, keep


def heldout_data(ds) -> tuple[list[dict], list[int]]:
    metas, keep = [], []
    for i, (ri, ep) in enumerate(ds.items):
        s = ds.repos[ri].states[ep]
        fc = first_close(s)
        if fc is None:
            continue
        metas.append({"task": task_name(ds.repos[ri].repo), "arm": fc[0], "config_at_close": s[fc[1], ARM_JOINTS[fc[0]]]})
        keep.append(i)
    return metas, keep


def saliency(policy, ds, n_episodes: int) -> dict:
    """Input-gradient saliency of the flow loss (fixed noise, flow time 0.5) w.r.t. the pooled demo features."""
    model = policy.model
    rng = np.random.default_rng(0)
    picks = rng.choice(len(ds), size=min(n_episodes, len(ds)), replace=False)
    ratios, cam_mass, sp_mass = [], [], []
    for i in picks:
        ri, ep = ds.items[int(i)]
        length = len(ds.repos[ri].states[ep])
        starts = list(range(0, length, ds.cfg.segment_len))
        demo = ds.demo(int(i))
        for k, st in enumerate(starts):
            batch = to_device(collate([{**demo, **ds.target(int(i), [st], generic=[False])}]))
            feats = batch["demo.feats"].float().requires_grad_(True)
            batch["demo.feats"] = feats
            a = batch[ACTION].shape
            noise = torch.randn((1, a[2], int(policy.config.max_action_dim)),
                                generator=torch.Generator().manual_seed(int(i) * 1000 + k)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                flow = model.forward_segments(batch, noise=noise, time=torch.full((1,), 0.5, device="cuda"))["flow"]
            valid = batch[ACTION_VALID_MASK].reshape(1, -1, 1).float()
            loss = (flow[..., : a[-1]] * valid).sum()
            (grad,) = torch.autograd.grad(loss, feats)
            sal = (grad * feats).abs().sum(-1)[0].detach().cpu().numpy()  # [32, 3, P]
            slot = sal.sum((1, 2))
            t = demo["demo.time"].numpy()
            tau = batch["seg.tau"][0, 0].cpu().numpy()
            inside = (t >= tau[0]) & (t <= tau[1])
            if inside.any() and slot.sum() > 0:
                ratios.append(float(slot[inside].sum() / slot.sum()) / float(inside.mean()))
            cam_mass.append(sal.sum((0, 2)) / max(1e-12, sal.sum()))
            sp_mass.append(sal.sum((0, 1)) / max(1e-12, sal.sum()))
    return {"episodes": int(len(picks)), "segments": int(len(cam_mass)),
            "own_window_mass_over_share": {"mean": float(np.mean(ratios)), "median": float(np.median(ratios)),
                                           "frac_gt_1": float(np.mean(np.array(ratios) > 1))} if ratios else None,
            "camera_mass": np.mean(cam_mass, 0).tolist(), "spatial_token_mass": np.mean(sp_mass, 0).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", choices=["probe", "heldout"], default="probe")
    ap.add_argument("--probe_repo_file", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--saliency", type=int, default=0, help="episodes for the gradient saliency diagnostic (0 = off)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    args, overrides = ap.parse_known_args()
    cfg = load_cfg(args.ckpt, overrides)
    if args.data == "probe":
        if not args.probe_repo_file or not args.labels:
            raise SystemExit("--data probe needs --probe_repo_file and --labels")
        ds = make_dataset(cfg, repo_id_file=args.probe_repo_file, data_root=None, split="all", segment_mode="fixed")
        metas, keep = probe_data(ds, json.loads(Path(args.labels).read_text()))
    else:
        ds = make_dataset(cfg, split="val", segment_mode="fixed")
        metas, keep = heldout_data(ds)
    trained = make_policy(cfg, args.ckpt)
    torch.manual_seed(0)
    random_init = make_policy(cfg, None)
    f_tr, f_rand = features(trained, ds), features(random_init, ds)
    feats = {"z_trained": f_tr["z"], "z_random": f_rand["z"], "vit_kf0_head": f_tr["vit_kf0_head"], "vit_mean": f_tr["vit_mean"]}
    feats = {k: v[keep] for k, v in feats.items()}

    task = np.array([m["task"] for m in metas])
    arm = np.array([m["arm"] for m in metas])
    group = np.array([f"{m['task']}|{m['arm']}" for m in metas])
    roles = ("grasp_id", "target_id", "arm") if args.data == "probe" else ("arm",)
    res = {"data": args.data, "n": len(metas), "tasks": sorted(set(task.tolist())), "identity": {}, "position": {}}
    for name, x in feats.items():
        per = {}
        for t in np.unique(task):
            sel = np.flatnonzero(task == t)
            for role in roles:
                y = arm[sel] if role == "arm" else np.array([str(metas[i][role]) for i in sel])
                if role != "arm" and all(metas[i][role] is None for i in sel):
                    continue
                r = classify(x[sel], y)
                if r:
                    per[f"{t}|{role}"] = r
        res["identity"][name] = per
    if args.data == "probe":
        ok = np.array([m["grasp_init"] is not None for m in metas])
        for key in ("grasp_init", "grasp_final"):
            y = np.zeros((len(metas), 2))
            y[ok] = np.array([m[key][:2] for m, o in zip(metas, ok) if o])
            y[ok] = demeaned(y[ok], group[ok])
            res["position"][key] = {name: position_r2(x[ok], y[ok]) for name, x in feats.items()}
    else:
        y = demeaned(np.stack([m["config_at_close"] for m in metas]), group)
        res["position"]["joint_config_at_first_close"] = {name: float(np.mean(position_r2(x, y))) for name, x in feats.items()}
    res["summary"] = {name: {role: float(np.mean([v["bacc"] for k, v in res["identity"][name].items() if k.endswith(role)]))
                             for role in roles if any(k.endswith(role) for k in res["identity"][name])}
                      for name in feats}
    if args.saliency:
        res["saliency"] = saliency(trained, ds, args.saliency)
    res["reproducibility"] = reproducibility(cfg, args.ckpt, trained, ds)
    path = write_result(args.out, args.name, res)
    print(path)
    print(json.dumps(res["summary"], indent=1))
    print(json.dumps(res["position"], indent=1))
    if args.saliency:
        print(json.dumps(res["saliency"], indent=1))


if __name__ == "__main__":
    main()
