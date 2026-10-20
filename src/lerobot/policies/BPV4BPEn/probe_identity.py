"""Does the BP encoder know WHAT the target object is? Probes on newly collected RoboTwin episodes with object ids.

Inputs per repo (converted by tools/bp_task_screen/robotwin_hdf5_to_lerobot.py):
  meta/scene_info.json   object model ids ("{A}": "041_shoe/base5", ...) and object poses (world frame)
  source HDF5            head-camera intrinsics / extrinsics (to project objects onto the 8x8 head-camera tokens)
  --labels               FK grasp / place labels (probe_labels.py) for the grasp keyframe and the grasp arm

Measurements
  identity   within-task classification of the grasped object / placement target model id
             (standardize -> PCA 32 -> logistic regression, stratified 5-fold CV, balanced accuracy, chance = 1 / #classes)
  position   pooled ridge (CV) of the grasped object's initial / final xy (residual after the (task, arm) mean)
  attention  compressor last-layer attention on the head-camera token(s) containing the grasped object at keyframe 0
Feature sets: raw ViT (keyframe 0; grasp-arm wrist camera at the first keyframe after the grasp, i.e. object in hand),
random encoder, trained BP under the visibility conditions of probe_shortcut (all / kf0 / event / event2).

Usage: python -m lerobot.policies.BPV4BPEn.probe_identity --repo_file <txt> --labels labels_probe.json --out out.json \
         --ckpt main10k=<pretrained_model> --ckpt nomask10k=<...>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.BPV4BPEn.modeling_bpv4_bpen import sinusoidal_encoding
from lerobot.policies.BPV4BPEn.probe_eval import compressor_with_attention, load_modules
from lerobot.policies.BPV4BPEn.probe_shortcut import CONDS, K, Trajectories

ROBOTWIN_DATA = Path("/vla/workspace/my_robotwin/data")
SOURCE_CONFIG = "bp_probe_clean"
# info keys of the grasped object and of the placement target / second object, per task
ROLES = {
    "place_shoe": {"grasp": "{A}"},
    "move_can_pot": {"grasp": "{B}", "target": "{A}"},
    "place_phone_stand": {"grasp": "{A}", "target": "{B}"},
    "place_can_basket": {"grasp": "{A}", "target": "{B}"},
    "place_mouse_pad": {"grasp": "{A}", "target": "{B}"},
    "scan_object": {"grasp": "{A}", "target": "{B}"},
    "handover_mic": {"grasp": "{A}"},
    "rotate_qrcode": {"grasp": "{A}"},
    "place_a2b_left": {"grasp": "{A}", "target": "{B}"},
    "place_object_basket": {"grasp": "{A}", "target": "{B}"},
    "place_container_plate": {"grasp": "{B}", "target": "{A}"},
    "place_object_stand": {"grasp": "{A}", "target": "{B}"},
}
STATIC = {"wall", "table"}


def episode_meta(repo: str, ep: int, scene: dict) -> dict:
    task = Path(repo).parent.name
    src_file = Path(repo, "meta", "source_episodes.json")
    if src_file.exists():
        ep = json.loads(src_file.read_text())["source_episodes"][ep]
    info = scene[f"episode_{ep}"]
    ids = info["info"]
    init, final = info.get("object_pose_init", {}), info.get("object_pose_final", {})
    moved = {n: float(np.linalg.norm(np.array(final[n][:3]) - np.array(init[n][:3]))) for n in init if n in final and n not in STATIC}
    grasped = max(moved, key=moved.get) if moved else None
    with h5py.File(ROBOTWIN_DATA / task / SOURCE_CONFIG / "data" / f"episode{ep}.hdf5", "r") as f:
        intr = f["observation/head_camera/intrinsic_cv"][0]
        extr = f["observation/head_camera/extrinsic_cv"][0]
    token = None
    if grasped:
        p = np.array(init[grasped][:3])
        cam = extr @ np.r_[p, 1.0]
        if cam[2] > 0:
            u, v = (intr @ cam)[:2] / cam[2]
            # 320x240 -> resize_with_pad 224: scale 0.7, 28 px pad top / bottom; 8x8 tokens of 28 px
            col, row = int(u * 0.7 // 28), int((v * 0.7 + 28) // 28)
            if 0 <= col < 8 and 0 <= row < 8:
                token = row * 8 + col
    role = ROLES.get(task, {})
    return {
        "task": task,
        "grasp_id": ids.get(role.get("grasp", ""), None),
        "target_id": ids.get(role.get("target", ""), None),
        "grasp_init": init[grasped][:3] if grasped else None,
        "grasp_final": final[grasped][:3] if grasped else None,
        "grasped_actor": grasped,
        "head_token": token,
    }


@torch.no_grad()
def encode(enc, res, x, t, vis):
    content = enc(x) + sinusoidal_encoding(t, enc.output_proj[-1].out_features)[:, :, None]
    b, k, q, _ = content.shape
    return res(content.reshape(b, k * q, -1), vis[:, :, None].expand(-1, -1, q).reshape(b, k * q)).mean(1)


def classify(x: np.ndarray, y: np.ndarray) -> dict | None:
    classes, counts = np.unique(y, return_counts=True)
    keep = np.isin(y, classes[counts >= 3])
    if len(np.unique(y[keep])) < 2:
        return None
    x, y = x[keep], y[keep]
    folds = min(5, int(np.unique(y, return_counts=True)[1].min()))
    clf = make_pipeline(StandardScaler(), PCA(n_components=min(32, int(len(y) * (folds - 1) / folds) - 1), random_state=0),
                        LogisticRegression(C=1.0, max_iter=5000))
    pred = cross_val_predict(clf, x, y, cv=StratifiedKFold(folds, shuffle=True, random_state=0))
    return {"bacc": float(balanced_accuracy_score(y, pred)), "chance": 1.0 / len(np.unique(y)), "n": int(len(y)),
            "classes": int(len(np.unique(y)))}


def position_r2(x: np.ndarray, y: np.ndarray) -> list[float]:
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13)))
    pred = cross_val_predict(model, x, y, cv=KFold(5, shuffle=True, random_state=0))
    return [float(v) for v in 1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_file", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache_dir", default="/vla/workspace/cache/bpv4_bpen_vit_base800m_k32_v2")
    ap.add_argument("--ckpt", action="append", default=[])
    args = ap.parse_args()

    labels = json.loads(Path(args.labels).read_text())
    repos = [l.strip() for l in Path(args.repo_file).read_text().splitlines() if l.strip()]
    items, metas = [], []
    for repo in repos:
        scene = json.loads(Path(repo, "meta", "scene_info.json").read_text())
        n = int(json.loads(Path(repo, "meta", "info.json").read_text())["total_episodes"])
        for ep in range(n):
            lab = labels.get(f"{repo}|{ep}") or {}
            if not lab.get("grasp"):
                continue
            items.append((repo, ep, True))
            metas.append({**episode_meta(repo, ep, scene), "arm": lab["arm"], "grasp_frame": int(lab["grasp_frame"])})
    print("episodes", len(items), flush=True)

    ckpts = dict(c.split("=", 1) for c in args.ckpt)
    cfg = PreTrainedConfig.from_pretrained(next(iter(ckpts.values())))
    models = {"random": load_modules(None, cfg), **{n: load_modules(p, cfg) for n, p in ckpts.items()}}
    ds = Trajectories(items, args.cache_dir, [m["grasp_frame"] for m in metas])
    loader = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=16)

    feats: dict[str, list] = {"vit.kf0_head": [], "vit.kf0_all": [], "vit.inhand_wrist": [], "vit.inhand_head": []}
    for m in models:
        for c in CONDS:
            feats[f"{m}.bp.{c}"] = []
        feats[f"{m}.attn"] = []
    for batch in loader:
        x, t, vis = batch["feats"].cuda(), batch["time"].cuda(), batch["vis"].cuda()
        b = x.shape[0]
        feats["vit.kf0_head"].append(x[:, 0, 0].float().mean(1).cpu())
        feats["vit.kf0_all"].append(x[:, 0].float().mean(2).reshape(b, -1).cpu())
        wrist, head = [], []
        for j, i in enumerate(batch["index"].tolist()):
            frames = batch["frames"][j].numpy()
            after = np.flatnonzero(frames > metas[i]["grasp_frame"])
            k = int(after[0]) if len(after) else int((frames > -10**9).sum()) - 1
            cam = 1 if metas[i]["arm"] == "L" else 2
            wrist.append(x[j, k, cam].float().mean(0).cpu())
            head.append(x[j, k, 0].float().mean(0).cpu())
        feats["vit.inhand_wrist"].append(torch.stack(wrist))
        feats["vit.inhand_head"].append(torch.stack(head))
        for name, (enc, res) in models.items():
            for ci, c in enumerate(CONDS):
                feats[f"{name}.bp.{c}"].append(encode(enc, res, x, t, vis[:, ci]).cpu())
            feats[f"{name}.attn"].append(compressor_with_attention(enc, x[:, :1].float())[1].cpu())
    feats = {k: torch.cat(v).numpy() for k, v in feats.items()}

    task = np.array([m["task"] for m in metas])
    result = {"n_episodes": len(items), "tasks": {}, "identity": {}, "position": {}, "attention": {}}
    for t in np.unique(task):
        sel = task == t
        result["tasks"][t] = {
            "n": int(sel.sum()),
            "grasp_ids": int(len({m["grasp_id"] for m, s in zip(metas, sel) if s and m["grasp_id"]})),
            "target_ids": int(len({m["target_id"] for m, s in zip(metas, sel) if s and m["target_id"]})),
            "arms": {a: int(((np.array([m["arm"] for m in metas]) == a) & sel).sum()) for a in ("L", "R")},
        }

    # identity: per (task, role), then averaged over the (task, role) pairs that have >= 2 classes
    arm = np.array([m["arm"] for m in metas])
    for name, x in feats.items():
        if name.endswith(".attn"):
            continue
        per = {}
        for t in np.unique(task):
            sel = np.flatnonzero(task == t)
            for role in ("grasp_id", "target_id", "arm"):
                y = arm[sel] if role == "arm" else np.array([str(metas[i][role]) for i in sel])
                if role != "arm" and all(metas[i][role] is None for i in sel):
                    continue
                r = classify(x[sel], y)
                if r:
                    per[f"{t}|{role}"] = r
        summary = {}
        for role in ("grasp_id", "target_id", "arm"):
            rs = [v for k, v in per.items() if k.endswith("|" + role)]
            if rs:
                summary[role] = {"bacc": float(np.mean([r["bacc"] for r in rs])), "chance": float(np.mean([r["chance"] for r in rs])),
                                 "pairs": len(rs)}
        result["identity"][name] = {"summary": summary, "per": per}
        print(name, {k: round(v["bacc"], 3) for k, v in summary.items()}, flush=True)

    # position of the grasped object (initial = where to grasp, final = where it was placed), residual per (task, arm)
    ok = np.array([m["grasp_init"] is not None for m in metas])
    for key in ("grasp_init", "grasp_final"):
        y = np.zeros((len(metas), 2))
        y[ok] = np.array([m[key][:2] for m, o in zip(metas, ok) if o])
        group = np.array([f"{m['task']}|{m['arm']}" for m in metas])
        for g in np.unique(group[ok]):
            s = ok & (group == g)
            y[s] -= y[s].mean(0)
        result["position"][key] = {"std_m": [float(v) for v in y[ok].std(0)]}
        for name, x in feats.items():
            if name.endswith(".attn"):
                continue
            result["position"][key][name] = position_r2(x[ok], y[ok])
        print(key, {k: round(float(np.mean(v)), 3) for k, v in result["position"][key].items() if k != "std_m"}, flush=True)

    # attention on the grasped object's head-camera token at keyframe 0 (and its 3x3 neighbourhood)
    tok = np.array([m["head_token"] if m["head_token"] is not None else -1 for m in metas])
    has = tok >= 0
    for name in models:
        a = feats[f"{name}.attn"][:, :64]
        a = a / a.sum(1, keepdims=True)
        on = a[np.flatnonzero(has), tok[has]]
        nb = []
        for i in np.flatnonzero(has):
            r, c = divmod(int(tok[i]), 8)
            idx = [rr * 8 + cc for rr in range(max(0, r - 1), min(8, r + 2)) for cc in range(max(0, c - 1), min(8, c + 2))]
            nb.append((a[i, idx].sum(), len(idx) / 64))
        nb = np.array(nb)
        result["attention"][name] = {"object_token": float(on.mean()), "uniform_token": 1 / 64,
                                     "object_3x3": float(nb[:, 0].mean()), "uniform_3x3": float(nb[:, 1].mean()),
                                     "n": int(has.sum())}
    print(json.dumps(result["attention"], indent=1), flush=True)
    result["episodes"] = [{"repo": r, "episode": e, **{k: v for k, v in m.items()}} for (r, e, _), m in zip(items, metas)]
    Path(args.out).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
