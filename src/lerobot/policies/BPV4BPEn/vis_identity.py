"""Heatmaps for the object-identity probe: episode x episode similarity sorted by object instance, and confusion
matrices of the instance classifier (same pipeline as probe_identity.classify).

Usage: python -m lerobot.policies.BPV4BPEn.vis_identity --out <dir> --ckpt main10k=<...> --ckpt nomask10k=<...>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.BPV4BPEn.probe_eval import load_modules
from lerobot.policies.BPV4BPEn.probe_identity import encode, episode_meta
from lerobot.policies.BPV4BPEn.probe_shortcut import Trajectories

TASKS = {"place_shoe": "shoe", "place_phone_stand": "phone"}
PANELS = [("vit.inhand", "ViT wrist cam, object in gripper"), ("vit.kf0", "ViT head cam, keyframe 0"),
          ("random", "random encoder, BP (all keyframes)")]


def model_panels(names: list[str]) -> tuple[list[tuple[str, str]], tuple[str, str]]:
    """One similarity / confusion panel per checkpoint; the arm-first panel uses the first checkpoint."""
    panels = PANELS + [(n, f"{n}, BP (all keyframes)") for n in names]
    return panels, (names[0], f"{names[0]}, BP, sorted by ARM first")


def load(repo_file: str, labels: dict):
    items, metas = [], []
    for repo in [l.strip() for l in Path(repo_file).read_text().splitlines() if l.strip()]:
        scene = json.loads(Path(repo, "meta", "scene_info.json").read_text())
        n = int(json.loads(Path(repo, "meta", "info.json").read_text())["total_episodes"])
        for ep in range(n):
            lab = labels.get(f"{repo}|{ep}") or {}
            if lab.get("grasp"):
                items.append((repo, ep, True))
                metas.append({**episode_meta(repo, ep, scene), "arm": lab["arm"], "grasp_frame": int(lab["grasp_frame"])})
    return items, metas


@torch.no_grad()
def features(items, metas, models, cache_dir):
    loader = torch.utils.data.DataLoader(Trajectories(items, cache_dir, [m["grasp_frame"] for m in metas]),
                                         batch_size=32, num_workers=16)
    out = {k: [] for k in ("vit.inhand", "vit.kf0", *models)}
    for batch in loader:
        x, t, vis = batch["feats"].cuda(), batch["time"].cuda(), batch["vis"].cuda()
        out["vit.kf0"].append(x[:, 0, 0].float().mean(1).cpu())
        wrist = []
        for j, i in enumerate(batch["index"].tolist()):
            frames = batch["frames"][j].numpy()
            after = np.flatnonzero(frames > metas[i]["grasp_frame"])
            k = int(after[0]) if len(after) else int((frames > -10**9).sum()) - 1
            wrist.append(x[j, k, 1 if metas[i]["arm"] == "L" else 2].float().mean(0).cpu())
        out["vit.inhand"].append(torch.stack(wrist))
        for name, (enc, res) in models.items():
            out[name].append(encode(enc, res, x, t, vis[:, 0]).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def short(model_id: str) -> str:
    return model_id.split("/")[-1].replace("base", "b")


def gain(sim: np.ndarray, y: np.ndarray) -> float:
    """Mean similarity of same-label pairs minus different-label pairs (diagonal excluded)."""
    same = y[:, None] == y[None]
    off = ~np.eye(len(y), dtype=bool)
    return float(sim[same & off].mean() - sim[~same].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo_file", default="/vla/workspace/my_tbot/configs/ds_ids/pro6k/robo_bpv4_bpen_probe.txt")
    ap.add_argument("--labels", default="/vla/workspace/my_tbot/outputs/BPv4/probe/labels_probe.json")
    ap.add_argument("--cache_dir", default="/vla/workspace/cache/bpv4_bpen_vit_base800m_k32_v2")
    ap.add_argument("--ckpt", action="append", default=[])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items, metas = load(args.repo_file, json.loads(Path(args.labels).read_text()))
    ckpts = dict(c.split("=", 1) for c in args.ckpt)
    cfg = PreTrainedConfig.from_pretrained(next(iter(ckpts.values())))
    models = {"random": load_modules(None, cfg), **{n: load_modules(p, cfg) for n, p in ckpts.items()}}
    feats = features(items, metas, models, args.cache_dir)
    task = np.array([m["task"] for m in metas])
    print("features", {k: v.shape for k, v in feats.items()}, flush=True)
    panels, arm_first = model_panels(list(ckpts))

    fig_s, ax_s = plt.subplots(len(TASKS), len(panels) + 1, figsize=(4.2 * (len(panels) + 1), 4.6 * len(TASKS)))
    fig_c, ax_c = plt.subplots(len(TASKS), len(panels), figsize=(3.9 * len(panels), 4.0 * len(TASKS)))
    for r, (tname, obj) in enumerate(TASKS.items()):
        sel = np.flatnonzero(task == tname)
        ids = np.array([str(metas[i]["grasp_id"]) for i in sel])
        arms = np.array([metas[i]["arm"] for i in sel])
        classes, counts = np.unique(ids, return_counts=True)
        keep = np.isin(ids, classes[counts >= 3])
        sel, ids, arms = sel[keep], ids[keep], arms[keep]
        classes = sorted(set(ids))
        order = np.lexsort((arms, ids))
        idx_num = np.array([classes.index(v) for v in ids])
        arm_num = (arms == "R").astype(int)
        order_obj = order
        for c, (key, title) in enumerate([*panels, arm_first]):
            order = np.lexsort((ids, arms)) if c == len(panels) else order_obj
            bounds = np.flatnonzero(np.diff((arm_num if c == len(panels) else idx_num)[order])) + 0.5
            x = feats[key][sel]
            z = (x - x.mean(0)) / (x.std(0) + 1e-6)
            z = z / np.linalg.norm(z, axis=1, keepdims=True)
            sim = z @ z.T
            g_obj, g_arm = gain(sim, idx_num), gain(sim, arm_num)

            ax = ax_s[r, c]
            ax.imshow(sim[np.ix_(order, order)], cmap="RdBu_r", vmin=-0.6, vmax=0.6, interpolation="nearest")
            for b in bounds:
                ax.axhline(b, color="k", lw=0.5), ax.axvline(b, color="k", lw=0.5)
            n = len(order)
            ax.imshow(idx_num[order][None], cmap="tab10", vmin=0, vmax=9, extent=(-0.5, n - 0.5, -0.5 - n * 0.06, -0.5),
                      aspect="auto", interpolation="nearest")
            ax.imshow(np.where(arm_num[order] == 0, 1, 0)[:, None], cmap="Oranges", vmin=0, vmax=1.3,
                      extent=(-0.5 - n * 0.06, -0.5, n - 0.5, -0.5), aspect="auto", interpolation="nearest")
            ax.set_xlim(-0.5 - n * 0.06, n - 0.5), ax.set_ylim(n - 0.5, -0.5 - n * 0.06)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{title}\nsame-{obj} gain {g_obj:+.2f} | same-arm gain {g_arm:+.2f}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{tname} ({len(classes)} {obj}s, {n} episodes)\nleft strip: orange = left arm", fontsize=9)
            if c == len(panels):
                continue

            folds = min(5, int(np.unique(ids, return_counts=True)[1].min()))
            clf = make_pipeline(StandardScaler(), PCA(n_components=min(32, int(len(ids) * (folds - 1) / folds) - 1), random_state=0),
                                LogisticRegression(C=1.0, max_iter=5000))
            pred = cross_val_predict(clf, x, ids, cv=StratifiedKFold(folds, shuffle=True, random_state=0))
            cm = confusion_matrix(ids, pred, labels=classes, normalize="true")
            ax = ax_c[r, c]
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(classes)), [short(v) for v in classes], rotation=90, fontsize=7)
            ax.set_yticks(range(len(classes)), [short(v) for v in classes], fontsize=7)
            ax.set_title(f"{title}\nbalanced acc {balanced_accuracy_score(ids, pred):.2f} (chance {1 / len(classes):.2f})",
                         fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{tname}: true {obj}", fontsize=9)
            ax.set_xlabel(f"predicted {obj}", fontsize=8)
            print(tname, key, round(g_obj, 3), round(g_arm, 3), round(balanced_accuracy_score(ids, pred), 3), flush=True)
    fig_s.suptitle("Episode x episode cosine similarity within a task. Columns 1-5: sorted by grasped object instance (top strip, "
                   "black lines) then arm (left strip); column 6: sorted by arm first. "
                   "'gain' = mean similarity of same-label pairs minus different-label pairs",
                   fontsize=10)
    fig_s.tight_layout(rect=(0, 0, 1, 0.96)), fig_s.savefig(out / "identity_similarity.png", dpi=110)
    fig_c.suptitle("Object-instance classifier (5-fold CV, PCA 32 + logistic regression): row-normalised confusion matrix",
                   fontsize=10)
    fig_c.tight_layout(rect=(0, 0, 1, 0.95)), fig_c.savefig(out / "identity_confusion.png", dpi=110)


if __name__ == "__main__":
    main()
