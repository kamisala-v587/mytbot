"""Figures for the bpv4_bpen probe analysis.

  images: keyframe strips under the probe visibility conditions, compressor attention on the first keyframe,
          t-SNE / PCA of BP tokens (needs the feature cache, videos and checkpoints; one GPU)
  charts: bar charts / scatter plots from probe_result.json, shortcut_result.json and offline_eval/*.json

Usage: python -m lerobot.policies.BPV4BPEn.vis_probe images --out <dir> --ckpt main10k=<...> --ckpt nomask10k=<...>
       python -m lerobot.policies.BPV4BPEn.vis_probe charts --out <dir>
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

PROBE_DIR = Path("/vla/workspace/my_tbot/outputs/BPv4/probe")
EVAL_DIR = Path("/vla/workspace/my_tbot/outputs/BPv4/offline_eval")
DATA = Path("/vla/workspace/data/robotwin2.0")
CACHE = "/vla/workspace/cache/bpv4_bpen_vit_base800m_k32_v2"
EXAMPLE_TASKS = ("place_shoe", "move_can_pot", "place_empty_cup", "place_phone_stand", "place_mouse_pad", "place_can_basket")
COND_LABEL = {"all": "all keyframes", "kf0": "first keyframe only", "event": "event mask (as in training)",
              "event2": "event + carry segment masked"}


# ----------------------------------------------------------------------------------------------------- images
def head_images(repo: str, ep: int, frames) -> list[np.ndarray]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.transforms.utils import resize_with_pad

    ds = LeRobotDataset(repo, video_backend="torchcodec")
    start = int(ds.meta.episodes[ep]["dataset_from_index"])
    out = []
    for f in frames:
        img = ds[start + int(f)]["observation.images.cam_high"]
        out.append(resize_with_pad(img[None], 224, 224)[0].permute(1, 2, 0).clamp(0, 1).numpy())
    return out


def keyframe_strip(repo: str, ep: int, labels: dict, out: Path) -> None:
    from lerobot.policies.BPV4BPEn.probe_shortcut import CONDS, K, Trajectories

    gf = int(labels[f"{repo}|{ep}"]["grasp_frame"])
    item = Trajectories([(repo, ep, True)], CACHE, [gf])[0]
    frames = item["frames"].numpy()
    k_eff = int((frames > -10**9).sum())
    imgs = head_images(repo, ep, frames[:k_eff])
    fig, axes = plt.subplots(len(CONDS), k_eff, figsize=(k_eff * 1.3, len(CONDS) * 1.45 + 0.6))
    for r, c in enumerate(CONDS):
        vis = item["vis"][r].numpy()
        for j in range(k_eff):
            ax = axes[r, j]
            ax.imshow(imgs[j] if vis[j] else imgs[j] * 0.15 + 0.6)
            ax.set_xticks([]), ax.set_yticks([])
            color = "red" if frames[j] == gf else ("black" if vis[j] else "lightgray")
            for s in ax.spines.values():
                s.set_edgecolor(color), s.set_linewidth(2.5 if frames[j] == gf else 0.8)
            if r == 0:
                ax.set_title(str(frames[j]), fontsize=7)
        axes[r, 0].set_ylabel(COND_LABEL[c], rotation=0, ha="right", va="center", fontsize=9)
    task = Path(repo).parent.name
    fig.suptitle(f"{task} ep{ep}: head-camera keyframes (frame index on top, red = grasp keyframe, faded = hidden)",
                 fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.998, top=0.88, bottom=0.02, wspace=0.05, hspace=0.08)
    fig.savefig(out, dpi=80)
    plt.close(fig)


def attention_maps(models: dict, items_by_task: dict, labels: dict, out: Path) -> None:
    """Per task: first-keyframe head image, across-episode variation of ViT tokens, compressor attention per model."""
    from lerobot.policies.BPV4BPEn.build_feature_cache import cache_path
    from lerobot.policies.BPV4BPEn.probe_eval import compressor_with_attention

    names = list(models)
    cols = 2 + len(names)
    fig, axes = plt.subplots(len(items_by_task), cols, figsize=(cols * 2.3, len(items_by_task) * 2.4))
    for r, (task, (repo, ep, train_eps)) in enumerate(items_by_task.items()):
        img = head_images(repo, ep, [torch.load(cache_path(CACHE, repo, ep), weights_only=False)["frames"][0]])[0]
        head = torch.stack([torch.load(cache_path(CACHE, repo, e), weights_only=False)["feats"][0, 0].float()
                            for e in train_eps])  # [E, 64, D]
        var = head.std(0).mean(-1).reshape(8, 8).numpy()
        top = np.zeros(64, bool)
        top[np.argsort(var.reshape(-1))[-6:]] = True
        feats = torch.load(cache_path(CACHE, repo, ep), weights_only=False)["feats"][:1].float()[None].cuda()
        panels = [("head image, keyframe 0", None), ("ViT variation across episodes", var)]
        for n in names:
            _, attn = compressor_with_attention(models[n][0], feats)
            a = attn[0].cpu().numpy()
            h = a[:64]
            panels.append((f"{n}: head {h.sum():.2f}, on var {h[top].sum() / h.sum():.2f}", (h / h.sum()).reshape(8, 8)))
        for c, (title, m) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img)
            if m is not None:
                ax.imshow(m, cmap="jet", alpha=0.5, extent=(0, 224, 224, 0), interpolation="nearest")
                if c >= 2:
                    ys, xs = np.nonzero(top.reshape(8, 8))
                    ax.scatter(xs * 28 + 14, ys * 28 + 14, s=30, facecolors="none", edgecolors="white", linewidths=1.2)
            ax.set_title(title, fontsize=7.5)
            ax.set_xticks([]), ax.set_yticks([])
        axes[r, 0].set_ylabel(task, fontsize=9)
    fig.suptitle("Compressor last-layer attention on the head camera at keyframe 0 (8x8 tokens; white circles = 6 most "
                 "varying tokens = object region; 'head' = share of attention on head camera, 'on var' = share on circles)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=110)
    plt.close(fig)


@torch.no_grad()
def encode(model, feats, time, vis):
    from lerobot.policies.BPV4BPEn.modeling_bpv4_bpen import sinusoidal_encoding

    enc, res = model
    content = enc(feats) + sinusoidal_encoding(time, enc.output_proj[-1].out_features)[:, :, None]
    b, k, q, _ = content.shape
    return res(content.reshape(b, k * q, -1), vis[:, :, None].expand(-1, -1, q).reshape(b, k * q)).mean(1)


def embeddings(model, labels: dict, repo_file: str, out: Path) -> None:
    """t-SNE of validation BP tokens coloured by task; PCA within one task coloured by the grasp-point residual."""
    from sklearn.manifold import TSNE

    from lerobot.policies.BPV4BPEn.probe_eval import episodes
    from lerobot.policies.BPV4BPEn.probe_shortcut import Trajectories

    items = [it for it in episodes(repo_file) if (labels.get(f"{it[0]}|{it[1]}") or {}).get("grasp")]
    focus = "place_shoe"
    sel = [it for it in items if it[2] or Path(it[0]).parent.name == focus]
    ds = Trajectories(sel, CACHE, [int(labels[f"{r}|{e}"]["grasp_frame"]) for r, e, _ in sel])
    loader = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=16)
    bp_all, bp_kf0, vit0 = [], [], []
    for batch in loader:
        x, t, vis = batch["feats"].cuda(), batch["time"].cuda(), batch["vis"].cuda()
        bp_all.append(encode(model, x, t, vis[:, 0]).cpu())
        bp_kf0.append(encode(model, x, t, vis[:, 1]).cpu())
        vit0.append(x[:, 0].float().mean(2).reshape(x.shape[0], -1).cpu())
    bp_all, bp_kf0, vit0 = (torch.cat(v).numpy() for v in (bp_all, bp_kf0, vit0))
    task = np.array([Path(r).parent.name for r, _, _ in sel])
    is_val = np.array([v for _, _, v in sel])
    tasks = sorted(set(task[is_val]))
    color = np.array([tasks.index(t) for t in task[is_val]])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (title, x) in zip(axes[0], (("ViT, keyframe 0 (token mean)", vit0), ("BP tokens, first keyframe only", bp_kf0),
                                        ("BP tokens, all keyframes", bp_all))):
        z = TSNE(perplexity=20, random_state=0, init="pca").fit_transform((x[is_val] - x[is_val].mean(0)) / (x[is_val].std(0) + 1e-6))
        ax.scatter(z[:, 0], z[:, 1], c=color, cmap="tab20", s=9)
        ax.set_title(f"t-SNE of validation episodes ({len(tasks)} tasks, colour = task)\n{title}", fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])

    f_mask = task == focus
    rec = [labels[f"{r}|{e}"] for (r, e, _), m in zip(sel, f_mask) if m]
    arms = np.array([x["arm"] for x in rec])
    clean = np.array(["clean" in r for (r, _, _), m in zip(sel, f_mask) if m])
    for ax, (title, x) in zip(axes[1], (("ViT, keyframe 0 (token mean)", vit0), ("BP tokens, first keyframe only", bp_kf0),
                                        ("BP tokens, all keyframes", bp_all))):
        xf = x[f_mask]
        xf = (xf - xf.mean(0)) / (xf.std(0) + 1e-6)
        u, s, _ = np.linalg.svd(xf, full_matrices=False)
        pc = u[:, :2] * s[:2]
        for arm, col in (("L", "tab:orange"), ("R", "tab:blue")):
            for is_clean, mk in ((True, "s"), (False, "o")):
                m = (arms == arm) & (clean == is_clean)
                ax.scatter(pc[m, 0], pc[m, 1], c=col, marker=mk, s=16, alpha=0.8,
                           label=f"{'left' if arm == 'L' else 'right'} arm, {'clean' if is_clean else 'rand'}")
        ax.set_title(f"{focus}: PCA of {len(xf)} episodes (train + val)\n{title}", fontsize=10)
        ax.set_xlabel("PC1"), ax.set_ylabel("PC2"), ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def run_images(args) -> None:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.BPV4BPEn.probe_eval import episodes, load_modules

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = json.loads((PROBE_DIR / "labels.json").read_text())
    ckpts = dict(c.split("=", 1) for c in args.ckpt)
    cfg = PreTrainedConfig.from_pretrained(next(iter(ckpts.values())))
    models = {"random": load_modules(None, cfg), **{n: load_modules(p, cfg) for n, p in ckpts.items()}}

    items = episodes(args.repo_file)
    by_task = {}
    for task in EXAMPLE_TASKS:
        repo = str(DATA / task / "aloha-agilex_clean_50")
        eps = [e for r, e, v in items if r == repo and v and (labels.get(f"{r}|{e}") or {}).get("grasp")]
        train = [e for r, e, v in items if r == repo and not v]
        by_task[task] = (repo, eps[0], train)
    for task in EXAMPLE_TASKS[:3]:
        repo, ep, _ = by_task[task]
        keyframe_strip(repo, ep, labels, out / f"keyframes_{task}.png")
        print("strip", task, flush=True)
    attention_maps(models, by_task, labels, out / "attention_kf0.png")
    print("attention done", flush=True)
    embeddings(models[args.embed_model], labels, args.repo_file, out / "embeddings.png")
    print("embeddings done", flush=True)


# ----------------------------------------------------------------------------------------------------- charts
def bars(ax, groups, series, values, title, ylabel="validation R²", ylim=(-0.15, 1.05)):
    w = 0.8 / len(series)
    for i, s in enumerate(series):
        v = [values[g][s] for g in groups]
        b = ax.bar(np.arange(len(groups)) + (i - (len(series) - 1) / 2) * w, v, w, label=s)
        for rect, val in zip(b, v):
            ax.text(rect.get_x() + rect.get_width() / 2, max(val, 0) + 0.01, f"{val:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(np.arange(len(groups)), groups, fontsize=8.5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(*ylim), ax.set_ylabel(ylabel), ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.5)


def run_charts(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    probe = json.loads((PROBE_DIR / "probe_result.json").read_text())["probe"]
    short = json.loads((PROBE_DIR / "shortcut_result.json").read_text())["conds"]

    # 1. first-keyframe probe: x / y mean R²
    feats = [("vit_head.tok", "ViT head\n(all tokens)"), ("vit.mean", "ViT 3 cams\n(token mean)"),
             ("random.enc.tok", "random\ncompressor"), ("main2.5k.enc.tok", "main 2.5k\ncompressor"),
             ("main10k.enc.tok", "main 10k\ncompressor"), ("nomask10k.enc.tok", "nomask 10k\ncompressor"),
             ("main10k.res.tok", "main 10k\nBP tokens")]
    vals = {lab: {t: float(np.mean(probe[t][k]["r2_xyz"][:2])) for t in ("grasp", "place")} for k, lab in feats}
    fig, ax = plt.subplots(figsize=(11, 4))
    bars(ax, [lab for _, lab in feats], ["grasp", "place"], vals,
         "Linear probe on keyframe 0 (scene only): within-task object position, mean R² of x and y", ylim=(-0.1, 0.4))
    fig.tight_layout(), fig.savefig(out / "probe_kf0.png", dpi=120), plt.close(fig)

    # 2. where the grasp-point information comes from
    conds = ["all", "kf0", "event", "event2"]
    lab = {c: COND_LABEL[c].replace(" (as in training)", "\n(as in training)").replace(" + carry", "\n+ carry") for c in conds}
    series = {"arm EE at nearest visible keyframes": lambda d: np.mean(d["arm"]),
              "BP random encoder": lambda d: d["bp:random.tok"]["r2_mean"],
              "BP main 10k": lambda d: d["bp:main10k.tok"]["r2_mean"],
              "BP nomask 10k": lambda d: d["bp:nomask10k.tok"]["r2_mean"]}
    vals = {lab[c]: {s: float(f(short[c])) for s, f in series.items()} for c in conds}
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.3), gridspec_kw={"width_ratios": [3, 1.3]})
    bars(axes[0], [lab[c] for c in conds], list(series), vals, "Grasp point (within-task residual) read out from ...")
    ee = {"random": short["event2"]["bp->ee:random.tok"]["r2_mean"], "main 10k": short["event2"]["bp->ee:main10k.tok"]["r2_mean"],
          "nomask 10k": short["event2"]["bp->ee:nomask10k.tok"]["r2_mean"]}
    gr = {"random": short["event2"]["bp:random.tok"]["r2_mean"], "main 10k": short["event2"]["bp:main10k.tok"]["r2_mean"],
          "nomask 10k": short["event2"]["bp:nomask10k.tok"]["r2_mean"]}
    bars(axes[1], list(ee), ["grasp point", "visible arm EE"], {k: {"grasp point": gr[k], "visible arm EE": ee[k]} for k in ee},
         "event + carry masked:\nBP -> grasp point vs. visible arm pose")
    fig.tight_layout(), fig.savefig(out / "probe_source.png", dpi=120), plt.close(fig)

    # 3. scatter of predicted vs. true grasp x residual
    panels = [("ViT head tokens, keyframe 0", probe["grasp"]["vit_head.tok"]),
              ("main 10k BP, keyframe 0 only", short["kf0"]["bp:main10k.tok"]),
              ("main 10k BP, event mask", short["event"]["bp:main10k.tok"]),
              ("arm EE feature, event mask", short["event"]["arm_pred"]),
              ("main 10k BP, event + carry masked", short["event2"]["bp:main10k.tok"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (title, d) in zip(axes, panels):
        p, t = np.array(d["pred"]) * 100, np.array(d["true"]) * 100
        r2 = 1 - ((p - t) ** 2).sum(0) / ((t - t.mean(0)) ** 2).sum(0)
        ax.scatter(t[:, 0], p[:, 0], s=5, alpha=0.5, label=f"x  R²={r2[0]:.2f}")
        ax.scatter(t[:, 1], p[:, 1], s=5, alpha=0.5, label=f"y  R²={r2[1]:.2f}")
        ax.plot([-15, 15], [-15, 15], "k--", lw=0.8)
        ax.set_xlim(-15, 15), ax.set_ylim(-15, 15), ax.set_aspect("equal")
        ax.set_xlabel("true residual (cm)"), ax.set_ylabel("predicted (cm)"), ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Grasp point within-task residual on validation episodes: prediction vs. truth", fontsize=11)
    fig.tight_layout(), fig.savefig(out / "probe_scatter.png", dpi=120), plt.close(fig)

    # 4. offline flow-loss ratios
    conds_eval = ["event", "window", "p_drop", "lang_drop", "swap_same", "event_swap", "swap_wrong"]
    names = {"event": "event mask", "window": "window mask", "p_drop": "drop p", "lang_drop": "drop language",
             "swap_same": "BP from another\nepisode (same task)", "event_swap": "event mask +\nswap same task",
             "swap_wrong": "BP from\nanother task"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.3), gridspec_kw={"width_ratios": [3, 1.2]})
    vals = {}
    for c in conds_eval:
        vals[names[c]] = {}
        for run in ("main", "nomask"):
            d = json.loads((EVAL_DIR / f"{run}_10000.json").read_text())["loss"]
            vals[names[c]][f"{run} 10k"] = d[c] / d["nomask"]
    bars(axes[0], [names[c] for c in conds_eval], ["main 10k", "nomask 10k"], vals,
         "Validation flow loss relative to the unmasked condition (1 = no effect)", ylabel="loss ratio", ylim=(0, 7.8))
    axes[0].axhline(1, color="gray", ls="--", lw=0.8)
    steps = (2500, 5000, 7500, 10000)
    for run, st in (("main", "-"), ("nomask", "--")):
        for c, col in (("nomask", "C0"), ("swap_same", "C1"), ("event", "C2")):
            d = [json.loads((EVAL_DIR / f"{run}_{s}.json").read_text())["loss"] for s in steps]
            axes[1].plot([s / 1000 for s in steps], [x[c] for x in d], st, color=col, marker="o", ms=3, label=f"{run}: {c}")
    axes[1].set_xlabel("step (k)"), axes[1].set_ylabel("flow loss"), axes[1].set_yscale("log")
    axes[1].set_title("Validation flow loss over training", fontsize=10), axes[1].legend(fontsize=7)
    fig.tight_layout(), fig.savefig(out / "offline_loss.png", dpi=120), plt.close(fig)
    print("charts done")


def run_identity(args) -> None:
    out = Path(args.out)
    r = json.loads((PROBE_DIR / "identity_result.json").read_text())
    feats = [("vit.kf0_head", "ViT head\nkeyframe 0"), ("vit.inhand_wrist", "ViT wrist\nobject in hand"),
             ("random.bp.all", "random enc.\nall kf"), ("main10k.bp.all", "main 10k\nall kf"),
             ("main10k.bp.event", "main 10k\nevent mask"), ("main10k.bp.kf0", "main 10k\nkf0 only"),
             ("nomask10k.bp.all", "nomask 10k\nall kf")]
    pairs = [("place_shoe|grasp_id", "shoe id (9 classes)"), ("place_phone_stand|grasp_id", "phone id (4)"),
             ("place_phone_stand|target_id", "stand id (2)"), ("place_shoe|arm", "grasp arm, shoe (2)")]
    vals = {lab: {p_lab: r["identity"][k]["per"][p]["bacc"] for p, p_lab in pairs} for k, lab in feats}
    fig, ax = plt.subplots(figsize=(14, 4.5))
    bars(ax, [lab for _, lab in feats], [p_lab for _, p_lab in pairs], vals,
         "Within-task object identity: balanced accuracy (5-fold CV, 100 new episodes per task)",
         ylabel="balanced accuracy", ylim=(0, 1.1))
    for (p, _), col in zip(pairs, plt.rcParams["axes.prop_cycle"].by_key()["color"]):
        ax.axhline(r["identity"]["vit.kf0_head"]["per"][p]["chance"], color=col, ls=":", lw=1)
    fig.tight_layout(), fig.savefig(out / "identity.png", dpi=120), plt.close(fig)
    print("identity done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["images", "charts", "identity"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--embed_model", default="main10k")
    ap.add_argument("--repo_file", default="/vla/workspace/my_tbot/configs/ds_ids/pro6k/robo_bpv4_bpen_clean50_rand100.txt")
    args = ap.parse_args()
    {"images": run_images, "charts": run_charts, "identity": run_identity}[args.mode](args)


if __name__ == "__main__":
    main()
