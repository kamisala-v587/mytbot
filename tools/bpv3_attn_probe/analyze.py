"""Summaries and plots for attn_probe.py shards."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

SEG_LABELS = ("head cam", "wrist cam 1", "wrist cam 2", "text/special", "BP", "state", "action")
EVAL_TASKS = (
    "handover_mic", "place_cans_plasticbox", "place_mouse_pad", "place_shoe", "rotate_qrcode",
    "scan_object", "stack_blocks_two", "stack_bowls_two",
)


def load(out: Path, names: list[str]) -> dict[str, dict]:
    return {n: torch.load(out / "shards" / f"{n}.pt", map_location="cpu", weights_only=False) for n in names}


def stack(results, cond, key):
    return torch.stack([r["attn"][cond][key] for r in results])  # [N, T, L, ...]


def summarize(run: dict) -> dict[str, Any]:
    res = run["results"]
    s: dict[str, Any] = {}
    mass = stack(res, "other", "mass")  # [N,T,L,H,S]
    contrib = stack(res, "other", "contrib")
    s["mass_share"] = mass.mean(dim=(0, 1, 2, 3)).tolist()
    s["mass_share_by_layer"] = mass.mean(dim=(0, 1, 3)).tolist()  # [L,S]
    s["mass_share_by_time"] = mass.mean(dim=(0, 2, 3)).tolist()  # [T,S]
    c = contrib.mean(dim=(0, 1, 2, 3))
    s["contrib_share"] = (c / c.sum()).tolist()
    cl = contrib.mean(dim=(0, 1, 3))
    s["contrib_share_by_layer"] = (cl / cl.sum(-1, keepdim=True)).tolist()
    bp_heads = mass[..., 4].mean(dim=(0, 1))  # [L,H]
    s["bp_mass_max_head"] = float(bp_heads.max())
    s["bp_mass_head_layer"] = [int(i) for i in np.unravel_index(int(bp_heads.argmax()), bp_heads.shape)]
    n_tokens = np.array([64, 64, 64, np.nan, np.nan, 1, 50], dtype=float)
    s["bp_mass_by_cond"] = {c_: float(stack(res, c_, "mass")[..., 4].mean()) for c_ in ("self", "other", "wrong")}
    s["img_to_bp"] = float(stack(res, "other", "img_to_bp").mean())
    s["bp_to_img"] = float(stack(res, "other", "bp_to_img").mean())
    bp_tokens = np.mean([r["bp_valid_chunks"] for r in res]) * 8
    n_tokens[4] = bp_tokens
    s["bp_tokens"] = float(bp_tokens)
    s["mass_per_token"] = (np.array(s["mass_share"]) / n_tokens).tolist()

    # BP keyframe attention vs current progress
    chunk = stack(res, "other", "chunk").mean(dim=(1, 3))  # [N, L, C]
    progress = np.array([r["progress"] for r in res])
    kv = np.array([r["bp_valid_chunks"] for r in res])
    corr_by_layer, dist_by_layer = [], []
    for layer in range(chunk.shape[1]):
        exp_pos = []
        dist = np.zeros((len(sorted(set(progress))), chunk.shape[-1]))
        for i, r in enumerate(res):
            m = chunk[i, layer, : kv[i]].numpy()
            m = m / max(m.sum(), 1e-12)
            exp_pos.append(float((m * np.linspace(0, 1, kv[i])).sum()) if kv[i] > 1 else 0.0)
            if kv[i] == chunk.shape[-1]:
                dist[sorted(set(progress)).index(progress[i])] += m
        corr_by_layer.append(float(spearmanr(exp_pos, progress).correlation))
        dist_by_layer.append((dist / dist.sum(-1, keepdims=True)).tolist())
    s["chunk_progress_corr_by_layer"] = corr_by_layer
    s["chunk_dist_by_layer"] = dist_by_layer

    # offline BP swap: flow loss under each BP condition
    loss = {c_: np.array([np.mean(r["loss"][c_]) for r in res]) for c_ in ("self", "other", "wrong", "mask")}
    s["loss"] = {c_: float(v.mean()) for c_, v in loss.items()}
    eval_idx = np.array([r["task"] in EVAL_TASKS for r in res])
    s["loss_eval_tasks"] = {c_: float(v[eval_idx].mean()) for c_, v in loss.items()}
    s["loss_by_time"] = {
        c_: np.mean([r["loss"][c_] for r in res], axis=0).tolist() for c_ in ("self", "other", "wrong", "mask")
    }
    tasks = sorted({r["task"] for r in res})
    s["loss_by_task"] = {
        t: {c_: float(loss[c_][[r["task"] == t for r in res]].mean()) for c_ in loss} for t in tasks
    }
    s["wrong_worse_frac"] = float(np.mean(loss["wrong"] > loss["other"]))
    s["mask_worse_frac"] = float(np.mean(loss["mask"] > loss["other"]))
    return s


def plot(summaries: dict[str, dict], out: Path, times: list[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    names = list(summaries)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, key, title in ((axes[0], "mass_share", "Attention mass share (action queries)"),
                           (axes[1], "contrib_share", "Value-weighted contribution share")):
        bottom = np.zeros(len(names))
        for k, label in enumerate(SEG_LABELS):
            vals = np.array([summaries[n][key][k] for n in names])
            ax.bar(names, vals, bottom=bottom, label=label)
            bottom += vals
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=30)
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(plots / "segment_share.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for n in names:
        layers = np.array(summaries[n]["mass_share_by_layer"])
        axes[0].plot(layers[:, 4], marker="o", label=n)
        axes[1].plot(np.array(summaries[n]["contrib_share_by_layer"])[:, 4], marker="o", label=n)
        axes[2].plot(summaries[n]["chunk_progress_corr_by_layer"], marker="o", label=n)
    axes[0].set_title("BP attention mass by full-attention layer")
    axes[1].set_title("BP value-weighted contribution by layer")
    axes[2].set_title("Spearman(expected BP keyframe, current progress)")
    for ax in axes:
        ax.set_xlabel("full-attention layer index")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "bp_by_layer.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.2))
    for ax, n in zip(np.atleast_1d(axes), names):
        corr = summaries[n]["chunk_progress_corr_by_layer"]
        layer = int(np.nanargmax(np.abs(corr)))
        im = ax.imshow(np.array(summaries[n]["chunk_dist_by_layer"][layer]), aspect="auto", cmap="viridis")
        ax.set_title(f"{n}\nlayer {layer}, rho={corr[layer]:+.2f}", fontsize=9)
        ax.set_xlabel("BP keyframe")
        ax.set_ylabel("current progress")
        ax.set_yticks(range(3), ["0.1", "0.5", "0.9"])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(plots / "bp_keyframe_vs_progress.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.2
    x = np.arange(len(names))
    for k, cond in enumerate(("self", "other", "wrong", "mask")):
        ax.bar(x + (k - 1.5) * width, [summaries[n]["loss"][cond] for n in names], width, label=cond)
    ax.set_xticks(x, names, rotation=30)
    ax.set_title("Flow loss under BP conditions (fixed noise, t averaged)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "bp_swap_loss.png", dpi=150)
    plt.close(fig)


def analyze_routes(runs: dict[str, dict], out: Path, conds: tuple[str, ...]) -> None:
    summary = {}
    for n, run in runs.items():
        res = run["results"]
        loss = {c: np.array([np.mean(r["loss"][c]) for r in res]) for c in conds}
        eval_idx = np.array([r["task"] in EVAL_TASKS for r in res])
        tasks = sorted({r["task"] for r in res})
        summary[n] = {
            "loss": {c: float(v.mean()) for c, v in loss.items()},
            "loss_eval_tasks": {c: float(v[eval_idx].mean()) for c, v in loss.items()},
            "loss_by_task": {t: {c: float(loss[c][[r["task"] == t for r in res]].mean()) for c in conds} for t in tasks},
        }
        by_task = summary[n]["loss_by_task"]
        summary[n]["median_task_ratio"] = {c: float(np.median([by_task[t][c] / by_task[t]["other"] for t in tasks])) for c in conds}
        summary[n]["loss_t0"] = {c: float(np.mean([r["loss"][c][0] for r in res])) for c in conds}
        progress = sorted({r["progress"] for r in res})
        summary[n]["loss_by_progress"] = {
            str(p): {c: float(loss[c][[r["progress"] == p for r in res]].mean()) for c in conds} for p in progress
        }
        print(f"== {n}  all: " + " ".join(f"{c}={summary[n]['loss'][c]:.4f}" for c in conds))
        print("   eval: " + " ".join(f"{c}={summary[n]['loss_eval_tasks'][c]:.4f}" for c in conds))
        print("   t=0.9: " + " ".join(f"{c}={summary[n]['loss_t0'][c]:.4f}" for c in conds))
        print("   median task ratio: " + " ".join(f"{c}={v:.2f}" for c, v in summary[n]["median_task_ratio"].items()))
        for p, row in summary[n]["loss_by_progress"].items():
            print(f"   progress {p}: " + " ".join(f"{c}={v:.4f}" for c, v in row.items()))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))


def analyze(cfg: dict[str, Any], out: Path) -> None:
    names = [ck["name"] for ck in cfg["checkpoints"] if (out / "shards" / f"{ck['name']}.pt").is_file()]
    runs = load(out, names)
    if cfg.get("route_ablation"):
        analyze_routes(runs, out, ("other", "block_direct", "block_prefix", "block_both", "mask"))
        return
    if cfg.get("modality_ablation"):
        analyze_routes(runs, out, ("other", "img_wrong", "state_wrong", "wrong", "mask"))
        return
    if cfg.get("temporal_ablation"):
        analyze_routes(runs, out, ("other", "reverse", "chunk_mean", "task_mean", "wrong", "mask"))
        return
    summaries = {n: summarize(runs[n]) for n in names}
    (out / "summary.json").write_text(json.dumps(summaries, indent=1, ensure_ascii=False))
    plot(summaries, out, cfg["times"])
    for n, s in summaries.items():
        ms = " ".join(f"{l}={v:.3f}" for l, v in zip(SEG_LABELS, s["mass_share"]))
        cs = " ".join(f"{l}={v:.3f}" for l, v in zip(SEG_LABELS, s["contrib_share"]))
        print(f"== {n}\n  mass    {ms}\n  contrib {cs}")
        print(f"  BP mass self/other/wrong {s['bp_mass_by_cond']}  img->bp {s['img_to_bp']:.4f} bp->img {s['bp_to_img']:.3f}")
        print(f"  chunk~progress rho by layer {[round(v, 2) for v in s['chunk_progress_corr_by_layer']]}")
        print(f"  loss {s['loss']}  wrong>other {s['wrong_worse_frac']:.2f} mask>other {s['mask_worse_frac']:.2f}")
