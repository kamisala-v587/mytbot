"""Figures + numbers for the evaluation report template (demo filled with the v1 baseline of BP 350k and q35).

Missing evaluations are filled by linear interpolation between the neighbouring evaluated checkpoints and marked as
fabricated (hollow markers / hatched bars, '*' in the printed tables). q35 has no offline action MSE: fabricated too.
usage: python code/eval_template_figs.py [--out <dir>]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval_350k_figs import EVAL, G9_DIR, TASKS16, task

RUNS = Path("/vla/workspace/my_tbot/outputs/BPv4CDX/stage_b")
CKPTS = (0, 20, 40, 60, 80, 100)
BEST = 100
M = {
    "BP 350k": {"pre": "b350", "color": "#009E73", "loss": RUNS / "b4_350k_sft/2026-11-02_12-04/2026-11-02_12-05/loss.log",
                "g9": "2026-11-02_12-04_{}k.json"},
    "q35": {"pre": "q35", "color": "#0072B2", "loss": RUNS / "q35_bp_baseline/2026-10-31_22-24/2026-10-31_22-25/loss.log",
            "g9": None},
}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.22, "legend.fontsize": 8,
                     "savefig.facecolor": "white"})


def rows(d: str) -> list[dict]:
    out, seen = [], set()
    for p in (EVAL / d).glob("*/episodes.jsonl"):
        for r in (json.loads(x) for x in p.read_text().splitlines() if x.strip()):
            if r.get("success") is not None and r["uid"] not in seen:
                seen.add(r["uid"])
                out.append(r)
    return out


def first_n(rs: list[dict], n: int, key: str = "task") -> list[dict]:
    """First n episodes (by plan index) per task / candidate set: the same scenes as an n-episode run."""
    by = defaultdict(list)
    for r in sorted(rs, key=lambda r: r["index"]):
        by[r[key]].append(r)
    return [r for v in by.values() for r in v[:n]]


def rate(rs: list[dict]) -> float:
    return float(np.mean([r["success"] for r in rs])) if rs else np.nan


def curve(model: str, kind: str, n: int) -> dict[int, tuple[float, int, bool]]:
    """{step_k: (success, episodes per task actually used, fabricated)} for the template checkpoints."""
    pre = M[model]["pre"] + ("train" if kind == "train" else "ood")
    real = {}
    for d in EVAL.glob(f"{pre}_c*k"):
        rs = rows(d.name)
        if kind == "train":
            rs = [r for r in rs if task(r) in TASKS16]
            if len({task(r) for r in rs}) != 16:
                continue
        elif len({task(r) for r in rs}) != 38:
            continue
        rs = first_n(rs, n)
        per_task = min(sum(1 for r in rs if r["task"] == t) for t in {r["task"] for r in rs})
        real[int(d.name.split("_c")[1][:-1])] = (rate(rs), per_task)
    xs = sorted(real)
    out = {}
    for c in CKPTS:
        if c in real:
            out[c] = (*real[c], False)
        else:
            out[c] = (float(np.interp(c, xs, [real[x][0] for x in xs])), 0, True)
    return out


def trend_fig(kind: str, n: int, title: str, fname: str) -> dict:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    res = {m: curve(m, kind, n) for m in M}
    for m, spec in M.items():
        cv = res[m]
        ax.plot(CKPTS, [cv[c][0] for c in CKPTS], color=spec["color"], lw=1.6, label=m)
        for c, (v, _, fab) in cv.items():
            other = [res[o][c][0] for o in M if o != m][0]
            ax.plot(c, v, "o", ms=6, color=spec["color"], mfc="white" if fab else spec["color"])
            # higher model labelled above the point, lower one below: labels never overlap
            ax.annotate(f"{v:.0%}" + ("*" if fab else ""), (c, v), textcoords="offset points",
                        xytext=(0, 7 if v >= other else -13), ha="center", fontsize=8, color=spec["color"])
    ax.plot([], [], "o", mfc="white", color="gray", label="* fabricated (interpolated, to be evaluated)")
    ax.set(ylim=(-0.08, 1.05), xticks=CKPTS, xticklabels=[f"{c}k" for c in CKPTS], title=title,
           xlabel="Post-training step", ylabel="Success rate")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=200)
    return res


def smooth_loss(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)[["step", "loss"]]
    df["ema"] = df.loss.ewm(span=200).mean()
    return df


def g9(model: str) -> dict[int, dict]:
    pat = M[model]["g9"]
    out = {}
    for c in CKPTS:
        p = G9_DIR / pat.format(c) if pat else None
        if p and p.is_file():
            d = json.loads(p.read_text())
            out[c] = {"train": d["by_group"]["train"]["matched_mse"], "held": d["by_group"]["heldout_sets"]["matched_mse"],
                      "blank": d["ratios"]["blank"]["ratio"], "fab": False}
    return out


def loss_fig() -> dict:
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    table = {}
    for m, spec in M.items():
        df = smooth_loss(spec["loss"])
        axs[0].plot(df.step / 1000, df.ema, color=spec["color"], lw=1.2, label=m)
        table[m] = {c: float(df[(df.step > c * 1000 - 1000) & (df.step <= c * 1000)].loss.mean()) if c else np.nan
                    for c in CKPTS}
    axs[0].set(yscale="log", title="Training loss (EMA 200)", xlabel="Post-training step (k)", ylabel="Loss")
    axs[0].legend(frameon=False)
    bp = g9("BP 350k")
    # q35: no offline action MSE yet -> fabricated as 1.2x the BP curve, only to show the layout
    q = {c: {"train": v["train"] * 1.2, "held": v["held"] * 0.9, "fab": True} for c, v in bp.items()}
    ax = axs[1]
    for m, g, col in (("BP 350k", bp, M["BP 350k"]["color"]), ("q35", q, M["q35"]["color"])):
        for grp, ls in (("train", "-"), ("held", "--")):
            xs = sorted(g)
            ys = [g[c][grp] for c in xs]
            fab = g[xs[0]]["fab"]
            ax.plot(xs, ys, ls, color=col, marker="o", mfc="white" if fab else col,
                    label=f"{m} - {'train sets' if grp == 'train' else 'held-out sets'}" + (" *fabricated" if fab else ""))
    ax.set(yscale="log", xticks=CKPTS[1:], xticklabels=[f"{c}k" for c in CKPTS[1:]], xlabel="Post-training step",
           ylabel="Action MSE (normalized delta actions)", title="Offline action MSE per checkpoint (G9 queries, matched demo)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "T1_loss_and_action_mse.png", dpi=200)
    return {"loss": table, "g9_bp": bp, "g9_q35": q}


def best_fig() -> dict:
    new = {m: first_n(rows(f"{s['pre']}ood_c{BEST}k"), 50) for m, s in M.items()}
    held = {m: rows(f"{s['pre']}held_c{BEST}k") for m, s in M.items()}
    names = list(M)
    # per task heat map: tasks sorted by the mean of both models, numbers in the cells
    tasks = sorted({task(r) for r in new[names[0]]})
    val = pd.DataFrame({m: {t: rate([r for r in new[m] if task(r) == t]) for t in tasks} for m in names})
    val["diff"] = val[names[0]] - val[names[1]]
    val = val.sort_values(names[0], ascending=False)
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 12), gridspec_kw={"width_ratios": [2, 1]}, sharey=True)
    axs[0].imshow(val[names].values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axs[1].imshow(val[["diff"]].values, cmap="PiYG", vmin=-0.5, vmax=0.5, aspect="auto")
    for i, t in enumerate(val.index):
        for j, m in enumerate(names):
            axs[0].text(j, i, f"{val.loc[t, m]:.0%}", ha="center", va="center", fontsize=8)
        axs[1].text(0, i, f"{val.loc[t, 'diff'] * 100:+.0f}", ha="center", va="center", fontsize=8)
    axs[0].set(xticks=range(2), xticklabels=names, yticks=range(len(val)), yticklabels=val.index)
    axs[1].set(xticks=[0], xticklabels=["BP - q35 (pp)"])
    for ax in axs:
        ax.grid(False)
        ax.xaxis.tick_top()
    per_task = min(sum(1 for r in new[names[0]] if r["task"] == t + "_bp") for t in tasks)
    fig.suptitle(f"New scenes, {BEST}k: success per task ({per_task} episodes each)", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "T4_best_per_task.png", dpi=200)

    cats = ("New scenes", "Held-out sets", "Held-out tasks")
    res = {m: [rate(new[m]), rate([r for r in held[m] if r["group"] == "heldout_sets"]),
               rate([r for r in held[m] if r["group"] == "heldout_tasks"])] for m in names}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x, w = np.arange(3), 0.36
    for i, m in enumerate(names):
        bars = ax.bar(x + (i - 0.5) * w, res[m], w, color=M[m]["color"], label=m)
        for b, v in zip(bars, res[m]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center", fontsize=8)
    ax.set(xticks=x, xticklabels=cats, ylim=(0, 1), ylabel="Success rate", title=f"Best checkpoint ({BEST}k) by evaluation type")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "T5_best_by_type.png", dpi=200)

    hs = {m: [r for r in held[m] if r["group"] == "heldout_sets"] for m in names}
    sets = sorted({r["candidate_set"] for r in hs[names[0]]})
    hv = pd.DataFrame({m: {s: rate([r for r in hs[m] if r["candidate_set"] == s]) for s in sets} for m in names})
    hv["diff"] = hv[names[0]] - hv[names[1]]
    hv = hv.sort_values(names[0], ascending=False)
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 6.5), gridspec_kw={"width_ratios": [2, 1]}, sharey=True)
    axs[0].imshow(hv[names].values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axs[1].imshow(hv[["diff"]].values, cmap="PiYG", vmin=-0.5, vmax=0.5, aspect="auto")
    for i, s in enumerate(hv.index):
        for j, m in enumerate(names):
            axs[0].text(j, i, f"{hv.loc[s, m]:.0%}", ha="center", va="center", fontsize=8)
        axs[1].text(0, i, f"{hv.loc[s, 'diff'] * 100:+.0f}", ha="center", va="center", fontsize=8)
    axs[0].set(xticks=range(2), xticklabels=names, yticks=range(len(hv)), yticklabels=[s.replace("_bp/", " ") for s in hv.index])
    axs[1].set(xticks=[0], xticklabels=["BP - q35 (pp)"])
    for ax in axs:
        ax.grid(False)
        ax.xaxis.tick_top()
    n_set = min(sum(1 for r in hs[names[0]] if r["candidate_set"] == s) for s in sets)
    fig.suptitle(f"Held-out sets, {BEST}k: success per candidate set ({n_set} episodes each)", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "T6_best_heldout_sets.png", dpi=200)
    return {"by_type": res, "n_new": {m: len(v) for m, v in new.items()}, "per_task": val, "held": hv,
            "n_held": {m: (len(hs[m]), sum(1 for r in held[m] if r["group"] == "heldout_tasks")) for m in names},
            "ht": {m: (sum(r["success"] for r in held[m] if r["group"] == "heldout_tasks"),
                       sum(1 for r in held[m] if r["group"] == "heldout_tasks")) for m in names}}


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/vla/workspace/my_robotwin_v2/doc/assets/eval_report_template")
    OUT = Path(ap.parse_args().out)
    OUT.mkdir(parents=True, exist_ok=True)
    lf = loss_fig()
    print("loss at ckpt (mean of the last 1k steps):", {m: {c: round(v, 4) for c, v in t.items()} for m, t in lf["loss"].items()})
    print("G9 BP:", {c: (round(v["train"], 4), round(v["held"], 4), round(v["blank"], 2)) for c, v in lf["g9_bp"].items()})
    print("G9 q35 (fabricated):", {c: (round(v["train"], 4), round(v["held"], 4)) for c, v in lf["g9_q35"].items()})
    for kind, n, title, fname in (("train", 10, "Training scenes (16 tasks x 10): success per checkpoint", "T2_train16_trend.png"),
                                  ("new", 20, "New scenes (38 tasks x 20): success per checkpoint", "T3_new_scenes_trend.png")):
        res = trend_fig(kind, n, title, fname)
        for m, cv in res.items():
            print(kind, m, {c: f"{v:.1%}{'*' if fab else ''}(n={k})" for c, (v, k, fab) in cv.items()})
    b = best_fig()
    print("best by type", {m: [f"{x:.1%}" for x in v] for m, v in b["by_type"].items()}, b["n_new"], b["n_held"], b["ht"])
    print(b["per_task"].round(3).to_string())
    print(b["held"].round(3).to_string())
    print(OUT)


if __name__ == "__main__":
    main()
