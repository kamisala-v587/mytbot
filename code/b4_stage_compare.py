"""Cross-stage comparison of the B-4 post-training on BP RoboTwin (B-4 SFT -> B-4 SFT full -> B-4 SFT full2), 0-100k.

Same figures as the "跨阶段对比" section of b4_sft_eval.ipynb, extended to the third stage:
  C1_train_loss   training flow loss (rank-0 loss.log, trailing 1,000-step mean)
  C2_matched      G9 matched action error: all 444 queries, and per split group
  C3_g9_ratios    G9 blank / alt / wrong / goal error ratios with the within-checkpoint 95% CI
Only the 444-query G9 index (gate_g9_bp) is used. x = stage step + stage start (cumulative step).
usage: python code/b4_stage_compare.py [--out <dir>]   (default outputs/BPv4CDX/stage_b/reports/stage_compare_100k)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STAGE_B = Path("/vla/workspace/my_tbot/outputs/BPv4CDX/stage_b")
G9_DIR = STAGE_B / "g9_bp"
STAGES = [
    {"name": "B-4 SFT", "run": STAGE_B / "b4_sft/2026-10-29_18-26/2026-10-29_18-27", "g9": "2026-10-29_18-26_*k.json", "start": 0,
     "color": "#0072B2"},
    {"name": "B-4 SFT full", "run": STAGE_B / "b4_sft_full/2026-10-30_02-46/2026-10-30_02-47", "g9": "2026-10-30_02-46_*k.json",
     "start": 20000, "color": "#D55E00"},
    {"name": "B-4 SFT full2", "run": STAGE_B / "b4_sft_full2/2026-10-30_13-45/2026-10-30_13-47", "g9": "2026-10-30_13-45_*k.json",
     "start": 50000, "color": "#009E73"},
]
CONDITIONS = ("blank", "alt", "wrong", "goal")
GROUPS = {"train": "train sets", "heldout_sets": "held-out sets", "heldout_tasks": "held-out tasks"}
XLABEL = "Cumulative step (k)"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.22, "legend.fontsize": 8,
                     "savefig.facecolor": "white"})


def load_g9(stage: dict) -> pd.DataFrame:
    rows = []
    for p in sorted(G9_DIR.glob(stage["g9"])):
        d = json.loads(p.read_text())
        if int(d.get("queries", 0)) != 444:
            continue
        step = int(Path(d["ckpt"]).parent.name)
        row = {"stage": stage["name"], "step": step, "cum": step + stage["start"], "matched": d["mean_error"]["matched"]}
        for c in CONDITIONS:
            r = d["ratios"].get(c)
            if r:
                row.update({f"{c}_ratio": r["ratio"], f"{c}_lo": r["ci95"][0], f"{c}_hi": r["ci95"][1]})
        for g, e in d["by_group"].items():
            row[f"matched_{g}"] = e["matched_mse"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cum") if rows else pd.DataFrame()


def load_loss(stage: dict) -> pd.DataFrame:
    df = pd.read_csv(stage["run"] / "loss.log", usecols=["step", "loss"]).dropna()
    df = df.drop_duplicates("step", keep="last").sort_values("step")
    freq = int(np.median(np.diff(df["step"].to_numpy())))
    df["smooth"] = df["loss"].rolling(max(1, 1000 // freq), min_periods=1).mean()
    df["cum"] = df["step"] + stage["start"]
    return df


def stage_lines(ax) -> None:
    for s in STAGES[1:]:
        ax.axvline(s["start"] / 1000, color=s["color"], ls=":", lw=1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STAGE_B / "reports" / "stage_compare_100k"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    g9 = {s["name"]: load_g9(s) for s in STAGES}
    loss = {s["name"]: load_loss(s) for s in STAGES}

    # C1 training loss
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for s in STAGES:
        df = loss[s["name"]]
        ax.plot(df["cum"] / 1000, df["loss"], color=s["color"], lw=0.5, alpha=0.2)
        ax.plot(df["cum"] / 1000, df["smooth"], color=s["color"], lw=1.8, label=f"{s['name']} (1k-step mean)")
    stage_lines(ax)
    ax.set_yscale("log")
    ax.set(title="Training flow loss across stages", xlabel=XLABEL, ylabel="Flow loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "C1_train_loss.png", dpi=200)

    # C2 matched error: all queries + per group
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4))
    for s in STAGES:
        df = g9[s["name"]]
        if df.empty:
            continue
        axs[0].plot(df["cum"] / 1000, df["matched"], marker="o", ms=3.5, color=s["color"], label=s["name"])
    for (g, label), ls in zip(GROUPS.items(), ("-", "--", ":")):
        df = pd.concat([g9[s["name"]] for s in STAGES]).sort_values("cum")
        axs[1].plot(df["cum"] / 1000, df[f"matched_{g}"], marker="o", ms=3, ls=ls, color="0.25", label=label)
    for ax, title in zip(axs, ("G9 matched action error (444 queries)", "G9 matched action error by split group")):
        stage_lines(ax)
        ax.set_yscale("log")
        ax.set(title=title, xlabel=XLABEL, ylabel="MSE, normalized delta actions (lower = better)")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "C2_matched.png", dpi=200)

    # C3 ratios with CI
    fig, axs = plt.subplots(1, len(CONDITIONS), figsize=(5.2 * len(CONDITIONS), 4.4))
    for ax, c in zip(axs, CONDITIONS):
        for s in STAGES:
            df = g9[s["name"]]
            if df.empty or f"{c}_ratio" not in df:
                continue
            x = df["cum"] / 1000
            ax.plot(x, df[f"{c}_ratio"], marker="o", ms=3.5, color=s["color"], label=s["name"])
            ax.fill_between(x, df[f"{c}_lo"], df[f"{c}_hi"], color=s["color"], alpha=0.15, lw=0)
        ax.axhline(1, color="k", lw=0.8, ls="--")
        stage_lines(ax)
        ax.set(title=f"{c} / matched error ratio", xlabel=XLABEL, ylabel="Ratio (> 1: the matched demo helps)")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "C3_g9_ratios.png", dpi=200)

    table = pd.concat(g9.values())
    table.to_csv(out / "g9_cross_stage.csv", index=False)
    print(table[["stage", "cum", "matched", "matched_train", "matched_heldout_sets", "matched_heldout_tasks",
                 "blank_ratio", "goal_ratio"]].round(3).to_string(index=False))
    print(out)


if __name__ == "__main__":
    main()
