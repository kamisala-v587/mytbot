"""Figures of the BP RoboTwin evaluation report with the B-4 350k post-training as the main line.

Same figure set as eval_overview_figs.py (v0 report), three models: BP (B-4 350k start, main), BP (B-4 10k start) and
q35 (pt-15w start). No failure-mode figure (traced rollouts of the 350k run not done yet).
usage: python code/eval_350k_figs.py [--out <dir>]   (default code/conclusion/bpv4CDX/figs_eval_350k)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVAL = Path("/vla/workspace/my_robotwin/eval_result_bp")
G9_DIR = Path("/vla/workspace/my_tbot/outputs/BPv4CDX/stage_b/g9_bp")
CKPTS = ("c50k", "c80k", "c100k")
MAIN = "c100k"
BP10_TRAIN = {"c50k": "b4full_full_30k", "c80k": "b4full2_full2_30k", "c100k": "b4full2_full2_50k"}
MODELS = {
    "BP (B-4 350k start)": {"train": lambda c: f"b350train_{c}", "new": "b350ood_{}", "held": "b350held_{}",
                            "gap": "_gap_b350_{}.jsonl", "color": "#009E73"},
    "BP (B-4 10k start)": {"train": BP10_TRAIN.get, "new": "ood_{}", "held": "held_{}", "gap": "_gap_bp_{}.jsonl",
                           "color": "#D55E00"},
    "q35 (pt-15w start)": {"train": lambda c: f"q35train_{c}", "new": "q35ood_{}", "held": "q35held_{}",
                           "gap": "_gap_q35_{}.jsonl", "color": "#0072B2"},
}
G9_RUNS = {  # (name, glob, cumulative offset, color)
    "BP (B-4 10k start)": [("2026-10-29_18-26_*k.json", 0), ("2026-10-30_02-46_*k.json", 20000), ("2026-10-30_13-45_*k.json", 50000)],
    "BP (B-4 350k start)": [("2026-11-02_12-04_*k.json", 0)],
}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.22, "legend.fontsize": 8,
                     "savefig.facecolor": "white"})


def eps(d: str | None) -> list[dict]:
    if not d or not (EVAL / d).is_dir():
        return []
    out, seen = [], set()
    for p in (EVAL / d).glob("*/episodes.jsonl"):
        for r in (json.loads(x) for x in p.read_text().splitlines() if x.strip()):
            # helper processes may have run an episode twice: keep the first record per uid
            if r.get("success") is not None and r["uid"] not in seen:
                seen.add(r["uid"])
                out.append(r)
    return out


def sr(rows: list[dict]) -> float:
    return float(np.mean([r["success"] for r in rows])) if rows else np.nan


def task(r: dict) -> str:
    return r["task"].removesuffix("_bp")


def fig_training_curve(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    pts = []
    for prefix, off in (("b4sft_sft_", 0), ("b4full_full_", 20000), ("b4full2_full2_", 50000)):
        for d in EVAL.glob(prefix + "*k"):
            m = re.fullmatch(re.escape(prefix) + r"(\d+)k", d.name)
            rows = eps(d.name)
            if m and len({task(r) for r in rows}) == 16:
                pts.append((int(m.group(1)) * 1000 + off, sr(rows)))
    curves = {"BP (B-4 10k start)": sorted(pts)}
    for m, prefix in (("BP (B-4 350k start)", "b350train_c"), ("q35 (pt-15w start)", "q35train_c")):
        p = []
        for d in EVAL.glob(prefix + "*k"):
            rows = eps(d.name)
            if len(rows) >= 100:  # 350k sweep may be partial (>= 100 of 160 episodes)
                p.append((int(d.name[len(prefix):-1]) * 1000, sr(rows)))
        curves[m] = sorted(p)
    for m, p in curves.items():
        ax.plot([x[0] / 1000 for x in p], [x[1] for x in p], marker="o", ms=4, color=MODELS[m]["color"], label=m)
    ax.set(ylim=(0, 1), title="Success on training scenes during post-training (16 tasks x 10)",
           xlabel="Post-training step (k; BP 10k: cumulative over 3 stages)", ylabel="Success rate")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "F1_training_scene_curve.png", dpi=200)


def fig_bars(out: Path) -> dict:
    cats = ("training scenes\n(16 tasks)", "new scenes\n(38 tasks)", "held-out sets\n(19)", "held-out tasks\n(2)")
    res = {}
    for m, spec in MODELS.items():
        for c in CKPTS:
            held = eps(spec["held"].format(c))
            res[(m, c)] = [sr(eps(spec["train"](c))), sr(eps(spec["new"].format(c))),
                           sr([r for r in held if r["group"] == "heldout_sets"]),
                           sr([r for r in held if r["group"] == "heldout_tasks"])]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
    x, w = np.arange(len(cats)), 0.27
    for ax, c in zip(axs, CKPTS):
        for i, (m, spec) in enumerate(MODELS.items()):
            v = res[(m, c)]
            bars = ax.bar(x + (i - 1) * w, v, w, color=spec["color"], label=m)
            for b, val in zip(bars, v):
                if np.isfinite(val):
                    ax.text(b.get_x() + b.get_width() / 2, val + 0.01, f"{val:.0%}", ha="center", fontsize=7)
        ax.set(xticks=x, xticklabels=cats, ylim=(0, 1), title=f"{c[1:]} steps")
    axs[0].set_ylabel("Success rate")
    axs[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Success by evaluation type (same post-training data and training frames)", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "F2_by_type.png", dpi=200, bbox_inches="tight")
    return res


def fig_per_task(out: Path) -> None:
    rows = {m: eps(spec["new"].format(MAIN)) for m, spec in MODELS.items()}
    names = list(MODELS)
    tasks = sorted({task(r) for r in rows[names[0]]} or {task(r) for r in rows[names[1]]})
    val = {m: {t: sr([r for r in rows[m] if task(r) == t]) for t in tasks} for m in names}
    tasks.sort(key=lambda t: -np.nansum([val[m][t] for m in names]))
    fig, ax = plt.subplots(figsize=(17, 4.8))
    x, w = np.arange(len(tasks)), 0.28
    for i, m in enumerate(names):
        ax.bar(x + (i - 1) * w, [val[m][t] for t in tasks], w, color=MODELS[m]["color"], label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=60, ha="right", fontsize=8)
    ax.set(ylim=(0, 1.05), xlim=(-0.7, len(tasks) - 0.3), ylabel="Success rate (10 episodes)",
           title=f"New scenes, {MAIN[1:]}: success per task")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / f"F3_per_task_new_scenes_{MAIN[1:]}.png", dpi=200)


def fig_novelty(out: Path) -> dict:
    bins = [(0, .01, "<=1"), (.01, .02, "1-2"), (.02, .03, "2-3"), (.03, .05, "3-5"), (.05, 9, ">5")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    table = {}
    for m, spec in MODELS.items():
        p = EVAL / spec["gap"].format(MAIN)
        if not p.is_file():
            continue
        g, seen = [], set()
        for r in (json.loads(x) for x in p.read_text().splitlines()):
            if r["group"] != "heldout_tasks" and (r["eval"], r["uid"]) not in seen:  # helper reruns: first record only
                seen.add((r["eval"], r["uid"]))
                g.append(r)
        ys = [sr([r for r in g if lo < r["train_gap"] <= hi]) for lo, hi, _ in bins]
        table[m] = (ys, [sum(1 for r in g if lo < r["train_gap"] <= hi) for lo, hi, _ in bins])
        ax.plot([b[2] for b in bins], ys, marker="o", color=spec["color"], label=m)
    ax.set(ylim=(0, 1), xlabel="Distance of the scene to the nearest training scene (cm, mean over objects)",
           ylabel="Success rate", title=f"{MAIN[1:]}: success vs scene novelty (new scenes + held-out sets)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"F5_success_vs_novelty_{MAIN[1:]}.png", dpi=200)
    return table


TRACES = {"BP (B-4 350k start)": "b350trace_{}", "BP (B-4 10k start)": "trace_{}", "q35 (pt-15w start)": "q35trace_{}"}
MODES = {"knocked_distractor": "knocked distractor", "knocked_before_grasp": "knocked / pushed / hit while carrying",
         "dropped_after_lift": "dropped after lift", "placed_wrong": "placed wrong", "missed_grasp": "missed grasp"}


def fig_failures(out: Path) -> dict:
    """Failure modes from traced rollouts (bp_tasks.eval_policy.failure_modes), 80k and 100k panels."""
    colors = ("#CC79A7", "#D55E00", "#E69F00", "#56B4E9", "#999999")
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.2), sharex=True)
    table = {}
    for ax, c in zip(axs, ("c80k", "c100k")):
        names, left = [], {}
        for m, pat in TRACES.items():
            p = EVAL / pat.format(c) / "_failure_modes.jsonl"
            if not p.is_file():
                continue
            rows = [json.loads(x) for x in p.read_text().splitlines()]
            names.append(m)
            fails = [r for r in rows if not r["success"] and r["task"] != "click_bell_alarm"]
            table[(m, c)] = ({k: sum(r["mode"] == k for r in fails) for k in MODES}, sum(r["success"] for r in rows), len(rows))
            left[m] = 0
        for (k, label), col in zip(MODES.items(), colors):
            vals = [table[(m, c)][0][k] for m in names]
            ax.barh(names, vals, left=[left[m] for m in names], color=col, label=label)
            for m, v in zip(names, vals):
                left[m] += v
        for i, m in enumerate(names):
            ax.text(left[m] + 1, i, f"success {table[(m, c)][1]}/{table[(m, c)][2]}", va="center", fontsize=8)
        ax.set(title=f"Failure modes at {c[1:]} (traced rollouts)", xlabel="Failed episodes (10 tasks x 10, click_bell_alarm excluded)")
    axs[0].set_xlim(0, 85)
    axs[0].legend(frameon=False, ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(1.05, -0.2))
    fig.tight_layout()
    fig.savefig(out / "F4_failure_modes.png", dpi=200, bbox_inches="tight")
    return table


def g9_table() -> pd.DataFrame:
    rows = []
    for m, runs in G9_RUNS.items():
        for pattern, off in runs:
            for p in G9_DIR.glob(pattern):
                d = json.loads(p.read_text())
                if int(d.get("queries", 0)) != 444:
                    continue
                step = int(Path(d["ckpt"]).parent.name)
                row = {"model": m, "step": step + off, "matched": d["mean_error"]["matched"]}
                for c in ("blank", "alt", "wrong", "goal"):
                    r = d["ratios"].get(c)
                    if r:
                        row.update({f"{c}": r["ratio"], f"{c}_lo": r["ci95"][0], f"{c}_hi": r["ci95"][1]})
                for g, e in d["by_group"].items():
                    row[f"matched_{g}"] = e["matched_mse"]
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "step"])


def fig_g9(out: Path) -> pd.DataFrame:
    df = g9_table()
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.3), sharey=True)
    for ax, (g, title) in zip(axs, (("train", "train sets"), ("heldout_sets", "held-out sets"), ("heldout_tasks", "held-out tasks"))):
        for m in G9_RUNS:
            s = df[df.model == m]
            ax.plot(s.step / 1000, s[f"matched_{g}"], marker="o", ms=3, color=MODELS[m]["color"], label=m)
        ax.set(yscale="log", title=f"G9 matched action error: {title}", xlabel="Post-training step (k)")
    axs[0].set_ylabel("MSE, normalized delta actions (lower = better)")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "C2_matched_by_group.png", dpi=200)
    conds = ("blank", "alt", "wrong", "goal")
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.3))
    for ax, c in zip(axs, conds):
        for m in G9_RUNS:
            s = df[df.model == m]
            ax.plot(s.step / 1000, s[c], marker="o", ms=3, color=MODELS[m]["color"], label=m)
            ax.fill_between(s.step / 1000, s[f"{c}_lo"], s[f"{c}_hi"], color=MODELS[m]["color"], alpha=0.15, lw=0)
        ax.axhline(1, color="k", lw=0.8, ls="--")
        ax.set(title=f"{c} / matched error ratio", xlabel="Post-training step (k)", ylabel="Ratio (> 1: the matched demo helps)")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "C3_g9_ratios.png", dpi=200)
    df.to_csv(out / "g9_two_bp_runs.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/vla/workspace/code/conclusion/bpv4CDX/figs_eval_350k")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_training_curve(out)
    res = fig_bars(out)
    fig_per_task(out)
    nov = fig_novelty(out)
    fails = fig_failures(out)
    for (m, c), (modes, ok, n) in sorted(fails.items(), key=lambda kv: kv[0][1]):
        print("failures", c, m, f"success {ok}/{n}", modes)
    g9 = fig_g9(out)
    for k, v in res.items():
        print(k, [f"{x:.1%}" if np.isfinite(x) else "-" for x in v])
    for m, (ys, ns) in nov.items():
        print("novelty", m, [f"{y:.0%}" for y in ys], ns)
    print(g9[g9.model == "BP (B-4 350k start)"][["step", "matched", "matched_train", "matched_heldout_sets",
                                                  "matched_heldout_tasks", "blank", "goal"]].round(3).to_string(index=False))
    print(out)


if __name__ == "__main__":
    main()
