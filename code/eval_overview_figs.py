"""Figures of the BP RoboTwin evaluation overview report (BP B-4 10k lineage vs q35 baseline).

usage: python code/eval_overview_figs.py [--out <dir>]   (default code/conclusion/bpv4CDX/figs_eval_overview)
Reads eval_result_bp/ (episodes.jsonl, _failure_modes.jsonl, _gap_*.jsonl); success = null episodes are excluded.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EVAL = Path("/vla/workspace/my_robotwin/eval_result_bp")
CKPTS = ("c50k", "c80k", "c100k")
MAIN = "c80k"  # checkpoint of the per-task / novelty figures (100k is not a formal checkpoint)
BP_TRAIN = {"c50k": "b4full_full_30k", "c80k": "b4full2_full2_30k", "c100k": "b4full2_full2_50k"}
MODELS = {"BP (B-4 10k start)": {"train": BP_TRAIN.get, "new": "ood_{}", "held": "held_{}", "trace": "trace_c100k",
                                 "gap": "_gap_bp_{}.jsonl", "color": "#D55E00"},
          "q35 (pt-15w start)": {"train": lambda c: f"q35train_{c}", "new": "q35ood_{}", "held": "q35held_{}",
                                 "trace": "q35trace_c100k", "gap": "_gap_q35_{}.jsonl", "color": "#0072B2"}}
MODES = {"knocked_distractor": "knocked distractor", "knocked_before_grasp": "knocked / pushed / hit while carrying",
         "dropped_after_lift": "dropped after lift", "placed_wrong": "placed wrong", "missed_grasp": "missed grasp"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.22, "legend.fontsize": 8,
                     "savefig.facecolor": "white"})


def eps(d: str | None) -> list[dict]:
    if not d:
        return []
    out = []
    for p in (EVAL / d).glob("*/episodes.jsonl"):
        out += [r for r in (json.loads(x) for x in p.read_text().splitlines() if x.strip()) if r.get("success") is not None]
    return out


def sr(rows: list[dict]) -> float:
    return float(np.mean([r["success"] for r in rows])) if rows else np.nan


def task(r: dict) -> str:
    return r["task"].removesuffix("_bp")


BP_DATA = Path("/vla/workspace/my_robotwin/data_bp/bp_robotwin_lerobot")
# (title, [(candidate set, column label, held out?)]): a held-out set next to training sets of the same task
HELDOUT_EXAMPLES = [
    ("place_a2b: toy car / tea box", [
        ("place_a2b_bp/I840/L0/A_left_of_B", "HELD OUT\ncar LEFT of tea box", True),
        ("place_a2b_bp/I840/L0/A_right_of_B", "train: same objects & layout\ncar RIGHT of tea box", False),
        ("place_a2b_bp/I1015/L1/A_left_of_B", "train: same goal, other objects\nbread LEFT of stapler", False)]),
    ("place_shoe: shoe #1 on the mat", [
        ("place_shoe_bp/I1/L1/toe_left", "HELD OUT\nshoe 1, layout 1, toe LEFT", True),
        ("place_shoe_bp/I1/L1/toe_right", "train: same shoe & layout\ntoe RIGHT", False),
        ("place_shoe_bp/I1/L2/toe_left", "train: same shoe & goal\nlayout 2, toe LEFT", False)]),
]


def fig_heldout_examples(out: Path) -> None:
    """First (initial layout) and last (goal reached) head-camera frame of episode 0 of each candidate set."""
    import sys

    sys.path.insert(0, "/vla/workspace/my_robotwin")
    from bp_tasks.eval_policy.compare_videos import demo_frames

    w, h = 320, 240
    fig, axs = plt.subplots(2 * len(HELDOUT_EXAMPLES), 3, figsize=(11.5, 4.3 * len(HELDOUT_EXAMPLES)))
    for e, (title, cols) in enumerate(HELDOUT_EXAMPLES):
        for c, (cs, label, held) in enumerate(cols):
            repo = BP_DATA / cs.split("/")[0] / "_".join(cs.split("/")[1:])
            frames = demo_frames(str(repo), 0, w, h, 2000)
            for r, (frame, which) in enumerate(((frames[0], "start"), (frames[-1], "end"))):
                ax = axs[2 * e + r, c]
                ax.imshow(frame[:, :, ::-1])
                ax.set_xticks([]), ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(True), s.set_color("#D55E00" if held else "0.6"), s.set_linewidth(3 if held else 1)
                if r == 0:
                    ax.set_title(label, fontsize=9, color="#D55E00" if held else "0.15")
                if c == 0:
                    ax.set_ylabel(f"{title}\n{which}" if r == 0 else which, fontsize=9)
    fig.suptitle("Held-out candidate sets vs training candidate sets (episode 0: first / last frame)", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "F0_heldout_examples.png", dpi=170, bbox_inches="tight")


def fig_bp_training_curve(out: Path) -> None:
    pts = []
    for prefix, off in (("b4sft_sft_", 0), ("b4full_full_", 20000), ("b4full2_full2_", 50000)):
        for d in EVAL.glob(prefix + "*k"):
            m = re.fullmatch(re.escape(prefix) + r"(\d+)k", d.name)
            rows = eps(d.name)
            if m and len({task(r) for r in rows}) == 16:
                pts.append((int(m.group(1)) * 1000 + off, sr(rows)))
    pts.sort()
    q35 = []
    for c in CKPTS:
        rows = eps(f"q35train_{c}")
        if len({task(r) for r in rows}) == 16:
            q35.append((int(c[1:-1]) * 1000, sr(rows)))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot([p[0] / 1000 for p in pts], [p[1] for p in pts], marker="o", color="#D55E00", label="BP (B-4 10k start)")
    ax.plot([p[0] / 1000 for p in q35], [p[1] for p in q35], marker="s", color="#0072B2", label="q35 (pt-15w start)")
    for x in (20, 50):
        ax.axvline(x, color="0.5", ls=":", lw=1)
    ax.set(ylim=(0, 1), title="Success on training scenes during post-training (16 tasks x 10)",
           xlabel="Cumulative step (k)", ylabel="Success rate")
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
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    x = np.arange(len(cats))
    for ax, c in zip(axs, CKPTS):
        for i, (m, spec) in enumerate(MODELS.items()):
            v = res[(m, c)]
            bars = ax.bar(x + (i - 0.5) * 0.38, v, 0.38, color=spec["color"], label=m)
            for b, val in zip(bars, v):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.01, f"{val:.0%}", ha="center", fontsize=7.5)
        ax.set(xticks=x, xticklabels=cats, ylim=(0, 1), title=f"{c[1:]} steps")
    axs[0].set_ylabel("Success rate")
    axs[0].legend(frameon=False, loc="upper right")
    fig.suptitle("BP vs q35: success by evaluation type (same data, same training frames)", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "F2_bp_vs_q35_by_type.png", dpi=200, bbox_inches="tight")
    return res


def fig_per_task(out: Path) -> None:
    rows = {m: eps(spec["new"].format(MAIN)) for m, spec in MODELS.items()}
    names = list(MODELS)
    tasks = sorted({task(r) for r in rows[names[0]]})
    val = {m: {t: sr([r for r in rows[m] if task(r) == t]) for t in tasks} for m in names}
    tasks.sort(key=lambda t: -val[names[1]][t] - val[names[0]][t])  # easiest (both models) first
    fig, ax = plt.subplots(figsize=(16, 4.8))
    x = np.arange(len(tasks))
    for i, m in enumerate(names):
        ax.bar(x + (i - 0.5) * 0.4, [val[m][t] for t in tasks], 0.4, color=MODELS[m]["color"], label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=60, ha="right", fontsize=8)
    ax.set(ylim=(0, 1.05), ylabel="Success rate (10 episodes)", xlim=(-0.7, len(tasks) - 0.3),
           title=f"New scenes, {MAIN[1:]}: success per task")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / f"F3_per_task_new_scenes_{MAIN[1:]}.png", dpi=200)


def fig_failures(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    left = {m: 0 for m in MODELS}
    colors = ("#CC79A7", "#D55E00", "#E69F00", "#56B4E9", "#999999")
    for (k, name), col in zip(MODES.items(), colors):
        vals = []
        for m, spec in MODELS.items():
            rs = [json.loads(x) for x in (EVAL / spec["trace"] / "_failure_modes.jsonl").read_text().splitlines()]
            vals.append(sum(1 for r in rs if not r["success"] and r["task"] != "click_bell_alarm" and r["mode"] == k))
        ax.barh(list(MODELS), vals, left=[left[m] for m in MODELS], color=col, label=name)
        for m, v in zip(MODELS, vals):
            left[m] += v
    for m, spec in MODELS.items():
        rs = [json.loads(x) for x in (EVAL / spec["trace"] / "_failure_modes.jsonl").read_text().splitlines()]
        ax.text(left[m] + 1, list(MODELS).index(m), f"success {sum(r['success'] for r in rs)}/100", va="center", fontsize=8)
    ax.set(xlabel="Failed episodes (10 tasks x 10, click_bell_alarm excluded)", title="Failure modes at 100k (traced rollouts)")
    ax.set_xlim(0, max(left.values()) + 18)
    ax.legend(frameon=False, ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(out / "F4_failure_modes_100k.png", dpi=200, bbox_inches="tight")


def fig_novelty(out: Path) -> None:
    bins = [(0, .01, "<=1"), (.01, .02, "1-2"), (.02, .03, "2-3"), (.03, .05, "3-5"), (.05, 9, ">5")]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for m, spec in MODELS.items():
        g = [json.loads(x) for x in (EVAL / spec["gap"].format(MAIN)).read_text().splitlines()]
        g = [r for r in g if r["group"] != "heldout_tasks"]
        ys = [sr([r for r in g if lo < r["train_gap"] <= hi]) for lo, hi, _ in bins]
        ax.plot([b[2] for b in bins], ys, marker="o", color=spec["color"], label=m)
    ax.set(ylim=(0, 1), xlabel="Distance of the scene to the nearest training scene (cm, mean over objects)",
           ylabel="Success rate", title=f"{MAIN[1:]}: success vs scene novelty (new scenes + held-out sets)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"F5_success_vs_novelty_{MAIN[1:]}.png", dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/vla/workspace/code/conclusion/bpv4CDX/figs_eval_overview")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_heldout_examples(out)
    fig_bp_training_curve(out)
    res = fig_bars(out)
    fig_per_task(out)
    fig_failures(out)
    fig_novelty(out)
    for k, v in res.items():
        print(k, [f"{x:.1%}" for x in v])
    print(out)


if __name__ == "__main__":
    main()
