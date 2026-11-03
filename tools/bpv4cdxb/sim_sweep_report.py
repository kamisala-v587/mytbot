"""Summarize BP RoboTwin quick sweeps of the B-4 post-training runs on one cumulative-step axis.

usage: python tools/bpv4cdxb/sim_sweep_report.py [--out_md ...] [--out_png ...]
Reads eval_result_bp/<prefix><k>k/*/episodes.jsonl for the three runs; episodes with success = null (scene build
failures) are excluded. Writes a markdown table (episode success, task-average success, per task) and a figure.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EVAL_ROOT = Path("/vla/workspace/my_robotwin/eval_result_bp")
RUNS = (("b4sft_sft_", 0, "run 1"), ("b4full_full_", 20000, "run 2"), ("b4full2_full2_", 50000, "run 3"))
TASKS6 = ["place_shoe", "stack_blocks_two", "handover_block", "stack_bowls_two", "click_bell_alarm", "place_object_stand"]


def load() -> list[dict]:
    rows = []
    for prefix, off, run in RUNS:
        for d in EVAL_ROOT.glob(prefix + "*k"):
            m = re.fullmatch(re.escape(prefix) + r"(\d+)k", d.name)
            if not m:
                continue
            cum = int(m.group(1)) * 1000 + off
            for p in d.glob("*/episodes.jsonl"):
                for line in p.read_text().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        if r.get("success") is not None:
                            rows.append({"run": run, "cum": cum, "task": r["task"].removesuffix("_bp"),
                                         "success": bool(r["success"])})
    return rows


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--out_md", default=str(EVAL_ROOT / "_sweep16_summary.md"))
    ap.add_argument("--out_png", default="/vla/workspace/code/conclusion/bpv4CDX/figs_b4_bp_sft/sim_success_16.png")
    a = ap.parse_args()

    df = pd.DataFrame(load())
    per = df.groupby(["cum", "run", "task"]).success.agg(["sum", "count"]).reset_index()
    all_tasks = sorted(per.task.unique(), key=lambda t: (t not in TASKS6, t))
    n_tasks = per.groupby("cum").task.nunique()
    full = n_tasks[n_tasks == len(all_tasks)].index  # checkpoints that have every task
    per_full = per[per.cum.isin(full)]

    lines = [f"# B-4 BP RoboTwin 快速测评汇总（{len(all_tasks)} 个任务 × 训练场景）", "",
             "累计步数：第一轮 0–20k，第二轮 20k–50k，第三轮 50k–100k。只列出全部任务都测完的 checkpoint。", "",
             "| 累计步数 | 轮次 | 回合成功率 | 任务平均成功率 | 原 6 任务 | 新 10 任务 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    curve = []
    for cum, g in per_full.groupby("cum"):
        rate = g["sum"] / g["count"]
        old = g[g.task.isin(TASKS6)]
        new = g[~g.task.isin(TASKS6)]
        ep = g["sum"].sum() / g["count"].sum()
        curve.append({"cum": cum, "episode": ep, "task_avg": rate.mean(),
                      "old6": old["sum"].sum() / max(old["count"].sum(), 1),
                      "new10": new["sum"].sum() / new["count"].sum() if len(new) else float("nan")})
        lines.append(f"| {cum // 1000}k | {g.run.iloc[0]} | {ep:.1%} ({g['sum'].sum()}/{g['count'].sum()}) | "
                     f"{rate.mean():.1%} | {old['sum'].sum()}/{old['count'].sum()} | {new['sum'].sum()}/{new['count'].sum()} |")
    lines += ["", "## 逐任务（成功 / 回合）", ""]
    cums = sorted(full)
    lines.append("| 任务 | " + " | ".join(f"{c // 1000}k" for c in cums) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in cums) + " |")
    for t in all_tasks:
        cells = []
        for c in cums:
            r = per_full[(per_full.cum == c) & (per_full.task == t)]
            cells.append(f"{int(r['sum'].iloc[0])}/{int(r['count'].iloc[0])}" if len(r) else "—")
        lines.append(f"| `{t}`{'' if t in TASKS6 else ' *'} | " + " | ".join(cells) + " |")
    never = [t for t in all_tasks if per_full[per_full.task == t]["sum"].sum() == 0]
    lines += ["", "`*` = 2026-10-31 新增的任务。", "",
              f"始终失败（所有 checkpoint 都是 0）：{', '.join(f'`{t}`' for t in never) if never else '无'}"]
    Path(a.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    cv = pd.DataFrame(curve)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.22, "legend.fontsize": 8})
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2))
    for col, lab, c in (("episode", "episode success", "#0072B2"), ("task_avg", "task-average success", "#D55E00"),
                        ("old6", "original 6 tasks", "0.6"), ("new10", "added 10 tasks", "#009E73")):
        axs[0].plot(cv.cum / 1000, cv[col], marker="o", ms=3, color=c, label=lab, ls="--" if col in ("old6", "new10") else "-")
    eps = int(per_full["count"].median()) if len(per_full) else 0
    axs[0].set_ylim(0, 1); axs[0].set_title(f"Sim success, training scenes ({len(all_tasks)} tasks x {eps} per ckpt)")
    for t in all_tasks:
        g = per_full[per_full.task == t].sort_values("cum")
        axs[1].plot(g.cum / 1000, g["sum"] / g["count"], marker="o", ms=2.5, lw=1.1, label=t)
    axs[1].set_ylim(-0.03, 1.03); axs[1].set_title("Per task"); axs[1].legend(frameon=False, ncol=2, fontsize=6.5)
    for ax in axs:
        for x in (20, 50):
            ax.axvline(x, color="0.5", ls=":", lw=1)
        ax.set_xlabel("cumulative step (k)")
    axs[0].legend(frameon=False)
    plt.tight_layout()
    fig.savefig(a.out_png, dpi=130)
    print(f"{a.out_md}\n{a.out_png}\ncheckpoints with all {len(all_tasks)} tasks: {[c // 1000 for c in cums]}")


if __name__ == "__main__":
    main()
