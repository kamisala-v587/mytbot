"""BP (B-4 post-training) vs plain VLA (q35 baseline) on the same BP RoboTwin evaluations, same cumulative steps.

usage: python tools/bpv4cdxb/q35_vs_bp_report.py [--out_md ...]
Reads (eval_result_bp/):
  BP   new scenes ood_c<k>, held-out held_c<k>, training scenes (quick sweep dirs below), traces trace_c100k
  q35  new scenes q35ood_c<k>, held-out q35held_c<k>, training scenes q35train_c<k>, traces q35trace_c100k
  scene novelty: _gap_bp_c100k.jsonl / _gap_q35_c100k.jsonl (bp_tasks.eval_policy.demo_scene_gap)
Episodes with success = null (scene build failures) are excluded.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

EVAL_ROOT = Path("/vla/workspace/my_robotwin/eval_result_bp")
CKPTS = ("c50k", "c80k", "c100k")
BP_TRAIN_SCENE = {"c50k": "b4full_full_30k", "c80k": "b4full2_full2_30k", "c100k": "b4full2_full2_50k"}
MODELS = {
    "BP": {"new": "ood_{c}", "held": "held_{c}", "train": lambda c: BP_TRAIN_SCENE[c], "trace": "trace_c100k",
           "gap": "_gap_bp_c100k.jsonl"},
    "q35": {"new": "q35ood_{c}", "held": "q35held_{c}", "train": lambda c: f"q35train_{c}", "trace": "q35trace_c100k",
            "gap": "_gap_q35_c100k.jsonl"},
}
MODE_NAMES = {"knocked_distractor": "撞开干扰物", "knocked_before_grasp": "碰倒 / 挤开 / 搬运碰撞",
              "dropped_after_lift": "抓起后掉落", "placed_wrong": "放置不到位", "missed_grasp": "抓空",
              "no_attempt": "无动作"}


def episodes(d: Path) -> list[dict]:
    out = []
    for p in sorted(d.glob("*/episodes.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("success") is not None:
                    out.append(r)
    return out


def rate(rows: list[dict]) -> str:
    if not rows:
        return "—"
    s = sum(bool(r["success"]) for r in rows)
    return f"{s / len(rows):.1%} ({s}/{len(rows)})"


def frac(rows: list[dict]) -> str:
    return f"{sum(bool(r['success']) for r in rows)}/{len(rows)}" if rows else "—"


def contrast(rows: list[dict]) -> str:
    """Same anchor and same evaluated scene, different goals: all goals succeeded."""
    g = defaultdict(list)
    for r in rows:
        g[(r["anchor_id"], r["scene_source"], r["scene_index"], r.get("repeat", 0))].append(r)
    groups = [v for v in g.values() if len({r["goal"] for r in v}) >= 2]
    if not groups:
        return "—"
    ok = sum(all(r["success"] for r in v) for v in groups)
    return f"{ok / len(groups):.1%} ({ok}/{len(groups)})"


def task(r: dict) -> str:
    return r["task"].removesuffix("_bp")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_md", default=str(EVAL_ROOT / "_q35_vs_bp_summary.md"))
    a = ap.parse_args()

    data = {}
    for m, spec in MODELS.items():
        for c in CKPTS:
            held = episodes(EVAL_ROOT / spec["held"].format(c=c))
            data[(m, c)] = {"train": episodes(EVAL_ROOT / spec["train"](c)), "new": episodes(EVAL_ROOT / spec["new"].format(c=c)),
                            "hs": [r for r in held if r["group"] == "heldout_sets"],
                            "ht": [r for r in held if r["group"] == "heldout_tasks"]}

    lines = ["# BP vs 普通 VLA（q35）：BP RoboTwin 测评对比", "",
             "两个模型同样的训练数据（split v1 训练集 674 repo）、同样的累计训练帧数（每步 256 帧），测评场景与回合计划完全相同。"
             "BP 用示教指定目标，q35 用语言指令（指令里写明了目标）。", "",
             "| 模型 | 累计步数 | 训练场景（16 任务） | 新场景（同 16 任务） | 新场景（38 任务） | 新场景对比组全对 | 留出候选集 | 留出任务 | 全部合计（新场景） |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in CKPTS:
        for m in MODELS:
            d = data[(m, c)]
            t16 = {task(r) for r in d["train"]}
            lines.append(f"| {m} | {c[1:]} | {rate(d['train'])} | {rate([r for r in d['new'] if task(r) in t16])} | "
                         f"{rate(d['new'])} | {contrast(d['new'])} | {rate(d['hs'])} | {rate(d['ht'])} | "
                         f"{rate(d['new'] + d['hs'] + d['ht'])} |")

    def by_task(rows):
        g = defaultdict(list)
        for r in rows:
            g[task(r)].append(r)
        return g

    bp, q = data[("BP", "c100k")], data[("q35", "c100k")]
    lines += ["", "## 100k 逐任务（新场景，训练候选集；括号内为训练场景，只测了 16 个任务）", "",
              "| 任务 | BP | q35 | 差（q35 − BP） |", "| --- | ---: | ---: | ---: |"]
    tb, tq, sb, sq = by_task(bp["new"]), by_task(q["new"]), by_task(bp["train"]), by_task(q["train"])
    for t in sorted(set(tb) | set(tq)):
        nb, nq = sum(r["success"] for r in tb.get(t, [])), sum(r["success"] for r in tq.get(t, []))
        extra_b = f"（{frac(sb[t])}）" if t in sb else ""
        extra_q = f"（{frac(sq[t])}）" if t in sq else ""
        lines.append(f"| `{t}` | {frac(tb.get(t, []))}{extra_b} | {frac(tq.get(t, []))}{extra_q} | {nq - nb:+d} |")

    lines += ["", "## 100k 留出候选集 / 留出任务", "", "| 候选集 / 任务 | BP | q35 |", "| --- | ---: | ---: |"]
    for key, field in (("candidate_set", "hs"), ("task", "ht")):
        gb, gq = defaultdict(list), defaultdict(list)
        for r in bp[field]:
            gb[r[key]].append(r)
        for r in q[field]:
            gq[r[key]].append(r)
        for k in sorted(set(gb) | set(gq)):
            lines.append(f"| `{k}` | {frac(gb.get(k, []))} | {frac(gq.get(k, []))} |")

    lines += ["", "## 100k 失败类型（10 个任务 × 10 回合，新场景，逐步轨迹自动分类；不含按压任务 click_bell_alarm）", "",
              "| 失败类型 | BP | q35 |", "| --- | ---: | ---: |"]
    counts = {}
    for m, spec in MODELS.items():
        p = EVAL_ROOT / spec["trace"] / "_failure_modes.jsonl"
        rows = [json.loads(x) for x in p.read_text().splitlines()] if p.is_file() else []
        counts[m] = (Counter(r["mode"] for r in rows if not r["success"] and r["task"] != "click_bell_alarm"),
                     sum(r["success"] for r in rows), len(rows))
    lines.append(f"| 成功 / 回合 | {counts['BP'][1]}/{counts['BP'][2]} | {counts['q35'][1]}/{counts['q35'][2]} |")
    for k, name in MODE_NAMES.items():
        lines.append(f"| {name} | {counts['BP'][0].get(k, 0)} | {counts['q35'][0].get(k, 0)} |")

    lines += ["", "## 100k 成功率随场景新颖度（测评场景到该候选集最近训练场景的物体平均距离；新场景 + 留出候选集）", "",
              "| 距离 | BP | q35 |", "| --- | ---: | ---: |"]
    bins = [(0, .01, "≤ 1 cm"), (.01, .02, "1–2 cm"), (.02, .03, "2–3 cm"), (.03, .05, "3–5 cm"), (.05, 9, "> 5 cm")]
    gaps = {}
    for m, spec in MODELS.items():
        p = EVAL_ROOT / spec["gap"]
        gaps[m] = [json.loads(x) for x in p.read_text().splitlines()] if p.is_file() else []
    for lo, hi, name in bins:
        cells = []
        for m in MODELS:
            sel = [r for r in gaps[m] if r["group"] != "heldout_tasks" and lo < r["train_gap"] <= hi]
            cells.append(f"{sum(r['success'] for r in sel) / len(sel):.0%} ({len(sel)})" if sel else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    Path(a.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(a.out_md)


if __name__ == "__main__":
    main()
