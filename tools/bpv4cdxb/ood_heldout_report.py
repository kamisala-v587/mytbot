"""Summarize tools/bpv4cdxb/ood_heldout_sweep.sh next to the training-scene quick sweep of the same checkpoints.

usage: python tools/bpv4cdxb/ood_heldout_report.py [--out_md ...]
Episodes with success = null (scene build failures) are excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EVAL_ROOT = Path("/vla/workspace/my_robotwin/eval_result_bp")
# checkpoint label -> training-scene quick-sweep dir of the same checkpoint
CKPTS = {"c38k": "b4full_full_18k", "c50k": "b4full_full_30k", "c80k": "b4full2_full2_30k", "c100k": "b4full2_full2_50k"}


def episodes(d: Path) -> list[dict]:
    out = []
    for p in d.glob("*/episodes.jsonl"):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_md", default=str(EVAL_ROOT / "_ood_heldout_summary.md"))
    a = ap.parse_args()

    data = {}
    for c, train_dir in CKPTS.items():
        held = episodes(EVAL_ROOT / f"held_{c}")
        data[c] = {"train_scene": episodes(EVAL_ROOT / train_dir), "new_scene": episodes(EVAL_ROOT / f"ood_{c}"),
                   "heldout_sets": [r for r in held if r.get("group") == "heldout_sets"],
                   "heldout_tasks": [r for r in held if r.get("group") == "heldout_tasks"]}
    cs = list(CKPTS)

    def by(rows, key):
        g = {}
        for r in rows:
            g.setdefault(key(r), []).append(r)
        return g

    task = lambda r: r["task"].removesuffix("_bp")
    n_all = max(len({task(r) for r in data[c]["new_scene"]}) for c in cs)
    lines = ["# B-4 泛化测评：新场景 / 留出候选集 / 留出任务", "",
             "除“训练场景”一列外都是专家验证过的新场景（scene bank，训练没见过），训练候选集每个任务 10 回合。"
             "“16 任务”两列是同样的任务、同样的候选集，只差场景。“全部合计”= 新场景下的训练候选集（全部任务）+ 留出候选集"
             " + 留出任务。", "",
             f"| 累计步数 | 训练场景（16 任务） | 新场景（同 16 任务） | 新场景（全部 {n_all} 任务） | 留出候选集（19 个） | "
             "留出任务（2 个） | 全部合计 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in cs:
        d = data[c]
        t16 = {task(r) for r in d["train_scene"]}
        n_c = len({task(r) for r in d["new_scene"]})
        all_new = rate(d["new_scene"]) + ("" if n_c == n_all else f"（仅 {n_c} 任务）")
        total = rate(d["new_scene"] + d["heldout_sets"] + d["heldout_tasks"]) if n_c == n_all else "—"
        lines.append(f"| {c[1:]} | {rate(d['train_scene'])} | {rate([r for r in d['new_scene'] if task(r) in t16])} | "
                     f"{all_new} | {rate(d['heldout_sets'])} | {rate(d['heldout_tasks'])} | {total} |")
    lines += ["", "## 训练候选集：训练场景 vs 新场景（逐任务；训练场景只测了 16 个任务，其余为 —）", "",
              "| 任务 | " + " | ".join(f"{c[1:]} 训练 / 新" for c in cs) + " |", "| --- | " + " | ".join("---:" for _ in cs) + " |"]
    tasks = sorted({task(r) for c in cs for r in data[c]["new_scene"]})
    for t in tasks:
        cells = []
        for c in cs:
            tr, nw = by(data[c]["train_scene"], task).get(t, []), by(data[c]["new_scene"], task).get(t, [])
            cells.append(f"{frac(tr)} / {frac(nw)}")
        lines.append(f"| `{t}` | " + " | ".join(cells) + " |")

    lines += ["", "## 留出候选集（逐个）", "", "| 候选集 | " + " | ".join(c[1:] for c in cs) + " |",
              "| --- | " + " | ".join("---:" for _ in cs) + " |"]
    cset = lambda r: r.get("candidate_set") or r.get("set") or task(r)
    for s in sorted({cset(r) for c in cs for r in data[c]["heldout_sets"]}):
        lines.append(f"| `{s}` | " + " | ".join(frac(by(data[c]["heldout_sets"], cset).get(s, [])) for c in cs) + " |")

    lines += ["", "## 留出任务", "", "| 任务 | " + " | ".join(c[1:] for c in cs) + " |",
              "| --- | " + " | ".join("---:" for _ in cs) + " |"]
    for t in sorted({task(r) for c in cs for r in data[c]["heldout_tasks"]}):
        lines.append(f"| `{t}` | " + " | ".join(frac(by(data[c]["heldout_tasks"], task).get(t, [])) for c in cs) + " |")
    Path(a.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(a.out_md)


if __name__ == "__main__":
    main()
