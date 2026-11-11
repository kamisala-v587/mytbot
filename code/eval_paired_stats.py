"""Paired comparison BP 350k vs q35 at 100k on identical scenes (same episode uid), per evaluation type.

For each type: paired episodes, both success rates, difference, discordant pairs (only BP / only q35), exact McNemar
p-value, and a 95% CI of the difference from a task-cluster bootstrap (tasks resampled with replacement), which
accounts for episodes of one task being correlated. Per task: paired counts.
usage: python code/eval_paired_stats.py [--out_md ...]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

EVAL = Path("/vla/workspace/my_robotwin/eval_result_bp/v1_baseline")
TYPES = {"训练场景": ("b350train_c100k", "q35train_c100k", "train"),
         "新场景": ("b350ood_c100k", "q35ood_c100k", "train"),
         "留出候选集": ("b350held_c100k", "q35held_c100k", "heldout_sets")}


def load(d: str, group: str) -> dict[str, dict]:
    out = {}
    for p in sorted((EVAL / d).glob("*/episodes.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("success") is None or r.get("group") != group or r["uid"] in out:
                continue  # first record per uid (helper reruns)
            out[r["uid"]] = r
    return out


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_md", default=str(EVAL / "_paired_bp350k_vs_q35_100k.md"))
    ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    lines = ["# BP 350k vs q35（100k）配对比较：同一批场景逐个配对", "",
             "差 = BP 350k − q35（百分点）。CI = 按任务重抽样的 95% 置信区间；p = 精确 McNemar 检验（只看结果不一致的场景）。", "",
             "| 测评 | 配对回合 | BP 350k | q35 | 差 | 95% CI | 只有 BP 成功 | 只有 q35 成功 | p |",
             "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |"]
    per_task_all = {}
    for name, (da, dq, group) in TYPES.items():
        A, Q = load(da, group), load(dq, group)
        uids = sorted(set(A) & set(Q))
        if not uids:
            lines.append(f"| {name} | 0 | — | — | — | — | — | — | — |")
            continue
        sa = np.array([A[u]["success"] for u in uids], float)
        sq = np.array([Q[u]["success"] for u in uids], float)
        key = "candidate_set" if group == "heldout_sets" else "task"
        tasks = defaultdict(list)
        for i, u in enumerate(uids):
            tasks[A[u][key]].append(i)
        groups = list(tasks.values())
        diffs = []
        for _ in range(a.boot):
            idx = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
            diffs.append(sa[idx].mean() - sq[idx].mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5]) * 100
        b, c = int(((sa == 1) & (sq == 0)).sum()), int(((sa == 0) & (sq == 1)).sum())
        lines.append(f"| {name} | {len(uids)} | {sa.mean():.1%} | {sq.mean():.1%} | {(sa.mean() - sq.mean()) * 100:+.1f} | "
                     f"[{lo:+.1f}, {hi:+.1f}] | {b} | {c} | {mcnemar_exact(b, c):.3f} |")
        per_task_all[name] = {t.removesuffix("_bp") if key == "task" else t: (int(sa[ix].sum()), int(sq[ix].sum()), len(ix))
                              for t, ix in tasks.items()}
    for name, pt in per_task_all.items():
        lines += ["", f"## {name}：逐{'候选集' if name == '留出候选集' else '任务'}（成功 / 回合）", "",
                  "| 名称 | BP 350k | q35 | 差 |", "| --- | ---: | ---: | ---: |"]
        for t, (x, y, n) in sorted(pt.items(), key=lambda kv: (kv[1][0] - kv[1][1], kv[0])):
            lines.append(f"| `{t}` | {x}/{n} | {y}/{n} | {x - y:+d} |")
    Path(a.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:9]))
    print(a.out_md)


if __name__ == "__main__":
    main()
