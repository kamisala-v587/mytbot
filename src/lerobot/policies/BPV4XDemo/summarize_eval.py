"""Markdown tables from bpv4_xdemo offline_eval jsons (<dir>/main_<step>.json).

Usage: python -m lerobot.policies.BPV4XDemo.summarize_eval <eval_dir> <out.md>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LABELS = {
    "normal": "正常（同任务另一条示教）",
    "swap_same": "换成同任务第三条示教（应接近 1）",
    "swap_wrong": "换成其他任务示教（应明显 > 1）",
    "no_bp": "去掉 BP（应明显 > 1）",
    "no_image": "去掉当前图像（应明显 > 1）",
    "scene_swap": "同场景异任务示教（shake_bottle ↔ horizontally，应 > 1）",
}


def main() -> None:
    eval_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    runs = {}
    for f in eval_dir.glob("*.json"):
        name, _, step = f.stem.rpartition("_")
        if step.isdigit():
            runs[(name, int(step))] = json.loads(f.read_text())
    cols = sorted(runs)
    head = "| 条件 | " + " | ".join(f"{n} @{s // 1000}k" for n, s in cols) + " |"
    sep = "| ---- |" + " ---- |" * len(cols)
    lines = ["# bpv4_xdemo 离线评测", "", "验证集：当前轨迹与示教都来自验证 episode；查询、示教、噪声、flow 时间固定。", "",
             "## flow loss 绝对值", "", head, sep]
    for c in LABELS:
        lines.append(f"| {LABELS[c]} | " + " | ".join(f"{runs[k]['loss'].get(c, float('nan')):.4f}" for k in cols) + " |")
    lines += ["", "## 相对「正常」的比值", "", head, sep]
    for c in LABELS:
        if c == "normal":
            continue
        lines.append(f"| {LABELS[c]} | " + " | ".join(f"{runs[k]['ratio_to_normal'][c]:.3f}" for k in cols) + " |")
    lines += ["", "## 同手 / 异手示教（正常条件）", "", head, sep]
    for arm, label in (("same_arm", "同手示教"), ("diff_arm", "异手示教")):
        lines.append(f"| {label} | " + " | ".join(
            f"{runs[k]['loss_by_arm'][arm]:.4f} (n={runs[k]['count_by_arm'][arm]})" for k in cols) + " |")
    lines += ["", "## clean / rand（正常条件）", "", head, sep]
    for split in ("clean", "rand"):
        lines.append(f"| {split} | " + " | ".join(f"{runs[k]['loss_by_split']['normal'][split]:.4f}" for k in cols) + " |")
    if cols:
        last = runs[cols[-1]]["loss_by_task"]
        lines += ["", f"## 各任务（{cols[-1][0]} @{cols[-1][1] // 1000}k，正常 / 换其他任务 / 去掉 BP）", "",
                  "| 任务 | 正常 | 换其他任务 | 去掉 BP |", "| ---- | ---- | ---- | ---- |"]
        for task, d in last.items():
            lines.append(f"| {task} | {d.get('normal', float('nan')):.4f} | {d.get('swap_wrong', float('nan')):.4f} | "
                         f"{d.get('no_bp', float('nan')):.4f} |")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
