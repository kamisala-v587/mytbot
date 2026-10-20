"""Turn offline_eval json files into a markdown report (main run vs no-mask control, per checkpoint)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

CONDITIONS = ("nomask", "event", "window", "p_drop", "lang_drop", "swap_same", "swap_wrong", "event_swap")
LABELS = {
    "nomask": "不遮挡",
    "event": "事件整段遮挡",
    "window": "比例窗口遮挡",
    "p_drop": "丢弃 p",
    "lang_drop": "丢弃语言",
    "swap_same": "BP 换成同任务另一条轨迹",
    "swap_wrong": "BP 换成其他任务轨迹",
    "event_swap": "事件整段遮挡 + BP 换成同任务另一条轨迹",
}


def load(eval_dir: Path) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {}
    for f in sorted(eval_dir.glob("*.json")):
        name, _, step = f.stem.rpartition("_")
        if not step.isdigit() or name.startswith("test"):
            continue
        runs.setdefault(name, {})[int(step)] = json.loads(f.read_text())
    return runs


def table(runs: dict[str, dict[int, dict]], key: str, ratio: bool) -> list[str]:
    cols = [(name, step) for name in sorted(runs) for step in sorted(runs[name])]
    head = "| 条件 | " + " | ".join(f"{n} @{s // 1000 if s % 1000 == 0 else s / 1000}k" for n, s in cols) + " |"
    lines = [head, "| ---- |" + " ---- |" * len(cols)]
    for c in CONDITIONS:
        cells = []
        for n, s in cols:
            loss = runs[n][s][key]
            v = loss[c] if key == "loss" else loss[c]
            base = loss["nomask"]
            cells.append(f"{v / base:.3f}" if ratio else f"{v:.4f}")
        lines.append(f"| {LABELS[c]} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    eval_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    runs = load(eval_dir)
    if not runs:
        raise SystemExit(f"no eval results in {eval_dir}")
    lines = [
        "# bpv4_bpen 离线验证报告",
        "",
        "自动生成：`my_tbot/src/lerobot/policies/BPV4BPEn/summarize_eval.py`。设计见 `bpv4 第一阶段详细方案.md`。",
        "",
        "- 数据：验证集（每个 repo 末尾 clean 5 条 / rand 10 条，共 750 条轨迹 × 4 个查询），不参与训练。",
        "- 指标：flow loss，在 t = 0.1 / 0.5 / 0.9 三个时间点平均；所有 checkpoint 使用完全相同的查询、遮挡、噪声。",
        "- main = 主实验（遮挡 + p 丢弃 10%）；nomask = 对照组（不遮挡、不丢弃 p）。两组语言都丢弃 20%。",
        "",
        "## 1. flow loss 绝对值",
        "",
        *table(runs, "loss", ratio=False),
        "",
        "## 2. 相对「不遮挡」的比值",
        "",
        "比值 > 1 表示该条件下更难；「BP 换轨迹」类条件的比值越大，说明模型越依赖这条轨迹特有的信息（如物体位置）。",
        "",
        *table(runs, "loss", ratio=True),
        "",
        "## 3. 按数据来源拆分（不遮挡 / 事件整段遮挡 / BP 换成同任务另一条轨迹）",
        "",
    ]
    cols = [(n, s) for n in sorted(runs) for s in sorted(runs[n])]
    lines.append("| checkpoint | 来源 | 不遮挡 | 事件整段遮挡 | BP 换同任务轨迹 |")
    lines.append("| ---------- | ---- | ------ | ------------ | --------------- |")
    for n, s in cols:
        split = runs[n][s]["loss_by_split"]
        for src in ("clean", "rand"):
            lines.append(f"| {n} @{s} | {src} | {split['nomask'][src]:.4f} | {split['event'][src]:.4f} | {split['swap_same'][src]:.4f} |")
    lines += [
        "",
        "## 4. 主实验 vs 对照组（同一 checkpoint 步数）",
        "",
        "| 步数 | 条件 | main | nomask | main / nomask |",
        "| ---- | ---- | ---- | ------ | ------------- |",
    ]
    if "main" in runs and "nomask" in runs:
        for s in sorted(set(runs["main"]) & set(runs["nomask"])):
            for c in ("nomask", "event", "window", "p_drop", "swap_same"):
                a, b = runs["main"][s]["loss"][c], runs["nomask"][s]["loss"][c]
                lines.append(f"| {s} | {LABELS[c]} | {a:.4f} | {b:.4f} | {a / b:.3f} |")
    lines += [
        "",
        "## 5. 怎么读",
        "",
        "- 不遮挡时 main / nomask ≫ 1：遮挡训练损害了基本拟合。",
        "- 遮挡条件下 main / nomask ≪ 1：遮挡训练让模型学会了从遮挡区间以外推出动作；接近 1 则遮挡训练没有带来额外能力。",
        "- 「BP 换成同任务另一条轨迹」比值明显 > 1：模型在用该轨迹特有的信息；「换成其他任务」应更大。",
        "- 2.5k 为 VLM 开始升温之前，与之后的 checkpoint 对比可看 VLM 参与训练的影响。",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
