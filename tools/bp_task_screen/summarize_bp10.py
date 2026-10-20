"""Summarize BP test-set (10 tasks) closed-loop results: same-arm (best50) vs cross-arm (worst50) demos, plus the
earlier fixed-demo runs on overlapping tasks."""
import re
from pathlib import Path

ROOT = Path("/vla/workspace/my_robotwin/eval_result")
TASKS = ["place_cans_plasticbox", "scan_object", "place_shoe", "rotate_qrcode", "place_mouse_pad", "place_empty_cup",
         "move_can_pot", "place_phone_stand", "handover_mic", "place_can_basket"]
SINGLE_ARM = {"place_shoe", "rotate_qrcode", "place_mouse_pad", "place_empty_cup", "move_can_pot", "place_phone_stand"}
DUAL_MIRROR = {"handover_mic", "place_can_basket"}
RUNS = {
    "R3 同手": "bp10_best__r3__7000__clean_ep20__best50",
    "R2 同手": "bp10_best__r2__7000__clean_ep20__best50",
    "taskcon 同手": "bp10_best__taskcon__7000__clean_ep20__best50",
    "R3 异手": "bp10_worst__r3__7000__clean_ep20__worst50",
    "taskcon 异手": "bp10_worst__taskcon__7000__clean_ep20__worst50",
    "R3 固定": "bpv31r__r3__7000__clean_ep20__normal",
    "R2 固定": "bpv31r__r2__7000__clean_ep20__normal",
    "taskcon 固定": "bpv3_training_curve__bpv3_bptaskcon__7000__clean_ep20__normal",
}


def task_counts(run: str) -> dict[str, tuple[int, int]]:
    out = {}
    for task in TASKS:
        f = ROOT / run / task / "_result.txt"
        if not f.is_file():
            continue
        m = re.search(r"Success:\s*(\d+)\s*/\s*(\d+)", f.read_text())
        if m and int(m.group(2)) > 0:
            out[task] = (int(m.group(1)), int(m.group(2)))
    return out


def rate(counts: dict, tasks) -> str:
    s = sum(counts[t][0] for t in tasks if t in counts)
    n = sum(counts[t][1] for t in tasks if t in counts)
    return f"{s / n:.1%} ({s}/{n})" if n else "-"


data = {name: task_counts(run) for name, run in RUNS.items()}
lines = ["| 任务 | " + " | ".join(RUNS) + " |", "| ---- |" + " ---- |" * len(RUNS)]
for t in TASKS:
    cells = []
    for name in RUNS:
        c = data[name].get(t)
        cells.append(f"{c[0] / c[1]:.0%}" if c else "-")
    lines.append(f"| {t} | " + " | ".join(cells) + " |")
OVERLAP7 = {"place_cans_plasticbox", "scan_object", "place_shoe", "rotate_qrcode", "place_mouse_pad", "place_empty_cup",
            "handover_mic"}
for label, group in (("**10 任务平均**", TASKS), ("与旧测评重合的 7 个", OVERLAP7), ("其中单臂 4 个", OVERLAP7 & SINGLE_ARM),
                     ("单臂 6 个", SINGLE_ARM), ("镜像双臂 2 个", DUAL_MIRROR),
                     ("A1 2 个", {"place_cans_plasticbox", "scan_object"})):
    lines.append(f"| {label} | " + " | ".join(rate(data[n], group) for n in RUNS) + " |")
print("\n".join(lines))
