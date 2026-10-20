"""Combine data metrics (screen_data.py) with code-level object randomization into BP suitability tiers."""
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
metrics = {r["task"]: r for r in json.loads((HERE / "data_metrics.json").read_text())}
train = {l.strip().split("/robotwin2.0/")[1].split("/")[0]
         for l in Path("/vla/workspace/my_tbot/configs/ds_ids/pro6k/robo_clean_bp.txt").read_text().splitlines() if l.strip()}
smoke = {"scan_object", "rotate_qrcode", "place_shoe", "stack_blocks_two", "place_empty_cup", "handover_mic",
         "place_cans_plasticbox", "stack_bowls_two", "place_mouse_pad"}

# 来自 envs/<task>.py：物体类别本身随机（不是同类换外观）或物体数量不固定。
OBJECT_CATEGORY_RANDOM = {
    "place_a2b_left": "A、B 都从 10 类物体中随机",
    "place_a2b_right": "A、B 都从 10 类物体中随机",
    "place_object_scale": "物体从 鼠标/订书机/铃 3 类中随机",
    "place_object_stand": "物体从 6 类中随机",
    "put_object_cabinet": "物体从 8 类中随机",
    "place_object_basket": "物体在 扑克牌/玩具车 2 类中随机",
    "place_bread_basket": "面包数量不固定（单臂/双臂混合）",
}
# 只有颜色 / 同类外观随机的目标垫，clean 场景中只有一块目标物。
COLOR_TARGET = {"place_mouse_pad", "move_stapler_pad", "place_fan", "stamp_seal"}
MILD_CATEGORY = {"place_container_plate": "容器在 碗/杯 2 类中随机"}
USER_EXCLUDED = {"move_pillbottle_pad", "move_stapler_pad", "place_a2b_left", "place_a2b_right", "place_bread_basket",
                 "place_object_basket", "place_object_scale", "place_object_stand", "stamp_seal"}


def tier(t: str, m: dict) -> tuple[str, str]:
    if t in OBJECT_CATEGORY_RANDOM:
        return "D", OBJECT_CATEGORY_RANDOM[t]
    if m["canon_seq_mode"] < 0.9:
        return "C", f"动作套路不固定（镜像后最常见套路只占 {m['canon_seq_mode']:.0%}）"
    notes = []
    if m["len_cv"] > 0.15 or (m["event_time_std"] or 0) > 0.04:
        notes.append(f"节奏不稳定（长度 CV {m['len_cv']:.2f}，事件时刻 std {m['event_time_std']:.3f}）")
    if t in MILD_CATEGORY:
        notes.append(MILD_CATEGORY[t])
    mirror = m["raw_seq_mode"] < 0.9
    base = "A2" if mirror else "A1"
    if notes:
        return base + "*", "；".join(notes)
    return base, ""


rows = []
for t, m in sorted(metrics.items()):
    if t.startswith("my"):
        continue
    tr, why = tier(t, m)
    kf = m["kf_align_matched"] or {}
    rows.append({
        "task": t, "tier": tr, "why": why,
        "arms": f"{m['arm_counts']['L']}/{m['arm_counts']['R']}/{m['arm_counts']['B']}",
        "pattern": m["canon_seq_top"][0][0] or "-",
        "raw": m["raw_seq_mode"], "canon": m["canon_seq_mode"],
        "kf_within1": kf.get("within1"), "len_cv": m["len_cv"],
        "in_train": t in train, "in_smoke": t in smoke, "user_excluded": t in USER_EXCLUDED,
        "color_target": t in COLOR_TARGET,
    })

order = {"A1": 0, "A1*": 1, "A2": 2, "A2*": 3, "C": 4, "D": 5}
rows.sort(key=lambda r: (order[r["tier"]], r["task"]))
(HERE / "task_tiers.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
yaml.safe_dump({k: [r["task"] for r in rows if r["tier"].rstrip("*") == k or r["tier"] == k]
                for k in ("A1", "A2", "C", "D")}, open(HERE / "task_tiers.yaml", "w"), allow_unicode=True)

lines = ["| 任务 | 档 | 左/右/双 | 镜像后套路 | 原始一致 | 镜像一致 | 关键帧 within-1 | 长度 CV | 训练 | 测评 | 你已排除 | 说明 |",
         "| ---- | -- | -------- | ---------- | -------- | -------- | --------------- | ------- | ---- | ---- | -------- | ---- |"]
for r in rows:
    lines.append(
        f"| {r['task']} | {r['tier']} | {r['arms']} | {r['pattern']} | {r['raw']:.0%} | {r['canon']:.0%} | "
        f"{r['kf_within1']:.2f} | {r['len_cv']:.2f} | {'✓' if r['in_train'] else ''} | {'✓' if r['in_smoke'] else ''} | "
        f"{'✓' if r['user_excluded'] else ''} | {r['why']}{'（颜色随机目标垫）' if r['color_target'] else ''} |")
(HERE / "tier_table.md").write_text("\n".join(lines) + "\n")
from collections import Counter
print(Counter(r["tier"] for r in rows))
for k in order:
    print(k, [r["task"] for r in rows if r["tier"] == k])
