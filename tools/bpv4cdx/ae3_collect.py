"""Collect the AE-3 evaluation (checkpoints 600k / 700k, run_ae3_eval.sh) next to the AE-2 large-scale results into
markdown tables in the layout of the AE-2 report v2. Missing files are shown as "—".

usage: python tools/bpv4cdx/ae3_collect.py [--out outputs/BPv4CDX/stage_a3_eval/summary.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/bpv4cdx"))
from ae2p_collect import f, hom, load, ratio, table  # noqa: E402

V2 = ROOT / "outputs/BPv4CDX/stage_a/v2"
E = ROOT / "outputs/BPv4CDX/stage_a3_eval"
R = E / "results"
STEPS = {"AE-3 60w": "60w", "AE-3 70w": "70w"}
# AE-2 reference models: (run dir, G4/G5 v1, G5 v2 K=1, bp_probe, G7-D prefix, G7-S prefix, attn, zdemo)
AE2 = {"AE-2 S-U": ("ae2l_su", "ae2l_eval/su.json", "v2_eval/su42_v2.json", "ae2l_test/su_bp_probe.json", "ae2l_g7d/su",
                    "g7s_v2/su", "attn/su_heldout.json", "zdemo/su_heldout.json"),
       "AE-2 S-A": ("ae2l_main", "ae2l_eval/main.json", "v2_eval/sa42_v2.json", "ae2l_test/sa_bp_probe.json", "ae2l_g7d/sa",
                    "g7s_v2/sa", "attn/sa_heldout.json", "zdemo/sa_heldout.json")}


def res(name: str, kind: str):
    """kind: eval / v2 / test / g7d_probe / g7d_heldout / g7s_heldout / g7s_probe / attn / zdemo / all_v2."""
    # bp_probe was re-recorded on 2026-10-24: every bp_probe result here is on the new data (caches and labels rebuilt)
    if name in AE2:
        m = AE2[name]
        key = {"AE-2 S-U": "ae2_su", "AE-2 S-A": "ae2_sa"}[name]
        if kind in ("test", "g7d_probe", "g7s_probe"):
            return load(R, {"test": f"test1024/{key}.json", "g7d_probe": f"g7d1024/{key}_probe.json",
                            "g7s_probe": f"g7s1024/{key}_probe.json"}[kind])
        rel = {"eval": m[1], "v2": m[2], "g7d_heldout": m[4] + "_heldout.json",
               "g7s_heldout": m[5] + "_heldout.json", "attn": m[6], "zdemo": m[7]}.get(kind)
        return load(V2, rel)
    s = STEPS[name]
    rel = {"eval": f"eval/rt_{s}.json", "v2": f"eval/rt_{s}_v2.json", "test": f"test/probe_{s}.json",
           "g7d_probe": f"g7d1024/ae3_{s}_probe.json", "g7d_heldout": f"g7d/{s}_heldout.json",
           "g7s_heldout": f"g7s/{s}_heldout.json", "g7s_probe": f"g7s1024/ae3_{s}_probe.json",
           "attn": f"attn/{s}_heldout.json", "zdemo": f"zdemo/{s}_heldout.json", "all_v2": f"eval/all_{s}_v2.json"}[kind]
    return load(R, rel)


def hom_rows(path: Path) -> dict:
    lines = path.read_text().splitlines() if path.is_file() else []
    return json.loads(lines[-1]) if lines else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(E / "summary.md"))
    args = ap.parse_args()
    names = list(STEPS) + list(AE2)
    md: list[str] = []

    md.append("### RoboTwin 留出集（750 条）：HOM 指标\n")
    rows = []
    for n in names:
        if n in AE2:
            r = hom(V2, AE2[n][0]).get(20000, {})
        else:
            r = hom_rows(E / "rt" / f"hom_rt_{STEPS[n]}.jsonl")
        m = r.get("mse_matched")
        rows.append([n, f(m, 4), f(r.get("mse_matched_generic"), 4),
                     f(r["mse_matched_textdrop"] / m, 3) if m and r.get("mse_matched_textdrop") else "—",
                     f(r.get("alt_over_match"), 2), f(r.get("ee_cm_matched"), 2), f(r.get("ee_cm_matched_moving"), 2)])
    for n, run in (("AE-2 S-U seed 43", "ae2l_su_seed43"), ("AE-2 S-A seed 43", "ae2l_main_seed43")):
        r = hom(V2, run).get(20000, {})
        rows.append([n, f(r.get("mse_matched"), 4), f(r.get("mse_matched_generic"), 4), "—", f(r.get("alt_over_match"), 2), "—", "—"])
    md.append(table(["模型", "matched", "通用文本", "屏蔽文本 ÷ matched", "同文本换示教（HOM）", "末端误差 cm（两手）",
                     "末端误差 cm（在动的手）"], rows))

    ev = {n: res(n, "eval") for n in names}
    md.append("\n### RoboTwin 留出集：G4（协议 v1，K = 1）\n")
    md.append(table(["条件"] + names, [
        ["matched"] + [f(r["mean_error"]["matched"], 4) if r else "—" for r in ev.values()],
        ["同文本换示教"] + [ratio(r, "alternate") for r in ev.values()],
        ["AE-2 只有文本的模型（参考）"] + [ratio(r, "textonly", ci=False) for r in ev.values()],
        ["通用文本下换示教"] + [ratio(r, "alternate_generic_vs_matched_generic", ci=False) for r in ev.values()],
        ["屏蔽文本 ÷ matched"] + [ratio(r, "matched_textdrop_vs_matched", 3, ci=False) for r in ev.values()],
        ["blank"] + [ratio(r, "blank", ci=False) for r in ev.values()],
        ["shuffle"] + [ratio(r, "shuffle", ci=False) for r in ev.values()],
        ["wrong-task"] + [ratio(r, "wrong", ci=False) for r in ev.values()],
        ["跟随（换示教）"] + [f(r["follow"]["alternate"]["frac_closer_to_partner"], 2) if r else "—" for r in ev.values()],
        ["G4"] + [f(r["G4"]["pass"]) if r else "—" for r in ev.values()],
    ]))
    v2 = {n: res(n, "v2") for n in names}
    z = lambda r: (f"{r['z_vs_action_distance']['mean']:.2f} [{r['z_vs_action_distance']['ci95'][0]:.2f}, "  # noqa: E731
                   f"{r['z_vs_action_distance']['ci95'][1]:.2f}]，{r['z_vs_action_distance']['tasks_positive']} / "
                   f"{r['z_vs_action_distance']['n']} 为正") if r else "—"
    md.append("\n### RoboTwin 留出集：G5\n")
    md.append(table(["类别"] + names, [
        ["目标位置 v1"] + [ratio(r, "position") for r in ev.values()],
        ["抓取臂 v1"] + [ratio(r, "arm", ci=False) for r in ev.values()],
        ["困难负样本 v1"] + [ratio(r, "hard", ci=False) for r in ev.values()],
        ["目标位置 v2"] + [ratio(r, "position") for r in v2.values()],
        ["困难负样本 v2"] + [ratio(r, "hard") for r in v2.values()],
        ["抓取臂 v2"] + [ratio(r, "arm", ci=False) for r in v2.values()],
        ["`Z_demo` 距离相关"] + [z(r) for r in ev.values()],
        ["G5 v1 / v2"] + [f"{f(a['G5']['pass']) if a else '—'} / {f(b['G5']['pass']) if b else '—'}"
                          for a, b in zip(ev.values(), v2.values())],
    ]))
    md.append("\n### 整条示教往返（RoboTwin）\n")
    md.append(table(["指标"] + names, [
        ["中位偏移（帧）"] + [f(r["round_trip"]["lag_frames"]["median"], 1) if r else "—" for r in ev.values()],
        ["偏移 ≤ 2 帧的比例"] + [f(r["round_trip"]["lag_frames"]["frac_abs_le_2"], 3) if r else "—" for r in ev.values()],
        ["尾段比"] + [f(r["round_trip"]["tail_ratio"], 2) if r else "—" for r in ev.values()],
    ]))

    te = {n: res(n, "test") for n in names}
    md.append("\n### 独立 test（bp_probe，2026-10-24 重新采集的数据）\n")
    md.append(table(["指标"] + names, [
        ["matched"] + [f(r["mean_error"]["matched"], 3) if r else "—" for r in te.values()],
        ["同文本换示教"] + [ratio(r, "alternate") for r in te.values()],
        ["AE-2 只有文本的模型（参考）"] + [ratio(r, "textonly", ci=False) for r in te.values()],
        ["目标位置"] + [ratio(r, "position") for r in te.values()],
        ["抓取臂"] + [ratio(r, "arm", ci=False) for r in te.values()],
        ["`Z_demo` 距离相关"] + [f(r["z_vs_action_distance"]["mean"], 2) if r else "—" for r in te.values()],
        ["shuffle"] + [ratio(r, "shuffle", ci=False) for r in te.values()],
        ["G4 / G5"] + [f"{f(r['G4']['pass'])} / {f(r['G5']['pass'])}" if r else "—" for r in te.values()],
    ]))

    g7 = {n: res(n, "g7d_probe") for n in names}
    xy = lambda r, k: (f"{r['position'][k]['z_trained'][0]:.2f} / {r['position'][k]['z_trained'][1]:.2f}"  # noqa: E731
                       if r else "—")
    md.append("\n### G7-D（bp_probe，`Z_demo` token 平均）\n")
    md.append(table(["探针"] + names, [
        ["被抓物体身份"] + [f(r["summary"]["z_trained"]["grasp_id"], 2) if r else "—" for r in g7.values()],
        ["放置目标身份"] + [f(r["summary"]["z_trained"]["target_id"], 2) if r else "—" for r in g7.values()],
        ["抓取臂"] + [f(r["summary"]["z_trained"]["arm"], 2) if r else "—" for r in g7.values()],
        ["被抓物体初始位置 R²（x / y）"] + [xy(r, "grasp_init") for r in g7.values()],
        ["放置后位置 R²（x / y）"] + [xy(r, "grasp_final") for r in g7.values()],
        ["随机编码器：物体身份 / 放置目标"] + [
            f"{r['summary']['z_random']['grasp_id']:.2f} / {r['summary']['z_random']['target_id']:.2f}" if r else "—"
            for r in g7.values()],
        ["原始 ViT：物体身份 / 放置目标"] + [
            f"{r['summary']['vit_mean']['grasp_id']:.2f} / {r['summary']['vit_mean']['target_id']:.2f}" if r else "—"
            for r in g7.values()],
    ]))

    md.append("\n### G7-S 显著性（相对均匀的倍数）\n")
    for part, label in (("heldout", "RoboTwin 留出（100 条）"), ("probe", "bp_probe（200 条）")):
        gs = {n: res(n, f"g7s_{part}") for n in names}
        g = lambda r, k: f(r["trained"]["all"][k]["mean"], 2) if r else "—"  # noqa: E731
        rows = [["首槽 / 末槽"] + [f"{g(r, 'first_lift')} / {g(r, 'last_lift')}" for r in gs.values()],
                ["抓取槽 / 放置槽"] + [f"{g(r, 'grasp_lift')} / {g(r, 'place_lift')}" for r in gs.values()],
                ["本段自己那段示教"] + [g(r, "own_window_lift") for r in gs.values()],
                ["相机（头 / 左腕 / 右腕）"] + [" / ".join(f"{x:.2f}" for x in r["trained"]["all"]["camera"].values())
                                          if r else "—" for r in gs.values()]]
        if part == "probe":
            rows += [["物体初始位置 3×3"] + [g(r, "grasp_point_lift") for r in gs.values()],
                     ["放置位置 3×3"] + [g(r, "place_point_lift") for r in gs.values()]]
        md.append(f"\n{label}\n")
        md.append(table(["指标"] + names, rows))

    at = {n: res(n, "attn") for n in names}
    a = lambda r, x, y: f(r["mass_mean_over_layers"][x][y], 3) if r else "—"  # noqa: E731
    md.append("\n### 注意力（RoboTwin 留出 100 条，6 个全注意力层平均）\n")
    md.append(table(["谁看谁"] + names, [
        ["`segment_query` → `Z_demo`"] + [a(r, "segment_query", "Z_demo") for r in at.values()],
        ["`segment_query` → 自己"] + [a(r, "segment_query", "segment_query") for r in at.values()],
        ["`segment_query` → `demo_end`"] + [a(r, "segment_query", "demo_end") for r in at.values()],
        ["`segment_query` → 文本"] + [a(r, "segment_query", "text") for r in at.values()],
        ["动作 token → `segment_query`"] + [a(r, "action", "segment_query") for r in at.values()],
        ["动作 token → state 槽位"] + [a(r, "action", "state") for r in at.values()],
    ]))
    zd = {n: res(n, "zdemo") for n in names}
    q = lambda r, l, k: f(r["qformer_trained"][l][k], 2) if r else "—"  # noqa: E731
    sp = lambda r, k: f(float(np.mean(r["slot_probes"]["z_trained"][k][1:])), 2) if r else "—"  # noqa: E731
    md.append("\n### 示教编码器内部（RoboTwin 留出 750 条）\n")
    md.append(table(["指标"] + names, [
        ["读自己那个槽（交叉 1 / 2 层）"] + [f"{q(r, 'cross0', 'own_slot_x_uniform')} / {q(r, 'cross1', 'own_slot_x_uniform')}" for r in zd.values()],
        ["注意力分布熵（交叉 1 / 2 层）"] + [f"{q(r, 'cross0', 'entropy_norm')} / {q(r, 'cross1', 'entropy_norm')}" for r in zd.values()],
        ["被读到 ≥ 0.5 倍平均的槽"] + [f"{q(r, 'cross0', 'coverage_frac_slots_ge_half')} / {q(r, 'cross1', 'coverage_frac_slots_ge_half')}" for r in zd.values()],
        ["相机（交叉第 1 层，头 / 左腕 / 右腕）"] + [" / ".join(f"{x:.2f}" for x in r["qformer_trained"]["cross0"]["camera"]) if r else "—" for r in zd.values()],
        ["槽探针 R²：该槽时刻机械臂状态"] + [sp(r, "local_state") for r in zd.values()],
        ["槽探针 R²：首次抓取构型"] + [sp(r, "grasp_config") for r in zd.values()],
        ["槽探针 R²：最后一帧状态"] + [sp(r, "final_state") for r in zd.values()],
        ["token 平均余弦 / 有效秩"] + [f"{r['geometry']['trained']['mean_token_cos']:.2f} / {r['geometry']['trained']['effective_rank']:.1f}"
                                 if r else "—" for r in zd.values()],
    ]))

    # all sources
    md.append("\n### 全部来源的留出集：HOM（按来源）\n")
    homs = {n: hom_rows(E / "ae3" / f"hom_all_{s}.jsonl") for n, s in STEPS.items()}
    srcs = sorted({s for h in homs.values() for s in h.get("by_source", {})})
    rows = []
    for s in srcs + ["全部"]:
        row = [s]
        for h in homs.values():
            b = h if s == "全部" else h.get("by_source", {}).get(s)
            if not b:
                row.append("—")
                continue
            row.append(f"{f(b.get('mse_matched'), 4)}（{b.get('segments', '—')} 段）；通用 {f(b.get('generic_over_match'), 3) if s != '全部' else f(b['mse_matched_generic'] / b['mse_matched'], 3)}；"
                       f"屏蔽 {f(b.get('textdrop_over_match'), 3)}")
        rows.append(row)
    md.append(table(["来源"] + list(homs), rows))
    for n, h in homs.items():
        if h:
            md.append(f"\n{n}：全部留出 {h.get('episodes')} 条 / {h.get('segments')} 段；同文本换示教（HOM）{f(h.get('alt_over_match'), 2)}；"
                      f"RoboTwin 末端误差 {f(h.get('ee_cm_matched'), 2)} cm（在动的手 {f(h.get('ee_cm_matched_moving'), 2)} cm）")

    md.append("\n### 全部来源的留出集：G4 / G5（协议 v2，K = 1，按任务分层 1,500 条）\n")
    av = {n: res(n, "all_v2") for n in STEPS}
    keys = [("alternate", "同文本换示教"), ("alternate_generic_vs_matched_generic", "通用文本下换示教"),
            ("matched_textdrop_vs_matched", "屏蔽文本 ÷ matched"), ("blank", "blank"), ("shuffle", "shuffle"),
            ("wrong", "wrong-task"), ("position", "目标位置（双臂 6+1）"), ("arm", "抓取臂（双臂 6+1）"),
            ("hard", "困难负样本（双臂 6+1）")]
    rows = [["matched"] + [f(r["mean_error"]["matched"], 4) if r else "—" for r in av.values()]]
    rows += [[label] + [ratio(r, k) for r in av.values()] for k, label in keys]
    rows += [["G4 / G5"] + [f"{f(r['G4']['pass'])} / {f(r['G5']['pass'])}" if r else "—" for r in av.values()]]
    md.append(table(["条件"] + list(av), rows))
    for n, r in av.items():
        if not r or "by_source" not in r:
            continue
        md.append(f"\n{n} 按来源（比值）\n")
        bs = r["by_source"]
        ks = [k for k, _ in keys if any(bs[s].get(k) for s in bs)]
        lab = dict(keys)
        md.append(table(["来源"] + [lab[k] for k in ks],
                        [[s] + [f"{bs[s][k]['ratio']:.2f}（{bs[s][k]['groups']}）" if bs[s].get(k) else "—" for k in ks]
                         for s in sorted(bs)]))

    text = "\n".join(md) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
