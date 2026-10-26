"""Collect the AE-2 supplement results (S-UF P5, S-UF P64, S-U P64) next to the AE-2 large-scale S-A / S-U (P5) results
into markdown tables in the layout of `bpv4cdx Stage A-AE 实验报告 v2.md`. Missing files are shown as "—".

usage: python tools/bpv4cdx/ae2p_collect.py [--root outputs/BPv4CDX/stage_a/v2] [--out <root>/ae2p/summary.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# name -> (run dir, G4/G5 v1, G5 v2 K=1, G5 v2 K=3, bp_probe, G7-D prefix, G7-S prefix, attn, zdemo)
MODELS = {
    "S-A P5": ("ae2l_main", "ae2l_eval/main.json", "v2_eval/sa42_v2.json", None, "ae2l_test/sa_bp_probe.json",
               "ae2l_g7d/sa", "g7s_v2/sa", "attn/sa_heldout.json", "zdemo/sa_heldout.json"),
    "S-U P5": ("ae2l_su", "ae2l_eval/su.json", "v2_eval/su42_v2.json", None, "ae2l_test/su_bp_probe.json",
               "ae2l_g7d/su", "g7s_v2/su", "attn/su_heldout.json", "zdemo/su_heldout.json"),
    **{name: (run, f"ae2p/eval/{s}.json", f"ae2p/eval/{s}_v2k1.json", f"ae2p/eval/{s}_v2.json",
              f"ae2p/test/{s}_bp_probe.json", f"ae2p/g7d/{s}", f"ae2p/g7s/{s}", f"ae2p/attn/{s}_heldout.json",
              f"ae2p/zdemo/{s}_heldout.json")
       for name, run, s in (("S-UF P5", "ae2l_suf", "suf"), ("S-UF P64", "ae2l_suf_p64", "suf_p64"),
                            ("S-U P64", "ae2l_su_p64", "su_p64"))},
}
SEED43 = {"S-A P5 seed 43": "ae2l_main_seed43", "S-U P5 seed 43": "ae2l_su_seed43"}
STEPS = (2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000)


def load(root: Path, rel: str | None):
    if not rel:
        return None
    p = root / rel
    return json.loads(p.read_text()) if p.is_file() else None


def run_dir(root: Path, run: str) -> Path | None:
    runs = sorted(p for p in (root / run).glob("*/") if p.is_dir())
    return runs[-1] if runs else None


def hom(root: Path, run: str) -> dict[int, dict]:
    d = run_dir(root, run)
    if d is None:
        return {}
    rows = {}
    # the later add-on files (end-effector error, text drop) only contribute their own fields
    for name, keep in (("heldout_monitor.jsonl", None), ("heldout_monitor_ee.jsonl", ("ee_",)),
                       ("heldout_monitor_textdrop.jsonl", ("mse_matched_textdrop", "textdrop_"))):
        f = d / name
        if f.is_file():
            for line in f.read_text().splitlines():
                r = json.loads(line)
                rows.setdefault(int(r["step"]), {}).update(
                    {k: v for k, v in r.items() if v is not None and (keep is None or k.startswith(keep))})
    return rows


def f(v, p=3):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "通过" if v else "未通过"
    return f"{v:.{p}f}"


def ratio(r, key, p=2, ci=True):
    if r is None or not r.get("ratios", {}).get(key):
        return "—"
    x = r["ratios"][key]
    s = f(x["ratio"], p)
    return f"{s} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}]" if ci and x.get("ci95") else s


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def video_onset(h: dict[int, dict]) -> str:
    prev = 0
    for s in sorted(h):
        if (h[s].get("alt_over_match") or 0) >= 2.0:
            return f"{prev // 1000 if prev % 1000 == 0 else prev / 1000}k ~ {s / 1000:g}k"
        prev = s
    return "—"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "outputs/BPv4CDX/stage_a/v2"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    md: list[str] = []
    names = list(MODELS)

    # HOM over training
    homs = {n: hom(root, m[0]) for n, m in MODELS.items()}
    homs.update({n: hom(root, r) for n, r in SEED43.items()})
    md.append("### HOM：matched 误差 / 同文本换示教比值（750 条留出轨迹）\n")
    cols = names + list(SEED43)
    md.append(table(["步数"] + cols, [[f"{s:,}"] + [
        f"{f(homs[n][s]['mse_matched'])} / {f(homs[n][s].get('alt_over_match'), 2)}" if s in homs[n] else "—"
        for n in cols] for s in STEPS]))
    md.append("\n### 第 20000 步：误差、末端误差、文本依赖、开始用视频\n")
    rows = []
    for n in cols:
        r = homs[n].get(20000, {})
        m = r.get("mse_matched")
        rows.append([n, f(m, 4), f(r.get("mse_matched_generic"), 4),
                     f(r["mse_matched_textdrop"] / m, 3) if m and r.get("mse_matched_textdrop") else "—",
                     f(r.get("ee_cm_matched"), 2), f(r.get("ee_cm_matched_moving"), 2),
                     f(r.get("alt_over_match"), 2), video_onset(homs[n])])
    md.append(table(["模型", "matched", "通用文本", "屏蔽文本 / matched", "末端误差 cm（两手）", "末端误差 cm（在动的手）",
                     "换示教比值", "开始用视频（比值首次 ≥ 2）"], rows))

    ev = {n: load(root, m[1]) for n, m in MODELS.items()}
    md.append("\n### G4（协议 v1，K = 1）\n")
    md.append(table(["条件"] + names, [
        ["matched"] + [f(r["mean_error"]["matched"], 4) if r else "—" for r in ev.values()],
        ["同文本换示教"] + [ratio(r, "alternate") for r in ev.values()],
        ["只有文本的对照"] + [ratio(r, "textonly", ci=False) for r in ev.values()],
        ["通用文本下换示教"] + [ratio(r, "alternate_generic_vs_matched_generic", ci=False) for r in ev.values()],
        ["屏蔽文本 / matched"] + [ratio(r, "matched_textdrop_vs_matched", 3, ci=False) for r in ev.values()],
        ["blank"] + [ratio(r, "blank", ci=False) for r in ev.values()],
        ["shuffle"] + [ratio(r, "shuffle", ci=False) for r in ev.values()],
        ["wrong-task"] + [ratio(r, "wrong", ci=False) for r in ev.values()],
        ["跟随（换示教）"] + [f(r["follow"]["alternate"]["frac_closer_to_partner"], 2) if r else "—" for r in ev.values()],
        ["G4"] + [f(r["G4"]["pass"]) if r else "—" for r in ev.values()],
    ]))
    md.append("\n### G5（协议 v1 与 v2，K = 1）\n")
    v2 = {n: load(root, m[2]) for n, m in MODELS.items()}
    v2k3 = {n: load(root, m[3]) for n, m in MODELS.items()}
    z = lambda r: (f"{r['z_vs_action_distance']['mean']:.2f} [{r['z_vs_action_distance']['ci95'][0]:.2f}, "  # noqa: E731
                   f"{r['z_vs_action_distance']['ci95'][1]:.2f}]，{r['z_vs_action_distance']['tasks_positive']} / "
                   f"{r['z_vs_action_distance']['n']} 为正") if r else "—"
    md.append(table(["类别"] + names, [
        ["目标位置 v1"] + [ratio(r, "position") for r in ev.values()],
        ["抓取臂 v1"] + [ratio(r, "arm", ci=False) for r in ev.values()],
        ["困难负样本 v1"] + [ratio(r, "hard", ci=False) for r in ev.values()],
        ["目标位置 v2"] + [ratio(r, "position") for r in v2.values()],
        ["困难负样本 v2"] + [ratio(r, "hard") for r in v2.values()],
        ["抓取臂 v2"] + [ratio(r, "arm", ci=False) for r in v2.values()],
        ["目标位置 / 困难负样本 v2（K = 3，敏感性）"] + [
            f"{ratio(r, 'position', ci=False)} / {ratio(r, 'hard', ci=False)}" if r else "—" for r in v2k3.values()],
        ["`Z_demo` 距离相关"] + [z(r) for r in ev.values()],
        ["G5 v1 / v2"] + [f"{f(a['G5']['pass']) if a else '—'} / {f(b['G5']['pass']) if b else '—'}"
                          for a, b in zip(ev.values(), v2.values())],
    ]))
    md.append("\n### 整条示教往返\n")
    md.append(table(["指标"] + names, [
        ["中位偏移（帧）"] + [f(r["round_trip"]["lag_frames"]["median"], 1) if r else "—" for r in ev.values()],
        ["偏移 ≤ 2 帧的比例"] + [f(r["round_trip"]["lag_frames"]["frac_abs_le_2"], 3) if r else "—" for r in ev.values()],
        ["尾段比"] + [f(r["round_trip"]["tail_ratio"], 2) if r else "—" for r in ev.values()],
    ]))

    te = {n: load(root, m[4]) for n, m in MODELS.items()}
    md.append("\n### 独立 test（bp_probe，只报告）\n")
    md.append(table(["指标"] + names, [
        ["matched"] + [f(r["mean_error"]["matched"], 3) if r else "—" for r in te.values()],
        ["同文本换示教"] + [ratio(r, "alternate") for r in te.values()],
        ["只有文本的对照"] + [ratio(r, "textonly", ci=False) for r in te.values()],
        ["目标位置"] + [ratio(r, "position") for r in te.values()],
        ["抓取臂"] + [ratio(r, "arm", ci=False) for r in te.values()],
        ["`Z_demo` 距离相关"] + [f(r["z_vs_action_distance"]["mean"], 2) if r else "—" for r in te.values()],
        ["shuffle"] + [ratio(r, "shuffle", ci=False) for r in te.values()],
        ["G4 / G5"] + [f"{f(r['G4']['pass'])} / {f(r['G5']['pass'])}" if r else "—" for r in te.values()],
    ]))

    g7 = {n: load(root, m[5] + "_probe.json") for n, m in MODELS.items()}
    xy = lambda r, k: (f"{r['position'][k]['z_trained'][0]:.2f} / {r['position'][k]['z_trained'][1]:.2f}"  # noqa: E731
                       if r else "—")
    md.append("\n### G7-D 探针（bp_probe，`Z_demo` token 平均，5 折）\n")
    md.append(table(["探针"] + names, [
        ["被抓物体身份"] + [f(r["summary"]["z_trained"]["grasp_id"], 2) if r else "—" for r in g7.values()],
        ["放置目标身份"] + [f(r["summary"]["z_trained"]["target_id"], 2) if r else "—" for r in g7.values()],
        ["抓取臂"] + [f(r["summary"]["z_trained"]["arm"], 2) if r else "—" for r in g7.values()],
        ["被抓物体初始位置 R²（x / y）"] + [xy(r, "grasp_init") for r in g7.values()],
        ["放置后位置 R²（x / y）"] + [xy(r, "grasp_final") for r in g7.values()],
        ["随机编码器：物体身份 / 放置目标"] + [
            f"{r['summary']['z_random']['grasp_id']:.2f} / {r['summary']['z_random']['target_id']:.2f}" if r else "—"
            for r in g7.values()],
    ]))

    md.append("\n### G7-S 显著性（相对均匀的倍数）\n")
    for part, label in (("heldout", "留出（100 条）"), ("probe", "bp_probe（200 条）")):
        gs = {n: load(root, m[6] + f"_{part}.json") for n, m in MODELS.items()}
        g = lambda r, k: f(r["trained"]["all"][k]["mean"], 2) if r else "—"  # noqa: E731
        rows = [["首槽"] + [g(r, "first_lift") for r in gs.values()],
                ["末槽"] + [g(r, "last_lift") for r in gs.values()],
                ["抓取槽 / 放置槽"] + [f"{g(r, 'grasp_lift')} / {g(r, 'place_lift')}" for r in gs.values()],
                ["本段自己那段示教"] + [g(r, "own_window_lift") for r in gs.values()],
                ["相机（头 / 左腕 / 右腕）"] + [" / ".join(f"{x:.2f}" for x in r["trained"]["all"]["camera"].values())
                                          if r else "—" for r in gs.values()]]
        if part == "probe":
            rows += [["物体初始位置 3×3"] + [g(r, "grasp_point_lift") for r in gs.values()],
                     ["放置位置 3×3"] + [g(r, "place_point_lift") for r in gs.values()]]
        md.append(f"\n{label}\n")
        md.append(table(["指标"] + names, rows))

    at = {n: load(root, m[7]) for n, m in MODELS.items()}
    a = lambda r, x, y: f(r["mass_mean_over_layers"][x][y], 3) if r else "—"  # noqa: E731
    md.append("\n### 注意力（6 个全注意力层平均，100 条留出轨迹）\n")
    md.append(table(["谁看谁"] + names, [
        ["`segment_query` → `Z_demo`"] + [a(r, "segment_query", "Z_demo") for r in at.values()],
        ["`segment_query` → 文本"] + [a(r, "segment_query", "text") for r in at.values()],
        ["动作 token → `segment_query`"] + [a(r, "action", "segment_query") for r in at.values()],
        ["动作 token → state 槽位"] + [a(r, "action", "state") for r in at.values()],
    ]))

    zd = {n: load(root, m[8]) for n, m in MODELS.items()}
    q = lambda r, l, k: f(r["qformer_trained"][l][k], 2) if r else "—"  # noqa: E731
    sp = lambda r, k: f(float(np.mean(r["slot_probes"]["z_trained"][k][1:])), 2) if r else "—"  # noqa: E731
    md.append("\n### 示教编码器内部（750 条留出轨迹）\n")
    md.append(table(["指标"] + names, [
        ["读自己那个槽（交叉 1 / 2 层，× 均匀）"] + [f"{q(r, 'cross0', 'own_slot_x_uniform')} / {q(r, 'cross1', 'own_slot_x_uniform')}" for r in zd.values()],
        ["注意力分布熵（交叉 1 / 2 层）"] + [f"{q(r, 'cross0', 'entropy_norm')} / {q(r, 'cross1', 'entropy_norm')}" for r in zd.values()],
        ["被读到 ≥ 0.5 倍平均的槽（交叉 1 / 2 层）"] + [f"{q(r, 'cross0', 'coverage_frac_slots_ge_half')} / {q(r, 'cross1', 'coverage_frac_slots_ge_half')}" for r in zd.values()],
        ["槽探针 R²：该槽时刻机械臂状态（槽 1 ~ 31 平均）"] + [sp(r, "local_state") for r in zd.values()],
        ["槽探针 R²：首次抓取构型"] + [sp(r, "grasp_config") for r in zd.values()],
        ["槽探针 R²：最后一帧状态"] + [sp(r, "final_state") for r in zd.values()],
        ["token 平均余弦 / 有效秩"] + [f"{r['geometry']['trained']['mean_token_cos']:.2f} / {r['geometry']['trained']['effective_rank']:.1f}"
                                 if r else "—" for r in zd.values()],
    ]))
    text = "\n".join(md) + "\n"
    out = Path(args.out) if args.out else root / "ae2p" / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
