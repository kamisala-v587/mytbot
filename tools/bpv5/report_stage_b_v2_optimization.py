#!/usr/bin/env python
"""Offline bs2 comparison. No torch, training, input writes or waiting.

Run from any directory; defaults are relative to this repository.
Only explicitly supplied --combined is inspected. Incomplete combinations are omitted.
Generated files can be updated iff their previous manifest hashes still match.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

from report_stage_b_v2_profile import (ReportError, batch_counts, load_profile,
                                      read_json, summarize_e2e)

ROOT = Path(__file__).resolve().parents[2]
RUNS = (
    ("baseline", "Baseline", "基线", "test_bs2_e2e_20261125_retry"),
    ("telemetry10", "Telemetry every 10", "遥测间隔10步", "opt_telemetry10_20261125"),
    ("persistent", "Persistent workers", "常驻loader workers", "opt_persistent_20261125"),
    ("unused_off", "DDP unused off", "关闭DDP unused检测", "opt_ddp_unused_off_20261125"),
)
ASSET_DIR = "bpv5_stage_b_v2_bs2_20261125"
REPORT = "BPv5_StageB_v2_bs2_端到端与优化.md"
OWNER = "report_stage_b_v2_optimization/v1"
FIGURES = ("e2e_time_share", "time_share", "total_allocated_share")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_diff(old, new, prefix=""):
    """Keep exact runtime snapshot deltas, including added/removed fields."""
    if isinstance(old, dict) and isinstance(new, dict):
        result = []
        for key in sorted(old.keys() | new.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in old or key not in new:
                result.append({"path": path, "baseline": old.get(key),
                               "current": new.get(key), "added": key not in old,
                               "removed": key not in new})
            else:
                result.extend(config_diff(old[key], new[key], path))
        return result
    return [] if old == new else [{"path": prefix, "baseline": old, "current": new}]


def load_run(source, key, label, chinese, epoch_steps):
    ranks = load_profile(source)
    if any(r["metadata"]["schema_version"] != 3 for r in ranks):
        raise ReportError(f"{source}: 需要完整 schema3 E2E 数据")
    batch = batch_counts(ranks)
    if batch["per_rank_trajectories"] != 2 or batch["global_trajectories"] != 8:
        raise ReportError(f"{source}: 需要四卡bs2")
    e2e = summarize_e2e(ranks, batch)
    rid = e2e["slowest_window_rank"]
    # A fixed rank chosen by the whole window, not a sum of per-step maxima.
    rows = ranks[rid]["records"]
    config = ranks[0]["metadata"]["resolved_config"]
    result = {"key": key, "label": label, "chinese_label": chinese,
              "source_directory": str(source), "batch": batch, "e2e": e2e,
              "resolved_config": config,
              "epoch_steps": epoch_steps,
              "epoch_boundary_steps": list(range(epoch_steps + 1, 41, epoch_steps)),
              "epoch_boundary_method": "26 rank-local steps = logged 104 sampler batches / 4 ranks; measured loop starts at fresh loader; override --epoch-steps for other datasets",
              "curve_rank": rid,
              "curve": [{"step": s, "mode": rows[s]["mode"],
                         "iteration_wall_s": rows[s]["iteration_wall_s"],
                         "dataloader_next_wall_s": rows[s]["dataloader_next_wall_s"]}
                        for s in range(1, 41)],
              "input_sha256": {str(r["directory"] / name): digest(r["directory"] / name)
                                for r in ranks for name in ("metadata.json", "summary.json", "steps.jsonl")}}
    return result


def make_plots(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(output / f"{name}.{extension}", dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [r["label"].replace(" ", "\n", 1) for r in runs]
    for ax, field, title, unit in (
        (axes[0], "global_trajectories_per_s", "E2E throughput (higher is better)", "Global trajectories / s"),
        (axes[1], "mean_step_wall_s", "E2E mean step (lower is better)", "Seconds / update")):
        values = [r["e2e"][field] for r in runs]
        ax.bar(range(len(runs)), values, color=["#666666"] + ["#3776a6"] * (len(runs) - 1))
        for i, (v, run) in enumerate(zip(values, runs)):
            annotation = f"{v:.3f}"
            if i and field == "global_trajectories_per_s":
                annotation += f"\n{run['throughput_gain_percent']:+.2f}%"
            ax.text(i, v + max(values) * .025, annotation, ha="center", fontsize=9)
        ax.set(xticks=range(len(runs)), xticklabels=labels, ylabel=unit, title=title,
               ylim=(0, max(values) * 1.24))
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Steps 11-30: 20 / max_rank(continuous 20-step elapsed)\nOne short run per variant; no repeat-based error bars", fontsize=11)
    save(fig, "optimization_throughput")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for run in runs:
        points = [p for p in run["curve"] if 11 <= p["step"] <= 30]
        for ax, field in zip(axes, ("iteration_wall_s", "dataloader_next_wall_s")):
            ax.plot([p["step"] for p in points], [p[field] for p in points],
                    marker=".", label=f"{run['label']} (rank {run['curve_rank']})")
    for ax in axes:
        for step in runs[0]["epoch_boundary_steps"]:
            if 11 <= step <= 30:
                ax.axvline(step, color="black", linestyle="--", alpha=.6)
                ax.text(step + .12, .94, f"epoch restart: step {step}",
                        transform=ax.get_xaxis_transform(), fontsize=9, va="top")
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set(ylabel="Full iteration wall (s)", title="Steady E2E / epoch boundary: fixed slowest-window rank per run")
    axes[1].set(xlabel="Measured update step (11-30 only)", ylabel="Main-process next wait (s)",
                title="Loader next includes implicit H2D enqueue; worker prefetch overlaps compute")
    save(fig, "optimization_speed_epoch")


def report_text(runs, base, root, combined_note):
    def image(name, alt):
        return f"![{alt}]({ASSET_DIR}/{name}.png)"
    def relative(path):
        return Path(os.path.relpath(path, root)).as_posix()
    e2e = runs[0]["e2e"]
    components = e2e["critical_window_mean_wall_s"]
    vit = next(m["per_call_inclusive_cuda_ms"] for m in base["modules_inclusive"]["rows"]
               if m["name"] == "function/vision/visual_only")
    lines = [f"<!-- generated-by: {OWNER}; edit generator, not this file -->",
             "# BPv5 Stage B v2：bs2 端到端与单变量优化", "",
             "## 结论先看", "",
             f"- **基线 E2E：{e2e['mean_step_wall_s']:.3f} s/step**；全局 {e2e['global_trajectories_per_s']:.3f} trajectories/s、{e2e['global_segments_per_s']:.3f} segments/s。",
             f"- **ViT 约 {vit['mean']:.1f} ms/调用**：仅 `vlm_visual.forward` 的 `visual_only` CUDA event inclusive 诊断，范围为35–40步、四rank；不是完整前向，也不是端到端独占时间。",
             f"- **loader next 均值 {components['dataloader_next'] * 1000:.1f} ms**（基线慢窗口rank {e2e['slowest_window_rank']}）：含 epoch 重启尖峰与 implicit H2D enqueue，不是每一步都等189 ms。", "",
             "## 1. 基线端到端：时间花在哪里", "",
             image("e2e_time_share", "基线E2E互斥时间占比"), "",
             "采用同一个最慢20步窗口rank的分项均值，严格闭合到其E2E步时；不把四卡均值拼成慢卡分母。", ""]
    names = {"dataloader_next": "loader next", "update": "同步update", "postupdate": "更新后日志/遥测等", "other residual": "其他残差"}
    for key, value in components.items():
        lines.append(f"- {names[key]}：{value * 1000:.1f} ms，{value / e2e['mean_step_wall_s']:.1%}。")
    lines += ["", "H2D 已在 next 内，遥测已在 postupdate 内；worker 预取可与计算/其他batch重叠，不能再次相加。",
              "other 包含更新前同步、worker统计提取与包装/导出等；不把所有等待都归因于遥测。", "",
              "## 2. 基线诊断占比：update 与显存", "",
              image("time_share", "基线同步诊断update阶段占比"), "",
              "时间占比来自31–34诊断步×四rank的互斥同步阶段，不是11–30稳态E2E，也不直接等于可优化收益。", "",
              image("total_allocated_share", "基线前向末端总allocated显存占比"), "",
              "显存分母是rank0、step35同刻forward-end CUDA allocated，不是reserved或全步峰值。",
              "parameter / optimizer / gradient / saved activation按storage去重且互斥；其余留在residual，不冒充精确activation。", "",
              "## 3. 单变量优化：按实际窗口实算", "",
              image("optimization_throughput", "E2E优化吞吐与平均步时"), ""]
    for run in runs:
        v = run["e2e"]
        gain = "基准" if run is runs[0] else f"吞吐 {run['throughput_gain_percent']:+.2f}%；步时下降 {run['step_time_reduction_percent']:.2f}%"
        lines.append(f"- **{run['chinese_label']}**：{v['mean_step_wall_s']:.4f} s/step，{v['global_trajectories_per_s']:.4f} trajectories/s；{gain}。")
    lines += ["", combined_note, "", "吞吐提升 = 当前trajectories/s ÷ 基线trajectories/s − 1；步时下降 = 1 − 当前步时 ÷ 基线步时。",
              "三项单测是各自相对同一基线，不可把收益直接相加预测组合。", "",
              "## 4. 速度曲线与epoch边界", "",
              image("optimization_speed_epoch", "稳态速度和loader等待含epoch边界"), "",
              "曲线对每个运行固定选取其最慢完整窗口rank，不是逐步拼接不同rank；显示真实11–30步，不平滑、不删尖峰。",
              f"epoch边界按每rank {runs[0]['epoch_steps']}步标注（默认来自本批日志104 sampler batches/4卡=26）；不按尖峰猜边界。",
              "基线next约189 ms的均值包含epoch重建iterator/worker等等待（默认下一epoch首批step27）；常驻workers降低重启成本，但不代表数据读取免费。", "",
              "## 5. 比较口径与限制", "",
              "- 每rank batch=2，全局8条轨迹/更新、32 segments/更新，梯度累积=1；四卡吞吐不再乘四。",
              "- 每轮仅1次短测；稳态只有连续20步（11–30），maxrank取各rank完整窗口elapsed最大值。",
              "- 更新吞吐=20/maxrank elapsed；轨迹吞吐=160/maxrank elapsed。不是20个逐步rank最大值之和，不跨rank相减monotonic时钟。",
              "- 没有独立重复试验，故无误差条；20步或四rank不是20次/4次独立复测，微小提升需重复验证。",
              "- 31–40步含同步、hooks/profiler/export扰动，只用于定位；128ms ViT不可除以E2E画独占占比。",
              "- persistent workers可能改变worker生命周期/RNG；DDP unused关闭须确保真实分支不遗漏梯度，短测成功不保证所有数据分支安全。",
              "- 40步loss不支持收敛、泛化或效果保证；本报告不承诺优化不影响收敛。", "",
              "## 6. 来源与复现", "",
              f"- [参数、逐步数据、配置差异与输入SHA256]({ASSET_DIR}/parameters_data.json)。",
              f"- [本工具产物所有权清单]({ASSET_DIR}/manifest.json)：原地更新前校验旧文件hash，不覆盖其他工具文件。"]
    for run in runs:
        source = Path(run["source_directory"])
        lines.append(f"- {run['chinese_label']}：[原始采集]({relative(source)}/)、[保留的详细报告]({relative(source / 'report_stage_b_v2' / 'REPORT.md')})。")
    lines += ["", "", "### 更新方法", "",
              "在仓库根运行 `python tools/bpv5/report_stage_b_v2_optimization.py` 可重生成当前已完成对照。",
              "组合完成后增加 `--combined outputs/BPv5/bv2/实际组合目录`；不传参绝不扫描组合目录。",
              "若组合未完成则省略组合指标；已有产物被手工改动时拒绝覆盖，需先另存或换report-root。", "",
              "所有图同时提供PNG/SVG；现有训练报告模版、原始采集和原详细报告均不修改。", ""]
    return "\n".join(lines)


def publish(staging, root):
    """Refuse foreign/edited targets. Never delete existing files or directories."""
    manifest_path = root / ASSET_DIR / "manifest.json"
    old = read_json(manifest_path) if manifest_path.exists() else {}
    if old and old.get("owner") != OWNER:
        raise ReportError("输出manifest不属于本工具，拒绝更新")
    previous = old.get("files", {})
    for name, checksum in previous.items():
        target = root / name
        if target.is_file() and digest(target) != checksum:
            raise ReportError(f"已有工具产物被外部修改，拒绝覆盖：{target}")
    files = sorted(p for p in staging.rglob("*") if p.is_file())
    for source in files:
        name = source.relative_to(staging).as_posix()
        target = root / name
        if target.is_symlink() or (target.exists() and name not in previous):
            raise ReportError(f"目标已存在且不属本工具：{target}")
        if any(parent.is_symlink() for parent in target.parents if parent != root.parent):
            raise ReportError(f"输出路径不能经由符号链接：{target}")
    manifest = {"owner": OWNER, "files": {p.relative_to(staging).as_posix(): digest(p) for p in files}}
    for source in files:
        target = root / source.relative_to(staging)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "outputs/BPv5/bv2")
    parser.add_argument("--combined", type=Path, help="仅显式指定才读取；须四rank完整40步，未完成则不纳入")
    parser.add_argument("--report-root", type=Path, default=ROOT / "outputs/reports/bpv5-B")
    parser.add_argument("--epoch-steps", type=int, default=26, help="rank-local epoch步数，本批日志104/4=26")
    args = parser.parse_args(argv)
    try:
        if args.epoch_steps < 1:
            raise ReportError("epoch-steps须为正整数")
        source_root = args.runs_root.expanduser().resolve()
        report_root = args.report_root.expanduser().resolve()
        runs = [load_run(source_root / directory, key, label, chinese, args.epoch_steps)
                for key, label, chinese, directory in RUNS]
        note = "组合优化未纳入：未提供已完成组合目录；不推算、不填零、不造数据。"
        if args.combined:
            source = args.combined.expanduser().resolve()
            complete = all((source / f"rank_{r:02d}" / "summary.json").is_file()
                           and read_json(source / f"rank_{r:02d}" / "summary.json").get("status") == "complete"
                           for r in range(4))
            if complete:
                runs.append(load_run(source, "combined", "Combined", "组合优化", args.epoch_steps))
                note = "组合优化已完成，按同一20步窗口口径纳入；不是由单项收益推算。"
            else:
                note = "组合优化尚未完整完成：本次省略其图线和指标；不等待、不启动训练。"
                print(note, file=sys.stderr)
        base = read_json(source_root / RUNS[0][3] / "report_stage_b_v2/summary.json")
        if base.get("status") != "complete" or not base.get("e2e"):
            raise ReportError("基线详细报告须完成且含E2E")
        if not math.isclose(base["e2e"]["mean_step_wall_s"], runs[0]["e2e"]["mean_step_wall_s"], abs_tol=1e-8):
            raise ReportError("基线详细报告与原始采集窗口不一致")
        for run in runs:
            if run["batch"] != runs[0]["batch"]:
                raise ReportError("各运行batch/segments不一致，不能直接比较")
            run["config_diff_vs_baseline"] = config_diff(runs[0]["resolved_config"], run["resolved_config"])
            run["throughput_gain_percent"] = (run["e2e"]["global_trajectories_per_s"] / runs[0]["e2e"]["global_trajectories_per_s"] - 1) * 100
            run["step_time_reduction_percent"] = (1 - run["e2e"]["mean_step_wall_s"] / runs[0]["e2e"]["mean_step_wall_s"]) * 100
        # Restrict writes to report artifacts, never inputs/rank data.
        if any(report_root == Path(r["source_directory"]) or report_root in Path(r["source_directory"]).parents
               or Path(r["source_directory"]) in report_root.parents for r in runs):
            raise ReportError("report-root不能与采集目录存在祖先/子目录关系")
        with tempfile.TemporaryDirectory(prefix="bpv5-optimization-") as temporary:
            staging = Path(temporary)
            assets = staging / ASSET_DIR
            assets.mkdir()
            hashes = {}
            for name in FIGURES:
                for ext in ("png", "svg"):
                    source = source_root / RUNS[0][3] / "report_stage_b_v2" / f"{name}.{ext}"
                    shutil.copyfile(source, assets / source.name)
                    hashes[str(source)] = digest(source)
            make_plots(runs, assets)
            data = {"owner": OWNER, "runs": runs, "combined_note": note,
                    "baseline_time_share": base["time_share"],
                    "baseline_memory_share": base["memory_share"],
                    "baseline_modules_inclusive": base["modules_inclusive"],
                    "baseline_figure_sha256": hashes, "independent_repeats_per_run": 1}
            (assets / "parameters_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            text = report_text(runs, base, report_root, note)
            (staging / REPORT).write_text(text, encoding="utf-8")
            publish(staging, report_root)
        print(f"报告：{report_root / REPORT}（{len(text.splitlines())}行）")
        for run in runs:
            print(f"{run['key']}: {run['e2e']['mean_step_wall_s']:.6f} s/step, {run['throughput_gain_percent']:+.3f}% throughput")
        return 0
    except (ReportError, OSError, KeyError, ValueError, TypeError, StopIteration) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
