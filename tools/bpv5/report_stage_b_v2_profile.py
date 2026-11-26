#!/usr/bin/env python
"""Offline report for profile_stage_b_v2.py (no torch import, CUDA or training).

Usage: python tools/bpv5/report_stage_b_v2_profile.py PROFILE_DIR [--output-dir DIR]
Only completed four-rank profiles are accepted. The input is never modified.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

PHASES = ("forward", "backward_ddp", "clip", "optimizer", "zero_grad", "scheduler")
GIB = 1024 ** 3
MEMORY_TITLE = "前向末端saved activation，不是总显存峰值占比"


class ReportError(ValueError):
    """Invalid, incomplete or inconsistent source data."""


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReportError(f"无法读取有效 JSON：{path}：{exc}") from exc


def number(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{label} 必须是有限数值，实际为 {value!r}")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ReportError(f"{label} 必须是{'正' if positive else '非负'}有限数值")
    return value


def stats(values):
    values = sorted(values)
    if not values:
        raise ReportError("统计样本为空")
    return {"count": len(values), "mean": statistics.mean(values),
            "p50": statistics.median(values),
            "p95": values[math.ceil(.95 * len(values)) - 1],
            "min": values[0], "max": values[-1]}


def load_profile(source):
    ranks = []
    for rank in range(4):
        directory = source / f"rank_{rank:02d}"
        path = directory / "steps.jsonl"
        if not path.is_file():
            raise ReportError(f"尚无完整四卡 profiling 数据：缺少 {path}。"
                              "请在采集结束后运行；本工具不会等待或启动训练。")
        if (directory / "failure.json").exists() or (directory / "failure_minimal.txt").exists():
            raise ReportError(f"rank {rank} 存在失败记录；拒绝将失败/部分运行作为完整报告")
        metadata = read_json(directory / "metadata.json")
        device = read_json(directory / "device.json")
        done = read_json(directory / "summary.json")
        if metadata.get("rank") != rank or metadata.get("world_size") != 4:
            raise ReportError(f"rank {rank} metadata rank/world_size 不一致")
        if metadata.get("schema_version") not in (1, 2, 3):
            raise ReportError(f"rank {rank} 不支持的 schema_version")
        if not isinstance(metadata.get("resolved_config"), dict):
            raise ReportError(f"rank {rank} 缺少 resolved_config 快照")
        if done.get("status") != "complete":
            raise ReportError(f"rank {rank} 未完成；不会等待采集")
        records = {}
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ReportError(f"{path}:{lineno} JSON 不完整") from exc
            step = row.get("step")
            if type(step) is not int or step not in range(1, 41) or step in records:
                raise ReportError(f"{path}:{lineno} step 缺失、重复或越界")
            if row.get("rank") != rank:
                raise ReportError(f"{path}:{lineno} rank 不一致")
            number(row.get("update_wall_s"), f"rank {rank}/step {step}/update_wall_s", True)
            expected = ("warmup" if step <= 10 else "steady" if step <= 30 else
                        "phases" if step <= 34 else "modules_saved" if step <= 38 else "profiler")
            if row.get("mode") != expected:
                raise ReportError(f"rank {rank}/step {step} mode 与采集 schedule 不一致")
            for kind in ("allocated_bytes", "reserved_bytes"):
                number(row.get("whole_step_peak", {}).get(kind), f"step {step}/{kind}")
            loss = row.get("loss")
            if loss is not None and (isinstance(loss, bool) or not isinstance(loss, (int, float))
                                     or not math.isfinite(loss)):
                raise ReportError(f"rank {rank}/step {step} loss 非有限数值")
            records[step] = row
        if set(records) != set(range(1, 41)):
            raise ReportError(f"rank {rank} 需要完整 1–40 步，实际 {len(records)} 步")
        ranks.append({"rank": rank, "directory": directory, "metadata": metadata,
                      "device": device, "records": records, "completion": done})
    if any(r["metadata"]["resolved_config"] != ranks[0]["metadata"]["resolved_config"]
           for r in ranks[1:]):
        raise ReportError("四卡 resolved_config 不一致，不能合并报告")
    if len({r["metadata"]["schema_version"] for r in ranks}) != 1:
        raise ReportError("四卡 schema_version 不一致")
    return ranks


def phase_split(row):
    """Sibling wrapper times only; unknown policy_update stays in residual."""
    values = dict.fromkeys(PHASES, 0.0)
    for item in row.get("phases", []):
        if item.get("status") != "ok":
            raise ReportError(f"step {row['step']} 存在未完成阶段")
        wall = number(item.get("wall_s"), "phase wall_s")
        if item.get("name") in values:
            values[item["name"]] += wall
    for required in ("forward", "backward_ddp", "optimizer", "zero_grad"):
        if not any(p.get("name") == required for p in row.get("phases", [])):
            raise ReportError(f"step {row['step']} 缺少 {required} 阶段")
    residual = row["update_wall_s"] - sum(values.values())
    if residual < -1e-9:
        raise ReportError(f"step {row['step']} 阶段和超过 update 墙时，拒绝重叠时间饼图")
    values["other residual"] = max(0.0, residual)
    return values


def inventory_totals(inventory):
    """Keep device types separate; CPU scalar optimizer state is not VRAM."""
    totals = {"cuda": dict.fromkeys(("parameter", "optimizer_state", "gradient"), 0),
              "cpu_or_other": dict.fromkeys(("parameter", "optimizer_state", "gradient"), 0)}
    if not isinstance(inventory, dict) or "groups_bytes" not in inventory:
        raise ReportError("缺少持久 storage inventory")
    for groups in inventory["groups_bytes"].values():
        for key, size in groups.items():
            kind, device = key.split(":", 1)
            if kind not in totals["cuda"]:
                raise ReportError(f"未知 storage 类别：{kind}")
            number(size, key)
            totals["cuda" if device.startswith("cuda") else "cpu_or_other"][kind] += size
    return totals


def validate_forward_decomposition(row, allocated):
    data = row.get("memory_forward_end")
    if data is None:
        return None  # Schema 1: do not splice snapshots from other moments.
    if data.get("allocated_bytes") != allocated:
        raise ReportError("同刻显存分解 allocated 与 forward after 不一致")
    inventory = row.get("storage_forward_end", {})
    saved = row.get("saved_activations_forward_end", {})
    records = inventory.get("storage_records")
    saved_records = saved.get("storage_records")
    if not isinstance(records, list) or not isinstance(saved_records, list):
        raise ReportError("新 schema 缺少同刻 storage identities")
    excluded = saved.get("excluded_storage_keys")
    if not isinstance(excluded, list):
        raise ReportError("新 schema 缺少 saved 排除项 storage identities")
    excluded = set(excluded)
    if any(record["key"] in excluded for record in saved_records):
        raise ReportError("saved storage 与 parameter/buffer/batch 排除项重叠")
    kinds = ("parameter", "optimizer_state", "gradient", "saved_activation", "other_residual")
    totals = dict.fromkeys(kinds, 0)
    seen = set()
    device = data.get("device")
    if not isinstance(device, str) or not device.startswith("cuda"):
        raise ReportError("同刻显存分解必须选择明确 CUDA device")
    for record in records + saved_records:
        size = number(record.get("bytes"), "storage bytes")
        if record.get("device") != device:
            continue
        key = record["key"]
        if key in seen:
            raise ReportError("同刻 storage 分类存在交叉/重复")
        seen.add(key)
        kind = record.get("kind")
        if kind not in kinds[:-1]:
            raise ReportError("未知同刻 storage 类别")
        totals[kind] += size
    totals["other_residual"] = allocated - sum(totals.values())
    if totals["other_residual"] < 0 or data.get("categories_bytes") != totals:
        raise ReportError("显存分类之和不等于 allocated 或 residual 为负")
    return {**data, "categories_fraction": {k: v / allocated for k, v in totals.items()},
            "disjoint_storage_check": True, "sum_equals_allocated_check": True}



def aggregate_trace_kernels(paths):
    """Chrome trace dur is microseconds; displayTimeUnit affects the viewer only.

    Accept only complete cat=kernel events, never key-average device attribution.
    Fail closed across the whole requested scope: no partial/fallback ranking.
    """
    kernels = defaultdict(lambda: {"duration_us": 0.0, "count": 0})
    errors = []
    sources = []
    for path in paths:
        try:
            trace = read_json(path)
            events = trace.get("traceEvents") if isinstance(trace, dict) else None
            if not isinstance(events, list):
                raise ReportError(f"{path} 缺少 traceEvents 数组")
            count = 0
            duration = 0.0
            for event in events:
                if not isinstance(event, dict):
                    raise ReportError(f"{path} traceEvents 存在非对象事件")
                if event.get("cat") != "kernel":
                    continue
                if event.get("ph") != "X":
                    raise ReportError(f"{path} kernel 不是完整 X duration 事件")
                name = event.get("name")
                if not isinstance(name, str) or not name.strip():
                    raise ReportError(f"{path} kernel 缺少有效 name")
                dur = number(event.get("dur"), f"{path} kernel dur")
                kernels[name]["duration_us"] += dur
                kernels[name]["count"] += 1
                count += 1
                duration += dur
            if not count:
                raise ReportError(f"{path} 无 cat=kernel 的实际 GPU kernel 事件")
            sources.append({"path": str(path), "kernel_event_count": count,
                            "duration_us": duration})
        except ReportError as exc:
            errors.append(str(exc))
    rows = [] if errors else sorted(
        ({"name": name, "device_type": "CUDA", "is_user_annotation": False,
          "cat": "kernel", "source_metric": "traceEvents.dur",
          # Compatibility alias for existing report consumers; not key averages.
          "self_device_time_us": value["duration_us"], **value}
         for name, value in kernels.items()),
        key=lambda op: op["duration_us"], reverse=True)
    return {"rows": rows, "status": "unavailable" if errors else "complete",
            "errors": errors, "source_stats": sources,
            "sources": [str(path) for path in paths]}


def batch_counts(ranks):
    config = ranks[0]["metadata"]["resolved_config"]
    batch = config.get("batch_size")
    segments = config.get("dataset", {}).get("segments_per_trajectory")
    if type(batch) is not int or batch <= 0 or config.get("gradient_accumulation_steps") != 1:
        raise ReportError("要求正整数 batch_size、gradient_accumulation_steps=1")
    if segments is not None and (type(segments) is not int or segments <= 0):
        raise ReportError("segments_per_trajectory 必须为正整数")
    if ranks[0]["metadata"]["schema_version"] == 3 and segments is None:
        raise ReportError("schema3 缺少 segments_per_trajectory，不能计算实际segment吞吐")
    result = {"per_rank_trajectories": batch, "global_trajectories": batch * len(ranks),
              "segments_per_trajectory": segments,
              "global_segments": batch * len(ranks) * segments if segments else None}
    for rank in ranks:
        recorded = rank["metadata"].get("batch")
        if recorded is not None and any(recorded.get(k) != result[k] for k in
                                       ("per_rank_trajectories", "global_trajectories", "global_segments")):
            raise ReportError("metadata.batch 与 resolved_config 不一致")
    return result


def iteration_split(row):
    """Disjoint loop siblings; H2D/telemetry remain nested, never added again."""
    whole = number(row.get("iteration_wall_s"), "iteration_wall_s", True)
    values = {"dataloader_next": number(row.get("dataloader_next_wall_s"), "dataloader_next_wall_s"),
              "update": number(row.get("update_wall_s"), "update_wall_s", True),
              "postupdate": number(row.get("post_update_wall_s"), "post_update_wall_s")}
    residual = whole - sum(values.values())
    if residual < -1e-8:
        raise ReportError("E2E sibling 时间和超过 iteration wall；拒绝非互斥分解")
    values["other residual"] = max(0.0, residual)
    return values


def summarize_e2e(ranks, batch):
    per_rank, splits, diagnostic = {}, [], []
    for rank in ranks:
        rid, rows = rank["rank"], rank["records"]
        for step, row in rows.items():
            values = iteration_split(row)
            start = number(row.get("iteration_start_monotonic_s"), "iteration start")
            end = number(row.get("iteration_end_monotonic_s"), "iteration end")
            if not math.isclose(end - start, row["iteration_wall_s"], rel_tol=1e-7, abs_tol=1e-7):
                raise ReportError("iteration 边界与 wall 不一致")
            if step < 40 and not math.isclose(end, rows[step + 1]["iteration_start_monotonic_s"], abs_tol=1e-7, rel_tol=0):
                raise ReportError("完整循环边界不连续")
            if 11 <= step <= 30:
                splits.append({"rank": rid, "step": step, **values})
            elif step >= 31:
                diagnostic.append({"rank": rid, "step": step, **values})
        steady = [rows[s] for s in range(11, 31)]
        elapsed = rows[30]["iteration_end_monotonic_s"] - rows[11]["iteration_start_monotonic_s"]
        done = rank["completion"]
        if done.get("count") != 20 or done.get("steady_steps") != [11, 30]:
            raise ReportError("schema3 完成标记缺少完整20步窗口")
        if not math.isclose(number(done.get("steady_window_elapsed_s"), "steady_window_elapsed_s", True),
                            elapsed, rel_tol=1e-7, abs_tol=1e-7):
            raise ReportError("summary 与 steps 的连续20步窗口不一致")
        telemetry, enqueue, worker = [], [], defaultdict(list)
        sync_h2d = []
        for row in steady:
            direct = sum(number(t.get("wall_s"), "telemetry wall_s") for t in row.get("gpu_telemetry", []))
            if direct > row["post_update_wall_s"] + 1e-8:
                raise ReportError("telemetry direct 超过 postupdate")
            telemetry.append(direct)
            h2d = sum(number(t.get("enqueue_wall_s"), "H2D enqueue") for t in row.get("h2d", []))
            if h2d > row["dataloader_next_wall_s"] + 1e-8:
                raise ReportError("H2D enqueue 超过 next 父区间")
            enqueue.append(h2d)
            for key, value in (row.get("worker_timing") or {}).items():
                worker[key].append(number(value, "worker " + key))
        for row in rows.values():
            for item in row.get("h2d", []):
                if item.get("diagnostic_sync"):
                    sync_h2d.append(number(item.get("synchronized_wall_s"), "diagnostic H2D"))
        per_rank[str(rid)] = {
            "steady_window_elapsed_s": elapsed,
            "iteration_wall_s": stats([r["iteration_wall_s"] for r in steady]),
            "mean_wall_s": {key: statistics.mean(s[key] for s in splits if s["rank"] == rid)
                            for key in ("dataloader_next", "update", "postupdate", "other residual")},
            "telemetry_direct_wall_s": stats(telemetry),
            "postupdate_nontelemetry_wall_s": stats([r["post_update_wall_s"] - t for r, t in zip(steady, telemetry)]),
            "h2d_enqueue_wall_s": stats(enqueue),
            "diagnostic_h2d_synchronized_wall_s": stats(sync_h2d) if sync_h2d else None,
            "worker_timing": {key: stats(values) for key, values in worker.items()},
        }
    elapsed = max(r["steady_window_elapsed_s"] for r in per_rank.values())
    critical = max(per_rank, key=lambda rid: per_rank[rid]["steady_window_elapsed_s"])
    means = {key: statistics.mean(s[key] for s in splits)
             for key in ("dataloader_next", "update", "postupdate", "other residual")}
    return {"steps": [11, 30], "samples": 80, "per_rank": per_rank,
            "max_rank_window_elapsed_s": elapsed, "slowest_window_rank": int(critical),
            "mean_step_wall_s": elapsed / 20,
            "global_updates_per_s": 20 / elapsed,
            "global_trajectories_per_s": 20 * batch["global_trajectories"] / elapsed,
            "global_segments_per_s": 20 * batch["global_segments"] / elapsed if batch["global_segments"] else None,
            "mean_wall_s": means, "per_rank_step": splits,
            "critical_window_mean_wall_s": per_rank[critical]["mean_wall_s"],
            "diagnostic_mean_wall_s": {key: statistics.mean(s[key] for s in diagnostic) for key in means},
            "semantics": "max_rank continuous 20-step elapsed; NOT sum of per-step rank maxima; no cross-rank clock subtraction",
            "residual_definition": "iteration - dataloader_next - update - postupdate; includes before-update synchronize, worker timing extraction, instrumentation/export before update return",
            "nested_warning": "H2D is inside next; telemetry is inside postupdate. Worker prefetch overlaps loop and cannot be added. Other-rank postupdate may include indirect telemetry/collective waits, not separately measured."}


def summarize(ranks, selected_rank, selected_step):
    batch = batch_counts(ranks)
    per_rank = {}
    phase_rows = []
    persistent = []
    module_rows = defaultdict(list)
    operators = defaultdict(lambda: {"self_device_time_us": 0.0, "self_cpu_time_us": 0.0, "count": 0})
    operator_sources = []
    for rank in ranks:
        rows = rank["records"]
        steady = [rows[s] for s in range(11, 31)]
        per_rank[str(rank["rank"])] = {
            "device": rank["device"],
            "steady_update_wall_s": stats([r["update_wall_s"] for r in steady]),
            "steady_peak_allocated_bytes": stats([r["whole_step_peak"]["allocated_bytes"] for r in steady]),
            "steady_peak_reserved_bytes": stats([r["whole_step_peak"]["reserved_bytes"] for r in steady]),
            "all40_max_peak_allocated_bytes": max(r["whole_step_peak"]["allocated_bytes"] for r in rows.values()),
            "all40_max_peak_reserved_bytes": max(r["whole_step_peak"]["reserved_bytes"] for r in rows.values()),
            "losses": [{"step": s, "loss": rows[s]["loss"],
                        "loss_components": rows[s].get("loss_components", {})} for s in range(1, 41)],
        }
        for step in range(31, 35):
            phase_rows.append({"rank": rank["rank"], "step": step, **phase_split(rows[step])})
        for step in range(35, 41 if rank["metadata"]["schema_version"] == 3 else 39):
            row = rows[step]
            for moment in (("storage_after_backward", "storage_after_update") if step <= 38 else ()):
                persistent.append({"rank": rank["rank"], "step": step, "moment": moment,
                                   **inventory_totals(row.get(moment))})
            for call in row.get("module_calls", []):
                module_rows[call["name"]].append(number(call["inclusive_cuda_ms"], "module inclusive_cuda_ms"))
        for step in (39, 40):
            path = rank["directory"] / f"step_{step:02d}.operators.json"
            data = read_json(path)
            operator_sources.append(str(path))
            for op in data.get("operators", []):
                if op.get("source_metric") not in ("self_device_time_total", "self_cuda_time_total"):
                    raise ReportError(f"{path} 算子必须使用 self device 时间")
                # Missing device_type in schema 1 cannot be reconstructed reliably.
                device_type = op.get("device_type", "UNKNOWN_LEGACY")
                key = (device_type, bool(op.get("is_user_annotation", False)), op["name"])
                operators[key]["self_device_time_us"] += number(op["self_device_time_us"], "operator self time")
                operators[key]["self_cpu_time_us"] += number(op.get("self_cpu_time_us", 0), "operator CPU self time")
                operators[key]["count"] += number(op["count"], "operator count")
    cross = [{"step": s, "max_rank_update_wall_s": max(r["records"][s]["update_wall_s"] for r in ranks),
              "slowest_rank": max(ranks, key=lambda r: r["records"][s]["update_wall_s"])["rank"]}
             for s in range(1, 41)]
    steady_cross = [r["max_rank_update_wall_s"] for r in cross if 11 <= r["step"] <= 30]
    phase_mean = {key: statistics.mean(r[key] for r in phase_rows) for key in (*PHASES, "other residual")}
    row = ranks[selected_rank]["records"][selected_step]
    saved = row.get("saved_activations_forward_end")
    if not isinstance(saved, dict) or not isinstance(saved.get("groups_storage_bytes"), dict):
        raise ReportError(f"rank {selected_rank}/step {selected_step} 缺少前向末端 saved activation 快照")
    groups = saved["groups_storage_bytes"]
    for key, value in groups.items():
        number(value, f"saved storage {key}")
    total = number(saved.get("total_storage_bytes"), "saved total_storage_bytes")
    if total <= 0 or sum(groups.values()) != total:
        raise ReportError("saved activation 为空或模块之和与 total_storage_bytes 不一致，无法生成真实饼图")
    forward = [p for p in row.get("phases", []) if p.get("name") == "forward" and p.get("status") == "ok"]
    if len(forward) != 1:
        raise ReportError("选定快照需要唯一完整 forward 阶段")
    allocated = number(forward[0].get("after", {}).get("allocated_bytes"), "forward-end allocated", True)
    if total > allocated:
        raise ReportError(f"saved activation storage {total} 超过同刻 allocated {allocated}，拒绝不可靠饼图")
    modules = [{"name": name, "per_call_inclusive_cuda_ms": stats(values)}
               for name, values in module_rows.items()]
    modules.sort(key=lambda m: m["per_call_inclusive_cuda_ms"]["mean"], reverse=True)
    ops = [{"name": name, "device_type": device, "is_user_annotation": annotation, **value}
           for (device, annotation, name), value in operators.items()]
    kernel_data = aggregate_trace_kernels([
        rank["directory"] / f"step_{step:02d}.trace.json"
        for rank in ranks for step in (39, 40)])
    cpu = sorted((op for op in ops if op["device_type"] == "CPU"),
                 key=lambda op: op["self_cpu_time_us"], reverse=True)
    legacy = [op for op in ops if op["device_type"] not in ("CPU", "CUDA")]
    decomposition = validate_forward_decomposition(row, allocated)

    return {
        "schema_version": 3 if ranks[0]["metadata"]["schema_version"] == 3 else 2,
        "status": "complete", "per_rank": per_rank, "batch": batch,
        "e2e": summarize_e2e(ranks, batch) if ranks[0]["metadata"]["schema_version"] == 3 else None,
        "steady_steps": [11, 30], "percentile_method": "p50 median; p95 nearest rank (ceil(.95*n)-1)",
        "synchronized_throughput": {
            "scope": "update_policy only; excludes loader, H2D, post-update logging; per-step max across four ranks",
            "steady_max_rank_update_wall_s": stats(steady_cross),
            "updates_per_second": 1 / statistics.mean(steady_cross),
            "global_samples_per_update": batch["global_trajectories"],
            "update_only_samples_per_second": batch["global_trajectories"] / statistics.mean(steady_cross),
            "update_only_trajectories_per_second": batch["global_trajectories"] / statistics.mean(steady_cross),
            "update_only_segments_per_second": batch["global_segments"] / statistics.mean(steady_cross) if batch["global_segments"] else None,
            "all_steps": cross},
        "time_share": {"steps": [31, 34], "ranks": [0, 1, 2, 3], "samples": len(phase_rows),
                       "mean_wall_s": phase_mean, "per_rank_step": phase_rows,
                       "scope": "synchronized sibling phases; equal mean over 16 rank-step pairs; not steady throughput"},
        "memory_share": {"kind": "saved_activation_module_share_only", "title": MEMORY_TITLE,
                         "rank": selected_rank, "step": selected_step,
                         "saved_snapshot": saved, "forward_end_allocated_bytes": allocated,
                         "saved_le_allocated_check": True,
                         "total_allocated_decomposition": decomposition,
                         "reason": ("Validated same-instant CUDA storage identities and allocated denominator" if decomposition else
                                    "Legacy schema lacks same-instant inventory and shared storage IDs; total allocated decomposition unavailable")},
        "persistent_storage": {"records": persistent,
                               "scope": "35–38 after-backward / after-update snapshots, not forward-end or peak; CUDA separate from CPU"},
        "modules_inclusive": {"steps": [35, 40 if ranks[0]["metadata"]["schema_version"] == 3 else 38], "rows": modules,
                              "warning": "inclusive per-call intervals; NEVER sum parent/child modules or infer exclusive time"},
        "operators_self_device": {"steps": [39, 40], "rank_step_samples": 8,
                                  "sources": kernel_data["sources"], "rows": kernel_data["rows"],
                                  "kernel_parse_status": kernel_data["status"],
                                  "kernel_parse_errors": kernel_data["errors"],
                                  "trace_source_stats": kernel_data["source_stats"],
                                  "cpu_sources": operator_sources,
                                  "kernel_metric": "sum traceEvents.dur (microseconds), count events; cat=kernel, ph=X",
                                  "cpu_rows": cpu, "legacy_unclassified_rows": legacy,
                                  "warning": "Only traceEvents cat=kernel, ph=X; excludes gpu_user_annotation, memcpy, memset and CPU events. Duration sums across streams/ranks are not wall or critical-path time. On any trace parse failure kernel rows are omitted, never inferred from operators.json. CPU self-time table retained separately."},
        "metadata_references": [{"rank": r["rank"], "metadata": str(r["directory"] / "metadata.json"),
                                 "device": str(r["directory"] / "device.json"),
                                 "config_pointer": "metadata.json#/resolved_config"} for r in ranks],
    }


def make_plots(summary, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as exc:
        raise ReportError("生成 PNG/SVG 需要 matplotlib；请使用已安装该依赖的 Python") from exc
    available = {f.name for f in font_manager.fontManager.ttflist}
    cjk = next((name for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei")
                if name in available), None)
    if cjk:
        plt.rcParams["font.sans-serif"] = [cjk, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"
    summary["plot_cjk_font"] = cjk
    summary["figures"] = []

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "svg"):
            filename = f"{name}.{ext}"
            fig.savefig(output / filename, dpi=150, bbox_inches="tight")
            summary["figures"].append(filename)
        plt.close(fig)

    def pie(values, labels, title, name):
        nonzero = [(v, label) for v, label in zip(values, labels) if v > 0]
        if not nonzero:
            raise ReportError(f"{name} 无正数样本，拒绝空饼图")
        fig, ax = plt.subplots(figsize=(10, 6))
        wedges, _, _ = ax.pie([x[0] for x in nonzero], autopct="%1.1f%%", startangle=90,
                             pctdistance=.78, textprops={"fontsize": 9})
        ax.legend(wedges, [x[1] for x in nonzero], loc="center left", bbox_to_anchor=(1, .5), fontsize=8)
        ax.set_title(title, fontsize=11)
        save(fig, name)

    times = summary["time_share"]["mean_wall_s"]
    pie(list(times.values()), [f"{k}: {v * 1000:.2f} ms" for k, v in times.items()],
        "Synchronized phase means | steps 31–34, all ranks\nDiagnostic update time; not steady throughput", "time_share")
    if summary.get("e2e"):
        e2e = summary["e2e"]
        times_e2e = e2e["critical_window_mean_wall_s"]
        pie(list(times_e2e.values()), [f"{k}: {v * 1000:.2f} ms" for k, v in times_e2e.items()],
            f"Full E2E loop | steady 11–30, slowest window rank {e2e['slowest_window_rank']}\nH2D nested in next; telemetry nested in postupdate", "e2e_time_share")
        fig, ax = plt.subplots(figsize=(10, 5))
        bottom = [0.] * 4
        for key in times_e2e:
            values = [e2e["per_rank"][str(r)]["mean_wall_s"][key] * 1000 for r in range(4)]
            ax.bar(range(4), values, bottom=bottom, label=key)
            bottom = [a + b for a, b in zip(bottom, values)]
        ax.set(xticks=range(4), xlabel="Rank", ylabel="Mean full iteration (ms)",
               title="Disjoint E2E loop components | steps 11–30")
        ax.legend(fontsize=8)
        save(fig, "e2e_rank_time")
        fig, ax = plt.subplots(figsize=(9, 4))
        for rid, data in e2e["per_rank"].items():
            points = [r for r in e2e["per_rank_step"] if str(r["rank"]) == rid]
            ax.plot([r["step"] for r in points], [sum(r[k] for k in times_e2e) for r in points], label=f"rank {rid}")
        ax.set(xlabel="Steady step", ylabel="Full iteration wall (s)", title="Full E2E loop (not update-only)")
        ax.legend(); ax.grid(alpha=.25)
        save(fig, "e2e_step_time")
    memory = summary["memory_share"]
    groups = sorted(memory["saved_snapshot"]["groups_storage_bytes"].items(), key=lambda x: x[1], reverse=True)
    if len(groups) > 10:
        groups = groups[:10] + [("other saved modules", sum(v for _, v in groups[10:]))]
    title = MEMORY_TITLE if cjk else "Forward-end saved activation; NOT total peak VRAM share"
    pie([v for _, v in groups], [f"{k}: {v / GIB:.3f} GiB" for k, v in groups],
        f"{title}\nrank {memory['rank']}, step {memory['step']}", "memory_share")

    decomposition = memory.get("total_allocated_decomposition")
    if decomposition:
        categories = decomposition["categories_bytes"]
        pie(list(categories.values()), [f"{k}: {v / GIB:.3f} GiB" for k, v in categories.items()],
            f"Forward-end total CUDA allocated (NOT peak/reserved)\nrank {memory['rank']}, step {memory['step']}",
            "total_allocated_share")

    fig, ax = plt.subplots(figsize=(9, 4))
    for rank, data in summary["per_rank"].items():
        ax.plot([r["step"] for r in data["losses"]],
                [r["loss"] if r["loss"] is not None else math.nan for r in data["losses"]],
                label=f"rank {rank}", alpha=.8)
    ax.set(xlabel="Update step", ylabel="Local loss", title="Short-run local loss (not a convergence test)")
    ax.legend(); ax.grid(alpha=.25)
    save(fig, "loss")

    fig, ax = plt.subplots(figsize=(9, 4))
    for rank, data in summary["per_rank"].items():
        # Per-rank durations are stored with the cross-rank trace below.
        ax.plot(range(1, 41), summary["rank_update_wall_s"][rank], label=f"rank {rank}", alpha=.55)
    cross = summary["synchronized_throughput"]["all_steps"]
    ax.plot([r["step"] for r in cross], [r["max_rank_update_wall_s"] for r in cross],
            color="black", linewidth=2, label="per-step max rank")
    for left, right, color in ((.5, 10.5, "gray"), (30.5, 34.5, "orange"), (34.5, 38.5, "purple"), (38.5, 40.5, "red")):
        ax.axvspan(left, right, color=color, alpha=.09)
    ax.set(xlabel="Update step (11–30 steady; 31–40 diagnostic)", ylabel="Synchronized update wall (s)", title="Update only: excludes loader / H2D / logging")
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    save(fig, "step_time")

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (field, label) in enumerate((("steady_peak_allocated_bytes", "Steady allocated"), ("steady_peak_reserved_bytes", "Steady reserved"))):
        ax.bar([r + (i - .5) * .35 for r in range(4)],
               [summary["per_rank"][str(r)][field]["max"] / GIB for r in range(4)], width=.35, label=label)
    ax.plot(range(4), [summary["per_rank"][str(r)]["all40_max_peak_allocated_bytes"] / GIB for r in range(4)],
            "ko", label="All 40 steps allocated max")
    ax.plot(range(4), [summary["per_rank"][str(r)]["all40_max_peak_reserved_bytes"] / GIB for r in range(4)],
            "kx", label="All 40 steps reserved max")
    ax.set(xticks=range(4), xlabel="Rank (do not sum ranks)", ylabel="GiB", title="Per-rank absolute allocator peaks")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
    save(fig, "rank_memory_peak")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    records = summary["persistent_storage"]["records"]
    for ax, moment, label in zip(axes, ("storage_after_backward", "storage_after_update"), ("After backward (before zero_grad)", "After update")):
        bottom = [0.] * 4
        for kind in ("parameter", "optimizer_state", "gradient"):
            vals = [statistics.mean(r["cuda"][kind] for r in records if r["rank"] == rank and r["moment"] == moment) / GIB for rank in range(4)]
            ax.bar(range(4), vals, bottom=bottom, label=kind)
            bottom = [a + b for a, b in zip(bottom, vals)]
        ax.set(xticks=range(4), xlabel="Rank", title=label)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("CUDA unique storage GiB (mean steps 35–38)")
    save(fig, "persistent_memory")

    ops = summary["operators_self_device"]["rows"][:15]
    if ops:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(range(len(ops)), [o["self_device_time_us"] / 1000 for o in ops])
        ax.set(yticks=range(len(ops)), yticklabels=[o["name"] for o in ops],
               xlabel="Kernel duration ms (trace dur sum, 4 ranks x 2 steps; not wall)", title="Actual GPU kernels: trace cat=kernel, ph=X")
        ax.invert_yaxis()
        save(fig, "operators_self_device")

    cpu = summary["operators_self_device"]["cpu_rows"][:15]
    if cpu:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(range(len(cpu)), [o["self_cpu_time_us"] / 1000 for o in cpu])
        ax.set(yticks=range(len(cpu)), yticklabels=[o["name"] for o in cpu],
               xlabel="Self CPU ms (sum 4 ranks × 2 steps; NOT device time)",
               title="CPU events / user annotations (separate from CUDA kernels)")
        ax.invert_yaxis()
        save(fig, "operators_self_cpu")


def overview_text(summary):
    batch = summary["batch"]
    update = summary["synchronized_throughput"]
    lines = ["## 直接回答：ViT、数据、完整步时和显存", ""]
    modules = {m["name"]: m["per_call_inclusive_cuda_ms"] for m in summary["modules_inclusive"]["rows"]}
    for name, label in (("function/vision/visual_only", "ViT visual_only：只包 vlm_visual.forward"),
                        ("function/prefix/vision", "prefix/vision：_vision，包括视觉周边处理"),
                        ("function/prefix/build_prefix", "prefix/build_prefix：完整 prefix 构建")):
        if name in modules:
            stat = modules[name]
            lines.append(f"- {label}：per-call inclusive CUDA mean={stat['mean']:.3f} ms，p50={stat['p50']:.3f}，p95={stat['p95']:.3f}，calls={stat['count']}；诊断 steps {summary['modules_inclusive']['steps']}、四 rank，不是 E2E 独占耗时。")
        else:
            lines.append(f"- {label}：该独立函数范围未采集，不用其他父子模块代替。")
    lines += ["三个范围有包含/重叠关系，分别列出，不相加、不把 prefix/vision 冒充纯 ViT；CUDA event 包含流中排队区间，不是 exclusive kernel 时间。",
              f"- update-only：逐步 max_rank mean={update['steady_max_rank_update_wall_s']['mean']:.6f} s，{update['update_only_trajectories_per_second']:.6f} trajectories/s；segments/s={update['update_only_segments_per_second']}。"]
    e2e = summary.get("e2e")
    if e2e:
        lines += [f"- E2E 完整20步窗口：max_rank elapsed={e2e['max_rank_window_elapsed_s']:.6f} s（rank {e2e['slowest_window_rank']}），折合 {e2e['mean_step_wall_s']:.6f} s/step；{e2e['global_updates_per_s']:.6f} updates/s，{e2e['global_trajectories_per_s']:.6f} trajectories/s，{e2e['global_segments_per_s']:.6f} segments/s。",
                  f"实际 batch：每 rank {batch['per_rank_trajectories']} 轨迹，全局 {batch['global_trajectories']} 轨迹/{batch['global_segments']} segments，每轨迹 {batch['segments_per_trajectory']} segments。吞吐分母是各 rank 自己连续20步窗口 elapsed 的最大值，不是20个逐步 max 的和，也不跨 rank 相减 monotonic 时钟。",
                  "### E2E 完整循环互斥分解", "饼图采用最慢完整窗口 rank 的20步均值，分类和严格闭合到该窗口 elapsed/20；各 rank 堆叠图也用各自20步均值。四 rank 等权均值另存 summary，不与慢卡窗口吞吐混作同一分母。"]
        for key, value in e2e["critical_window_mean_wall_s"].items():
            lines.append(f"- {key}：{value * 1000:.3f} ms/step（完整步内 {value / e2e['mean_step_wall_s']:.2%}）。")
        lines += ["定义：dataloader_next 含 Accelerate implicit H2D enqueue（诊断时含其同步等待）；update 为同步 update_policy；postupdate 从 update wrapper 返回到下一循环，含遥测、日志/pbar、跨 rank scalar reduce；other residual = iteration − next − update − postupdate，含 update 前 CUDA synchronize、worker timing 提取、wrapper 准备/退出、统计及诊断导出等。阶段计时前的 beforeupdate 同步没有进入 update，但必须保留在 E2E other，不能丢弃。",
                  "H2D 是 next 子项，telemetry 是 postupdate 子项，下列仅诊断展示，不再次加入饼图。31–40 诊断存在同步、hooks/profiler/export 扰动，不能用作稳态吞吐。",
                  "![完整循环占比](e2e_time_share.png)", "![各卡完整循环互斥耗时](e2e_rank_time.png)", "![稳态完整步时](e2e_step_time.png)",
                  "### 数据加载多久：主进程 next 等待与 worker 工作量分开"]
        for rid, data in e2e["per_rank"].items():
            lines.append(f"- rank {rid}：next mean={data['mean_wall_s']['dataloader_next'] * 1000:.3f} ms；其中 H2D enqueue mean={data['h2d_enqueue_wall_s']['mean'] * 1000:.3f} ms（不是 GPU 完成传输时间）。")
            for key, stat in data["worker_timing"].items():
                lines.append(f"  - worker {key}：20步 batch 内样本均值的 mean={stat['mean'] * 1000:.3f} ms，p95={stat['p95'] * 1000:.3f} ms（getitem_max_s 为 batch 样本 max 的分布）。")
            h2d = data["diagnostic_h2d_synchronized_wall_s"]
            if h2d:
                lines.append(f"  - 诊断 H2D synchronized mean={h2d['mean'] * 1000:.3f} ms，来自31–40步，含末端完成等待、不含开始前同步；非稳态传输完成时间。")
        lines += ["worker getitem/cache/frames/xform 在预取 worker 中运行，可与本步计算和其他 batch 重叠；cache/frames/xform 是 getitem 内部工作，非 next 额外互斥时间。不得将 worker 各项、getitem 或 H2D 再加到 iteration，next 等待接近零不代表数据处理免费。",
                  "### 遥测：rank0 直接时间与其他 rank 间接等待"]
        for rid, data in e2e["per_rank"].items():
            lines.append(f"- rank {rid}：直接 _collect_gpu_stats mean={data['telemetry_direct_wall_s']['mean'] * 1000:.3f} ms；postupdate 扣除此直接测量后 mean={data['postupdate_nontelemetry_wall_s']['mean'] * 1000:.3f} ms。")
        lines += ["nvidia-smi 仅 rank0 调用；其他 rank 的 postupdate 可能在 scalar reduce 等待 rank0，或在后续 update/DDP 间接等待。后者没有独立计时，不能声称全部 postupdate/残差就是遥测等待，也不能用 rank0 的直接耗时乘四。"]
    else:
        lines.append("- E2E/加载：旧采集只有 update-only，未采完整循环/next/postupdate，无法还原或比较 E2E。")
    lines.append(f"- 显存：稳态单卡 allocated 峰值（四卡 max，不相加）={max(d['steady_peak_allocated_bytes']['max'] for d in summary['per_rank'].values()) / GIB:.3f} GiB；reserved={max(d['steady_peak_reserved_bytes']['max'] for d in summary['per_rank'].values()) / GIB:.3f} GiB。完整同刻分解和 saved storage 图见下文。")
    lines += ["", "### 与上轮 bs1 的可比范围", "bs1 上轮只有 update-only，不能比较 E2E、加载或完整循环加速比。batch2 throughput per trajectory 的提升只能按相同 update-only 口径：新 trajectories/s ÷ bs1 trajectories/s − 1；不能仅比较 update 秒数，也不能把 segments/s 与 trajectories/s 混用。"]
    baseline = summary.get("baseline_comparison")
    if baseline:
        lines.append(f"本次 update-only={baseline['current_trajectories_per_s']:.6f} trajectories/s；上轮 bs1={baseline['baseline_trajectories_per_s']:.6f} trajectories/s；比值={baseline['ratio']:.4f}，提升={baseline['improvement_percent']:.2f}%（仅 update-only 可比，非 E2E）。来源：`{baseline['source']}`。")
    else:
        lines.append("未提供 --baseline-summary，故不虚构提升百分比；传入上轮已完成报告 summary.json 后自动计算。")
    return lines + [""]


def compare_baseline(summary, path):
    old = read_json(path)
    if old.get("status") != "complete" or old.get("schema_version") not in (2, 3):
        raise ReportError("baseline 必须是已完成的报告 summary schema2/3")
    if old.get("steady_steps") != [11, 30] or set(old.get("per_rank", {})) != {"0", "1", "2", "3"}:
        raise ReportError("baseline 必须同为四卡稳态11–30步报告")
    throughput = old["synchronized_throughput"]
    if throughput.get("global_samples_per_update") != 4:
        raise ReportError("baseline 必须为四卡 bs1（全局4轨迹）")
    previous = 4 / number(throughput["steady_max_rank_update_wall_s"]["mean"], "baseline update mean", True)
    current = summary["synchronized_throughput"]["update_only_trajectories_per_second"]
    return {"source": str(path), "baseline_trajectories_per_s": previous,
            "current_trajectories_per_s": current, "ratio": current / previous,
            "improvement_percent": (current / previous - 1) * 100,
            "scope": "update-only per-step max_rank; bs1 lacks comparable E2E"}


def report_text(summary, source):
    steady = summary["synchronized_throughput"]
    times = summary["time_share"]["mean_wall_s"]
    total = sum(times.values())
    memory = summary["memory_share"]
    lines = ["# Stage B v2 四卡真实 profiling 报告", "", f"数据源：`{source}`。仅离线读取，未启动、同步或修改训练。", "",
             "## 稳态与吞吐", "11–30 步共 20 步，p50 使用中位数，p95 使用 nearest-rank。四卡同一步 update 墙时取 max，作为同步吞吐口径；不是各卡均值。",
             f"边界仅 update_policy：不含 loader 等待、前置 H2D、后置日志。全局 batch={summary['batch']['global_trajectories']}、每 rank batch={summary['batch']['per_rank_trajectories']}、累积=1；samples 指 trajectory，不是 segment；此处只是 update-only 口径，不是端到端吞吐。"]
    for rank, data in summary["per_rank"].items():
        stat = data["steady_update_wall_s"]
        lines.append(f"- rank {rank}：mean={stat['mean']:.4f}s，p50={stat['p50']:.4f}s，p95={stat['p95']:.4f}s；稳态 allocated 峰值={data['steady_peak_allocated_bytes']['max'] / GIB:.3f} GiB，reserved 峰值={data['steady_peak_reserved_bytes']['max'] / GIB:.3f} GiB。")
    stat = steady["steady_max_rank_update_wall_s"]
    lines += [f"- 跨 rank 每步 max：mean={stat['mean']:.4f}s，p50={stat['p50']:.4f}s，p95={stat['p95']:.4f}s；{steady['updates_per_second']:.3f} updates/s、{steady['update_only_samples_per_second']:.3f} samples/s。", "",
              "## 诊断阶段时间饼图", "31–34 步 × 四 rank 共 16 个样本，按同一批样本的阶段 wall_s 分别取均值。仅使用互不重叠的 sibling 阶段 forward/backward_ddp/clip/optimizer/zero_grad/scheduler；other residual = update wall − 上述阶段之和，包含 policy_update（如有）、Python、盘点和同步开销等。",
              "阶段前后 cuda.synchronize；计时包含阶段末同步等待、DDP 等待，未包含计时器之前的同步。诊断阶段额外同步和 instrumentation 会扰动执行，不能作为稳态加速比。未用 inclusive 模块时间相加。"]
    for key, value in times.items():
        lines.append(f"- {key}：{value * 1000:.3f} ms（{value / total:.1%}）。")
    lines += ["", "![阶段时间](time_share.png)", "", "## 显存：独立 saved 占比与总 allocated 分解", f"**{MEMORY_TITLE}**。选取 rank {memory['rank']}、step {memory['step']} 单一快照，不混合 rank 或步骤。",
              f"该快照 saved storage={memory['saved_snapshot']['total_storage_bytes'] / GIB:.3f} GiB ≤ forward-end allocated={memory['forward_end_allocated_bytes'] / GIB:.3f} GiB，检查通过。饼图分母仅 saved storage，而非 allocated、reserved 或全步峰值。",
              ("新 schema 已在同一 forward-end 边界采集 inventory、saved 与 allocated，并按 storage 身份验证 CUDA 分类无交叉；总分解另图展示。"
               if memory.get("total_allocated_decomposition") else
               "旧 schema 缺少同刻 inventory/storage 身份；仅展示 saved 模块占比，不拼接其他时刻 storage，不推算总 allocated residual。"),
              "saved tensor 排除参数、buffer、batch 输入，按 storage 去重、归首个存活 save 的最深函数/模块。弱引用 wrapper 失效可能低估 saved；遗漏项留在 other residual，不伪装为精确 activation。旧 schema 无逐 storage 设备类型；新 schema 总分解仅计选定 CUDA device。",
              "![saved activation](memory_share.png)", "", "### 参数 / 优化器 / 梯度持久 storage", "独立展示 35–38 步 after-backward（zero_grad 前）与 after-update 的 CUDA storage 均值；这两者都不是 forward-end。全局 storage 去重，parameter > gradient > optimizer 优先分配；CPU 优化器状态留在 summary，不计入显存。实际零梯度按零显示，不用参数大小推算。", "![持久 storage](persistent_memory.png)", "",
              "## Loss、step time 与各卡峰值", "loss 为各 rank 本地真实观测；null 留空。40 步短程 loss 不代表收敛、泛化或训练效果，诊断段也不应用来判断性能回归。显存图区分 allocated / reserved、稳态 / 全 40 步，各卡峰值不相加。", "![loss](loss.png)", "![step time](step_time.png)", "![各卡显存](rank_memory_peak.png)", "",
              "## 模块与算子", "模块 CUDA event 为 inclusive 流区间；父子、调用和排队工作可重叠，summary 仅列 per-call 分布，不求和或画模块时间占比，也不将其解释为 exclusive kernel 时间。",
              "39–40 步 × 四 rank 的八份原始 step_39/40.trace.json：仅纳入 traceEvents 中 cat=kernel、ph=X 的实际 GPU kernel，按 name 聚合 dur（微秒）并计事件次数。排除 gpu_user_annotation、CPU、memcpy、memset 等；不使用 operators.json 的 device_type 或归因时间判断 kernel。CPU 事件/用户标注仍按 operators.json 的 self CPU 时间独立排序。任一 trace 解析失败或缺少 kernel，省略整个 kernel 排行且列出错误，不回退或伪造。跨卡两步 duration 累计非 wall/关键路径，并行 kernel 仍可重叠。"]
    for module in summary["modules_inclusive"]["rows"][:10]:
        stat = module["per_call_inclusive_cuda_ms"]
        lines.append(f"- inclusive 模块 `{module['name']}`：per-call mean={stat['mean']:.3f} ms，p50={stat['p50']:.3f} ms，p95={stat['p95']:.3f} ms，calls={stat['count']}；不可与父子相加。")
    for error in summary["operators_self_device"]["kernel_parse_errors"]:
        lines.append(f"- CUDA kernel 排行不可用：{error}")
    for op in summary["operators_self_device"]["rows"][:10]:
        lines.append(f"- CUDA kernel `{op['name']}`：四卡两步累计 {op['self_device_time_us'] / 1000:.3f} ms，calls={op['count']}。")
    if summary["operators_self_device"]["rows"]:
        lines.append("![实际 GPU kernel duration](operators_self_device.png)")
    if summary["operators_self_device"]["cpu_rows"]:
        lines += ["", "### CPU 事件 / user annotations（按 self CPU 独立排序）"]
        for op in summary["operators_self_device"]["cpu_rows"][:10]:
            lines.append(f"- CPU `{op['name']}`：累计 self CPU {op['self_cpu_time_us'] / 1000:.3f} ms，calls={op['count']}。")
        lines.append("![CPU events](operators_self_cpu.png)")
    if summary["operators_self_device"]["legacy_unclassified_rows"]:
        lines.append("- 存在旧 schema 未分类事件：见 summary 中 legacy_unclassified_rows；缺少 device_type，不声称算子独占排行。")
    decomposition = memory.get("total_allocated_decomposition")
    if decomposition:
        lines += ["", "### 同刻总 CUDA allocated 占比", "分母为 forward-end allocated，非 reserved、非全步峰值；storage 身份无交叉且分类之和等于 allocated。other residual 包含 buffer、batch 输入、未保存 activation、DDP bucket、workspace、allocator rounding 等，不强行归因。"]
        for kind, size in decomposition["categories_bytes"].items():
            lines.append(f"- {kind}：{size / GIB:.3f} GiB（{size / decomposition['allocated_bytes']:.1%}）。")
        lines.append("![总 allocated 分解](total_allocated_share.png)")
    lines += ["", "### 前向末端 saved storage 完整细分", "同一快照原始 groups_storage_bytes，storage 去重；占比分母仅 saved total，不是总显存峰值。"]
    for name, size in sorted(memory["saved_snapshot"]["groups_storage_bytes"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- `{name}`：{size} bytes，{size / GIB:.6f} GiB（saved 内 {size / memory['saved_snapshot']['total_storage_bytes']:.3%}）。")
    largest = max(times, key=times.get)
    lines += ["", "## 基于本次数据的优化核查建议", f"- 本次同步诊断中 `{largest}` 最大（{times[largest] / total:.1%}）；先用稳态复测及 trace 验证其是否也是实际关键路径，不能直接将诊断占比当作可实现加速。"]
    if times["backward_ddp"] >= times["forward"]:
        lines.append("- backward_ddp 不小于 forward：优先在 trace 分辨反向计算、NCCL 与 rank 等待，不能把全部 backward_ddp 当作通信成本。")
    else:
        lines.append("- forward 高于 backward_ddp：结合 self-device 算子与 trace 检查前向算力/访存热点；inclusive 模块排行仅用于定位候选边界。")
    if memory["saved_snapshot"]["groups_storage_bytes"]:
        owner = max(memory["saved_snapshot"]["groups_storage_bytes"], key=memory["saved_snapshot"]["groups_storage_bytes"].get)
        lines.append(f"- saved storage 最大归因模块为 `{owner}`：若确有显存压力，可评估该区域 activation checkpointing，但需验证重计算开销、正确性与吞吐；本数据不证明其决定总峰值。")
    lines += ["- p95、跨 rank 慢卡和 allocated/reserved 差距可用于发现复测线索；仅 20 个稳态样本，不据此断言通信瓶颈、内存泄漏或碎片化。", "",
              "## 配置、环境和可复现性", "直接引用各 rank 原有 metadata.json（含 resolved_config、torch_version、schedule 和测量口径）及 device.json，不读取当前配置冒充运行时配置，也不额外探测/修改训练环境。metadata 未记录的驱动、CUDA runtime、git 状态、数据版本不能追溯，报告不猜测。resolved_config 是 housekeeping override 后、trainer validate 前快照，并非保证最终所有内部状态。"]
    for ref in summary["metadata_references"]:
        lines.append(f"- rank {ref['rank']}：metadata `{ref['metadata']}`；device `{ref['device']}`；配置 JSON pointer `{ref['config_pointer']}`。")
    if not summary.get("plot_cjk_font"):
        lines.append("- 绘图环境没有已识别的中文字体，图内用等义英文标题避免缺字；中文显存标题在本文及 summary.memory_share.title 中完整保留。")
    lines += ["", "所有图均同时提供 PNG 和 SVG；完整数值、来源、样本范围与限制见 summary.json。", ""]
    lines[4:4] = overview_text(summary)
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="离线生成四卡 Stage B v2 profiling 报告；不启动/等待训练")
    parser.add_argument("profile_dir", type=Path, help="包含 rank_00…rank_03 的采集输出目录")
    parser.add_argument("--output-dir", type=Path, help="默认 PROFILE_DIR/report_stage_b_v2；须为空或不存在")
    parser.add_argument("--baseline-summary", type=Path, help="上轮bs1已完成报告summary.json；仅比较update-only吞吐")
    parser.add_argument("--memory-rank", type=int, choices=range(4), default=0)
    parser.add_argument("--memory-step", type=int, choices=range(35, 39), default=35)
    args = parser.parse_args(argv)
    try:
        source = args.profile_dir.expanduser().resolve()
        ranks = load_profile(source)
        summary = summarize(ranks, args.memory_rank, args.memory_step)
        if args.baseline_summary:
            summary["baseline_comparison"] = compare_baseline(summary, args.baseline_summary.expanduser().resolve())
        summary["source_directory"] = str(source)
        summary["rank_update_wall_s"] = {str(r["rank"]): [r["records"][s]["update_wall_s"] for s in range(1, 41)] for r in ranks}
        output = (args.output_dir or source / "report_stage_b_v2").expanduser().resolve()
        if output == source or output in source.parents or any(output == r["directory"] or r["directory"] in output.parents for r in ranks):
            raise ReportError("输出目录不能是采集根目录、其祖先或 rank 子目录")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ReportError(f"输出目录非空，拒绝覆盖：{output}")
        output.mkdir(parents=True, exist_ok=True)
        make_plots(summary, output)
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        (output / "REPORT.md").write_text(report_text(summary, source), encoding="utf-8")
        print(f"报告已生成：{output}")
        return 0
    except (ReportError, OSError, KeyError, TypeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
