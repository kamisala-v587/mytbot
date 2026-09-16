"""Metric records and aggregation without third-party dependencies."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence


@dataclass
class StageRecord:
    stage: str
    elapsed_s: float
    rank: int = 0
    step: int | None = None
    worker_id: int | None = None
    pid: int | None = None
    device_elapsed_s: float | None = None
    count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            pass
    return str(value)


def percentile(values: Sequence[float], q: float) -> float | None:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return None
    q = min(100.0, max(0.0, float(q)))
    position = (len(clean) - 1) * q / 100.0
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - position) + clean[hi] * (position - lo)


def summarize_values(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    total = sum(clean)
    return {
        "count": len(clean),
        "total_s": total,
        "mean_s": total / len(clean) if clean else None,
        "min_s": min(clean) if clean else None,
        "max_s": max(clean) if clean else None,
        "p50_s": percentile(clean, 50),
        "p90_s": percentile(clean, 90),
        "p95_s": percentile(clean, 95),
        "p99_s": percentile(clean, 99),
    }


def derive_bp_compressor_estimates(
    records: Iterable[StageRecord],
) -> list[StageRecord]:
    """Estimate BP non-visual cost as ``bp_encoder - bp_visual_encode`` per step.

    BPVAv2's compressor is not a single child module; pairing the whole BP
    encoder wall/device time with the timed ``_encode_images`` path attributes
    how much of BP is Qwen visual vs the rest (state/action + query compressor).
    """
    by_key: dict[tuple[int, int | None], dict[str, StageRecord]] = {}
    for record in records:
        if record.stage not in {"bp_encoder", "bp_visual_encode"}:
            continue
        by_key.setdefault((record.rank, record.step), {})[record.stage] = record

    derived: list[StageRecord] = []
    for (rank, step), pair in sorted(by_key.items()):
        parent = pair.get("bp_encoder")
        visual = pair.get("bp_visual_encode")
        if parent is None or visual is None:
            continue
        cpu = max(0.0, float(parent.elapsed_s) - float(visual.elapsed_s))
        device = None
        if parent.device_elapsed_s is not None and visual.device_elapsed_s is not None:
            device = max(
                0.0, float(parent.device_elapsed_s) - float(visual.device_elapsed_s)
            )
        derived.append(
            StageRecord(
                "bp_compressor_est",
                cpu,
                rank=rank,
                step=step,
                device_elapsed_s=device,
                metadata={
                    "definition": "bp_encoder - bp_visual_encode",
                    "bp_encoder_s": parent.elapsed_s,
                    "bp_visual_encode_s": visual.elapsed_s,
                },
            )
        )
    return derived


def summarize_records(records: Iterable[StageRecord]) -> dict[str, Any]:
    grouped: dict[str, list[StageRecord]] = {}
    by_rank: dict[int, list[StageRecord]] = {}
    base_records = [
        record for record in records if record.stage != "bp_compressor_est"
    ]
    derived = derive_bp_compressor_estimates(base_records)
    all_records = base_records + derived
    for record in all_records:
        grouped.setdefault(record.stage, []).append(record)
        by_rank.setdefault(record.rank, []).append(record)
    stages = {}
    for stage, rows in grouped.items():
        summary = summarize_values([r.elapsed_s for r in rows])
        gpu = [r.device_elapsed_s for r in rows if r.device_elapsed_s is not None]
        if gpu:
            summary["device"] = summarize_values(gpu)
        summary["ranks"] = sorted({r.rank for r in rows})
        stages[stage] = summary
    rank_summary = {
        str(rank): {
            stage: summarize_values([r.elapsed_s for r in rows if r.stage == stage])
            for stage in sorted({r.stage for r in rows})
        }
        for rank, rows in sorted(by_rank.items())
    }
    bottlenecks = sorted(
        ({"stage": stage, **stats} for stage, stats in stages.items()),
        key=lambda x: (
            (x.get("device") or {}).get("mean_s")
            if (x.get("device") or {}).get("mean_s") is not None
            else (x.get("mean_s") or 0.0)
        ),
        reverse=True,
    )
    return _json_safe(
        {
            "record_count": len(all_records),
            "derived_record_count": len(derived),
            "stages": stages,
            "ranks": rank_summary,
            "bottlenecks": bottlenecks,
            "bp_attribution": _bp_attribution_block(stages),
        }
    )


def _bp_attribution_block(stages: dict[str, Any]) -> dict[str, Any]:
    """Compact proof block: decode wait vs BP visual vs BP remainder."""

    def pick(stage: str) -> dict[str, Any] | None:
        stats = stages.get(stage)
        if not stats:
            return None
        device = stats.get("device") or {}
        return {
            "stage": stage,
            "mean_s": stats.get("mean_s"),
            "device_mean_s": device.get("mean_s"),
            "p95_s": stats.get("p95_s"),
            "device_p95_s": device.get("p95_s"),
            "count": stats.get("count"),
        }

    return {
        "data_wait": pick("data_wait"),
        "bp_encoder": pick("bp_encoder"),
        "bp_visual_encode": pick("bp_visual_encode"),
        "bp_qwen_visual": pick("bp_qwen_visual"),
        "bp_compressor_est": pick("bp_compressor_est"),
        "qwen_visual_current": pick("qwen_visual"),
        "method.embed_prefix": pick("method.embed_prefix"),
        "forward": pick("forward"),
        "interpretation": (
            "If data_wait << bp_visual_encode and bp_visual_encode ≈ bp_encoder, "
            "the BP bottleneck is Qwen visual encode of BP frames, not video decode. "
            "Cross-check TBot: compare qwen_visual / method.embed_prefix (current obs only) "
            "vs BPVA bp_visual_encode (extra 8-frame pass)."
        ),
    }


def merge_rank_records(
    rank_records: Iterable[Iterable[StageRecord | dict[str, Any]]],
) -> list[StageRecord]:
    merged = []
    for rows in rank_records:
        for row in rows:
            merged.append(row if isinstance(row, StageRecord) else StageRecord(**row))
    return merged


def aggregate_step_stragglers(
    records: Iterable[StageRecord],
    *,
    stages: Sequence[str] = (
        "microstep_wall",
        "optimizer_step_wall",
        "data_wait",
        "train_compute_wall",
        "forward",
        "bp_encoder",
        "bp_visual_encode",
        "bp_qwen_visual",
        "bp_compressor_est",
    ),
) -> list[dict[str, Any]]:
    """Aggregate equal report-step/stage rows across ranks without losing raw rows."""
    grouped: dict[tuple[str, int], list[StageRecord]] = {}
    allowed = set(stages)
    for record in records:
        if record.stage in allowed and record.step is not None:
            grouped.setdefault((record.stage, record.step), []).append(record)
    result = []
    for (stage, step), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: row.elapsed_s)
        values = [row.elapsed_s for row in ordered]
        maximum = ordered[-1]
        median = percentile(values, 50)
        result.append(_json_safe({
            "stage": stage, "step": step, "rank_count": len(rows),
            "rank_max_s": maximum.elapsed_s, "straggler_rank": maximum.rank,
            "rank_median_s": median,
            "rank_spread_s": maximum.elapsed_s - ordered[0].elapsed_s,
            "max_over_median_s": maximum.elapsed_s - median if median is not None else None,
            "optimizer_step": maximum.metadata.get("optimizer_step"),
            "microstep": maximum.metadata.get("microstep"),
            "ranks": [{"rank": row.rank, "elapsed_s": row.elapsed_s} for row in sorted(rows, key=lambda row: row.rank)],
        }))
    return result
