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


def summarize_records(records: Iterable[StageRecord]) -> dict[str, Any]:
    grouped: dict[str, list[StageRecord]] = {}
    by_rank: dict[int, list[StageRecord]] = {}
    all_records = list(records)
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
        key=lambda x: (x.get("mean_s") or 0.0),
        reverse=True,
    )
    return _json_safe(
        {
            "record_count": len(all_records),
            "stages": stages,
            "ranks": rank_summary,
            "bottlenecks": bottlenecks,
        }
    )


def merge_rank_records(
    rank_records: Iterable[Iterable[StageRecord | dict[str, Any]]],
) -> list[StageRecord]:
    merged = []
    for rows in rank_records:
        for row in rows:
            merged.append(row if isinstance(row, StageRecord) else StageRecord(**row))
    return merged
