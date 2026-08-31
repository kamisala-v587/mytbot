"""Atomic benchmark report writers and output-directory helpers."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .metrics import StageRecord, summarize_records


def resolve_output_dir(
    base: str | Path,
    *,
    exact: bool = False,
    now: datetime | None = None,
) -> Path:
    """Return an exact path or a collision-resistant timestamped run directory."""
    output = Path(base)
    if exact:
        return output
    stamp = (now or datetime.now()).strftime("%Y-%m-%d/%H-%M-%S-%f")
    return output / stamp


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    import io

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fields})
    return out.getvalue()


def write_report(
    output_dir: str | Path,
    records: Iterable[StageRecord],
    *,
    gpu_samples: Iterable[dict[str, Any]] = (),
    slow_samples: Iterable[dict[str, Any]] = (),
    slow_videos: Iterable[dict[str, Any]] = (),
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    rows = list(records)
    gpu = list(gpu_samples)
    summary = summarize_records(rows)
    summary["metadata"] = metadata or {}
    _atomic_text(
        output / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    stage_rows = [record.to_dict() for record in rows]
    stage_fields = [
        "stage",
        "elapsed_s",
        "device_elapsed_s",
        "rank",
        "step",
        "worker_id",
        "pid",
        "count",
        "metadata",
    ]
    for row in stage_rows:
        row["metadata"] = json.dumps(
            row.get("metadata", {}), ensure_ascii=False, sort_keys=True
        )
    _atomic_text(output / "stages.csv", _csv_text(stage_rows, stage_fields))
    gpu_fields = sorted({key for row in gpu for key in row}) or ["sample_time", "index"]
    _atomic_text(output / "gpu_samples.csv", _csv_text(gpu, gpu_fields))
    for name, values in (
        ("slow_samples.jsonl", slow_samples),
        ("slow_videos.jsonl", slow_videos),
    ):
        text = "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) + "\n"
            for value in values
        )
        _atomic_text(output / name, text)
    return summary


def format_terminal_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"记录数: {summary.get('record_count', 0)}",
        "阶段耗时（mean / p95 / max）:",
    ]
    for row in summary.get("bottlenecks", []):

        def fmt(value: float | None) -> str:
            return "-" if value is None else f"{value:.4f}s"

        lines.append(
            f"  {row['stage']}: {fmt(row.get('mean_s'))} / "
            f"{fmt(row.get('p95_s'))} / {fmt(row.get('max_s'))}"
        )
    return "\n".join(lines)
