"""Atomic benchmark reports, progressive snapshots, and shared run helpers."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .metrics import StageRecord, _json_safe, merge_rank_records, summarize_records


def resolve_output_dir(
    base: str | Path,
    *,
    exact: bool = False,
    now: datetime | None = None,
) -> Path:
    output = Path(base)
    if exact:
        return output
    stamp = (now or datetime.now()).strftime("%Y-%m-%d/%H-%M-%S-%f")
    return output / stamp


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_json(path: str | Path, value: Any) -> None:
    text = json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True)
    _atomic_text(Path(path), text + "\n")


@dataclass(frozen=True)
class RunSession:
    output_dir: Path
    generation: str
    rank: int
    world_size: int
    started_at: float

    @property
    def partial_path(self) -> Path:
        return self.output_dir / "partial" / f"rank-{self.rank:05d}.json"


def create_run_session(
    base: str | Path, accelerator: Any, *, exact: bool = False
) -> RunSession:
    rank = int(accelerator.process_index)
    world_size = int(accelerator.num_processes)
    payload: list[Any] = [None]
    if rank == 0:
        payload[0] = {
            "output_dir": str(resolve_output_dir(base, exact=exact).absolute()),
            "generation": (
                f"{datetime.now().strftime('%Y%m%dT%H%M%S%f')}-"
                f"{uuid.uuid4().hex[:8]}"
            ),
            "started_at": time.time(),
        }
        data = payload[0]
        session = RunSession(
            output_dir=Path(data["output_dir"]),
            generation=data["generation"],
            rank=rank,
            world_size=world_size,
            started_at=float(data["started_at"]),
        )
        session.output_dir.mkdir(parents=True, exist_ok=True)
        partial_dir = session.output_dir / "partial"
        if exact and partial_dir.exists():
            for stale in partial_dir.glob("rank-*.json"):
                stale.unlink(missing_ok=True)
        write_manifest(session, "running")

    if world_size > 1:
        try:
            import torch.distributed as distributed
        except ImportError:
            distributed = None
        if distributed is not None and distributed.is_initialized():
            distributed.broadcast_object_list(payload, src=0)
        else:
            from accelerate.utils import broadcast_object_list

            broadcast_object_list(payload, from_process=0)

    data = payload[0]
    if not isinstance(data, dict):
        raise RuntimeError("未能广播 benchmark 输出目录")
    return RunSession(
        output_dir=Path(data["output_dir"]),
        generation=data["generation"],
        rank=rank,
        world_size=world_size,
        started_at=float(data["started_at"]),
    )


def write_manifest(
    session: RunSession,
    status: str,
    *,
    error: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if session.rank != 0:
        return
    value = {
        "generation": session.generation,
        "status": status,
        "started_at": session.started_at,
        "updated_at": time.time(),
        "world_size": session.world_size,
        "output_dir": str(session.output_dir),
    }
    if error is not None:
        value["error"] = error
    if metadata:
        value["metadata"] = metadata
    atomic_json(session.output_dir / "manifest.json", value)


def snapshot_boundaries(total: int) -> tuple[int, ...]:
    """Return deterministic ceil-decile completed-item boundaries."""
    if total <= 0:
        return ()
    return tuple(sorted({math.ceil(total * tenth / 10) for tenth in range(1, 11)}))


@dataclass
class PhaseProgress:
    """Independent snapshot schedule for one warmup or measure phase."""

    phase: str
    total: int
    started: float
    completed: int = 0

    def __post_init__(self) -> None:
        self.boundaries = snapshot_boundaries(self.total)

    def advance(self, completed: int) -> bool:
        self.completed = completed
        return completed in self.boundaries



def log_phase(accelerator: Any, phase: str, message: str = "") -> None:
    if not accelerator.is_main_process:
        return
    suffix = f": {message}" if message else ""
    print(f"[bpva-benchmark] {phase}{suffix}", flush=True)


def log_progress(
    accelerator: Any,
    *,
    phase: str,
    completed: int,
    total: int,
    last_elapsed_s: float,
    started: float,
    path: str | Path,
) -> None:
    if not accelerator.is_main_process:
        return
    elapsed = max(0.0, time.perf_counter() - started)
    rate = completed / elapsed if elapsed else 0.0
    eta = (total - completed) / rate if rate else None
    eta_text = "-" if eta is None else f"{eta:.1f}s"
    print(
        f"[bpva-benchmark] {phase} {completed}/{total} "
        f"({completed / total:.0%}) last={last_elapsed_s:.4f}s "
        f"elapsed={elapsed:.1f}s rate={rate:.2f}/s ETA={eta_text} "
        f"partial={path}",
        flush=True,
    )


def exception_details(exc: BaseException) -> dict[str, Any]:
    return _json_safe(
        {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        }
    )


def record_failure(
    session: RunSession,
    accelerator: Any,
    exc: BaseException,
    snapshot: Any,
) -> dict[str, Any]:
    """Best-effort failure publication that never masks the original error."""
    details = exception_details(exc)
    try:
        snapshot(details)
    except BaseException as snapshot_exc:
        print(
            f"[bpva-benchmark] rank={accelerator.process_index} "
            f"failed snapshot error: {snapshot_exc}",
            flush=True,
        )
    try:
        write_manifest(session, "failed", error=details)
    except BaseException as manifest_exc:
        print(
            f"[bpva-benchmark] rank={accelerator.process_index} "
            f"failed manifest error: {manifest_exc}",
            flush=True,
        )
    print(
        f"[bpva-benchmark] rank={accelerator.process_index} failed: "
        f"{details['type']}: {details['message']}",
        flush=True,
    )
    return details


def write_partial(
    session: RunSession,
    *,
    kind: str,
    phase: str,
    completed: int,
    total: int,
    records: Iterable[StageRecord],
    collector: dict[str, Any] | None = None,
    memory: dict[str, Any] | None = None,
    monitor: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    status: str | None = None,
) -> Path:
    if status not in (None, "running", "failed", "local_complete"):
        raise ValueError(f"不支持的 partial status: {status}")
    effective_status = status or ("failed" if error else "running")
    if error is not None and effective_status != "failed":
        raise ValueError("带 error 的 partial 必须使用 failed status")
    value = {
        "schema_version": 1,
        "kind": kind,
        "generation": session.generation,
        "rank": session.rank,
        "world_size": session.world_size,
        "status": effective_status,
        "phase": phase,
        "completed": completed,
        "total": total,
        "updated_at": time.time(),
        "records": [
            record.to_dict() if isinstance(record, StageRecord) else _json_safe(record)
            for record in records
        ],
        "collector": collector,
        "memory": memory,
        "monitor": monitor,
        "error": error,
        "metadata": metadata or {},
    }
    atomic_json(session.partial_path, value)
    return session.partial_path


def load_local_complete_partials(
    output_dir: str | Path,
    *,
    generation: str,
    world_size: int,
    timeout_s: float,
    poll_interval_s: float,
) -> list[dict[str, Any]]:
    """Wait for and load one matching local-complete partial per rank."""
    if world_size <= 0:
        raise ValueError("world_size 必须为正整数")
    if timeout_s <= 0 or poll_interval_s <= 0:
        raise ValueError("timeout_s 和 poll_interval_s 必须为正数")

    partial_dir = Path(output_dir) / "partial"
    deadline = time.monotonic() + timeout_s
    last_state: dict[int, str] = {}
    while True:
        loaded: list[dict[str, Any] | None] = [None] * world_size
        for rank in range(world_size):
            path = partial_dir / f"rank-{rank:05d}.json"
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                last_state[rank] = "missing"
                continue
            except (OSError, json.JSONDecodeError) as exc:
                last_state[rank] = f"unreadable({type(exc).__name__})"
                continue

            if not isinstance(payload, dict):
                last_state[rank] = "invalid(non-object)"
                continue
            actual_generation = payload.get("generation")
            status = payload.get("status")
            if actual_generation != generation:
                last_state[rank] = f"stale_generation({actual_generation!r})"
                continue
            if status != "local_complete":
                last_state[rank] = f"status({status!r})"
                continue
            if payload.get("rank") != rank:
                last_state[rank] = f"rank_mismatch({payload.get('rank')!r})"
                continue
            if payload.get("world_size") != world_size:
                last_state[rank] = f"world_size_mismatch({payload.get('world_size')!r})"
                continue
            if not isinstance(payload.get("records"), list):
                last_state[rank] = "invalid(records)"
                continue
            collector = payload.get("collector")
            if not isinstance(collector, dict):
                last_state[rank] = "invalid(collector)"
                continue
            if not isinstance(collector.get("top_events"), list) or not isinstance(
                collector.get("stats"), dict
            ):
                last_state[rank] = "invalid(collector_contents)"
                continue
            monitor = payload.get("monitor")
            if monitor is not None and (
                not isinstance(monitor, dict)
                or not isinstance(monitor.get("samples"), list)
                or not isinstance(monitor.get("errors"), list)
            ):
                last_state[rank] = "invalid(monitor)"
                continue
            if (
                not isinstance(payload.get("phase"), str)
                or not isinstance(payload.get("completed"), int)
                or not isinstance(payload.get("total"), int)
            ):
                last_state[rank] = "invalid(progress)"
                continue
            loaded[rank] = payload
            last_state.pop(rank, None)

        if all(item is not None for item in loaded):
            return [item for item in loaded if item is not None]
        if time.monotonic() >= deadline:
            missing = [
                rank for rank in range(world_size) if last_state.get(rank) == "missing"
            ]
            stale = {
                rank: last_state.get(rank, "missing")
                for rank in range(world_size)
                if rank not in missing
            }
            raise TimeoutError(
                "等待 local_complete partial 超时: "
                f"generation={generation!r}, missing_ranks={missing}, "
                f"stale_or_incomplete_ranks={stale}"
            )
        time.sleep(min(poll_interval_s, max(0.0, deadline - time.monotonic())))


def _csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    import io

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fields})
    return output.getvalue()


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
    atomic_json(output / "summary.json", summary)

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

    def format_seconds(value: float | None) -> str:
        return "-" if value is None else f"{value:.4f}s"

    for row in summary.get("bottlenecks", []):
        lines.append(
            f"  {row['stage']}: {format_seconds(row.get('mean_s'))} / "
            f"{format_seconds(row.get('p95_s'))} / "
            f"{format_seconds(row.get('max_s'))}"
        )
    return "\n".join(lines)
