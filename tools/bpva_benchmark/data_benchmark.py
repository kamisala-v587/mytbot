"""BPVA data-pipeline throughput benchmark."""

from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Any

from .data_instrumentation import (
    DataInstrumentation,
    EventCollector,
    WorkerInstrumentation,
)
from .metrics import StageRecord, merge_rank_records
from .reporting import (
    PhaseProgress,
    create_run_session,
    format_terminal_summary,
    load_local_complete_partials,
    log_phase,
    log_progress,
    record_failure,
    write_manifest,
    write_partial,
    write_report,
)
from .system_monitor import SystemMonitor, memory_snapshot
from .train_benchmark import (
    nonnegative_float,
    nonnegative_int,
    positive_float,
    positive_int,
)


def probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("必须位于 [0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BPVA 数据吞吐测评（不会修改输入配置）"
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-dir", default="outputs/bpva_benchmark/data")
    parser.add_argument("--exact-output-dir", action="store_true")
    parser.add_argument("--warmup-batches", type=nonnegative_int, default=5)
    parser.add_argument("--measure-batches", type=positive_int, default=50)
    parser.add_argument("--num-workers", type=nonnegative_int)
    parser.add_argument("--bp-num-chunks", type=positive_int)
    parser.add_argument("--repo-id-file")
    parser.add_argument("--sample-rate", type=probability, default=1.0)
    parser.add_argument("--slow-sample-threshold", type=nonnegative_float, default=1.0)
    parser.add_argument("--slow-video-threshold", type=nonnegative_float, default=0.5)
    parser.add_argument("--top-k", type=nonnegative_int, default=100)
    parser.add_argument("--queue-size", type=positive_int, default=4096)
    parser.add_argument("--disable-video-instrumentation", action="store_true")
    parser.add_argument(
        "--skip-send-to-device",
        action="store_true",
        help="仅对 dist_loading 生效；跳过显式 H2D",
    )
    parser.add_argument("--monitor-interval", type=positive_float, default=1.0)
    parser.add_argument("--finalize-timeout", type=positive_float, default=300.0)
    parser.add_argument("--finalize-poll-interval", type=positive_float, default=0.2)
    return parser


def _load_cfg(path: str):
    from .config_utils import register_bpva_configs

    register_bpva_configs()
    from lerobot.configs.train import TrainPipelineConfig

    return copy.deepcopy(TrainPipelineConfig.from_pretrained(path))


def disable_dist_loading_for_single_process(cfg: Any, num_processes: int) -> bool:
    """Disable incompatible dist loading on the in-memory data-benchmark config."""
    if num_processes == 1 and bool(getattr(cfg.dataset, "dist_loading", False)):
        cfg.dataset.dist_loading = False
        return True
    return False


def _overrides(cfg: Any, args: argparse.Namespace) -> None:
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.bp_num_chunks is not None:
        if hasattr(cfg.dataset, "bp_num_chunks"):
            cfg.dataset.bp_num_chunks = args.bp_num_chunks
        if cfg.policy is not None and hasattr(cfg.policy, "bp_num_chunks"):
            cfg.policy.bp_num_chunks = args.bp_num_chunks
    if args.repo_id_file is not None:
        if not hasattr(cfg.dataset, "repo_id_file"):
            raise ValueError("此 dataset 配置不支持 repo_id_file")
        cfg.dataset.repo_id_file = args.repo_id_file


def _effective_num_workers(cfg: Any) -> int:
    """Return the DataLoader worker count actually used by this benchmark."""
    return 1 if cfg.dataset.streaming else cfg.num_workers


def _loader(dataset: Any, cfg: Any, device: Any, worker_init_fn: Any = None):
    import torch
    from lerobot.datasets.sampler import MultiLeRobotWeightedSampler

    weighted = (
        not cfg.dataset.streaming
        and getattr(dataset, "dataset_weights", None) is not None
    )
    sampler = MultiLeRobotWeightedSampler(dataset=dataset) if weighted else None
    num_workers = _effective_num_workers(cfg)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=not cfg.dataset.streaming and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=(
            (4 if cfg.dataset.streaming else 2) if num_workers > 0 else None
        ),
        worker_init_fn=worker_init_fn,
    )


def merge_data_partials(partials: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge fully loaded per-rank data partials without distributed calls."""
    records = merge_rank_records(partial["records"] for partial in partials)
    detail: list[dict[str, Any]] = []
    collector_stats: list[dict[str, Any]] = []
    gpu_samples: list[dict[str, Any]] = []
    monitor_errors: list[Any] = []
    for partial in partials:
        collector = partial.get("collector") or {}
        detail.extend(collector.get("top_events") or [])
        collector_stats.append(collector.get("stats") or {})
        monitor = partial.get("monitor") or {}
        gpu_samples.extend(monitor.get("samples") or [])
        monitor_errors.extend(monitor.get("errors") or [])
    return {
        "records": records,
        "slow_samples": [event for event in detail if event.get("kind") == "sample"],
        "slow_videos": [event for event in detail if event.get("kind") == "video"],
        "event_collector_by_rank": collector_stats,
        "gpu_samples": gpu_samples,
        "monitor_errors": monitor_errors,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from accelerate import Accelerator
    from accelerate.utils import send_to_device
    from lerobot.datasets.factory import make_dataset

    accelerator = Accelerator()
    session = create_run_session(
        args.output_dir, accelerator, exact=args.exact_output_dir
    )
    if accelerator.is_main_process:
        print(f"[bpva-benchmark] 输出目录: {session.output_dir}", flush=True)

    cfg = None
    collector = None
    monitor = None
    instrumentation = None
    records: list[StageRecord] = []
    memory_start = memory_snapshot()
    memory_after_dataset = None
    progress = PhaseProgress("dataset_init", 0, time.perf_counter())
    dist_loading_overridden = False

    def snapshot(error: dict[str, Any] | None = None, *, status: str | None = None):
        collector_state = collector.snapshot() if collector is not None else None
        monitor_state = monitor.snapshot() if monitor is not None else None
        gradient_metadata = {"config_path": args.config_path}
        return write_partial(
            session,
            kind="data",
            phase=progress.phase,
            completed=progress.completed,
            total=progress.total,
            records=records,
            collector=collector_state,
            memory={
                "start": memory_start,
                "after_dataset": memory_after_dataset,
                "current": memory_snapshot(),
            },
            monitor=monitor_state,
            error=error,
            metadata=gradient_metadata,
            status=status,
        )

    try:
        snapshot()
        cfg = _load_cfg(args.config_path)
        _overrides(cfg, args)
        dist_loading_overridden = disable_dist_loading_for_single_process(
            cfg, accelerator.num_processes
        )
        collector = EventCollector(args.queue_size, args.top_k).start()
        if accelerator.is_main_process:
            monitor = SystemMonitor(args.monitor_interval).start()
        instrumentation = DataInstrumentation(
            event_queue=collector.queue,
            sample_rate=args.sample_rate,
            slow_sample_s=args.slow_sample_threshold,
            slow_video_s=args.slow_video_threshold,
            video=not args.disable_video_instrumentation,
        ).install()

        log_phase(accelerator, "dataset_init")
        began = time.perf_counter()
        dataset, _ = make_dataset(cfg)
        records.append(
            StageRecord(
                "dataset_init",
                time.perf_counter() - began,
                accelerator.process_index,
                pid=os.getpid(),
            )
        )
        memory_after_dataset = memory_snapshot()

        effective_workers = _effective_num_workers(cfg)
        worker_init = None
        if effective_workers > 0:
            worker_init = WorkerInstrumentation(
                collector.queue,
                args.sample_rate,
                args.slow_sample_threshold,
                args.slow_video_threshold,
                not args.disable_video_instrumentation,
            )
        dataloader = _loader(dataset, cfg, accelerator.device, worker_init)
        if not cfg.dataset.dist_loading:
            dataloader = accelerator.prepare(dataloader)
        iterator = iter(dataloader)

        progress = PhaseProgress("warmup", args.warmup_batches, time.perf_counter())
        log_phase(accelerator, "warmup", f"0/{progress.total}")
        for warmup_index in range(progress.total):
            began = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)
            if cfg.dataset.dist_loading and not args.skip_send_to_device:
                batch = send_to_device(batch, accelerator.device, non_blocking=True)
            elapsed = time.perf_counter() - began
            if progress.advance(warmup_index + 1):
                path = snapshot()
                log_progress(
                    accelerator,
                    phase=progress.phase,
                    completed=progress.completed,
                    total=progress.total,
                    last_elapsed_s=elapsed,
                    started=progress.started,
                    path=path,
                )

        progress = PhaseProgress("measure", args.measure_batches, time.perf_counter())
        log_phase(accelerator, "measure", f"0/{progress.total}")
        for step in range(progress.total):
            began = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)
            elapsed = time.perf_counter() - began
            records.append(
                StageRecord(
                    "next_dataloader",
                    elapsed,
                    accelerator.process_index,
                    step,
                    pid=os.getpid(),
                )
            )
            if cfg.dataset.dist_loading and not args.skip_send_to_device:
                began = time.perf_counter()
                batch = send_to_device(batch, accelerator.device, non_blocking=True)
                records.append(
                    StageRecord(
                        "send_to_device",
                        time.perf_counter() - began,
                        accelerator.process_index,
                        step,
                        pid=os.getpid(),
                    )
                )
            if progress.advance(step + 1):
                path = snapshot()
                log_progress(
                    accelerator,
                    phase=progress.phase,
                    completed=progress.completed,
                    total=progress.total,
                    last_elapsed_s=elapsed,
                    started=progress.started,
                    path=path,
                )

        collector.stop()
        if monitor is not None:
            monitor.stop()

        progress = PhaseProgress("local_finalize", 1, time.perf_counter())
        progress.completed = 1
        log_phase(accelerator, progress.phase)
        snapshot(status="local_complete")

        if not accelerator.is_main_process:
            return

        progress = PhaseProgress("filesystem_merge", 0, time.perf_counter())
        log_phase(accelerator, progress.phase)
        partials = load_local_complete_partials(
            session.output_dir,
            generation=session.generation,
            world_size=session.world_size,
            timeout_s=args.finalize_timeout,
            poll_interval_s=args.finalize_poll_interval,
        )
        merged = merge_data_partials(partials)

        progress = PhaseProgress("report", 0, time.perf_counter())
        log_phase(accelerator, progress.phase)
        metadata = {
            "kind": "data",
            "completion": "completed",
            "world_size": accelerator.num_processes,
            "config_path": args.config_path,
            "output_dir": str(session.output_dir),
            "generation": session.generation,
            "dist_loading_single_process_override": (dist_loading_overridden),
            "memory_start": memory_start,
            "memory_after_dataset": memory_after_dataset,
            "memory_end": memory_snapshot(),
            "event_collector_by_rank": merged["event_collector_by_rank"],
            "monitor_errors": merged["monitor_errors"],
            "video_instrumentation": (not args.disable_video_instrumentation),
            "video_instrumentation_capability": (
                "whole decode call duration and requested timestamps only; "
                "open/seek/decode/close, actual decoded frames, and "
                "amplification unavailable"
            ),
        }
        summary = write_report(
            session.output_dir,
            merged["records"],
            gpu_samples=merged["gpu_samples"],
            slow_samples=merged["slow_samples"],
            slow_videos=merged["slow_videos"],
            metadata=metadata,
        )
        write_manifest(
            session,
            "completed",
            metadata={
                "record_count": len(merged["records"]),
                "monitor_errors": merged["monitor_errors"],
            },
        )
        log_phase(
            accelerator,
            "completed",
            f"output={session.output_dir}",
        )
        print(format_terminal_summary(summary), flush=True)
    except BaseException as exc:
        record_failure(session, accelerator, exc, snapshot)
        raise
    finally:
        if instrumentation is not None:
            instrumentation.uninstall()
        if collector is not None:
            collector.stop()
        if monitor is not None:
            monitor.stop()


if __name__ == "__main__":
    main()
