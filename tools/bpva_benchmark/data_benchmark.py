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
from .reporting import format_terminal_summary, resolve_output_dir, write_report
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


def _loader(dataset: Any, cfg: Any, device: Any, worker_init_fn: Any = None):
    import torch
    from lerobot.datasets.sampler import MultiLeRobotWeightedSampler

    weighted = (
        not cfg.dataset.streaming
        and getattr(dataset, "dataset_weights", None) is not None
    )
    sampler = MultiLeRobotWeightedSampler(dataset=dataset) if weighted else None
    num_workers = 1 if cfg.dataset.streaming else cfg.num_workers
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


def _gather_object(value: Any, accelerator: Any):
    if accelerator.num_processes == 1:
        return [value]
    from lerobot.utils.utils import gather_object

    return gather_object(value, accelerator)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from accelerate import Accelerator
    from accelerate.utils import send_to_device
    from lerobot.datasets.factory import make_dataset

    accelerator = Accelerator()
    cfg = _load_cfg(args.config_path)
    _overrides(cfg, args)
    dist_loading_overridden = disable_dist_loading_for_single_process(
        cfg, accelerator.num_processes
    )
    collector = EventCollector(args.queue_size, args.top_k).start()
    monitor = (
        SystemMonitor(args.monitor_interval).start()
        if accelerator.is_main_process
        else None
    )
    records: list[StageRecord] = []
    events: list[dict[str, Any]] = []
    gpu: list[dict[str, Any]] = []
    memory_start = memory_snapshot()
    memory_after_dataset = None

    instrumentation = DataInstrumentation(
        event_queue=collector.queue,
        sample_rate=args.sample_rate,
        slow_sample_s=args.slow_sample_threshold,
        slow_video_s=args.slow_video_threshold,
        video=not args.disable_video_instrumentation,
    )
    try:
        with instrumentation:
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
            worker_init = (
                WorkerInstrumentation(
                    collector.queue,
                    args.sample_rate,
                    args.slow_sample_threshold,
                    args.slow_video_threshold,
                    not args.disable_video_instrumentation,
                )
                if cfg.num_workers > 0
                else None
            )
            dataloader = _loader(dataset, cfg, accelerator.device, worker_init)
            if not cfg.dataset.dist_loading:
                dataloader = accelerator.prepare(dataloader)
            iterator = iter(dataloader)
            total = args.warmup_batches + args.measure_batches
            for microstep in range(total):
                began = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(dataloader)
                    batch = next(iterator)
                elapsed = time.perf_counter() - began
                measured = microstep >= args.warmup_batches
                report_step = microstep - args.warmup_batches
                if measured:
                    records.append(
                        StageRecord(
                            "next_dataloader",
                            elapsed,
                            accelerator.process_index,
                            report_step,
                            pid=os.getpid(),
                        )
                    )
                if cfg.dataset.dist_loading and not args.skip_send_to_device:
                    began = time.perf_counter()
                    batch = send_to_device(batch, accelerator.device, non_blocking=True)
                    if measured:
                        records.append(
                            StageRecord(
                                "send_to_device",
                                time.perf_counter() - began,
                                accelerator.process_index,
                                report_step,
                                pid=os.getpid(),
                            )
                        )
    finally:
        instrumentation.uninstall()
        events = collector.stop()
        gpu = monitor.stop() if monitor is not None else []

    accelerator.wait_for_everyone()
    all_records = merge_rank_records(
        _gather_object([record.to_dict() for record in records], accelerator)
    )
    gathered_events = _gather_object(events, accelerator)
    gathered_stats = _gather_object(collector.stats, accelerator)
    gathered_gpu = _gather_object(gpu, accelerator)
    if accelerator.is_main_process:
        detail = [event for rows in gathered_events for event in rows]
        flat_gpu = [item for rows in gathered_gpu for item in rows]
        output_dir = resolve_output_dir(args.output_dir, exact=args.exact_output_dir)
        summary = write_report(
            output_dir,
            all_records,
            gpu_samples=flat_gpu,
            slow_samples=[event for event in detail if event.get("kind") == "sample"],
            slow_videos=[event for event in detail if event.get("kind") == "video"],
            metadata={
                "kind": "data",
                "world_size": accelerator.num_processes,
                "config_path": args.config_path,
                "output_dir": str(output_dir),
                "dist_loading_single_process_override": dist_loading_overridden,
                "memory_start": memory_start,
                "memory_after_dataset": memory_after_dataset,
                "memory_end": memory_snapshot(),
                "event_collector_by_rank": gathered_stats,
                "video_instrumentation": not args.disable_video_instrumentation,
                "video_instrumentation_capability": (
                    "whole decode call duration and requested timestamps only; "
                    "open/seek/decode/close, actual decoded frames, and amplification unavailable"
                ),
            },
        )
        print(f"输出目录: {output_dir}")
        print(format_terminal_summary(summary))


if __name__ == "__main__":
    main()
