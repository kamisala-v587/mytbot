"""Short BPVA training-throughput benchmark without eval/checkpoint/W&B."""

from __future__ import annotations

import argparse
import copy
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from .metrics import StageRecord, aggregate_step_stragglers, merge_rank_records
from .data_instrumentation import (
    DataInstrumentation,
    EventCollector,
    WorkerInstrumentation,
    compose_worker_init,
    sample_records_from_batch,
    strip_benchmark_metadata,
    wrap_dataset_for_instrumentation,
)
from .model_instrumentation import (
    DeviceStageTimer,
    ModelInstrumentation,
    resolve_pending,
)
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


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("必须大于等于 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("必须大于等于 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("必须位于 [0, 1]")
    return parsed


def report_microstep_id(measured_microsteps: int) -> int:
    """Return the next unique measured-microstep identifier."""
    if measured_microsteps < 0:
        raise ValueError("measured_microsteps 必须大于等于 0")
    return measured_microsteps


def _unwrap_for_inventory(dataset: Any) -> Any:
    """Peel common wrappers so inventory reads the concrete train dataset."""
    ds = dataset
    for _ in range(6):
        if any(hasattr(ds, name) for name in ("datasets", "_datasets", "repo_ids", "repo_id")):
            if hasattr(ds, "num_frames") or hasattr(ds, "__len__"):
                return ds
        nxt = (
            getattr(ds, "current_ds", None)
            or getattr(ds, "dataset", None)
            or getattr(ds, "_base", None)
        )
        if nxt is None or nxt is ds:
            break
        ds = nxt
    return ds


def dataset_inventory(dataset: Any) -> dict[str, int | None]:
    """Return dataset_count and total_frames for the constructed train dataset.

    Counts sub-datasets (repos) when present; otherwise treats a single
    ``repo_id`` dataset as count=1. Frame total prefers ``num_frames``, then
    ``len(dataset)``.
    """
    ds = _unwrap_for_inventory(dataset)
    datasets = getattr(ds, "datasets", None)
    if datasets is None:
        datasets = getattr(ds, "_datasets", None)
    if datasets is not None:
        count = len(datasets)
    else:
        repo_ids = getattr(ds, "repo_ids", None)
        if repo_ids is not None:
            count = len(repo_ids)
        elif getattr(ds, "repo_id", None) is not None:
            count = 1
        else:
            count = 0

    frames: int | None
    num_frames = getattr(ds, "num_frames", None)
    if callable(num_frames):
        try:
            frames = int(num_frames())
        except TypeError:
            frames = None
    elif num_frames is not None:
        frames = int(num_frames)
    else:
        try:
            frames = int(len(ds))
        except TypeError:
            frames = None

    return {"dataset_count": int(count), "total_frames": frames}


@dataclass
class OptimizerStepWallTracker:
    """Track one accumulation cycle and never carry a completed warmup cycle forward."""

    began: float | None = None

    def begin_microstep(self, now: float) -> None:
        if self.began is None:
            self.began = now

    def finish_cycle(self, now: float, *, measured: bool, optimizer_step: int,
                     warmup_steps: int) -> tuple[int, float] | None:
        if self.began is None:
            raise RuntimeError("optimizer step cycle 尚未开始")
        elapsed = now - self.began
        self.began = None
        if not measured:
            return None
        report_step = optimizer_step - warmup_steps
        if report_step < 0:
            raise RuntimeError("测量 optimizer step 不能位于 warmup 之前")
        return report_step, elapsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BPVA 短程训练吞吐测评（不保存、不评测、不启用 W&B）"
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-dir", default="outputs/bpva_benchmark/train")
    parser.add_argument("--exact-output-dir", action="store_true")
    parser.add_argument("--warmup-steps", type=nonnegative_int, default=2)
    parser.add_argument("--measure-steps", type=positive_int, default=10)
    parser.add_argument("--batch-size", type=positive_int)
    parser.add_argument("--num-workers", type=nonnegative_int)
    parser.add_argument("--out-of-order", action="store_true",
                        help="设置 DataLoader(in_order=False)；默认保持顺序交付")
    parser.add_argument("--sample-rate", type=probability, default=1.0)
    parser.add_argument("--slow-sample-threshold", type=nonnegative_float, default=1.0)
    parser.add_argument("--slow-video-threshold", type=nonnegative_float, default=0.5)
    parser.add_argument("--top-k", type=nonnegative_int, default=100)
    parser.add_argument("--queue-size", type=positive_int, default=4096)
    parser.add_argument("--monitor-interval", type=positive_float, default=1.0)
    parser.add_argument("--finalize-timeout", type=positive_float, default=300.0)
    parser.add_argument("--finalize-poll-interval", type=positive_float, default=0.2)
    return parser


def _load(path: str):
    from .config_utils import register_bpva_configs

    register_bpva_configs()
    from lerobot.configs.train import TrainPipelineConfig

    cfg = copy.deepcopy(TrainPipelineConfig.from_pretrained(path))
    # cfg.validate() is intentionally not called: it appends a timestamp to output_dir and
    # reparses process CLI policy arguments. Apply the relevant training preset directly.
    if cfg.policy is None:
        raise ValueError("配置缺少 policy")
    if cfg.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps 必须大于 0")
    if cfg.use_policy_training_preset and not cfg.resume:
        cfg.optimizer = cfg.policy.get_optimizer_preset()
        cfg.scheduler = cfg.policy.get_scheduler_preset()
    if cfg.optimizer is None or cfg.scheduler is None:
        raise ValueError(
            "配置未生成 optimizer/scheduler；请检查 policy training preset"
        )
    cfg.save_checkpoint = False
    cfg.eval_freq = 0
    if hasattr(cfg, "wandb") and hasattr(cfg.wandb, "enable"):
        cfg.wandb.enable = False
    if hasattr(cfg.policy, "log_da3_teacher_timing"):
        cfg.policy.log_da3_teacher_timing = False
    return cfg


def _apply_overrides(cfg: Any, args: argparse.Namespace) -> dict[str, bool]:
    """Apply runtime-only CLI overrides before any training components are built."""
    overrides = {
        "batch_size": args.batch_size is not None,
        "num_workers": args.num_workers is not None,
    }
    if overrides["batch_size"]:
        cfg.batch_size = args.batch_size
    if overrides["num_workers"]:
        cfg.num_workers = args.num_workers
    return overrides


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _make_accelerator(cfg: Any):
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs

    timeout_s = int(
        os.environ.get(
            "LEROBOT_DDP_TIMEOUT_SEC", os.environ.get("DDP_TIMEOUT_SEC", "1800")
        )
    )
    handlers = [
        DistributedDataParallelKwargs(
            find_unused_parameters=_env_flag("LEROBOT_DDP_FIND_UNUSED_PARAMETERS", True)
        ),
        InitProcessGroupKwargs(timeout=timedelta(seconds=timeout_s)),
    ]
    return Accelerator(
        step_scheduler_with_optimizer=False,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        kwargs_handlers=handlers,
    )


def _loader(
    dataset: Any,
    cfg: Any,
    device: Any,
    *,
    sampler_dataset: Any | None = None,
    in_order: bool = True,
    worker_init_fn: Any = None,
):
    import inspect
    import torch
    from lerobot.datasets.sampler import MultiLeRobotWeightedSampler

    effective_sampler_dataset = dataset if sampler_dataset is None else sampler_dataset
    weighted = (
        not cfg.dataset.streaming
        and getattr(effective_sampler_dataset, "dataset_weights", None) is not None
    )
    sampler = (
        MultiLeRobotWeightedSampler(dataset=effective_sampler_dataset)
        if weighted
        else None
    )
    num_workers = 1 if cfg.dataset.streaming else cfg.num_workers
    kwargs = dict(
        dataset=dataset, batch_size=cfg.batch_size,
        shuffle=not cfg.dataset.streaming and sampler is None, sampler=sampler,
        num_workers=num_workers, pin_memory=device.type == "cuda", drop_last=False,
        prefetch_factor=((4 if cfg.dataset.streaming else 2) if num_workers > 0 else None),
        worker_init_fn=worker_init_fn,
    )
    if "in_order" not in inspect.signature(torch.utils.data.DataLoader).parameters:
        if not in_order:
            raise RuntimeError("当前 PyTorch DataLoader 不支持 in_order=False；请升级 PyTorch")
    else:
        kwargs["in_order"] = in_order
    return torch.utils.data.DataLoader(**kwargs)


def _optimizer_parameters(optimizer: Any):
    for group in optimizer.param_groups:
        yield from group["params"]


def merge_train_partials(partials: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-rank train partials without a final distributed collective."""
    records = merge_rank_records(partial["records"] for partial in partials)
    gpu_samples: list[dict[str, Any]] = []
    monitor_errors: list[Any] = []
    sample_loads: list[dict[str, Any]] = []
    collector_events: list[dict[str, Any]] = []
    collector_stats: list[dict[str, Any]] = []
    for partial in partials:
        collector = partial.get("collector") or {}
        collector_events.extend(collector.get("top_events") or [])
        collector_stats.append(collector.get("stats") or {})
        sample_loads.extend(partial.get("metadata", {}).get("sample_loads") or [])
        monitor = partial.get("monitor") or {}
        gpu_samples.extend(monitor.get("samples") or [])
        monitor_errors.extend(monitor.get("errors") or [])
    return {
        "records": records,
        "gpu_samples": gpu_samples,
        "monitor_errors": monitor_errors,
        "sample_loads": sample_loads,
        "collector_events": collector_events,
        "collector_stats": collector_stats,
    }


def main(
    argv: list[str] | None = None,
    *,
    load_config: Callable[[str], Any] | None = None,
) -> None:
    args = build_parser().parse_args(argv)
    import torch
    from accelerate.utils import send_to_device
    from lerobot.datasets.factory import make_dataset
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.policies.factory import make_policy
    from lerobot.utils.random_utils import set_seed
    from lerobot.utils.utils import has_method

    config_loader = _load if load_config is None else load_config
    cfg = config_loader(args.config_path)
    cli_overrides = _apply_overrides(cfg, args)
    accelerator = _make_accelerator(cfg)
    session = create_run_session(
        args.output_dir, accelerator, exact=args.exact_output_dir
    )
    if accelerator.is_main_process:
        print(f"[bpva-benchmark] 输出目录: {session.output_dir}", flush=True)

    memory_start = memory_snapshot()
    memory_after_dataset = None
    dataset_stats: dict[str, int | None] = {
        "dataset_count": None,
        "total_frames": None,
    }
    records: list[StageRecord] = []
    sample_loads: list[dict[str, Any]] = []
    collector = None
    data_instrument = None
    instrument = None
    monitor = None
    microstep = 0
    measured_microstep = 0
    optimizer_step = 0
    optimizer_wall = OptimizerStepWallTracker()
    progress = PhaseProgress("initialization", 0, time.perf_counter())

    def snapshot(error: dict[str, Any] | None = None, *, status: str | None = None):
        monitor_state = monitor.snapshot() if monitor is not None else None
        collector_state = collector.snapshot() if collector is not None else {"top_events": [], "stats": {}}
        return write_partial(
            session,
            kind="train",
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
            metadata={
                "config_path": args.config_path,
                "microsteps": microstep,
                "optimizer_steps": optimizer_step,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "batch_size_per_rank": cfg.batch_size,
                "num_workers_per_rank": (
                    1 if cfg.dataset.streaming else cfg.num_workers
                ),
                "cli_overrides": cli_overrides,
                "dataloader_in_order": not args.out_of_order,
                "sample_loads": sample_loads,
                "dataset_count": dataset_stats.get("dataset_count"),
                "total_frames": dataset_stats.get("total_frames"),
            },
            status=status,
        )

    try:
        snapshot()
        if cfg.dataset.dist_loading and accelerator.num_processes <= 1:
            raise ValueError("训练 benchmark 的 dist_loading 需要多 rank")
        if cfg.seed is not None:
            set_seed(cfg.seed, accelerator=accelerator)
        cfg.policy.device = str(accelerator.device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        progress = PhaseProgress("dataset_init", 0, time.perf_counter())
        log_phase(accelerator, progress.phase)
        collector = EventCollector(args.queue_size, args.top_k).start()
        data_instrument = DataInstrumentation(
            event_queue=collector.queue, sample_rate=args.sample_rate,
            slow_sample_s=args.slow_sample_threshold,
            slow_video_s=args.slow_video_threshold, video=True,
        ).install()
        raw_dataset, _ = make_dataset(cfg)
        dataset_stats = dataset_inventory(raw_dataset)
        if accelerator.is_main_process:
            frames = dataset_stats.get("total_frames")
            frames_text = f"{frames:,}" if frames is not None else "未知"
            print(
                f"[bpva-benchmark] 数据集个数={dataset_stats.get('dataset_count')} "
                f"总帧数={frames_text}",
                flush=True,
            )
        instrumented_dataset = wrap_dataset_for_instrumentation(raw_dataset)
        memory_after_dataset = memory_snapshot()
        policy = make_policy(cfg.policy)
        optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
        worker_init = None
        effective_workers = 1 if cfg.dataset.streaming else cfg.num_workers
        if effective_workers > 0:
            worker_init = compose_worker_init(None, WorkerInstrumentation(
                collector.queue, args.sample_rate, args.slow_sample_threshold,
                args.slow_video_threshold, True))
        loader = _loader(
            instrumented_dataset,
            cfg,
            accelerator.device,
            sampler_dataset=raw_dataset,
            in_order=not args.out_of_order,
            worker_init_fn=worker_init,
        )
        if cfg.dataset.dist_loading:
            policy, optimizer, scheduler = accelerator.prepare(
                policy, optimizer, scheduler
            )
        else:
            policy, optimizer, loader, scheduler = accelerator.prepare(
                policy, optimizer, loader, scheduler
            )

        raw = accelerator.unwrap_model(policy)
        instrument = ModelInstrumentation(
            getattr(raw, "model", raw), accelerator.process_index
        )
        instrument.install()
        if accelerator.is_main_process:
            monitor = SystemMonitor(args.monitor_interval).start()
        iterator = iter(loader)
        policy.train()
        optimizer.zero_grad(set_to_none=True)

        progress = PhaseProgress(
            "warmup", args.warmup_steps, time.perf_counter()
        )
        log_phase(accelerator, progress.phase, f"0/{progress.total}")
        target_steps = args.warmup_steps + args.measure_steps
        while optimizer_step < target_steps:
            measured = optimizer_step >= args.warmup_steps
            if measured and progress.phase != "measure":
                progress = PhaseProgress(
                    "measure", args.measure_steps, time.perf_counter()
                )
                log_phase(accelerator, progress.phase, f"0/{progress.total}")

            report_step = report_microstep_id(measured_microstep) if measured else microstep
            instrument.step = report_step if measured else None
            microstep_began = time.perf_counter()
            optimizer_wall.begin_microstep(microstep_began)
            began = microstep_began
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            if measured:
                records.append(
                    StageRecord(
                        "data_wait",
                        time.perf_counter() - began,
                        accelerator.process_index,
                        report_step,
                        pid=os.getpid(),
                        metadata={
                            "microstep": microstep,
                            "optimizer_step": optimizer_step,
                        },
                    )
                )
            if measured:
                samples, worker_batches = sample_records_from_batch(
                    batch, rank=accelerator.process_index, step=report_step,
                    optimizer_step=optimizer_step, microstep=microstep)
                sample_loads.extend(samples)
                for worker in worker_batches:
                    records.append(StageRecord(
                        "worker_batch_envelope", worker["elapsed_s"],
                        accelerator.process_index, report_step, worker_id=worker["worker_id"],
                        pid=worker["pid"], count=worker["sample_count"],
                        metadata={"optimizer_step": optimizer_step, "microstep": microstep,
                                  "definition": "returned samples min(start)-max(end) envelope"}))
            batch = strip_benchmark_metadata(batch)
            compute_began = time.perf_counter()
            if cfg.dataset.dist_loading:
                began = time.perf_counter()
                batch = send_to_device(
                    batch, accelerator.device, non_blocking=True
                )
                if measured:
                    records.append(
                        StageRecord(
                            "h2d",
                            time.perf_counter() - began,
                            accelerator.process_index,
                            report_step,
                            pid=os.getpid(),
                            metadata={
                                "microstep": microstep,
                                "optimizer_step": optimizer_step,
                            },
                        )
                    )

            stage_timers: list[DeviceStageTimer] = []
            optimizer_step_began = time.perf_counter()
            with accelerator.accumulate(policy):
                with DeviceStageTimer(
                    "forward", accelerator.process_index, report_step
                ) as timer:
                    with accelerator.autocast():
                        output = policy(batch)
                        loss = (
                            output[0]
                            if isinstance(output, (tuple, list))
                            else output
                        )
                stage_timers.append(timer)

                with DeviceStageTimer(
                    "backward", accelerator.process_index, report_step
                ) as timer:
                    accelerator.backward(loss)
                stage_timers.append(timer)

                if accelerator.sync_gradients:
                    with DeviceStageTimer(
                        "grad_clip", accelerator.process_index, report_step
                    ) as timer:
                        accelerator.clip_grad_norm_(
                            _optimizer_parameters(optimizer),
                            cfg.optimizer.grad_clip_norm,
                        )
                    stage_timers.append(timer)

                with DeviceStageTimer(
                    "optimizer", accelerator.process_index, report_step
                ) as timer:
                    optimizer.step()
                stage_timers.append(timer)

                if scheduler is not None and accelerator.sync_gradients:
                    with DeviceStageTimer(
                        "scheduler", accelerator.process_index, report_step
                    ) as timer:
                        scheduler.step()
                    stage_timers.append(timer)

                with DeviceStageTimer(
                    "zero_grad", accelerator.process_index, report_step
                ) as timer:
                    optimizer.zero_grad(set_to_none=True)
                stage_timers.append(timer)
                did_step = accelerator.sync_gradients
                unwrapped = accelerator.unwrap_model(
                    policy, keep_fp32_wrapper=True
                )
                if did_step and has_method(unwrapped, "update"):
                    unwrapped.update()

            pending = [
                timer.pending
                for timer in stage_timers
                if timer.pending is not None
            ]
            pending.extend(instrument.pop_pending(report_step))
            resolved = resolve_pending(pending)
            if measured:
                for record in resolved:
                    record.metadata.update(
                        {
                            "microstep": microstep,
                            "optimizer_step": optimizer_step,
                            "sync_gradients": did_step,
                        }
                    )
                records.extend(resolved)
                common = {"microstep": microstep, "optimizer_step": optimizer_step,
                          "sync_gradients": did_step,
                          "includes_cuda_resolution_sync": True}
                records.append(StageRecord("train_compute_wall", time.perf_counter() - compute_began,
                                           accelerator.process_index, report_step, pid=os.getpid(), metadata=dict(common)))
                records.append(StageRecord("microstep_wall", time.perf_counter() - microstep_began,
                                           accelerator.process_index, report_step, pid=os.getpid(), metadata=dict(common)))
            microstep += 1
            if measured:
                measured_microstep += 1

            if did_step:
                optimizer_wall_result = optimizer_wall.finish_cycle(
                    time.perf_counter(), measured=measured, optimizer_step=optimizer_step,
                    warmup_steps=args.warmup_steps)
                if optimizer_wall_result is not None:
                    optimizer_report_step, optimizer_elapsed = optimizer_wall_result
                    records.append(StageRecord(
                        "optimizer_step_wall", optimizer_elapsed,
                        accelerator.process_index, optimizer_report_step,
                        pid=os.getpid(), metadata={
                            "optimizer_step": optimizer_step,
                            "ending_microstep": microstep - 1,
                            "includes_cuda_resolution_sync": True,
                            "boundary": "first microstep next() start through final microstep CUDA resolution",
                        }))
                optimizer_step += 1
                phase_completed = (
                    optimizer_step - args.warmup_steps
                    if measured
                    else optimizer_step
                )
                if progress.advance(phase_completed):
                    path = snapshot()
                    log_progress(
                        accelerator,
                        phase=progress.phase,
                        completed=progress.completed,
                        total=progress.total,
                        last_elapsed_s=(
                            time.perf_counter() - optimizer_step_began
                        ),
                        started=progress.started,
                        path=path,
                    )

        if collector is not None:
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
        merged = merge_train_partials(partials)

        progress = PhaseProgress("report", 0, time.perf_counter())
        log_phase(accelerator, progress.phase)
        metadata = {
            "kind": "train",
            "completion": "completed",
            "world_size": accelerator.num_processes,
            "config_path": args.config_path,
            "output_dir": str(session.output_dir),
            "generation": session.generation,
            "warmup_optimizer_steps": args.warmup_steps,
            "measure_optimizer_steps": args.measure_steps,
            "gradient_accumulation_steps": (
                cfg.gradient_accumulation_steps
            ),
            "batch_size_per_rank": cfg.batch_size,
            "cli_overrides": cli_overrides,
            "measured_microsteps": sum(
                1 for record in merged["records"] if record.stage == "forward"
            ),
            "memory_start": memory_start,
            "memory_after_dataset": memory_after_dataset,
            "memory_end": memory_snapshot(),
            "monitor_errors": merged["monitor_errors"],
            "compile_model": getattr(cfg.policy, "compile_model", None),
            "gradient_checkpointing": getattr(
                cfg.policy, "gradient_checkpointing", None
            ),
            "log_da3_teacher_timing": False,
            "cuda_event_resolution": "once_per_microstep",
            "dataloader_in_order": not args.out_of_order,
            "num_workers_per_rank": 1 if cfg.dataset.streaming else cfg.num_workers,
            "event_collector_by_rank": merged["collector_stats"],
            "instrumentation_capability": (
                "sample metadata is assigned to the step after DataLoader delivery; "
                "worker production has no fabricated training step; worker_batch_envelope "
                "is min(start)-max(end), not summed sample time"
            ),
            "dataset_count": dataset_stats.get("dataset_count"),
            "total_frames": dataset_stats.get("total_frames"),
        }
        events = merged["collector_events"]
        stragglers = aggregate_step_stragglers(merged["records"])
        summary = write_report(
            session.output_dir, merged["records"], gpu_samples=merged["gpu_samples"],
            slow_samples=[event for event in events if event.get("kind") == "sample"],
            slow_videos=[event for event in events if event.get("kind") == "video"],
            sample_loads=merged["sample_loads"], step_stragglers=stragglers,
            metadata=metadata,
        )
        write_manifest(
            session,
            "completed",
            metadata={
                "record_count": len(merged["records"]),
                "monitor_errors": merged["monitor_errors"],
                "dataloader_in_order": not args.out_of_order,
                "batch_size_per_rank": cfg.batch_size,
                "num_workers_per_rank": (
                    1 if cfg.dataset.streaming else cfg.num_workers
                ),
                "cli_overrides": cli_overrides,
                "sample_load_count": len(merged["sample_loads"]),
                "dataset_count": dataset_stats.get("dataset_count"),
                "total_frames": dataset_stats.get("total_frames"),
            },
        )
        log_phase(
            accelerator,
            "completed",
            f"output={session.output_dir}",
        )
        print(format_terminal_summary(summary), flush=True)
        frames = dataset_stats.get("total_frames")
        frames_text = f"{frames:,}" if frames is not None else "未知"
        print(
            f"[bpva-benchmark] 数据集总个数={dataset_stats.get('dataset_count')} "
            f"数据集总帧数={frames_text}",
            flush=True,
        )
    except BaseException as exc:
        record_failure(session, accelerator, exc, snapshot)
        raise
    finally:
        if instrument is not None:
            instrument.uninstall()
        if data_instrument is not None:
            data_instrument.uninstall()
        if collector is not None:
            collector.stop()
        if monitor is not None:
            monitor.stop()


if __name__ == "__main__":
    main()
