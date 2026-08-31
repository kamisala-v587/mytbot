"""Short BPVA training-throughput benchmark without eval/checkpoint/W&B."""

from __future__ import annotations

import argparse
import copy
import os
import time
from datetime import timedelta
from typing import Any

from .metrics import StageRecord, merge_rank_records
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BPVA 短程训练吞吐测评（不保存、不评测、不启用 W&B）"
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-dir", default="outputs/bpva_benchmark/train")
    parser.add_argument("--exact-output-dir", action="store_true")
    parser.add_argument("--warmup-steps", type=nonnegative_int, default=2)
    parser.add_argument("--measure-steps", type=positive_int, default=10)
    parser.add_argument("--num-workers", type=nonnegative_int)
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


def _loader(dataset: Any, cfg: Any, device: Any):
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
    )


def _optimizer_parameters(optimizer: Any):
    for group in optimizer.param_groups:
        yield from group["params"]


def merge_train_partials(partials: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-rank train partials without a final distributed collective."""
    records = merge_rank_records(partial["records"] for partial in partials)
    gpu_samples: list[dict[str, Any]] = []
    monitor_errors: list[Any] = []
    for partial in partials:
        monitor = partial.get("monitor") or {}
        gpu_samples.extend(monitor.get("samples") or [])
        monitor_errors.extend(monitor.get("errors") or [])
    return {
        "records": records,
        "gpu_samples": gpu_samples,
        "monitor_errors": monitor_errors,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    import torch
    from accelerate.utils import send_to_device
    from lerobot.datasets.factory import make_dataset
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.policies.factory import make_policy
    from lerobot.utils.random_utils import set_seed
    from lerobot.utils.utils import has_method

    cfg = _load(args.config_path)
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    accelerator = _make_accelerator(cfg)
    session = create_run_session(
        args.output_dir, accelerator, exact=args.exact_output_dir
    )
    if accelerator.is_main_process:
        print(f"[bpva-benchmark] 输出目录: {session.output_dir}", flush=True)

    memory_start = memory_snapshot()
    memory_after_dataset = None
    records: list[StageRecord] = []
    instrument = None
    monitor = None
    microstep = 0
    optimizer_step = 0
    progress = PhaseProgress("initialization", 0, time.perf_counter())

    def snapshot(error: dict[str, Any] | None = None):
        monitor_state = monitor.snapshot() if monitor is not None else None
        return write_partial(
            session,
            kind="train",
            phase=progress.phase,
            completed=progress.completed,
            total=progress.total,
            records=records,
            collector={"top_events": [], "stats": {}},
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
                "gradient_accumulation_steps": (
                    cfg.gradient_accumulation_steps
                ),
            },
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
        dataset, _ = make_dataset(cfg)
        memory_after_dataset = memory_snapshot()
        policy = make_policy(cfg.policy)
        optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
        loader = _loader(dataset, cfg, accelerator.device)
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

            report_step = optimizer_step - args.warmup_steps
            instrument.step = report_step if measured else None
            began = time.perf_counter()
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
            microstep += 1

            if did_step:
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
        }
        summary = write_report(
            session.output_dir,
            merged["records"],
            gpu_samples=merged["gpu_samples"],
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
        if instrument is not None:
            instrument.uninstall()
        if monitor is not None:
            monitor.stop()


if __name__ == "__main__":
    main()
