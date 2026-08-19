#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CUDA environment smoke/stress test with configurable VRAM usage and duration."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, replace

import torch
import torch.multiprocessing as mp


BYTES_PER_GIB = 1024**3
FLOAT32_BYTES = torch.empty((), dtype=torch.float32).element_size()


@dataclass(frozen=True)
class StressArgs:
    devices: tuple[int, ...]
    duration_sec: float
    memory_gib: float | None
    memory_fraction: float | None
    tensor_chunk_mib: int
    matrix_size: int
    compute: bool
    log_interval_sec: float


def parse_args() -> StressArgs:
    parser = argparse.ArgumentParser(description="Run a bounded CUDA smoke/stress test.")
    parser.add_argument("--device", default="0", help="CUDA device index or comma-separated indices, for example 0 or 0,1,2,3.")
    parser.add_argument("--duration_sec", type=float, default=60.0, help="How long to run before exiting.")
    parser.add_argument("--memory_gib", type=float, default=None, help="Target VRAM allocation in GiB.")
    parser.add_argument(
        "--memory_fraction",
        type=float,
        default=0.5,
        help="Target fraction of free VRAM to allocate when --memory_gib is not set.",
    )
    parser.add_argument("--tensor_chunk_mib", type=int, default=256, help="Allocation chunk size in MiB.")
    parser.add_argument("--matrix_size", type=int, default=4096, help="Square matrix size for compute load.")
    parser.add_argument("--no_compute", action="store_true", help="Only allocate VRAM without matmul workload.")
    parser.add_argument("--log_interval_sec", type=float, default=5.0, help="Status log interval in seconds.")
    parsed = parser.parse_args()

    if parsed.duration_sec <= 0:
        parser.error("--duration_sec must be > 0")
    if parsed.memory_gib is not None and parsed.memory_gib <= 0:
        parser.error("--memory_gib must be > 0")
    if parsed.memory_fraction is not None and not 0 < parsed.memory_fraction < 1:
        parser.error("--memory_fraction must be between 0 and 1")
    if parsed.tensor_chunk_mib <= 0:
        parser.error("--tensor_chunk_mib must be > 0")
    if parsed.matrix_size < 256:
        parser.error("--matrix_size must be >= 256")
    if parsed.log_interval_sec <= 0:
        parser.error("--log_interval_sec must be > 0")

    devices = parse_devices(parsed.device, parser)

    return StressArgs(
        devices=devices,
        duration_sec=parsed.duration_sec,
        memory_gib=parsed.memory_gib,
        memory_fraction=parsed.memory_fraction,
        tensor_chunk_mib=parsed.tensor_chunk_mib,
        matrix_size=parsed.matrix_size,
        compute=not parsed.no_compute,
        log_interval_sec=parsed.log_interval_sec,
    )


def parse_devices(raw_devices: str, parser: argparse.ArgumentParser) -> tuple[int, ...]:
    try:
        devices = tuple(int(item.strip()) for item in raw_devices.split(",") if item.strip())
    except ValueError:
        parser.error("--device must be an integer or comma-separated integers, for example 0 or 0,1,2,3")
    if not devices:
        parser.error("--device must include at least one CUDA device index")
    if any(device < 0 for device in devices):
        parser.error("--device values must be >= 0")
    if len(set(devices)) != len(devices):
        parser.error("--device contains duplicate CUDA device indices")
    return devices


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / BYTES_PER_GIB:.2f} GiB"


def select_device(device_index: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable: torch.cuda.is_available() returned False")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device_index} does not exist; available count: {torch.cuda.device_count()}")
    torch.cuda.set_device(device_index)
    return torch.device(f"cuda:{device_index}")


def target_allocation_bytes(args: StressArgs, device: torch.device) -> int:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if args.memory_gib is not None:
        target_bytes = int(args.memory_gib * BYTES_PER_GIB)
    else:
        assert args.memory_fraction is not None
        target_bytes = int(free_bytes * args.memory_fraction)
    if target_bytes >= free_bytes:
        raise RuntimeError(
            f"Target VRAM {format_gib(target_bytes)} exceeds current free VRAM {format_gib(free_bytes)}; "
            "lower --memory_gib or --memory_fraction"
        )
    logging.info("GPU: %s", torch.cuda.get_device_name(device))
    logging.info("Total VRAM: %s, free before allocation: %s", format_gib(total_bytes), format_gib(free_bytes))
    logging.info("Target allocation: %s", format_gib(target_bytes))
    return target_bytes


def allocate_vram(target_bytes: int, chunk_mib: int, device: torch.device) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    chunk_bytes = chunk_mib * 1024**2
    allocated_bytes = 0
    while allocated_bytes < target_bytes:
        current_bytes = min(chunk_bytes, target_bytes - allocated_bytes)
        num_elements = max(1, current_bytes // FLOAT32_BYTES)
        tensors.append(torch.empty(num_elements, dtype=torch.float32, device=device))
        allocated_bytes += num_elements * FLOAT32_BYTES
    torch.cuda.synchronize(device)
    logging.info("Allocated approx %s across %d tensors", format_gib(allocated_bytes), len(tensors))
    return tensors


def run_compute_loop(args: StressArgs, device: torch.device) -> None:
    lhs = torch.randn((args.matrix_size, args.matrix_size), device=device, dtype=torch.float16)
    rhs = torch.randn((args.matrix_size, args.matrix_size), device=device, dtype=torch.float16)
    result = torch.empty_like(lhs)

    deadline = time.monotonic() + args.duration_sec
    next_log_at = time.monotonic()
    iterations = 0
    while time.monotonic() < deadline:
        torch.matmul(lhs, rhs, out=result)
        iterations += 1
        now = time.monotonic()
        if now >= next_log_at:
            torch.cuda.synchronize(device)
            log_status(device, iterations)
            next_log_at = now + args.log_interval_sec
    torch.cuda.synchronize(device)
    logging.info("Compute iterations: %d", iterations)


def hold_memory(args: StressArgs, device: torch.device) -> None:
    deadline = time.monotonic() + args.duration_sec
    next_log_at = time.monotonic()
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_log_at:
            log_status(device, iterations=0)
            next_log_at = now + args.log_interval_sec
        time.sleep(min(0.2, max(0.0, deadline - now)))


def log_status(device: torch.device, iterations: int) -> None:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    used_total = total_bytes - free_bytes
    logging.info(
        "status iterations=%d torch_allocated=%s torch_reserved=%s device_used=%s free=%s",
        iterations,
        format_gib(allocated),
        format_gib(reserved),
        format_gib(used_total),
        format_gib(free_bytes),
    )


def run_on_device(args: StressArgs, device_index: int) -> None:
    device = select_device(device_index)
    logging.info("Starting CUDA stress test on device %d", device_index)
    target_bytes = target_allocation_bytes(args, device)
    allocated_tensors = allocate_vram(target_bytes, args.tensor_chunk_mib, device)
    try:
        if args.compute:
            run_compute_loop(args, device)
        else:
            hold_memory(args, device)
    finally:
        allocated_tensors.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        logging.info("Released test allocations on device %d", device_index)


def main(args: StressArgs) -> None:
    if len(args.devices) == 1:
        run_on_device(args, args.devices[0])
        return

    mp.set_start_method("spawn", force=True)
    processes = []
    for device_index in args.devices:
        worker_args = replace(args, devices=(device_index,))
        process = mp.Process(target=run_on_device, args=(worker_args, device_index), name=f"cuda-stress-{device_index}")
        process.start()
        processes.append(process)

    failed_devices = []
    for device_index, process in zip(args.devices, processes, strict=True):
        process.join()
        if process.exitcode != 0:
            failed_devices.append((device_index, process.exitcode))
    if failed_devices:
        failures = ", ".join(f"device {device}: exit {exitcode}" for device, exitcode in failed_devices)
        raise RuntimeError(f"One or more CUDA stress workers failed: {failures}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    main(parse_args())
