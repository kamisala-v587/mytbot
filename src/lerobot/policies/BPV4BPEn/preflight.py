"""Pre-flight memory check: 2 real training steps with every learning rate set to 0, then restore.

All bpv4_bpen tensors have fixed shapes (keyframes padded to K, language padded to max length, 50 action steps), so a
real batch is the worst case. Optimizer states are really allocated, so the measured peak equals training's peak.
"""

from __future__ import annotations

import logging

import torch


def run_preflight(policy, optimizer, dataloader, accelerator, *, limit_frac: float = 0.9, grad_clip_norm: float = 1.0) -> None:
    if not torch.cuda.is_available():
        return
    device = accelerator.device
    inner = getattr(optimizer, "optimizer", optimizer)
    saved_lrs = [g["lr"] for g in inner.param_groups]
    model = accelerator.unwrap_model(policy).model
    step_buffer = getattr(model, "train_step", None)
    saved_step = step_buffer.clone() if step_buffer is not None else None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    batch = next(iter(dataloader))
    policy.train()
    try:
        for g in inner.param_groups:
            g["lr"] = 0.0
        for _ in range(2):
            with accelerator.autocast():
                loss, _ = policy.forward(batch)
            accelerator.backward(loss)
            if grad_clip_norm > 0:
                accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize(device)
        peak = torch.tensor(float(torch.cuda.max_memory_allocated(device)), device=device)
        peak = accelerator.gather(peak[None]).max().item()
        total = float(torch.cuda.get_device_properties(device).total_memory)
        accelerator.print(f"bpv4_bpen preflight: peak allocated {peak / 2**30:.1f} GiB / {total / 2**30:.1f} GiB "
                          f"({peak / total:.0%}), limit {limit_frac:.0%}")
        if peak > limit_frac * total:
            raise RuntimeError(
                f"bpv4_bpen preflight: peak memory {peak / 2**30:.1f} GiB exceeds {limit_frac:.0%} of "
                f"{total / 2**30:.1f} GiB; reduce batch_size (trajectories per rank), dataset.queries_per_trajectory "
                "or policy.bp_resampler_num_tokens"
            )
    finally:
        for g, lr in zip(inner.param_groups, saved_lrs, strict=True):
            g["lr"] = lr
        inner.state.clear()
        optimizer.zero_grad(set_to_none=True)
        if saved_step is not None:
            step_buffer.copy_(saved_step)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    logging.info("bpv4_bpen preflight passed; optimizer state cleared")
