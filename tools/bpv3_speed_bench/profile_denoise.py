"""单独 profile 一次 sample_actions，区分 GPU kernel 时间和 CPU 下发开销。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from bench import build_policy, make_batch  # noqa: E402
from lerobot.utils.constants import OBS_PREFIX, OBS_STATE  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--qwen-path", default="/vla/workspace/models/q35/base_800m")
    parser.add_argument("--no-kernelize", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    device = torch.device("cuda")
    policy = build_policy(args, device)
    policy.eval()
    model = policy.model
    cfg = model.core.backbone.vlm.config
    data = make_batch(args.batch, args.num_chunks, device, train=False, image_token_id=cfg.image_token_id,
                      vs_id=cfg.vision_start_token_id, ve_id=cfg.vision_end_token_id)
    kwargs = dict(
        pixel_values=data[f"{OBS_PREFIX}pixel_values"],
        image_grid_thw=data[f"{OBS_PREFIX}image_grid_thw"],
        input_ids=data[f"{OBS_PREFIX}input_ids"],
        mm_token_type_ids=data[f"{OBS_PREFIX}mm_token_type_ids"],
        attention_mask=data[f"{OBS_PREFIX}attention_mask"],
        state=data[OBS_STATE],
        behavior_prompt=data["behavior_prompt"],
    )
    for _ in range(20):
        model.sample_actions(**kwargs)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        model.sample_actions(**kwargs)
    torch.cuda.synchronize()
    print(f"e2e after 20 warmup: {(time.perf_counter() - t) * 100:.1f} ms")
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        model.sample_actions(**kwargs)
        torch.cuda.synchronize()
    events = prof.key_averages()
    gpu_total = sum(e.self_device_time_total for e in events) / 1e3
    print(f"GPU kernel time total: {gpu_total:.1f} ms, kernel launches: "
          f"{sum(e.count for e in events if e.key.startswith(('cudaLaunchKernel', 'cuLaunchKernel')))}")
    print(events.table(sort_by="self_cpu_time_total", row_limit=15))
    print(events.table(sort_by="self_device_time_total", row_limit=12))


if __name__ == "__main__":
    main()
