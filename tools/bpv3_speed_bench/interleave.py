"""单进程、两种模式轮流测墙钟时间，抵消系统负载波动。

一个 policy 用 bp_num_chunks=32 构建，K<32 时只喂前 K 个 chunk（BPObsEncoder 允许 num_chunks <= max）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from bench import build_policy, embed_prefix_raw, make_batch, summarize  # noqa: E402
from lerobot.utils.constants import OBS_PREFIX, OBS_STATE  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-path", default="/vla/workspace/models/q35/base_800m")
    parser.add_argument("--chunks", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--infer-batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--train-batches", type=int, nargs="*", default=[8, 16, 32])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--infer-iters", type=int, default=10)
    parser.add_argument("--train-iters", type=int, default=4)
    parser.add_argument("--no-kernelize", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    args.num_chunks = max(args.chunks)

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    policy = build_policy(args, device)
    model = policy.model
    cfg = model.core.backbone.vlm.config
    token_ids = dict(image_token_id=int(cfg.image_token_id), vs_id=int(cfg.vision_start_token_id),
                     ve_id=int(cfg.vision_end_token_id))
    embeds = {"compressed": model.embed_prefix_with_bp, "raw": types.MethodType(embed_prefix_raw, model)}
    params = [p for p in policy.parameters() if p.requires_grad]
    results: dict = {"inference": {}, "training": {}}

    def timed(fn, iters):
        out = []
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            out.append((time.perf_counter() - start) * 1e3)
        return out

    for k in args.chunks:
        for batch in args.infer_batches:
            data = make_batch(batch, k, device, train=False, **token_ids)
            kwargs = dict(
                pixel_values=data[f"{OBS_PREFIX}pixel_values"],
                image_grid_thw=data[f"{OBS_PREFIX}image_grid_thw"],
                input_ids=data[f"{OBS_PREFIX}input_ids"],
                mm_token_type_ids=data[f"{OBS_PREFIX}mm_token_type_ids"],
                attention_mask=data[f"{OBS_PREFIX}attention_mask"],
                state=data[OBS_STATE],
                behavior_prompt=data["behavior_prompt"],
            )
            noise = torch.randn(batch, 50, 32, device=device)
            policy.eval()
            samples = {m: [] for m in embeds}
            for mode, embed in embeds.items():
                model.embed_prefix_with_bp = embed
                timed(lambda: model.sample_actions(noise=noise, **kwargs), 5)
            for r in range(args.rounds):
                order = list(embeds) if r % 2 == 0 else list(reversed(embeds))
                for mode in order:
                    model.embed_prefix_with_bp = embeds[mode]
                    samples[mode] += timed(lambda: model.sample_actions(noise=noise, **kwargs), args.infer_iters)
            for mode, values in samples.items():
                results["inference"][f"k{k}_{mode}_b{batch}"] = summarize(values)
            print(f"[K={k}] infer B={batch} " + " ".join(
                f"{m}={results['inference'][f'k{k}_{m}_b{batch}']['mean']:.1f}ms" for m in embeds), flush=True)
            del data, kwargs
            torch.cuda.empty_cache()

        for batch in args.train_batches:
            data = make_batch(batch, k, device, train=True, **token_ids)
            policy.train()

            def step():
                loss, _ = policy.forward(data)
                loss.backward()
                for p in params:
                    p.grad = None

            samples = {m: [] for m in embeds}
            failed = set()
            for mode, embed in embeds.items():
                model.embed_prefix_with_bp = embed
                try:
                    timed(step, 2)
                except torch.OutOfMemoryError:
                    failed.add(mode)
                    policy.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
            for r in range(args.rounds):
                order = list(embeds) if r % 2 == 0 else list(reversed(embeds))
                for mode in order:
                    if mode in failed:
                        continue
                    model.embed_prefix_with_bp = embeds[mode]
                    samples[mode] += timed(step, args.train_iters)
            for mode in embeds:
                key = f"k{k}_{mode}_b{batch}"
                results["training"][key] = {"oom": True} if mode in failed else summarize(samples[mode])
            print(f"[K={k}] train B={batch} " + " ".join(
                f"{m}=" + ("OOM" if m in failed else f"{results['training'][f'k{k}_{m}_b{batch}']['mean']:.1f}ms")
                for m in embeds), flush=True)
            policy.eval()
            del data
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
