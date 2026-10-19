"""BPVAv3 推理 / 训练速度测评：BP query 压缩 vs. BP ViT token 直接拼进 prefix。

两种模式共用同一套 Qwen3.5 ViT + VLM + action expert，只替换 `embed_prefix_with_bp`：
  compressed: 每个 chunk 经 compressor 压成 bp_num_query_tokens 个 token（现有实现）
  raw:        跳过 compressor，BP 图像的 ViT 输出 token 直接拼到当前观测 prefix 后面

权重数值不影响耗时，这里只加载 Qwen3.5 base，BP encoder 随机初始化。

用法:
  python tools/bpv3_speed_bench/bench.py --num-chunks 8 --out results/k8.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import types
from contextlib import contextmanager
from pathlib import Path

import torch

from lerobot.policies.BPVAv3.configuration_bpvav3 import BPVAv3Config
from lerobot.policies.BPVAv3.modeling_bpvav3 import BPVAv3Policy
from lerobot.utils.constants import ACTION, OBS_PREFIX, OBS_STATE

BP_KEY = "observation.images.image0"


def embed_prefix_raw(
    self,
    *,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    behavior_prompt: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_pixels = pixel_values.reshape(-1, pixel_values.shape[-1])
    current_grids = image_grid_thw.reshape(-1, 3)
    bp_pixels, bp_grids, layout, _ = self._pack_bp_images(behavior_prompt, device=current_pixels.device)
    features = self._visual_features(
        torch.cat([current_pixels, bp_pixels], dim=0),
        torch.cat([current_grids, bp_grids], dim=0),
    )
    n_current = current_grids.shape[0]
    current_features = features[:n_current]
    prefix = self._scatter_current_prefix(current_features, input_ids)
    (_, batch, chunks), = layout
    bp_tokens = torch.stack(features[n_current:], dim=0)
    tokens_per_image = bp_tokens.shape[1]
    bp_embs = bp_tokens.reshape(batch, chunks * tokens_per_image, bp_tokens.shape[-1]).to(prefix.dtype)
    chunk_mask = torch.as_tensor(behavior_prompt["mask"], dtype=torch.bool, device=prefix.device)
    bp_mask = chunk_mask[..., None].expand(-1, -1, tokens_per_image).reshape(batch, -1)
    prefix_mask = attention_mask.to(device=prefix.device, dtype=torch.bool)
    return torch.cat([prefix, bp_embs], dim=1), torch.cat([prefix_mask, bp_mask], dim=1)


class StageTimer:
    """包住模型内部函数，在调用前后同步 CUDA 记录每个阶段的耗时。"""

    def __init__(self) -> None:
        self.records: dict[str, list[float]] = {}
        self.enabled = False
        self.annotate = False

    def wrap(self, obj, name: str, label: str) -> None:
        original = getattr(obj, name)

        def timed(*args, **kwargs):
            if self.annotate:
                with torch.profiler.record_function(f"stage::{label}"):
                    return original(*args, **kwargs)
            if not self.enabled:
                return original(*args, **kwargs)
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = original(*args, **kwargs)
            torch.cuda.synchronize()
            self.records.setdefault(label, []).append((time.perf_counter() - start) * 1e3)
            return out

        setattr(obj, name, timed)

    @contextmanager
    def active(self):
        self.enabled = True
        try:
            yield
        finally:
            self.enabled = False


def build_policy(args, device: torch.device) -> BPVAv3Policy:
    config = BPVAv3Config(
        qwen3_5_pretrained_path=args.qwen_path,
        qwen3_5_processor_path=args.qwen_path,
        dtype="bfloat16",
        use_language=False,
        bp_num_chunks=args.num_chunks,
        bp_camera_keys=[BP_KEY],
        bp_num_query_tokens=8,
        chunk_size=50,
        n_action_steps=50,
        num_inference_steps=10,
        attention_implementation="sdpa",
        device=str(device),
    )
    policy = BPVAv3Policy(config)
    if args.no_kernelize:
        policy._qwen35_training_kernels_enabled = True
    policy = policy.to(device)
    return policy


def make_batch(batch: int, num_chunks: int, device: torch.device, *, image_token_id, vs_id, ve_id, train: bool):
    grid = 224 // 16
    patches = grid * grid
    patch_dim = 3 * 2 * 16 * 16
    tokens_per_image = patches // 4
    cam_ids = [vs_id] + [image_token_id] * tokens_per_image + [ve_id]
    cam_types = [0] + [1] * tokens_per_image + [0]
    input_ids = torch.tensor(cam_ids * 3, device=device).unsqueeze(0).expand(batch, -1).contiguous()
    mm_types = torch.tensor(cam_types * 3, device=device).unsqueeze(0).expand(batch, -1).contiguous()
    gen = torch.Generator(device="cpu").manual_seed(0)
    data = {
        f"{OBS_PREFIX}pixel_values": torch.randn(batch, 3 * patches, patch_dim, generator=gen).to(device),
        f"{OBS_PREFIX}image_grid_thw": torch.tensor([1, grid, grid], device=device).repeat(batch, 3, 1),
        f"{OBS_PREFIX}input_ids": input_ids,
        f"{OBS_PREFIX}mm_token_type_ids": mm_types,
        f"{OBS_PREFIX}attention_mask": torch.ones_like(input_ids),
        OBS_STATE: torch.randn(batch, 32, generator=gen).to(device),
        "behavior_prompt": {
            "bp_pixel_values": {
                BP_KEY: torch.randn(batch, num_chunks, patches, patch_dim, generator=gen).to(device)
            },
            "bp_image_grid_thw": {BP_KEY: torch.tensor([1, grid, grid], device=device).repeat(batch, num_chunks, 1)},
            "mask": torch.ones(batch, num_chunks, dtype=torch.bool, device=device),
            "chunk_indices": torch.arange(num_chunks, device=device).repeat(batch, 1),
            "state": torch.randn(batch, num_chunks, 32, generator=gen).to(device),
            "action": torch.randn(batch, num_chunks, 50, 32, generator=gen).to(device),
            "action_is_pad": torch.zeros(batch, num_chunks, 50, dtype=torch.bool, device=device),
        },
    }
    if train:
        data[ACTION] = torch.randn(batch, 50, 32, generator=gen).to(device)
    return data


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    n = len(ordered)
    return {
        "mean": statistics.fmean(ordered),
        "p50": ordered[n // 2],
        "p90": ordered[min(n - 1, int(n * 0.9))],
        "std": statistics.pstdev(ordered),
        "n": n,
    }


def run_inference(policy, batch_data, *, warmup: int, iters: int, timer: StageTimer) -> dict:
    model = policy.model
    policy.eval()
    kwargs = dict(
        pixel_values=batch_data[f"{OBS_PREFIX}pixel_values"],
        image_grid_thw=batch_data[f"{OBS_PREFIX}image_grid_thw"],
        input_ids=batch_data[f"{OBS_PREFIX}input_ids"],
        mm_token_type_ids=batch_data[f"{OBS_PREFIX}mm_token_type_ids"],
        attention_mask=batch_data[f"{OBS_PREFIX}attention_mask"],
        state=batch_data[OBS_STATE],
        behavior_prompt=batch_data["behavior_prompt"],
    )
    noise = torch.randn(batch_data[OBS_STATE].shape[0], 50, 32, device=batch_data[OBS_STATE].device)
    for _ in range(warmup):
        model.sample_actions(noise=noise, **kwargs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    e2e = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.sample_actions(noise=noise, **kwargs)
        torch.cuda.synchronize()
        e2e.append((time.perf_counter() - start) * 1e3)
    peak = torch.cuda.max_memory_allocated() / 2**30
    timer.records.clear()
    with timer.active():
        for _ in range(iters):
            model.sample_actions(noise=noise, **kwargs)
    stages = {label: summarize(values) for label, values in timer.records.items()}
    gpu = profile_gpu_time(lambda: model.sample_actions(noise=noise, **kwargs), timer, repeats=3)
    return {"e2e_ms": summarize(e2e), "stages_ms": stages, "gpu_kernel_ms": gpu, "peak_mem_gb": peak}


def profile_gpu_time(fn, timer: StageTimer, *, repeats: int) -> dict:
    """GPU 上实际执行 kernel 的时间（不含 CPU 下发开销），总量和各阶段。"""
    import tempfile

    from torch.profiler import ProfilerActivity, profile

    timer.annotate = True
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(repeats):
                fn()
            torch.cuda.synchronize()
    finally:
        timer.annotate = False
    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        prof.export_chrome_trace(handle.name)
        trace = json.loads(Path(handle.name).read_text())["traceEvents"]
    kernels = [(e["ts"], e["ts"] + e["dur"]) for e in trace if e.get("cat") == "kernel"]
    spans: dict[str, list[tuple[float, float]]] = {}
    for e in trace:
        if e.get("cat") == "gpu_user_annotation" and e.get("name", "").startswith("stage::"):
            spans.setdefault(e["name"].removeprefix("stage::"), []).append((e["ts"], e["ts"] + e["dur"]))
    result = {"total": sum(end - start for start, end in kernels) / 1e3 / repeats, "launches": len(kernels) / repeats}
    for label, ranges in spans.items():
        busy = sum(
            end - start
            for start, end in kernels
            if any(lo <= start and end <= hi for lo, hi in ranges)
        )
        result[label] = busy / 1e3 / repeats
    return result


def run_training(policy, batch_data, *, warmup: int, iters: int, timer: StageTimer) -> dict:
    policy.train()
    params = [p for p in policy.parameters() if p.requires_grad]

    def step():
        loss, _ = policy.forward(batch_data)
        loss.backward()
        for p in params:
            p.grad = None

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
    peak = torch.cuda.max_memory_allocated() / 2**30
    gpu = profile_gpu_time(step, timer, repeats=2)
    policy.eval()
    return {"step_ms": summarize(times), "gpu_kernel_ms": gpu, "peak_mem_gb": peak}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--qwen-path", default="/vla/workspace/models/q35/base_800m")
    parser.add_argument("--infer-batches", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--train-batches", type=int, nargs="*", default=[8, 32])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--train-iters", type=int, default=20)
    parser.add_argument("--no-kernelize", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    policy = build_policy(args, device)
    model = policy.model
    vlm_config = model.core.backbone.vlm.config
    token_ids = dict(
        image_token_id=int(vlm_config.image_token_id),
        vs_id=int(vlm_config.vision_start_token_id),
        ve_id=int(vlm_config.vision_end_token_id),
    )

    timer = StageTimer()
    timer.wrap(model, "_visual_features", "vit")
    timer.wrap(model.bp_obs_encoder, "forward", "bp_compressor")
    timer.wrap(model.core, "prepare_inference_cache", "vlm_prefill")
    timer.wrap(model, "_sample_action_flow", "denoise_10steps")
    timer.wrap(model.core, "forward_embedded", "train_vlm_expert_fwd")

    compressed_embed = model.embed_prefix_with_bp
    raw_embed = types.MethodType(embed_prefix_raw, model)
    modes = {"compressed": compressed_embed, "raw": raw_embed}

    results = {
        "num_chunks": args.num_chunks,
        "kernelized": bool(getattr(policy, "_qwen35_training_kernels_enabled", False)) and not args.no_kernelize,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "inference": {},
        "training": {},
    }
    for mode, embed in modes.items():
        model.embed_prefix_with_bp = embed
        for batch in args.infer_batches:
            data = make_batch(batch, args.num_chunks, device, train=False, **token_ids)
            with torch.no_grad():
                hidden, _ = model.embed_prefix_with_bp(
                    pixel_values=data[f"{OBS_PREFIX}pixel_values"],
                    image_grid_thw=data[f"{OBS_PREFIX}image_grid_thw"],
                    input_ids=data[f"{OBS_PREFIX}input_ids"],
                    attention_mask=data[f"{OBS_PREFIX}attention_mask"],
                    behavior_prompt=data["behavior_prompt"],
                )
            prefix_len = int(hidden.shape[1])
            res = run_inference(policy, data, warmup=args.warmup, iters=args.iters, timer=timer)
            res["prefix_len"] = prefix_len
            res["bp_tokens"] = prefix_len - int(data[f"{OBS_PREFIX}input_ids"].shape[1])
            results["inference"][f"{mode}_b{batch}"] = res
            print(f"[K={args.num_chunks}] infer {mode} B={batch} prefix={prefix_len} "
                  f"e2e={res['e2e_ms']['mean']:.2f}ms gpu={res['gpu_kernel_ms']['total']:.2f}ms "
                  f"peak={res['peak_mem_gb']:.2f}GB", flush=True)
            del data
            torch.cuda.empty_cache()
        for batch in args.train_batches:
            data = make_batch(batch, args.num_chunks, device, train=True, **token_ids)
            try:
                res = run_training(policy, data, warmup=3, iters=args.train_iters, timer=timer)
            except torch.OutOfMemoryError:
                res = {"oom": True}
                policy.zero_grad(set_to_none=True)
            results["training"][f"{mode}_b{batch}"] = res
            msg = "OOM" if res.get("oom") else (
                f"step={res['step_ms']['mean']:.1f}ms gpu={res['gpu_kernel_ms']['total']:.1f}ms "
                f"peak={res['peak_mem_gb']:.2f}GB"
            )
            print(f"[K={args.num_chunks}] train {mode} B={batch} {msg}", flush=True)
            del data
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
