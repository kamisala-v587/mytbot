"""BP KV cache 场景下的推理 GPU 耗时：compressed vs raw。

可缓存的前提：prefix 顺序改为 [BP, 当前观测]，并且 BP token 不看当前观测。
  - linear attention 层本身是因果的，BP 在前即可缓存 conv / recurrent 状态；
  - full attention 层把 BP 行对当前观测列的 mask 置 False，BP 的 K/V 就与当前观测无关。
每个 episode 只算一次 BP（ViT + compressor + VLM 各层），之后每步只算：
  当前 3 图 ViT → 153 个当前 token 接着 BP cache 做 prefill → 10 步去噪（读 BP+当前 的 KV）。

对比四种方案的每步 GPU kernel 时间，另外给出每个 episode 构建一次 BP cache 的耗时。
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch
from torch.profiler import record_function

sys.path.insert(0, str(Path(__file__).parent))
from bench import StageTimer, build_policy, embed_prefix_raw, make_batch, profile_gpu_time  # noqa: E402
from lerobot.policies.q35.experts import Q35FullPrefixState, Q35PrefixCache  # noqa: E402
from lerobot.policies.q35.token_layout import Q35TokenLayout  # noqa: E402
from lerobot.utils.constants import OBS_PREFIX, OBS_STATE  # noqa: E402


def positions(start: int, length: int, batch: int, device) -> torch.Tensor:
    return (start + torch.arange(length, device=device))[None, None].expand(3, batch, -1)


def encode_bp_tokens(model, data, mode: str) -> torch.Tensor:
    """BP 图像过 ViT（+ compressor），返回 [B, N_bp, D]。每个 episode 一次。"""
    prompt = data["behavior_prompt"]
    bp_pixels, bp_grids, layout, skipped = model._pack_bp_images(prompt, device=data[OBS_STATE].device)
    features = model._visual_features(bp_pixels, bp_grids)
    if mode == "raw":
        (_, batch, chunks), = layout
        stacked = torch.stack(features, dim=0)
        return stacked.reshape(batch, chunks * stacked.shape[1], stacked.shape[-1])
    visual = model._group_bp_features(features, layout, skipped, prompt, features[0])
    encoder = model.bp_obs_encoder.chunk_encoder
    encoder._visual_tokens = visual
    try:
        encoded = model.bp_obs_encoder(prompt, visual=None, return_dict=True)
    finally:
        encoder._visual_tokens = None
    return encoded.chunk_tokens.to(features[0].dtype)


def build_bp_cache(model, bp_hidden: torch.Tensor):
    """BP 单独过 VLM 各层，缓存 linear 层状态和 full 层 K/V。"""
    from transformers.cache_utils import DynamicCache

    backbone = model.core.backbone
    lm = backbone.vlm_language_model
    bp_hidden = bp_hidden.to(lm.norm.weight.dtype)
    batch, n_bp = bp_hidden.shape[:2]
    cache = DynamicCache(config=lm.config)
    pos = positions(0, n_bp, batch, bp_hidden.device)
    rope = backbone._rotary_embeddings(bp_hidden, pos)
    additive = bp_hidden.new_zeros(batch, 1, n_bp, n_bp)
    valid = torch.ones(batch, n_bp, dtype=torch.bool, device=bp_hidden.device)
    full_kv = []
    hidden = bp_hidden
    for idx, layer in enumerate(lm.layers):
        if layer.block_type == "linear_attention":
            hidden = backbone._run_linear_stream(
                hidden, input_norm=layer.input_layernorm, mixer=layer.linear_attn, valid_mask=valid, cache=cache
            )
            full_kv.append(None)
        else:
            residual = hidden
            q, k, v, gate = backbone._project_full(layer.self_attn, layer.input_layernorm(hidden))
            q, k = backbone._qwen35.apply_rotary_pos_emb(q, k, *rope)
            full_kv.append((k, v))
            attended = backbone._apply_output_gate(backbone._run_attention(layer.self_attn, q, k, v, additive), gate)
            hidden = residual + layer.self_attn.o_proj(attended.to(layer.self_attn.o_proj.weight.dtype))
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    snapshot = {}
    for idx, layer in enumerate(cache.layers):
        if hasattr(layer, "recurrent_states") and layer.recurrent_states.get(0) is not None:
            snapshot[idx] = (layer.conv_states[0].clone(), layer.recurrent_states[0].clone())
    return {"cache": cache, "snapshot": snapshot, "full_kv": full_kv, "n_bp": n_bp}


def restore(bp_cache) -> None:
    for idx, (conv, rec) in bp_cache["snapshot"].items():
        layer = bp_cache["cache"].layers[idx]
        layer.conv_states[0].copy_(conv)
        layer.recurrent_states[0].copy_(rec)


def prefill_with_cache(model, cur_hidden: torch.Tensor, state: torch.Tensor, bp_cache) -> Q35PrefixCache:
    backbone = model.core.backbone
    lm = backbone.vlm_language_model
    batch, n_cur = cur_hidden.shape[:2]
    n_bp = bp_cache["n_bp"]
    restore(bp_cache)
    cache = bp_cache["cache"]
    pos = positions(n_bp, n_cur, batch, cur_hidden.device)
    rope = backbone._rotary_embeddings(cur_hidden, pos)
    additive = cur_hidden.new_zeros(batch, 1, n_cur, n_bp + n_cur)
    valid = torch.ones(batch, n_cur, dtype=torch.bool, device=cur_hidden.device)
    states = []
    hidden = cur_hidden
    for layer, bp_kv in zip(lm.layers, bp_cache["full_kv"], strict=True):
        if layer.block_type == "linear_attention":
            hidden = backbone._run_linear_stream(
                hidden, input_norm=layer.input_layernorm, mixer=layer.linear_attn, valid_mask=valid, cache=cache
            )
            states.append(None)
        else:
            residual = hidden
            q, k, v, gate = backbone._project_full(layer.self_attn, layer.input_layernorm(hidden))
            q, k = backbone._qwen35.apply_rotary_pos_emb(q, k, *rope)
            key = torch.cat((bp_kv[0], k), dim=2)
            value = torch.cat((bp_kv[1], v), dim=2)
            states.append(Q35FullPrefixState(key=key, value=value))
            attended = backbone._apply_output_gate(
                backbone._run_attention(layer.self_attn, q, key, value, additive), gate
            )
            hidden = residual + layer.self_attn.o_proj(attended.to(layer.self_attn.o_proj.weight.dtype))
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    core = model.core
    state_hidden = core.state_projection(state.to(core.state_projection.weight.dtype)).to(cur_hidden.dtype)[:, None]
    return Q35PrefixCache(expert_type="action", full_attention=tuple(states), condition_hidden=state_hidden)


def action_mask_and_positions(model, batch: int, n_vlm: int, device):
    chunk = model.config.chunk_size
    layout = Q35TokenLayout.for_step("robot", expert_type="action", vlm=n_vlm, state=1, action=chunk)
    valid = torch.ones(batch, n_vlm + 1 + chunk, dtype=torch.bool, device=device)
    full = layout.build_attention_mask(valid_mask=valid, causal_vlm=False)
    suffix = slice(layout.span("state").start, layout.total_tokens)
    return full[..., suffix, :], positions(n_vlm, 1 + chunk, batch, device)


def current_hidden(model, data) -> torch.Tensor:
    pixels = data[f"{OBS_PREFIX}pixel_values"]
    grids = data[f"{OBS_PREFIX}image_grid_thw"]
    feats = model._visual_features(pixels.reshape(-1, pixels.shape[-1]), grids.reshape(-1, 3))
    hidden = model._scatter_current_prefix(feats, data[f"{OBS_PREFIX}input_ids"])
    return hidden.to(model.core.backbone.vlm_language_model.norm.weight.dtype)


def cached_step(model, data, bp_cache, noise, mask_pos):
    with record_function("stage::vit"):
        cur = current_hidden(model, data)
    with record_function("stage::vlm_prefill"):
        prefix_cache = prefill_with_cache(model, cur, data[OBS_STATE], bp_cache)
    with record_function("stage::denoise_10steps"):
        return model._sample_action_flow(noise, prefix_cache, *mask_pos)


def reference_actions(model, data, bp_tokens, noise):
    """不用 cache，整段 [BP, 当前] 一次 prefill，BP 行看不到当前列。用来验证 cache 路径等价。"""
    cur = current_hidden(model, data)
    batch, n_bp = bp_tokens.shape[:2]
    n_cur = cur.shape[1]
    hidden = torch.cat([bp_tokens.to(cur.dtype), cur], dim=1)
    n_vlm = n_bp + n_cur
    chunk = model.config.chunk_size
    layout = Q35TokenLayout.for_step("robot", expert_type="action", vlm=n_vlm, state=1, action=chunk)
    full = layout.build_attention_mask(
        valid_mask=torch.ones(batch, n_vlm + 1 + chunk, dtype=torch.bool, device=cur.device), causal_vlm=False
    ).clone()
    full[..., :n_bp, n_bp:n_vlm] = False
    pos = positions(0, n_vlm + 1 + chunk, batch, cur.device)
    state_hidden = model.core.state_projection(
        data[OBS_STATE].to(model.core.state_projection.weight.dtype)
    ).to(cur.dtype)[:, None]
    _, _, prefix_cache = model.core.backbone.prepare_inference(
        hidden, state_hidden, attention_mask=full[..., :n_vlm, :n_vlm], position_ids=pos[..., :n_vlm]
    )
    suffix = slice(layout.span("state").start, layout.total_tokens)
    return model._sample_action_flow(noise, prefix_cache, full[..., suffix, :], pos[..., suffix])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-path", default="/vla/workspace/models/q35/base_800m")
    parser.add_argument("--chunks", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--no-kernelize", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    args.num_chunks = max(args.chunks)

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    policy = build_policy(args, device)
    policy.eval()
    model = policy.model
    cfg = model.core.backbone.vlm.config
    token_ids = dict(image_token_id=int(cfg.image_token_id), vs_id=int(cfg.vision_start_token_id),
                     ve_id=int(cfg.vision_end_token_id))

    timer = StageTimer()
    timer.wrap(model, "_visual_features", "vit")
    timer.wrap(model.bp_obs_encoder, "forward", "bp_compressor")
    timer.wrap(model.core, "prepare_inference_cache", "vlm_prefill")
    timer.wrap(model, "_sample_action_flow", "denoise_10steps")
    dummy = StageTimer()
    embeds = {"compressed": model.embed_prefix_with_bp, "raw": types.MethodType(embed_prefix_raw, model)}
    results: dict = {"no_cache": {}, "cache_step": {}, "cache_build": {}, "check": {}}

    with torch.no_grad():
        for k in args.chunks:
            for batch in args.batches:
                data = make_batch(batch, k, device, train=False, **token_ids)
                noise = torch.randn(batch, 50, 32, device=device)
                kwargs = dict(
                    pixel_values=data[f"{OBS_PREFIX}pixel_values"],
                    image_grid_thw=data[f"{OBS_PREFIX}image_grid_thw"],
                    input_ids=data[f"{OBS_PREFIX}input_ids"],
                    mm_token_type_ids=data[f"{OBS_PREFIX}mm_token_type_ids"],
                    attention_mask=data[f"{OBS_PREFIX}attention_mask"],
                    state=data[OBS_STATE],
                    behavior_prompt=data["behavior_prompt"],
                )
                for mode, embed in embeds.items():
                    key = f"k{k}_{mode}_b{batch}"
                    model.embed_prefix_with_bp = embed
                    run = lambda: model.sample_actions(noise=noise, **kwargs)  # noqa: E731
                    for _ in range(5):
                        run()
                    results["no_cache"][key] = profile_gpu_time(run, timer, repeats=3)

                    build = lambda: build_bp_cache(model, encode_bp_tokens(model, data, mode))  # noqa: E731
                    for _ in range(3):
                        bp_cache = build()
                    results["cache_build"][key] = profile_gpu_time(build, dummy, repeats=3)
                    bp_cache = build()
                    bp_tokens = encode_bp_tokens(model, data, mode)
                    mask_pos = action_mask_and_positions(model, batch, bp_cache["n_bp"] + 153, device)
                    step = lambda: cached_step(model, data, bp_cache, noise, mask_pos)  # noqa: E731
                    for _ in range(5):
                        step()
                    results["cache_step"][key] = profile_gpu_time(step, dummy, repeats=3)

                    if batch == 1:
                        ref = reference_actions(model, data, bp_tokens, noise).float()
                        got = step().float()
                        results["check"][key] = {
                            "max_abs_diff": float((ref - got).abs().max()),
                            "ref_abs_mean": float(ref.abs().mean()),
                        }
                    nc = results["no_cache"][key]["total"]
                    cs = results["cache_step"][key]["total"]
                    cb = results["cache_build"][key]["total"]
                    chk = results["check"].get(key, {})
                    print(f"[K={k} B={batch} {mode}] no_cache={nc:.1f}ms cache_step={cs:.1f}ms "
                          f"cache_build={cb:.1f}ms check={chk}", flush=True)
                    del bp_cache, bp_tokens
                del data
                torch.cuda.empty_cache()

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
