"""B-3: Qwen upper-layer LoRA on top of a B-2 checkpoint (Stage B 开发文档 2.7 / 5.4; StageB实施.md D6).

  load_init_weights   the B-2 model weights (`<pretrained_model>/model.safetensors`, keys `model.*`) loaded strictly
                      into a freshly built BPV4CDXBModel BEFORE the LoRA injection (injection renames the targeted
                      linears to `...base_layer`)
  add_qwen_lora       peft LoRA (no dropout) on the attention / linear-attention projections and the MLP linears of the
                      last `layers` decoder layers of the Qwen language model; the ViT and every original Qwen weight
                      stay frozen; at injection the model output is unchanged (LoRA B = 0)
"""

from __future__ import annotations

from pathlib import Path

import torch

TARGET_LINEARS = r"(linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"


def load_init_weights(model: torch.nn.Module, pretrained_dir: str) -> None:
    from safetensors.torch import load_file

    state = load_file(str(Path(pretrained_dir) / "model.safetensors"))
    state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    res = model.load_state_dict(state, strict=False)
    # tied weights are saved once (e.g. Qwen embed_tokens = lm_head): a missing key is fine only if its tensor shares
    # storage with a loaded one; anything else missing or unexpected is an error
    own = model.state_dict(keep_vars=True)
    loaded_ptrs = {own[k].data_ptr() for k in state if k in own}
    bad = [k for k in res.missing_keys if own[k].data_ptr() not in loaded_ptrs]
    if bad or res.unexpected_keys:
        raise RuntimeError(f"init_from {pretrained_dir}: missing {bad[:5]}, unexpected {res.unexpected_keys[:5]}")


def add_qwen_lora(vlm: torch.nn.Module, layers: int, rank: int, alpha: float, layer_ids=None) -> int:
    """LoRA on `layer_ids` (explicit decoder layer indices) or else the last `layers` layers. In the LAST decoder layer
    the prefix only feeds the Action Expert through its key / value projections (its attention output and MLP are
    never read), so there only k_proj / v_proj get LoRA."""
    from peft import LoraConfig, inject_adapter_in_model

    total = int(vlm.config.text_config.num_hidden_layers)
    ids = sorted({int(i) for i in layer_ids}) if layer_ids else list(range(total - int(layers), total))
    if any(i < 0 or i >= total for i in ids):
        raise ValueError(f"qwen LoRA layer ids {ids} outside 0..{total - 1}")
    body = [str(i) for i in ids if i != total - 1]
    parts = [rf".*language_model\.layers\.({'|'.join(body)})\.{TARGET_LINEARS}"] if body else []
    if total - 1 in ids:
        parts.append(rf".*language_model\.layers\.{total - 1}\.self_attn\.(k_proj|v_proj)")
    target = "|".join(f"(?:{p})" for p in parts)
    inject_adapter_in_model(LoraConfig(r=int(rank), lora_alpha=float(alpha), lora_dropout=0.0, target_modules=target), vlm)
    lora = 0
    for name, p in vlm.named_parameters():
        is_lora = "lora_" in name
        p.requires_grad_(is_lora)
        lora += int(is_lora)
    if lora == 0:
        raise RuntimeError(f"qwen LoRA: no module matched {target}")
    return lora
