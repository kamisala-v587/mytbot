"""Qwen3.5 checkpoint loading and width-scaled action-expert configuration."""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Qwen35Fingerprint:
    name: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    vision_model_type: str | None = None
    vision_depth: int | None = None
    vision_hidden_size: int | None = None
    vision_intermediate_size: int | None = None
    vision_num_heads: int | None = None
    vision_patch_size: int | None = None
    vision_spatial_merge_size: int | None = None
    vision_temporal_patch_size: int | None = None
    vision_out_hidden_size: int | None = None
    vision_num_position_embeddings: int | None = None
    image_token_id: int | None = None
    video_token_id: int | None = None
    vision_start_token_id: int | None = None
    vision_end_token_id: int | None = None


SUPPORTED_QWEN35_FINGERPRINTS = (
    Qwen35Fingerprint(
        "0.8B", 1024, 3584, 24, 8, 2, 256, 16, 16, 128, 128,
        "qwen3_5_vision", 12, 768, 3072, 12, 16, 2, 2, 1024, 2304,
        248056, 248057, 248053, 248054,
    ),
    Qwen35Fingerprint(
        "2B", 2048, 6144, 24, 8, 2, 256, 16, 16, 128, 128,
        "qwen3_5_vision", 24, 1024, 4096, 16, 16, 2, 2, 2048, 2304,
        248056, 248057, 248053, 248054,
    ),
    Qwen35Fingerprint(
        "4B", 2560, 9216, 32, 16, 4, 256, 16, 32, 128, 128,
        "qwen3_5_vision", 24, 1024, 4096, 16, 16, 2, 2, 2560, 2304,
        248056, 248057, 248053, 248054,
    ),
)


def _qwen35_classes():
    """Import Qwen3.5 only when a Q35 model is constructed."""

    try:
        from transformers.models.qwen3_5 import (
            Qwen3_5ForConditionalGeneration,
            Qwen3_5TextModel,
            modeling_qwen3_5,
        )
    except ImportError as exc:
        raise RuntimeError("Q35 requires a Qwen3.5-enabled Transformers build") from exc
    return Qwen3_5ForConditionalGeneration, Qwen3_5TextModel, modeling_qwen3_5


def qwen35_modeling_module():
    return _qwen35_classes()[2]


def _fingerprint(text_config) -> tuple[int, ...]:
    return (
        int(text_config.hidden_size),
        int(text_config.intermediate_size),
        int(text_config.num_hidden_layers),
        int(text_config.num_attention_heads),
        int(text_config.num_key_value_heads),
        int(text_config.head_dim),
        int(text_config.linear_num_key_heads),
        int(text_config.linear_num_value_heads),
        int(text_config.linear_key_head_dim),
        int(text_config.linear_value_head_dim),
    )


def validate_qwen35_text_config(text_config) -> Qwen35Fingerprint:
    actual = _fingerprint(text_config)
    matched = next(
        (
            candidate
            for candidate in SUPPORTED_QWEN35_FINGERPRINTS
            if actual
            == (
                candidate.hidden_size,
                candidate.intermediate_size,
                candidate.num_hidden_layers,
                candidate.num_attention_heads,
                candidate.num_key_value_heads,
                candidate.head_dim,
                candidate.linear_num_key_heads,
                candidate.linear_num_value_heads,
                candidate.linear_key_head_dim,
                candidate.linear_value_head_dim,
            )
        ),
        None,
    )
    if matched is None:
        supported = ", ".join(item.name for item in SUPPORTED_QWEN35_FINGERPRINTS)
        raise ValueError(
            f"Unsupported Qwen3.5 text config fingerprint {actual}; "
            f"Q35 supports official {supported} checkpoints."
        )

    layer_types = tuple(text_config.layer_types)
    expected = ("linear_attention", "linear_attention", "linear_attention", "full_attention") * (
        matched.num_hidden_layers // 4
    )
    if layer_types != expected:
        raise ValueError(
            f"Unsupported Qwen3.5 layer_types for {matched.name}: expected the official 3:1 hybrid pattern"
        )
    return matched


def validate_qwen35_model_config(model_config) -> Qwen35Fingerprint:
    if getattr(model_config, "model_type", None) != "qwen3_5":
        raise ValueError(
            f"Q35 requires model_type='qwen3_5', got {getattr(model_config, 'model_type', None)!r}"
        )
    matched = validate_qwen35_text_config(model_config.text_config)
    vision = model_config.vision_config
    actual = (
        str(vision.model_type),
        int(vision.depth),
        int(vision.hidden_size),
        int(vision.intermediate_size),
        int(vision.num_heads),
        int(vision.patch_size),
        int(vision.spatial_merge_size),
        int(vision.temporal_patch_size),
        int(vision.out_hidden_size),
        int(vision.num_position_embeddings),
        int(model_config.image_token_id),
        int(model_config.video_token_id),
        int(model_config.vision_start_token_id),
        int(model_config.vision_end_token_id),
    )
    expected = (
        matched.vision_model_type,
        matched.vision_depth,
        matched.vision_hidden_size,
        matched.vision_intermediate_size,
        matched.vision_num_heads,
        matched.vision_patch_size,
        matched.vision_spatial_merge_size,
        matched.vision_temporal_patch_size,
        matched.vision_out_hidden_size,
        matched.vision_num_position_embeddings,
        matched.image_token_id,
        matched.video_token_id,
        matched.vision_start_token_id,
        matched.vision_end_token_id,
    )
    if actual != expected:
        raise ValueError(
            f"Unsupported Qwen3.5 vision/token fingerprint for {matched.name}: "
            f"expected {expected}, got {actual}"
        )
    return matched


def _scaled_positive_int(value: int, width_ratio: float, *, multiple: int = 1) -> int:
    """Scale a structural dimension continuously, then round to its nearest positive multiple."""

    if multiple <= 0:
        raise ValueError("scaled dimension multiple must be positive")
    units = math.floor(int(value) * width_ratio / multiple + 0.5)
    return max(1, units) * multiple


def make_expert_config(text_config, *, width_ratio: float):
    if not math.isfinite(width_ratio) or width_ratio <= 0:
        raise ValueError("expert_width_ratio must be a finite positive number")

    expert_config = copy.deepcopy(text_config)
    # Width ratios remain continuous for all supported VLM scales. The hidden
    # size alone is rounded to an even integer because the sinusoidal action
    # time embedding splits it into equal sin/cos halves.
    expert_config.hidden_size = _scaled_positive_int(
        text_config.hidden_size,
        width_ratio,
        multiple=2,
    )
    expert_config.intermediate_size = _scaled_positive_int(
        text_config.intermediate_size,
        width_ratio,
    )

    base_key_heads = int(text_config.linear_num_key_heads)
    base_value_heads = int(text_config.linear_num_value_heads)
    if base_value_heads % base_key_heads:
        raise ValueError("Qwen3.5 linear value heads must be divisible by key heads")
    expert_config.linear_num_key_heads = _scaled_positive_int(base_key_heads, width_ratio)
    # Preserve the checkpoint's value-heads-per-key-head ratio after rounding
    # the scaled key-head count to a positive integer.
    expert_config.linear_num_value_heads = (
        expert_config.linear_num_key_heads * (base_value_heads // base_key_heads)
    )

    # Full-attention geometry is deliberately unchanged. VLM and action Q/K/V
    # must share a head space before concatenation along the token dimension.
    expert_config.num_attention_heads = int(text_config.num_attention_heads)
    expert_config.num_key_value_heads = int(text_config.num_key_value_heads)
    expert_config.head_dim = int(text_config.head_dim)
    expert_config.linear_key_head_dim = int(text_config.linear_key_head_dim)
    expert_config.linear_value_head_dim = int(text_config.linear_value_head_dim)
    # The action expert has no token embedding. Avoid allocating the 248k-token
    # vocabulary while constructing the official text-layer container.
    expert_config.vocab_size = 1
    expert_config.pad_token_id = 0
    expert_config.tie_word_embeddings = False
    return expert_config


def _cast_module_precision(module: nn.Module, precision: Literal["bfloat16", "float32"]) -> None:
    if precision == "bfloat16":
        module.to(dtype=torch.bfloat16)
    elif precision == "float32":
        module.to(dtype=torch.float32)
    else:
        raise ValueError(f"Invalid precision: {precision}")


def kernelize_qwen35_training_backbone(backbone: nn.Module) -> bool:
    """Enable official CUDA training kernels after device placement."""

    device = next(backbone.parameters()).device
    if device.type != "cuda":
        return False

    try:
        from kernels import LayerRepository, Mode, kernelize, register_kernel_mapping
    except ImportError:
        logger.warning(
            "Package `kernels` is not installed; Q35 will train with the PyTorch "
            "linear-attention fallback. Install `kernels` plus flash-linear-attention "
            "to enable the CUDA fast path."
        )
        return False

    # Torch 2.8 has an official FLA v1 variant, while the available mamba-ssm
    # v2 kernels require newer Torch builds. Register the compatible GDN
    # mappings and leave causal convolution on its native grouped-conv path.
    fla = LayerRepository(
        repo_id="kernels-community/fla",
        layer_name="chunk_gated_delta_rule",
        revision="aa89c609e8ca934132edc520e430716932ec5dde",
        trust_remote_code=True,
    )
    recurrent_fla = LayerRepository(
        repo_id="kernels-community/fla",
        layer_name="recurrent_gated_delta_rule",
        revision="aa89c609e8ca934132edc520e430716932ec5dde",
        trust_remote_code=True,
    )
    register_kernel_mapping(
        {
            "chunk_gated_delta_rule": {"cuda": {Mode.TRAINING: fla}},
            "fused_recurrent_gated_delta_rule": {"cuda": {Mode.TRAINING: recurrent_fla}},
        }
    )
    # The pinned Transformers build does not register these kernels under the
    # combined TRAINING|TORCH_COMPILE mode. The outer policy may still compile.
    try:
        kernelize(backbone, mode=Mode.TRAINING, device=device)
    except Exception as exc:
        logger.warning("Failed to enable Q35 CUDA training kernels: %s", exc)
        return False
    return True


def build_q35_backbone_parts(config):
    """Load an official Qwen3.5 VLM and construct one width-scaled action expert."""

    Qwen3_5ForConditionalGeneration, Qwen3_5TextModel, _ = _qwen35_classes()
    precision = config.dtype
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
    load_kwargs = {
        "dtype": dtype,
        "attn_implementation": config.attention_implementation,
    }
    model_revision = getattr(config, "qwen3_5_model_revision", None)
    if model_revision:
        load_kwargs["revision"] = model_revision

    vlm = Qwen3_5ForConditionalGeneration.from_pretrained(
        config.qwen3_5_pretrained_path,
        **load_kwargs,
    )
    fingerprint = validate_qwen35_model_config(vlm.config)
    expert_config = make_expert_config(
        vlm.config.text_config,
        width_ratio=config.expert_width_ratio,
    )
    expert = Qwen3_5TextModel(expert_config)
    expert.embed_tokens = None

    # Match from_pretrained(dtype=...) semantics. Qwen RMSNorm promotes
    # internally where required; forcing every norm parameter to FP32 would
    # also change the vision tower's dtype contract.
    _cast_module_precision(vlm, precision)
    _cast_module_precision(expert, precision)
    return vlm, expert, fingerprint
