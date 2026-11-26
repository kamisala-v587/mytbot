"""FSDP-safe joint VLM+expert layer units for BPv5 Q35.

Keeps the existing fused joint-attention math (QKV cat across VLM and suffix)
while giving each paired layer a real ``nn.Module.__call__`` boundary so FSDP
can all-gather both streams' parameters before the body runs.

Does NOT switch to prefix/suffix-separated attention.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class _LayerAlias(nn.Module):
    """Indexable view onto ``JointLayerPair`` children without re-owning them.

    Must be an ``nn.Module`` so it can replace a former ``ModuleList`` attribute
    on HF / expert modules (PyTorch rejects non-Module assignments there).
    """

    def __init__(self, pairs: nn.ModuleList, attr: str) -> None:
        super().__init__()
        # Keep a non-registered reference so parameters stay under ``joint_pairs``.
        object.__setattr__(self, "_pairs", pairs)
        object.__setattr__(self, "_attr", attr)

    def __getitem__(self, index: int) -> nn.Module:
        return getattr(self._pairs[index], self._attr)

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class JointLayerPair(nn.Module):
    """One VLM decoder layer + matching action-expert layer as a single FSDP unit."""

    def __init__(self, vlm_layer: nn.Module, expert_layer: nn.Module) -> None:
        super().__init__()
        if getattr(vlm_layer, "block_type", None) != getattr(expert_layer, "block_type", None):
            raise ValueError("VLM and action expert layer_types differ")
        self.vlm_layer = vlm_layer
        self.expert_layer = expert_layer
        self.block_type = vlm_layer.block_type

    def forward(
        self,
        executor: Any,
        vlm_hidden: torch.Tensor,
        suffix_hidden: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, ...],
        additive_mask: torch.Tensor,
        vlm_valid: torch.Tensor,
        suffix_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same per-layer body as ``ActionExpert.forward_joint`` (linear + joint full attn + mlp)."""
        vlm_layer = self.vlm_layer
        expert_layer = self.expert_layer

        if vlm_layer.block_type == "linear_attention":
            vlm_hidden = executor._run_linear_stream(
                vlm_hidden,
                input_norm=vlm_layer.input_layernorm,
                mixer=vlm_layer.linear_attn,
                valid_mask=vlm_valid,
                cache=None,
            )
            suffix_hidden = executor._run_linear_stream(
                suffix_hidden,
                input_norm=expert_layer.input_layernorm,
                mixer=expert_layer.linear_attn,
                valid_mask=suffix_valid,
                cache=None,
            )
        else:
            vlm_residual, suffix_residual = vlm_hidden, suffix_hidden
            vlm_norm = vlm_layer.input_layernorm(vlm_hidden)
            suffix_norm = expert_layer.input_layernorm(suffix_hidden)
            vlm_q, vlm_k, vlm_v, vlm_gate = executor._project_full(vlm_layer.self_attn, vlm_norm)
            suffix_q, suffix_k, suffix_v, suffix_gate = executor._project_full(
                expert_layer.self_attn,
                suffix_norm,
            )
            query = torch.cat((vlm_q, suffix_q), dim=2)
            key = torch.cat((vlm_k, suffix_k), dim=2)
            value = torch.cat((vlm_v, suffix_v), dim=2)
            query, key = executor._qwen35.apply_rotary_pos_emb(
                query,
                key,
                *position_embeddings,
            )
            attended = executor._run_attention(
                vlm_layer.self_attn,
                query,
                key,
                value,
                additive_mask,
            )
            vlm_length = vlm_hidden.shape[1]
            vlm_attended = executor._apply_output_gate(attended[:, :vlm_length], vlm_gate)
            suffix_attended = executor._apply_output_gate(attended[:, vlm_length:], suffix_gate)
            vlm_hidden = vlm_residual + vlm_layer.self_attn.o_proj(
                vlm_attended.to(vlm_layer.self_attn.o_proj.weight.dtype)
            )
            suffix_hidden = suffix_residual + expert_layer.self_attn.o_proj(
                suffix_attended.to(expert_layer.self_attn.o_proj.weight.dtype)
            )

        vlm_residual = vlm_hidden
        vlm_hidden = vlm_residual + vlm_layer.mlp(vlm_layer.post_attention_layernorm(vlm_hidden))
        suffix_residual = suffix_hidden
        suffix_hidden = suffix_residual + expert_layer.mlp(
            expert_layer.post_attention_layernorm(suffix_hidden)
        )
        return vlm_hidden, suffix_hidden


def enable_fsdp_joint_pairs(backbone: nn.Module) -> nn.ModuleList:
    """Steal VLM/expert layers into ``backbone.joint_pairs`` and install layer aliases.

    After this call, ``forward_joint`` must iterate ``backbone.joint_pairs`` so each
    FSDP unit gathers both layers before the joint body runs. ``*.layers[i]`` still
    returns the underlying layer modules for prepare_inference/denoise indexing.
    """
    if getattr(backbone, "_fsdp_joint_pairs", False):
        return backbone.joint_pairs

    language_model = backbone.vlm_language_model
    expert = backbone.expert
    vlm_layers = list(language_model.layers)
    expert_layers = list(expert.layers)
    if len(vlm_layers) != len(expert_layers):
        raise ValueError("VLM and action expert must have the same number of layers")

    pairs = nn.ModuleList(
        [JointLayerPair(vlm_layer, expert_layer) for vlm_layer, expert_layer in zip(vlm_layers, expert_layers, strict=True)]
    )
    backbone.joint_pairs = pairs
    # Replace ModuleLists with aliases so indexing still resolves to the stolen modules.
    language_model.layers = _LayerAlias(pairs, "vlm_layer")  # type: ignore[assignment]
    expert.layers = _LayerAlias(pairs, "expert_layer")  # type: ignore[assignment]
    backbone._fsdp_joint_pairs = True
    return pairs


__all__ = ["JointLayerPair", "enable_fsdp_joint_pairs"]
