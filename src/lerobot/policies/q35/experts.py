from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Q35FullPrefixState:
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class Q35PrefixCache:
    # Identifies which expert produced the cache so future expert adapters do
    # not accidentally consume an incompatible prefix representation.
    expert_type: str
    full_attention: tuple[Q35FullPrefixState | None, ...]
    condition_hidden: torch.Tensor | None = None


class ActionExpert(nn.Module):
    """Qwen3.5 action expert over a physical [State, Action] suffix."""

    expert_type = "action"

    def __init__(self, expert: nn.Module) -> None:
        super().__init__()
        self.layers = expert.layers
        self.norm = expert.norm
        self.config = copy.deepcopy(expert.config)

    @property
    def action_norm(self) -> nn.Module:
        return self.norm

    def forward_joint(
        self,
        executor,
        vlm_hidden: torch.Tensor,
        condition_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        language_model = executor.vlm_language_model
        if len(language_model.layers) != len(self.layers):
            raise ValueError("VLM and action expert must have the same number of layers")

        condition_length = condition_hidden.shape[1]
        suffix_hidden = torch.cat((condition_hidden, action_hidden), dim=1)
        lengths = (vlm_hidden.shape[1], suffix_hidden.shape[1])
        position_embeddings = executor._rotary_embeddings(
            vlm_hidden,
            position_ids,
            sequence_length=sum(lengths),
        )
        additive_mask = executor._additive_attention_mask(attention_mask, vlm_hidden.dtype)
        validity = executor._token_validity(attention_mask)
        vlm_valid, suffix_valid = torch.split(validity, lengths, dim=1)

        for vlm_layer, expert_layer in zip(language_model.layers, self.layers, strict=True):
            if vlm_layer.block_type != expert_layer.block_type:
                raise ValueError("VLM and action expert layer_types differ")
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
                vlm_q, vlm_k, vlm_v, vlm_gate = executor._project_full(
                    vlm_layer.self_attn,
                    vlm_norm,
                )
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
                suffix_attended = executor._apply_output_gate(
                    attended[:, vlm_length:],
                    suffix_gate,
                )
                vlm_hidden = vlm_residual + vlm_layer.self_attn.o_proj(
                    vlm_attended.to(vlm_layer.self_attn.o_proj.weight.dtype)
                )
                suffix_hidden = suffix_residual + expert_layer.self_attn.o_proj(
                    suffix_attended.to(expert_layer.self_attn.o_proj.weight.dtype)
                )

            vlm_residual = vlm_hidden
            vlm_hidden = vlm_residual + vlm_layer.mlp(
                vlm_layer.post_attention_layernorm(vlm_hidden)
            )
            suffix_residual = suffix_hidden
            suffix_hidden = suffix_residual + expert_layer.mlp(
                expert_layer.post_attention_layernorm(suffix_hidden)
            )

        suffix_hidden = self.norm(suffix_hidden)
        return (
            language_model.norm(vlm_hidden),
            suffix_hidden[:, :condition_length],
            suffix_hidden[:, condition_length:],
        )

    def prepare_inference(
        self,
        executor,
        vlm_hidden: torch.Tensor,
        condition_hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Q35PrefixCache]:
        from transformers.cache_utils import DynamicCache

        language_model = executor.vlm_language_model
        # Qwen3.5 uses this cache while evaluating the VLM linear-attention
        # prefix. Denoising never reads it, so it is discarded after prefill.
        vlm_cache = DynamicCache(config=language_model.config)
        full_states: list[Q35FullPrefixState | None] = []
        position_embeddings = executor._rotary_embeddings(vlm_hidden, position_ids)
        additive_mask = executor._additive_attention_mask(attention_mask, vlm_hidden.dtype)
        validity = executor._token_validity(attention_mask)

        for layer in language_model.layers:
            if layer.block_type == "linear_attention":
                vlm_hidden = executor._run_linear_stream(
                    vlm_hidden,
                    input_norm=layer.input_layernorm,
                    mixer=layer.linear_attn,
                    valid_mask=validity,
                    cache=vlm_cache,
                )
                full_states.append(None)
            else:
                residual = vlm_hidden
                normalized = layer.input_layernorm(vlm_hidden)
                query, key, value, gate = executor._project_full(layer.self_attn, normalized)
                query, key = executor._qwen35.apply_rotary_pos_emb(
                    query,
                    key,
                    *position_embeddings,
                )
                full_states.append(Q35FullPrefixState(key=key, value=value))
                attended = executor._run_attention(
                    layer.self_attn,
                    query,
                    key,
                    value,
                    additive_mask,
                )
                attended = executor._apply_output_gate(attended, gate)
                vlm_hidden = residual + layer.self_attn.o_proj(
                    attended.to(layer.self_attn.o_proj.weight.dtype)
                )
            residual = vlm_hidden
            vlm_hidden = residual + layer.mlp(layer.post_attention_layernorm(vlm_hidden))

        return (
            language_model.norm(vlm_hidden),
            condition_hidden,
            Q35PrefixCache(
                expert_type=self.expert_type,
                full_attention=tuple(full_states),
                condition_hidden=condition_hidden,
            ),
        )

    def denoise(
        self,
        executor,
        action_hidden: torch.Tensor,
        prefix_cache: Q35PrefixCache,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        from transformers.cache_utils import DynamicCache

        if prefix_cache.expert_type != self.expert_type:
            raise ValueError("ActionExpert received a cache from another expert")
        if prefix_cache.condition_hidden is None:
            raise ValueError("ActionExpert cache is missing State condition hidden")
        if len(prefix_cache.full_attention) != len(self.layers):
            raise ValueError("ActionExpert prefix cache depth does not match the backbone")

        condition_length = prefix_cache.condition_hidden.shape[1]
        suffix_hidden = torch.cat((prefix_cache.condition_hidden, action_hidden), dim=1)
        suffix_cache = DynamicCache(config=self.config)
        suffix_valid = executor._query_validity(attention_mask, suffix_hidden.shape[1])
        additive_mask = executor._additive_attention_mask(attention_mask, suffix_hidden.dtype)
        position_embeddings = executor._rotary_embeddings(suffix_hidden, position_ids)

        for expert_layer, prefix_state in zip(
            self.layers,
            prefix_cache.full_attention,
            strict=True,
        ):
            if expert_layer.block_type == "linear_attention":
                suffix_hidden = executor._run_linear_stream(
                    suffix_hidden,
                    input_norm=expert_layer.input_layernorm,
                    mixer=expert_layer.linear_attn,
                    valid_mask=suffix_valid,
                    cache=suffix_cache,
                )
            else:
                if prefix_state is None:
                    raise ValueError("Missing full-attention state in ActionExpert cache")
                residual = suffix_hidden
                normalized = expert_layer.input_layernorm(suffix_hidden)
                query, suffix_key, suffix_value, gate = executor._project_full(
                    expert_layer.self_attn,
                    normalized,
                )
                query, suffix_key = executor._qwen35.apply_rotary_pos_emb(
                    query,
                    suffix_key,
                    *position_embeddings,
                )
                key = torch.cat((prefix_state.key, suffix_key), dim=2)
                value = torch.cat((prefix_state.value, suffix_value), dim=2)
                attended = executor._run_attention(
                    expert_layer.self_attn,
                    query,
                    key,
                    value,
                    additive_mask,
                )
                attended = executor._apply_output_gate(attended, gate)
                suffix_hidden = residual + expert_layer.self_attn.o_proj(
                    attended.to(expert_layer.self_attn.o_proj.weight.dtype)
                )
            residual = suffix_hidden
            suffix_hidden = residual + expert_layer.mlp(
                expert_layer.post_attention_layernorm(suffix_hidden)
            )
        return self.norm(suffix_hidden)[:, condition_length:]


def build_expert_adapter(expert_type: str, expert: nn.Module) -> ActionExpert:
    """Build the configured expert; extend this factory for future expert types."""

    if expert_type == ActionExpert.expert_type:
        return ActionExpert(expert)
    raise ValueError(f"Unsupported Q35 expert_type: {expert_type!r}")


__all__ = ["ActionExpert", "Q35FullPrefixState", "Q35PrefixCache", "build_expert_adapter"]
