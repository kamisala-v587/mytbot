from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .experts import Q35FullPrefixState, Q35PrefixCache, build_expert_adapter
from .qwen3_5_backbone import (
    Qwen35Fingerprint,
    build_q35_backbone_parts,
    qwen35_modeling_module,
)


class Q35Backbone(nn.Module):
    """Qwen3.5 VLM paired with a width-scaled action expert."""

    def __init__(self, config) -> None:
        super().__init__()
        vlm, expert, fingerprint = build_q35_backbone_parts(config)
        self._adopt_qwen_parts(
            vlm,
            expert,
            expert_type=config.expert_type,
            fingerprint=fingerprint,
            attention_implementation=config.attention_implementation,
        )

    def _adopt_qwen_parts(
        self,
        vlm: nn.Module,
        expert: nn.Module,
        *,
        expert_type: str,
        fingerprint: Qwen35Fingerprint,
        attention_implementation: str,
    ) -> None:
        language_model = getattr(vlm, "model", None)
        language_model = (
            language_model.language_model if language_model is not None else vlm.language_model
        )
        if len(language_model.layers) != len(expert.layers):
            raise ValueError("VLM and action expert must have the same number of layers")

        self.vlm = vlm
        self.expert = build_expert_adapter(expert_type, expert)
        self.fingerprint = fingerprint
        self.attention_implementation = attention_implementation
        self._qwen35 = qwen35_modeling_module()

    @property
    def vlm_language_model(self) -> nn.Module:
        model = getattr(self.vlm, "model", None)
        return model.language_model if model is not None else self.vlm.language_model

    @property
    def vlm_visual(self) -> nn.Module:
        model = getattr(self.vlm, "model", None)
        return model.visual if model is not None else self.vlm.visual

    @property
    def expert_config(self):
        return self.expert.config

    @property
    def expert_depth(self) -> int:
        return len(self.expert.layers)

    @property
    def action_norm(self) -> nn.Module:
        return self.expert.action_norm

    @staticmethod
    def _project_full(attention: nn.Module, hidden: torch.Tensor):
        input_shape = hidden.shape[:-1]
        hidden = hidden.to(dtype=attention.q_proj.weight.dtype)
        head_dim = attention.head_dim
        query_heads = attention.q_proj.out_features // (head_dim * 2)
        key_value_heads = attention.k_proj.out_features // head_dim
        query, gate = torch.chunk(
            attention.q_proj(hidden).view(*input_shape, query_heads, head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, query_heads * head_dim)
        query = attention.q_norm(query).transpose(1, 2)
        key = attention.k_norm(
            attention.k_proj(hidden).view(*input_shape, key_value_heads, head_dim)
        ).transpose(1, 2)
        value = attention.v_proj(hidden).view(
            *input_shape,
            key_value_heads,
            head_dim,
        ).transpose(1, 2)
        return query, key, value, gate

    @staticmethod
    def _additive_attention_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask[:, None]
        elif mask.ndim != 4:
            raise ValueError("attention mask must have shape [B, Q, K] or [B, H, Q, K]")
        if mask.dtype != torch.bool:
            return mask.to(dtype=dtype)
        zero = torch.zeros((), device=mask.device, dtype=dtype)
        blocked = torch.full((), torch.finfo(dtype).min, device=mask.device, dtype=dtype)
        return torch.where(mask, zero, blocked)

    @staticmethod
    def _token_validity(mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 4:
            mask = mask[:, 0]
        if mask.ndim != 3 or mask.shape[-2] != mask.shape[-1]:
            raise ValueError("token validity requires a square attention mask")
        if mask.dtype == torch.bool:
            return mask.diagonal(dim1=-2, dim2=-1)
        return mask.diagonal(dim1=-2, dim2=-1) == 0

    @staticmethod
    def _query_validity(mask: torch.Tensor, query_length: int) -> torch.Tensor:
        if mask.ndim == 4:
            mask = mask[:, 0]
        if mask.ndim != 3 or mask.shape[-2] != query_length:
            raise ValueError("query mask must have shape [B, query_length, key_length]")
        query_block = mask[..., -query_length:]
        if mask.dtype == torch.bool:
            return query_block.diagonal(dim1=-2, dim2=-1)
        return query_block.diagonal(dim1=-2, dim2=-1) == 0

    def _run_attention(
        self,
        attention: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        additive_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.attention_implementation == "sdpa":
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=additive_mask[..., : key.shape[-2]],
                dropout_p=0.0,
                is_causal=False,
                scale=attention.scaling,
                enable_gqa=True,
            ).transpose(1, 2).contiguous()
        attended, _ = self._qwen35.eager_attention_forward(
            attention,
            query,
            key,
            value,
            additive_mask,
            dropout=0.0,
            scaling=attention.scaling,
        )
        return attended

    @staticmethod
    def _apply_output_gate(attended: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return attended.reshape(*gate.shape) * torch.sigmoid(gate)

    def _rotary_embeddings(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        sequence_length: int | None = None,
    ):
        language_model = self.vlm_language_model
        sequence_length = hidden.shape[1] if sequence_length is None else sequence_length
        rotary_input = hidden.new_zeros(
            hidden.shape[0],
            sequence_length,
            language_model.config.head_dim,
        )
        return language_model.rotary_emb(rotary_input, position_ids)

    @staticmethod
    def _run_linear_stream(
        hidden: torch.Tensor,
        *,
        input_norm: nn.Module,
        mixer: nn.Module,
        valid_mask: torch.Tensor,
        cache,
    ) -> torch.Tensor:
        if hidden.shape[1] == 0:
            return hidden
        residual = hidden
        mixed = mixer(
            hidden_states=input_norm(hidden),
            cache_params=cache,
            attention_mask=valid_mask,
        )
        return residual + mixed

    def _run_vlm_full_layer(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        *,
        additive_mask: torch.Tensor,
        position_embeddings,
    ) -> torch.Tensor:
        residual = hidden
        normalized = layer.input_layernorm(hidden)
        query, key, value, gate = self._project_full(layer.self_attn, normalized)
        query, key = self._qwen35.apply_rotary_pos_emb(query, key, *position_embeddings)
        attended = self._run_attention(layer.self_attn, query, key, value, additive_mask)
        attended = self._apply_output_gate(attended, gate)
        hidden = residual + layer.self_attn.o_proj(
            attended.to(layer.self_attn.o_proj.weight.dtype)
        )
        residual = hidden
        return residual + layer.mlp(layer.post_attention_layernorm(hidden))

    def forward_vlm(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        language_model = self.vlm_language_model
        position_embeddings = self._rotary_embeddings(hidden, position_ids)
        additive_mask = self._additive_attention_mask(attention_mask, hidden.dtype)
        validity = self._token_validity(attention_mask)
        for layer in language_model.layers:
            if layer.block_type == "linear_attention":
                hidden = self._run_linear_stream(
                    hidden,
                    input_norm=layer.input_layernorm,
                    mixer=layer.linear_attn,
                    valid_mask=validity,
                    cache=None,
                )
                residual = hidden
                hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
            else:
                hidden = self._run_vlm_full_layer(
                    layer,
                    hidden,
                    additive_mask=additive_mask,
                    position_embeddings=position_embeddings,
                )
        return language_model.norm(hidden)

    def forward_joint(
        self,
        vlm_hidden: torch.Tensor,
        condition_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.expert.forward_joint(
            self,
            vlm_hidden,
            condition_hidden,
            action_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def prepare_inference(
        self,
        vlm_hidden: torch.Tensor,
        condition_hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Q35PrefixCache]:
        return self.expert.prepare_inference(
            self,
            vlm_hidden,
            condition_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def denoise(
        self,
        action_hidden: torch.Tensor,
        prefix_cache: Q35PrefixCache,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.expert.denoise(
            self,
            action_hidden,
            prefix_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )


__all__ = ["Q35Backbone", "Q35FullPrefixState", "Q35PrefixCache"]
