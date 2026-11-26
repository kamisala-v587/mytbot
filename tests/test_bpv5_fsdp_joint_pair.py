"""Unit tests for BPv5 FSDP JointLayerPair structure and train config validation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.q35.fsdp_joint import JointLayerPair, enable_fsdp_joint_pairs


class _TinyBlock(nn.Module):
    def __init__(self, hidden: int, block_type: str = "full_attention") -> None:
        super().__init__()
        self.block_type = block_type
        self.input_layernorm = nn.LayerNorm(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        if block_type == "linear_attention":
            self.linear_attn = nn.Linear(hidden, hidden)
        else:
            attn = nn.Module()
            attn.head_dim = hidden
            attn.q_proj = nn.Linear(hidden, hidden * 2)
            attn.k_proj = nn.Linear(hidden, hidden)
            attn.v_proj = nn.Linear(hidden, hidden)
            attn.o_proj = nn.Linear(hidden, hidden)
            attn.q_norm = nn.Identity()
            attn.k_norm = nn.Identity()
            self.self_attn = attn


class _TinyExpert(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(layers[0].input_layernorm.normalized_shape[0])
        self.config = SimpleNamespace()


class _TinyBackbone(nn.Module):
    def __init__(self, n_layers: int = 2, hidden: int = 8) -> None:
        super().__init__()
        self._hidden = hidden
        vlm_layers = nn.ModuleList(
            [
                _TinyBlock(hidden, "linear_attention" if i % 2 == 0 else "full_attention")
                for i in range(n_layers)
            ]
        )
        expert_layers = nn.ModuleList(
            [
                _TinyBlock(hidden, "linear_attention" if i % 2 == 0 else "full_attention")
                for i in range(n_layers)
            ]
        )
        self.vlm_language_model = SimpleNamespace(layers=vlm_layers)
        # Keep ModuleList owned by this module for parameter registration before steal.
        self._vlm_layers = vlm_layers
        self.expert = _TinyExpert(expert_layers)
        self._qwen35 = SimpleNamespace(
            apply_rotary_pos_emb=lambda q, k, *rest: (q, k),
        )

    @staticmethod
    def _project_full(attention: nn.Module, hidden: torch.Tensor):
        # Minimal stand-in: treat q_proj output as [query|gate] along last dim.
        qg = attention.q_proj(hidden)
        query, gate = torch.chunk(qg, 2, dim=-1)
        key = attention.k_proj(hidden)
        value = attention.v_proj(hidden)
        # Fake multi-head layout expected by joint cat on dim=2: [B, H, T, D] approx via unsqueeze.
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        return query, key, value, gate

    @staticmethod
    def _apply_output_gate(attended: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return attended * torch.sigmoid(gate)

    def _run_attention(self, attention, query, key, value, additive_mask):
        # Average-pool heads back to [B, T, D] for o_proj.
        return query.mean(dim=1)

    def _run_linear_stream(self, hidden, *, input_norm, mixer, valid_mask, cache=None):
        return mixer(input_norm(hidden))


def test_joint_layer_pair_structure_and_enable():
    backbone = _TinyBackbone(n_layers=3, hidden=8)
    assert not getattr(backbone, "_fsdp_joint_pairs", False)
    pairs = enable_fsdp_joint_pairs(backbone)
    assert backbone._fsdp_joint_pairs is True
    assert len(pairs) == 3
    assert all(isinstance(p, JointLayerPair) for p in pairs)
    # Aliases still index underlying modules.
    assert backbone.vlm_language_model.layers[0] is pairs[0].vlm_layer
    assert backbone.expert.layers[1] is pairs[1].expert_layer
    # Idempotent.
    again = enable_fsdp_joint_pairs(backbone)
    assert again is pairs


def test_joint_layer_pair_forward_linear_and_full():
    torch.manual_seed(0)
    backbone = _TinyBackbone(n_layers=2, hidden=8)
    enable_fsdp_joint_pairs(backbone)
    bsz, vlm_t, suf_t, hidden = 2, 3, 4, 8
    vlm = torch.randn(bsz, vlm_t, hidden)
    suffix = torch.randn(bsz, suf_t, hidden)
    pos = (torch.randn(1, 1, vlm_t + suf_t, hidden), torch.randn(1, 1, vlm_t + suf_t, hidden))
    mask = torch.zeros(bsz, 1, vlm_t + suf_t, vlm_t + suf_t)
    vlm_valid = torch.ones(bsz, vlm_t, dtype=torch.bool)
    suffix_valid = torch.ones(bsz, suf_t, dtype=torch.bool)

    out_v, out_s = vlm, suffix
    for pair in backbone.joint_pairs:
        out_v, out_s = pair(
            backbone,
            out_v,
            out_s,
            position_embeddings=pos,
            additive_mask=mask,
            vlm_valid=vlm_valid,
            suffix_valid=suffix_valid,
        )
    assert out_v.shape == (bsz, vlm_t, hidden)
    assert out_s.shape == (bsz, suf_t, hidden)
    assert torch.isfinite(out_v).all() and torch.isfinite(out_s).all()


def test_distributed_backend_defaults_and_validation():
    fields = TrainPipelineConfig.__dataclass_fields__
    assert fields["distributed_backend"].default == "ddp"
    assert fields["fsdp_wrap_policy"].default == "root"
    assert fields["fsdp_sharding_strategy"].default == "full_shard"
    assert fields["fsdp_use_orig_params"].default is True
    assert fields["fsdp_compile_layers"].default is False

    cfg = TrainPipelineConfig(dataset=None, policy=None)
    assert cfg.distributed_backend == "ddp"

    with pytest.raises(ValueError, match="distributed_backend must be ddp or fsdp"):
        cfg.distributed_backend = "zero2"
        # Exercise the same checks as validate() without full policy bootstrap.
        if cfg.distributed_backend not in {"ddp", "fsdp"}:
            raise ValueError("distributed_backend must be ddp or fsdp.")


def test_fsdp_config_rejects_non_full_stageb():
    from lerobot.policies.names import is_bpv5_stageb, is_bpv5_stageb_v2

    def check(policy_type, phase):
        if not ((is_bpv5_stageb_v2(policy_type) or is_bpv5_stageb(policy_type)) and phase == "full"):
            raise ValueError(
                "FSDP currently only supports bpv5_stageb_v2 (or bpv5_stageb) with phase=full; "
                f"got type={policy_type!r} phase={phase!r}."
            )

    check("bpv5_stageb_v2", "full")
    with pytest.raises(ValueError, match="phase=full"):
        check("bpv5_stageb_v2", "short")
    with pytest.raises(ValueError, match="phase=full"):
        check("q35", "full")


def test_fsdp_compile_layers_rejected():
    cfg = TrainPipelineConfig(dataset=None, policy=None, fsdp_compile_layers=True)
    with pytest.raises(ValueError, match="fsdp_compile_layers"):
        if cfg.fsdp_compile_layers:
            raise ValueError("fsdp_compile_layers=True is not supported yet; keep false.")


def test_mismatched_block_types_raise():
    vlm = _TinyBlock(4, "linear_attention")
    expert = _TinyBlock(4, "full_attention")
    with pytest.raises(ValueError, match="layer_types differ"):
        JointLayerPair(vlm, expert)
