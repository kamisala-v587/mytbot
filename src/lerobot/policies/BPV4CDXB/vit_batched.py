"""Batched vision attention for the Qwen3.5 ViT (StageB_ViT与像素传输提速开发.md, option `vit_batched_attention`).

With the `sdpa` implementation, transformers' `Qwen3_5VisionAttention` splits the concatenated patch sequence back
into one chunk per image and calls attention once per image in a Python loop (192 images x 12 layers = 2,304 calls
per B-2 step). When every image has the same number of patches, the sequence is reshaped to
[images, heads, patches, head_dim] and attended in ONE scaled_dot_product_attention call per layer; every image still
attends only to its own patches, so the result is the same computation. Mixed sizes fall back to the original forward.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def _batched_forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    first = int(lengths[0])
    if position_embeddings is None or not bool((lengths == first).all()):
        return self._unbatched_forward(hidden_states, cu_seqlens, position_embeddings=position_embeddings, **kwargs)
    seq = hidden_states.shape[0]
    n = seq // first
    q, k, v = self.qkv(hidden_states).reshape(seq, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)  # [seq, heads, head_dim]
    q, k, v = (t.reshape(n, first, self.num_heads, -1).transpose(1, 2) for t in (q, k, v))  # [n, heads, L, d]
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0 if not self.training else self.attention_dropout,
                                         scale=self.scaling)
    out = out.transpose(1, 2).reshape(seq, -1).contiguous()  # image-major, same order as the per-image loop
    return self.proj(out)


def patch_vision_attention(module: torch.nn.Module) -> int:
    """Replace the forward of every Qwen3_5VisionAttention under `module`; returns the number patched."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionAttention

    n = 0
    for m in module.modules():
        if isinstance(m, Qwen3_5VisionAttention) and not hasattr(m, "_unbatched_forward"):
            m._unbatched_forward = m.forward
            m.forward = types.MethodType(_batched_forward, m)
            n += 1
    if n == 0:
        raise RuntimeError("vit_batched_attention: no Qwen3_5VisionAttention module found (transformers changed?)")
    return n
