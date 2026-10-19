from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from lerobot.policies.BPVAv2.bp_transformer_obs_encoder import (
    BPObsEncoder,
    BPObsEncoderOutput,
    BPTransformerObsEncoder,
)


def sinusoidal_progress_encoding(progress: torch.Tensor, dim: int) -> torch.Tensor:
    """progress ∈ [0, 1]，输出与 chunk token 同宽的正弦进度编码。

    频率按几何级数排布，但最高倍频封顶在 2^max_octaves，避免旧实现
    ``π·2^i`` 在 dim≥256（如 Qwen hidden=1024）时 float32 溢出成 inf/nan。
    """
    if dim % 2:
        raise ValueError(f"progress encoding dim must be even, got {dim}")
    progress = progress.float().clamp(0.0, 1.0)
    half = dim // 2
    max_octaves = 8.0  # 最高约 256 个周期，覆盖 [0,1] 进度足够
    i = torch.arange(half, device=progress.device, dtype=torch.float32)
    if half == 1:
        freq = torch.full((1,), math.pi, device=progress.device, dtype=torch.float32)
    else:
        freq = math.pi * (2.0 ** (i * (max_octaves / float(half - 1))))
    angles = progress[..., None] * freq
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class BPVAv31ChunkEncoder(BPTransformerObsEncoder):
    """压缩 BP chunk。视觉 token 由共享 Qwen3.5 ViT 预先算好，这里不再跑 visual。"""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["freeze_shared_visual"] = False
        kwargs["use_type_embedding"] = True
        kwargs["use_camera_embedding"] = True
        super().__init__(*args, **kwargs)
        self._visual_tokens: dict[str, torch.Tensor] | None = None
        self._tokens_per_image: int | None = None

    def _encode_images(
        self,
        prompt,
        visual,
        batch_size: int,
        num_chunks: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del visual
        if self._visual_tokens is None or self._tokens_per_image is None:
            raise RuntimeError("BPVAv31 visual tokens must be produced by the shared Qwen3.5 ViT")

        masks = prompt.get("image_masks")
        if masks is not None and not isinstance(masks, dict):
            raise TypeError("image_masks must be a dictionary")

        encoded_by_key: dict[str, torch.Tensor] = {}
        valid_by_key: dict[str, torch.Tensor] = {}
        projection_weight = self.visual_projection[0].weight
        tokens_per_image = int(self._tokens_per_image)
        for key, raw in self._visual_tokens.items():
            if key not in self.image_keys:
                raise KeyError(f"Visual tokens contain an undeclared BP camera: {key}")
            if raw.ndim != 4 or raw.shape[:2] != (batch_size, num_chunks):
                raise ValueError(f"Unexpected visual token shape for {key}: {tuple(raw.shape)}")
            if raw.shape[2] != tokens_per_image or raw.shape[-1] != self.visual_output_dim:
                raise ValueError(
                    f"Visual tokens for {key} must be "
                    f"[B,K,{tokens_per_image},{self.visual_output_dim}], got {tuple(raw.shape)}"
                )
            projected = self.visual_projection(raw.to(device=projection_weight.device, dtype=projection_weight.dtype))
            encoded_by_key[key] = projected
            if masks is None or key not in masks:
                valid_by_key[key] = torch.ones(batch_size, num_chunks, dtype=torch.bool, device=device)
            else:
                valid_by_key[key] = self._mask(masks[key], batch_size, num_chunks, device)

        zero_tokens = projection_weight.new_zeros(
            batch_size, num_chunks, tokens_per_image, self.compressor_dim
        )
        zero_valid = torch.zeros(batch_size, num_chunks, dtype=torch.bool, device=device)
        slot_tokens = []
        slot_valid = []
        for key in self.image_keys:
            slot_tokens.append(encoded_by_key.get(key, zero_tokens))
            camera_valid = valid_by_key.get(key, zero_valid)
            slot_valid.append(camera_valid.unsqueeze(-1).expand(-1, -1, tokens_per_image))
        return torch.stack(slot_tokens, dim=2), torch.stack(slot_valid, dim=2)


class BPVAv31ObsEncoder(BPObsEncoder):
    """相对 v3：可用归一化时间的正弦进度编码，替代可学习 chunk_position_embedding。"""

    def __init__(
        self,
        chunk_encoder: BPTransformerObsEncoder,
        max_num_chunks: int = 4,
        *,
        use_progress_encoding: bool = True,
    ) -> None:
        super().__init__(
            chunk_encoder=chunk_encoder,
            max_num_chunks=max_num_chunks,
            use_chunk_position_embedding=not use_progress_encoding,
        )
        self.use_progress_encoding = bool(use_progress_encoding)
        self.last_content: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(
        self,
        sample_or_behavior_prompt: dict[str, Any],
        visual: nn.Module,
        return_dict: bool = False,
    ) -> torch.Tensor | BPObsEncoderOutput:
        prompt = sample_or_behavior_prompt.get("behavior_prompt", sample_or_behavior_prompt)
        encoded = self.chunk_encoder(prompt, visual=visual, return_dict=True)
        tokens, chunk_mask = encoded.chunk_tokens, encoded.mask
        added_batch = tokens.ndim == 3
        if added_batch:
            tokens = tokens.unsqueeze(0)
            chunk_mask = None if chunk_mask is None else chunk_mask.unsqueeze(0)
        batch_size, num_chunks, num_queries, output_dim = tokens.shape
        if num_chunks > self.max_num_chunks:
            raise ValueError(f"BP chunk count {num_chunks} exceeds configured maximum {self.max_num_chunks}")

        indices = prompt.get("chunk_indices")
        if indices is None:
            indices = torch.arange(num_chunks, device=tokens.device).view(1, -1).expand(batch_size, -1)
        else:
            indices = torch.as_tensor(indices, device=tokens.device, dtype=torch.long)
            if indices.ndim == 1:
                indices = indices.unsqueeze(0)
        if indices.shape != (batch_size, num_chunks):
            raise ValueError(f"Unexpected chunk_indices shape {tuple(indices.shape)}")
        if torch.any(indices < 0) or torch.any(indices >= self.max_num_chunks):
            raise ValueError("chunk_indices out of range")
        if chunk_mask is None:
            chunk_mask = torch.ones(batch_size, num_chunks, dtype=torch.bool, device=tokens.device)
        # 加时间编码之前的纯内容 [B, C, Q, D]，供内容检索 / 时序对比读取（保留梯度）。
        self.last_content = (tokens, chunk_mask)

        if self.use_progress_encoding:
            ratios = prompt.get("source_time_ratio")
            if ratios is None:
                ratios = indices.float() / max(1, self.max_num_chunks - 1)
            else:
                ratios = torch.as_tensor(ratios, device=tokens.device, dtype=torch.float32)
                if ratios.ndim == 1:
                    ratios = ratios.unsqueeze(0)
            if ratios.shape != (batch_size, num_chunks):
                raise ValueError(f"Unexpected source_time_ratio shape {tuple(ratios.shape)}")
            position = sinusoidal_progress_encoding(ratios, output_dim).to(dtype=tokens.dtype).unsqueeze(2)
            tokens = tokens + position
        elif self.chunk_position_embedding is not None:
            position = self.chunk_position_embedding(indices).to(tokens.dtype).unsqueeze(2)
            tokens = tokens + position

        tokens = tokens.masked_fill(~chunk_mask[..., None, None], 0)
        flat_tokens = tokens.reshape(batch_size, num_chunks * num_queries, output_dim)
        flat_mask = chunk_mask.unsqueeze(-1).expand(-1, -1, num_queries).reshape(batch_size, -1)
        flat_indices = indices.unsqueeze(-1).expand(-1, -1, num_queries).reshape(batch_size, -1)

        if added_batch:
            flat_tokens = flat_tokens.squeeze(0)
            flat_mask = flat_mask.squeeze(0)
            flat_indices = flat_indices.squeeze(0)
            tokens = tokens.squeeze(0)
        output = BPObsEncoderOutput(flat_tokens, flat_mask, flat_indices, tokens)
        return output if return_dict else flat_tokens


def build_bp_obs_encoder(config, *, visual_output_dim: int, output_dim: int) -> BPVAv31ObsEncoder:
    chunk_encoder = BPVAv31ChunkEncoder(
        image_keys=list(config.bp_camera_keys),
        visual_output_dim=visual_output_dim,
        output_dim=output_dim,
        compressor_dim=config.bp_compressor_dim,
        state_action_hidden_dim=config.bp_state_action_hidden_dim,
        num_query_tokens=config.bp_num_query_tokens,
        num_layers=config.bp_compressor_num_layers,
        num_heads=config.bp_compressor_num_heads,
        ff_mult=config.bp_compressor_ff_mult,
        state_dim=config.max_state_dim,
        action_dim=config.max_action_dim,
        action_chunk_size=config.bp_action_chunk_size,
    )
    return BPVAv31ObsEncoder(
        chunk_encoder=chunk_encoder,
        max_num_chunks=config.bp_num_chunks,
        use_progress_encoding=bool(getattr(config, "bp_use_progress_encoding", True)),
    )
