from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_BP_CAMERA_KEYS = [f"{OBS_IMAGES}.image0"]


@dataclass
class BPTransformerObsEncoderOutput:
    chunk_tokens: torch.Tensor
    modality_tokens: torch.Tensor | None = None
    mask: torch.Tensor | None = None


@dataclass
class BPObsEncoderOutput:
    chunk_tokens: torch.Tensor
    mask: torch.Tensor | None = None
    chunk_indices: torch.Tensor | None = None
    modality_tokens: torch.Tensor | None = None


class CrossSelfBlock(nn.Module):
    """Pre-norm cross-attention, query self-attention, and FFN block."""

    def __init__(self, dim: int, num_heads: int, ff_mult: int) -> None:
        super().__init__()
        self.cross_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_valid: torch.Tensor,
    ) -> torch.Tensor:
        cross, _ = self.cross_attn(
            self.cross_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=~memory_valid,
            need_weights=False,
        )
        queries = queries + cross
        self_attended, _ = self.self_attn(
            self.self_norm(queries),
            self.self_norm(queries),
            self.self_norm(queries),
            need_weights=False,
        )
        queries = queries + self_attended
        return queries + self.ff(self.ff_norm(queries))


class BPTransformerObsEncoder(nn.Module):
    """Compress each fixed-camera BP chunk into a small set of learned query tokens."""

    def __init__(
        self,
        image_keys: list[str] | None = None,
        visual_output_dim: int = 2048,
        output_dim: int = 2048,
        compressor_dim: int = 512,
        state_action_hidden_dim: int = 256,
        num_query_tokens: int = 5,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_mult: int = 4,
        freeze_shared_visual: bool = True,
        use_type_embedding: bool = True,
        use_camera_embedding: bool = True,
        state_dim: int = 32,
        action_dim: int = 32,
        action_chunk_size: int = 50,
    ) -> None:
        super().__init__()
        self.image_keys = list(image_keys or DEFAULT_BP_CAMERA_KEYS)
        if not self.image_keys or len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError("BPTransformerObsEncoder requires one or more unique policy image slots")
        for name, value in (
            ("visual_output_dim", visual_output_dim),
            ("output_dim", output_dim),
            ("compressor_dim", compressor_dim),
            ("state_action_hidden_dim", state_action_hidden_dim),
            ("num_query_tokens", num_query_tokens),
            ("num_layers", num_layers),
            ("num_heads", num_heads),
            ("ff_mult", ff_mult),
            ("state_dim", state_dim),
            ("action_dim", action_dim),
            ("action_chunk_size", action_chunk_size),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if compressor_dim % num_heads:
            raise ValueError("compressor_dim must be divisible by num_heads")

        self.visual_output_dim = int(visual_output_dim)
        self.output_dim = int(output_dim)
        self.compressor_dim = int(compressor_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.action_chunk_size = int(action_chunk_size)
        self.freeze_shared_visual = bool(freeze_shared_visual)

        # Visual projection
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(self.visual_output_dim),
            nn.Linear(self.visual_output_dim, self.compressor_dim),
        )
        # State MLP
        self.state_mlp = nn.Sequential(
            nn.Linear(self.state_dim, state_action_hidden_dim),
            nn.SiLU(),
            nn.Linear(state_action_hidden_dim, self.compressor_dim),
            nn.LayerNorm(self.compressor_dim),
        )
        # Action MLP
        self.action_mlp = nn.Sequential(
            nn.Linear(self.action_dim, state_action_hidden_dim),
            nn.SiLU(),
            nn.Linear(state_action_hidden_dim, self.compressor_dim),
            nn.LayerNorm(self.compressor_dim),
        )
        # Action position embedding
        self.action_position_embedding = nn.Embedding(self.action_chunk_size, self.compressor_dim)
        # Type and camera embeddings
        self.type_embedding = nn.Embedding(3, self.compressor_dim) if use_type_embedding else None
        self.camera_embedding = nn.Embedding(len(self.image_keys), self.compressor_dim) if use_camera_embedding else None
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.empty(self.num_query_tokens, self.compressor_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        # Compressor layers 每层包括 cross-attention, self-attention, and FFN
        # 1. Cross-Attention：5 个 query 去读 BP memory，提取和当前压缩目标最相关的视觉/state/action信息。
        # 2. Query Self-Attention：5 个 query 彼此交流，让不同 query 学会分工。
        # 3. FFN：做非线性变换和残差更新。
        self.compressor_layers = nn.ModuleList(
            [CrossSelfBlock(self.compressor_dim, num_heads, ff_mult) for _ in range(num_layers)]
        )
        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.compressor_dim),
            nn.Linear(self.compressor_dim, self.output_dim),
        )

    def forward(
        self,
        behavior_prompt: dict[str, Any],
        visual: nn.Module,
        return_dict: bool = False,
    ) -> torch.Tensor | BPTransformerObsEncoderOutput:
        state = self._get_field(behavior_prompt, "state", OBS_STATE)
        action = self._get_field(behavior_prompt, "action", ACTION)
        state, added_batch = self._ensure_batch(state, 2, 3, "state")
        action, _ = self._ensure_batch(action, 3, 4, "action")
        batch_size, num_chunks = state.shape[:2]
        if state.shape != (batch_size, num_chunks, self.state_dim):
            raise ValueError(f"Unexpected state shape {tuple(state.shape)}")
        if action.shape != (batch_size, num_chunks, self.action_chunk_size, self.action_dim):
            raise ValueError(f"Unexpected action shape {tuple(action.shape)}")

        chunk_mask = behavior_prompt.get("mask")
        if chunk_mask is None:
            chunk_mask = torch.ones(batch_size, num_chunks, dtype=torch.bool, device=state.device)
        else:
            chunk_mask = self._mask(chunk_mask, batch_size, num_chunks, state.device)

        visual_tokens, visual_valid = self._encode_images(
            behavior_prompt, visual, batch_size, num_chunks, state.device
        )
        visual_tokens = self._add_visual_embeddings(visual_tokens)
        visual_tokens = visual_tokens.masked_fill(~visual_valid.unsqueeze(-1), 0)
        visual_tokens = visual_tokens.flatten(start_dim=2, end_dim=3)
        visual_valid = visual_valid.flatten(start_dim=2, end_dim=3)

        state_valid_value = behavior_prompt.get("state_is_available")
        state_valid = (
            torch.ones(batch_size, num_chunks, dtype=torch.bool, device=state.device)
            if state_valid_value is None
            else self._mask(state_valid_value, batch_size, num_chunks, state.device)
        )
        state_weight = self.state_mlp[0].weight
        state_input = state.to(device=state_weight.device, dtype=state_weight.dtype)
        state_tokens = self.state_mlp(state_input).unsqueeze(2)
        state_tokens = self._add_type_embedding(state_tokens, 1)
        state_tokens = state_tokens.masked_fill(~state_valid[..., None, None], 0)

        action_is_pad = behavior_prompt.get("action_is_pad")
        if action_is_pad is None:
            action_valid = torch.ones(
                batch_size, num_chunks, self.action_chunk_size, dtype=torch.bool, device=action.device
            )
        else:
            action_is_pad, _ = self._ensure_batch(
                torch.as_tensor(action_is_pad), 2, 3, "action_is_pad"
            )
            action_valid = ~action_is_pad.to(device=action.device, dtype=torch.bool)
            if action_valid.shape != (batch_size, num_chunks, self.action_chunk_size):
                raise ValueError(f"Unexpected action_is_pad shape {tuple(action_is_pad.shape)}")
        action_weight = self.action_mlp[0].weight
        action_input = action.to(device=action_weight.device, dtype=action_weight.dtype)
        action_tokens = self.action_mlp(action_input)
        action_positions = self.action_position_embedding(
            torch.arange(self.action_chunk_size, device=action.device)
        ).to(action_tokens.dtype)
        action_tokens = action_tokens + action_positions.view(1, 1, self.action_chunk_size, self.compressor_dim)
        action_tokens = self._add_type_embedding(action_tokens, 2)
        action_tokens = action_tokens.masked_fill(~action_valid.unsqueeze(-1), 0)

        memory = torch.cat([visual_tokens, state_tokens, action_tokens], dim=2)
        memory_valid = torch.cat([visual_valid, state_valid.unsqueeze(-1), action_valid], dim=2)
        all_invalid = ~memory_valid.any(dim=-1)
        invalid_active = all_invalid & chunk_mask
        if invalid_active.any():
            positions = invalid_active.nonzero(as_tuple=False).tolist()
            raise ValueError(f"BPVAv2 active chunk(s) have no valid memory tokens: {positions}")

        # Padded chunks are zeroed later, but MHA still needs one unmasked key to avoid NaNs.
        if all_invalid.any():
            memory_valid = memory_valid.clone()
            memory = memory.clone()
            memory_valid[..., 0] |= all_invalid
            memory[..., 0, :] = memory[..., 0, :].masked_fill(all_invalid.unsqueeze(-1), 0)

        flat_memory = memory.reshape(batch_size * num_chunks, memory.shape[2], self.compressor_dim)
        flat_valid = memory_valid.reshape(batch_size * num_chunks, memory_valid.shape[2])
        queries = self.query_tokens.view(1, self.num_query_tokens, self.compressor_dim).expand(
            batch_size * num_chunks, -1, -1
        )
        for layer in self.compressor_layers:
            queries = layer(queries, flat_memory, flat_valid)
        chunk_tokens = self.output_projection(queries).reshape(
            batch_size, num_chunks, self.num_query_tokens, self.output_dim
        )
        chunk_tokens = chunk_tokens.masked_fill(~chunk_mask[..., None, None], 0)

        if added_batch:
            chunk_tokens = chunk_tokens.squeeze(0)
            chunk_mask = chunk_mask.squeeze(0)
        output = BPTransformerObsEncoderOutput(chunk_tokens, chunk_tokens, chunk_mask)
        return output if return_dict else chunk_tokens

    def _encode_images(
        self,
        prompt: dict[str, Any],
        visual: nn.Module,
        batch_size: int,
        num_chunks: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = prompt.get("bp_pixel_values")
        grids = prompt.get("bp_image_grid_thw")
        if not isinstance(pixels, dict) or not isinstance(grids, dict):
            raise KeyError("BPVAv2 behavior_prompt requires bp_pixel_values and bp_image_grid_thw dictionaries")
        if set(pixels) != set(grids):
            raise KeyError("bp_pixel_values and bp_image_grid_thw must contain the same active camera keys")
        active_keys = list(pixels)
        if not active_keys:
            raise ValueError("BPVAv2 requires at least one active processed camera")
        unknown = set(active_keys) - set(self.image_keys)
        if unknown:
            raise KeyError(f"Active BP camera keys are not declared policy slots: {sorted(unknown)}")

        masks = prompt.get("image_masks")
        if masks is not None:
            if not isinstance(masks, dict):
                raise TypeError("image_masks must be a dictionary")
            unknown_masks = set(masks) - set(self.image_keys)
            if unknown_masks:
                raise KeyError(f"Unsupported image_masks keys: {sorted(unknown_masks)}")

        encoded_by_key: dict[str, torch.Tensor] = {}
        valid_by_key: dict[str, torch.Tensor] = {}
        expected_tokens_per_image: int | None = None
        context = torch.no_grad() if self.freeze_shared_visual else nullcontext()
        with context:
            for key in active_keys:
                grid = torch.as_tensor(grids[key], device=device)
                if grid.ndim == 2:
                    grid = grid.unsqueeze(0)
                if grid.shape != (batch_size, num_chunks, 3):
                    raise ValueError(f"Unexpected BP grid shape for {key}: {tuple(grid.shape)}")
                pixel = pixels[key]
                if pixel.ndim == 3:
                    pixel = pixel.unsqueeze(0)
                if pixel.ndim != 4 or pixel.shape[:2] != (batch_size, num_chunks):
                    raise ValueError(f"Unexpected BP pixel shape for {key}: {tuple(pixel.shape)}")
                flat_grid = grid.reshape(-1, 3)
                flat_pixel = pixel.reshape(-1, pixel.shape[-1])
                visual_output = visual(flat_pixel, flat_grid)
                encoded = visual_output[0] if isinstance(visual_output, tuple) else visual_output
                if encoded.ndim != 2 or encoded.shape[-1] != self.visual_output_dim:
                    raise ValueError(
                        f"Qwen visual output for {key} must be [tokens,{self.visual_output_dim}], "
                        f"got {tuple(encoded.shape)}"
                    )
                merge = int(getattr(visual, "spatial_merge_size", 0))
                if merge <= 0:
                    raise ValueError("Qwen visual spatial_merge_size must be positive")
                numerators = flat_grid.prod(dim=-1)
                if torch.any(numerators % (merge**2) != 0):
                    raise ValueError(f"BP image grid for {key} is not divisible by spatial merge area")
                lengths = [int(value) for value in (numerators // (merge**2)).tolist()]
                if not lengths or any(length <= 0 for length in lengths):
                    raise ValueError(f"BP image grid for {key} produced invalid visual token lengths")
                if sum(lengths) != encoded.shape[0]:
                    raise ValueError(f"Qwen visual token count does not match image grids for {key}")
                if len(set(lengths)) != 1:
                    raise ValueError(f"BPVAv2 requires equal Qwen output token counts within camera {key!r}")
                tokens_per_image = lengths[0]
                if expected_tokens_per_image is None:
                    expected_tokens_per_image = tokens_per_image
                elif tokens_per_image != expected_tokens_per_image:
                    raise ValueError(
                        "All active BP images in one forward must have the same Qwen output token count; "
                        f"expected {expected_tokens_per_image}, got {tokens_per_image} for {key!r}"
                    )
                projection_weight = self.visual_projection[0].weight
                projection_input = encoded.to(
                    device=projection_weight.device,
                    dtype=projection_weight.dtype,
                )
                projected = self.visual_projection(projection_input)
                encoded_by_key[key] = projected.reshape(
                    batch_size, num_chunks, tokens_per_image, self.compressor_dim
                )
                valid_by_key[key] = (
                    torch.ones(batch_size, num_chunks, dtype=torch.bool, device=device)
                    if masks is None or key not in masks
                    else self._mask(masks[key], batch_size, num_chunks, device)
                )

        assert expected_tokens_per_image is not None
        zero_tokens = next(iter(encoded_by_key.values())).new_zeros(
            batch_size, num_chunks, expected_tokens_per_image, self.compressor_dim
        )
        zero_valid = torch.zeros(batch_size, num_chunks, dtype=torch.bool, device=device)
        slot_tokens = []
        slot_valid = []
        for key in self.image_keys:
            slot_tokens.append(encoded_by_key.get(key, zero_tokens))
            camera_valid = valid_by_key.get(key, zero_valid)
            slot_valid.append(camera_valid.unsqueeze(-1).expand(-1, -1, expected_tokens_per_image))
        return torch.stack(slot_tokens, dim=2), torch.stack(slot_valid, dim=2)

    def _add_visual_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.type_embedding is not None:
            tokens = tokens + self.type_embedding.weight[0].to(tokens.dtype)
        if self.camera_embedding is not None:
            camera_ids = torch.arange(len(self.image_keys), device=tokens.device)
            camera_embeddings = self.camera_embedding(camera_ids).to(tokens.dtype)
            tokens = tokens + camera_embeddings.view(1, 1, len(self.image_keys), 1, self.compressor_dim)
        return tokens

    def _add_type_embedding(self, tokens: torch.Tensor, type_index: int) -> torch.Tensor:
        if self.type_embedding is not None:
            tokens = tokens + self.type_embedding.weight[type_index].to(tokens.dtype)
        return tokens

    @staticmethod
    def _mask(value: Any, batch_size: int, num_chunks: int, device: torch.device) -> torch.Tensor:
        mask = torch.as_tensor(value, dtype=torch.bool, device=device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape == (1, num_chunks) and batch_size > 1:
            mask = mask.expand(batch_size, -1)
        if mask.shape != (batch_size, num_chunks):
            raise ValueError(f"Unexpected mask shape {tuple(mask.shape)}")
        return mask

    @staticmethod
    def _ensure_batch(
        value: torch.Tensor, no_batch_ndim: int, batch_ndim: int, name: str
    ) -> tuple[torch.Tensor, bool]:
        if value.ndim == no_batch_ndim:
            return value.unsqueeze(0), True
        if value.ndim == batch_ndim:
            return value, False
        raise ValueError(f"{name} has unexpected shape {tuple(value.shape)}")

    @staticmethod
    def _get_field(data: dict[str, Any], short_key: str, canonical_key: str) -> torch.Tensor:
        if short_key in data:
            return data[short_key]
        if canonical_key in data:
            return data[canonical_key]
        raise KeyError(f"Missing {short_key!r} in behavior_prompt")


class BPObsEncoder(nn.Module):
    def __init__(
        self,
        chunk_encoder: BPTransformerObsEncoder,
        max_num_chunks: int = 4,
        use_chunk_position_embedding: bool = True,
    ) -> None:
        super().__init__()
        if max_num_chunks <= 0:
            raise ValueError("max_num_chunks must be positive")
        self.chunk_encoder = chunk_encoder
        self.max_num_chunks = int(max_num_chunks)
        self.chunk_dim = chunk_encoder.output_dim
        self.num_query_tokens = chunk_encoder.num_query_tokens
        self.chunk_position_embedding = (
            nn.Embedding(self.max_num_chunks, self.chunk_dim) if use_chunk_position_embedding else None
        )

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

        if self.chunk_position_embedding is not None:
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
