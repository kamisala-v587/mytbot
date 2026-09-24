from __future__ import annotations

import torch

from lerobot.policies.BPVAv2.bp_transformer_obs_encoder import (
    BPObsEncoder,
    BPTransformerObsEncoder,
)


class BPVAv3ChunkEncoder(BPTransformerObsEncoder):
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
            raise RuntimeError("BPVAv3 visual tokens must be produced by the shared Qwen3.5 ViT")

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


def build_bp_obs_encoder(config, *, visual_output_dim: int, output_dim: int) -> BPObsEncoder:
    chunk_encoder = BPVAv3ChunkEncoder(
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
    return BPObsEncoder(
        chunk_encoder=chunk_encoder,
        max_num_chunks=config.bp_num_chunks,
        use_chunk_position_embedding=True,
    )
