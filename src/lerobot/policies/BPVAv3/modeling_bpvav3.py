from __future__ import annotations

from collections import deque
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch._dynamo import disable as dynamo_disable

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.q35.modeling_q35 import (
    Q35Model,
    debug_check_finite,
    flow_euler_sample,
    masked_flow_loss,
)
from lerobot.policies.q35.qwen3_5_backbone import kernelize_qwen35_training_backbone
from lerobot.policies.q35.token_layout import Q35TokenLayout
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION, OBS_PREFIX, OBS_STATE, SAMPLE_ACTION_LOSS_MASK

from .bp_encoder import build_bp_obs_encoder
from .configuration_bpvav3 import BPVAv3Config


def block_align_query_attention(mask: torch.Tensor, n_prefix: int, n_align: int) -> torch.Tensor:
    """Align query 能看见 prefix，prefix 和 state/action 都不能看见这些 query。

    线性注意力层本身是因果的，query 放在 VLM 序列末尾时不会把未来写回 prefix。
    全注意力是双向的，所以这里要把对应列清掉。
    """
    if n_align <= 0:
        return mask
    blocked = mask.clone()
    align = slice(n_prefix, n_prefix + n_align)
    blocked[:, :, :n_prefix, align] = False
    blocked[:, :, n_prefix + n_align :, align] = False
    return blocked


def bp_align_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """逐 token 的 1 - cos。先对每个样本的有效 token 求平均，再对有有效 token 的样本求平均。"""
    if pred.shape != target.shape:
        raise ValueError(
            f"Align prediction shape {tuple(pred.shape)} does not match target {tuple(target.shape)}"
        )
    token_mask = token_mask.to(device=pred.device, dtype=torch.bool)
    if token_mask.shape != pred.shape[:2]:
        raise ValueError(
            f"Align mask shape {tuple(token_mask.shape)} does not match tokens {tuple(pred.shape[:2])}"
        )
    unit = pred.new_zeros(pred.shape, dtype=torch.float32)
    unit[..., 0] = 1
    valid = token_mask[..., None]
    pred_n = F.normalize(torch.where(valid, pred.float(), unit), dim=-1)
    target_n = F.normalize(torch.where(valid, target.float(), unit), dim=-1)
    per_token = (1.0 - (pred_n * target_n).sum(dim=-1)) * token_mask.to(dtype=torch.float32)
    counts = token_mask.to(dtype=torch.float32).sum(dim=-1)
    sample_loss = per_token.sum(dim=-1) / counts.clamp_min(1)
    valid_samples = counts > 0
    if not bool(valid_samples.any()):
        return pred.float().sum() * 0.0, counts.sum()
    return sample_loss[valid_samples].mean(), counts.sum()


def bp_align_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw-feature MSE, averaged over features, valid tokens, then valid samples."""
    if pred.shape != target.shape:
        raise ValueError(f"Align prediction shape {tuple(pred.shape)} does not match target {tuple(target.shape)}")
    token_mask = token_mask.to(device=pred.device, dtype=torch.bool)
    if token_mask.shape != pred.shape[:2]:
        raise ValueError(f"Align mask shape {tuple(token_mask.shape)} does not match tokens {tuple(pred.shape[:2])}")
    valid = token_mask[..., None]
    pred_valid = torch.where(valid, pred.float(), 0.0)
    target_valid = torch.where(valid, target.float(), 0.0)
    per_token = (pred_valid - target_valid).square().mean(dim=-1)
    counts = token_mask.float().sum(dim=-1)
    sample_loss = per_token.sum(dim=-1) / counts.clamp_min(1)
    valid_samples = counts > 0
    if not bool(valid_samples.any()):
        return pred_valid.sum() * 0.0, counts.sum()
    return sample_loss[valid_samples].mean(), counts.sum()


def bp_task_supcon_loss(z: torch.Tensor, labels: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervised contrastive loss; anchors without a same-task partner in the batch are skipped."""
    z = F.normalize(z.float(), dim=-1)
    labels = labels.reshape(-1).to(z.device)
    self_mask = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    logits = (z @ z.T / tau).masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive = (labels[:, None] == labels[None, :]) & ~self_mask
    num_positive = positive.sum(dim=1)
    valid = num_positive > 0
    if not bool(valid.any()):
        return z.sum() * 0.0, valid.sum()
    per_anchor = log_prob.masked_fill(~positive, 0.0).sum(dim=1)[valid] / num_positive[valid]
    return -per_anchor.mean(), valid.sum()


def bp_align_loss(pred, target, token_mask, loss_type="cosine"):
    if loss_type == "cosine":
        return bp_align_cosine_loss(pred, target, token_mask)
    if loss_type == "mse":
        return bp_align_mse_loss(pred, target, token_mask)
    raise ValueError(f"Unknown BP alignment loss type: {loss_type!r}")


class BPVAv3Model(Q35Model):
    """Q35 前向，BP 图像和当前观测共用一次 Qwen3.5 ViT。"""

    def __init__(self, config: BPVAv3Config) -> None:
        super().__init__(config)
        text_hidden = int(self.core.backbone.vlm.config.text_config.hidden_size)
        visual_out = int(self.core.backbone.vlm.config.vision_config.out_hidden_size)
        self.bp_obs_encoder = build_bp_obs_encoder(
            config,
            visual_output_dim=visual_out,
            output_dim=text_hidden,
        )
        n_align = int(config.bp_num_chunks) * int(config.bp_num_query_tokens)
        self.align_queries = torch.nn.Parameter(torch.empty(n_align, text_hidden))
        torch.nn.init.normal_(self.align_queries, std=0.02)
        if float(config.lambda_bp_encoder) == 0.0:
            self.align_queries.requires_grad_(False)
        if float(getattr(config, "lambda_bp_task", 0.0)) != 0.0:
            self.bp_task_head = torch.nn.Sequential(
                torch.nn.Linear(text_hidden, text_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(text_hidden, int(config.bp_task_proj_dim)),
            )
        if config.frozen_bp_encoder:
            self.freeze_bp_encoder()
        if config.frozen_vit:
            self.freeze_vit()

    def freeze_bp_encoder(self) -> None:
        """冻结 BP compressor。policy.train() 之后仍保持 eval，避免被重新打开。"""
        self.bp_obs_encoder.eval()
        for parameter in self.bp_obs_encoder.parameters():
            parameter.requires_grad_(False)

    def freeze_vit(self) -> None:
        """冻结共享 Qwen3.5 ViT。当前观测和 BP 图像用的是同一份参数。"""
        visual = self.core.backbone.vlm_visual
        visual.eval()
        for parameter in visual.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.frozen_bp_encoder:
            self.bp_obs_encoder.eval()
        if self.config.frozen_vit:
            self.core.backbone.vlm_visual.eval()
        return self

    @dynamo_disable
    def _visual_features(self, pixels: torch.Tensor, grids: torch.Tensor) -> list[torch.Tensor]:
        vlm = self.core.backbone.vlm
        vision_dtype = next(self.core.backbone.vlm_visual.parameters()).dtype
        flat_pixels = pixels.reshape(-1, pixels.shape[-1]).to(dtype=vision_dtype)
        flat_grids = grids.reshape(-1, 3).to(device=flat_pixels.device)
        context = torch.no_grad() if self.config.frozen_vit else nullcontext()
        with context:
            outputs = vlm.model.get_image_features(flat_pixels, flat_grids)
        chunks = outputs.pooler_output
        if not isinstance(chunks, (tuple, list)):
            merge = int(self.core.backbone.vlm_visual.spatial_merge_size)
            lengths = [int(value) for value in (flat_grids.prod(-1) // merge**2).tolist()]
            chunks = list(torch.split(chunks, lengths))
        chunks = list(chunks)
        if len(chunks) != flat_grids.shape[0]:
            raise ValueError(
                f"Qwen3.5 ViT returned {len(chunks)} images, expected {flat_grids.shape[0]}"
            )
        return chunks

    def _pack_bp_images(
        self,
        behavior_prompt: dict,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list[tuple[str, int, int]], list[str]]:
        pixels = behavior_prompt.get("bp_pixel_values")
        grids = behavior_prompt.get("bp_image_grid_thw")
        if not isinstance(pixels, dict) or not isinstance(grids, dict):
            raise KeyError("behavior_prompt requires bp_pixel_values and bp_image_grid_thw")
        masks = behavior_prompt.get("image_masks") or {}
        flat_pixels = []
        flat_grids = []
        layout: list[tuple[str, int, int]] = []
        skipped: list[str] = []
        for key in self.config.bp_camera_keys:
            if key not in pixels or key not in grids:
                raise KeyError(f"Missing processed BP camera {key}")
            pixel = pixels[key]
            grid = grids[key]
            if pixel.ndim == 3:
                pixel = pixel.unsqueeze(0)
            if grid.ndim == 2:
                grid = grid.unsqueeze(0)
            if pixel.ndim != 4 or grid.shape[:2] != pixel.shape[:2]:
                raise ValueError(
                    f"BP camera {key} pixels {tuple(pixel.shape)} do not match grids {tuple(grid.shape)}"
                )
            mask = masks.get(key)
            if mask is not None and not bool(torch.as_tensor(mask).any()):
                skipped.append(key)
                continue
            flat_pixels.append(pixel.reshape(-1, pixel.shape[-1]).to(device=device))
            flat_grids.append(grid.reshape(-1, 3).to(device=device))
            layout.append((key, int(pixel.shape[0]), int(pixel.shape[1])))
        if not flat_pixels:
            return None, None, [], skipped
        return torch.cat(flat_pixels, dim=0), torch.cat(flat_grids, dim=0), layout, skipped

    def _group_bp_features(
        self,
        features: list[torch.Tensor],
        layout: list[tuple[str, int, int]],
        skipped: list[str],
        behavior_prompt: dict,
        reference: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        grouped: dict[str, torch.Tensor] = {}
        cursor = 0
        tokens_per_image = None
        for key, batch, chunks in layout:
            count = batch * chunks
            group = features[cursor : cursor + count]
            cursor += count
            lengths = {int(item.shape[0]) for item in group}
            if len(lengths) != 1:
                raise ValueError(f"BP camera {key} produced mixed visual token counts: {sorted(lengths)}")
            token_count = int(group[0].shape[0])
            if tokens_per_image is None:
                tokens_per_image = token_count
            elif token_count != tokens_per_image:
                raise ValueError(
                    "All BP images in one forward must share the same Qwen3.5 visual token count: "
                    f"expected {tokens_per_image}, got {token_count} for {key}"
                )
            stacked = torch.stack(group, dim=0).reshape(batch, chunks, tokens_per_image, group[0].shape[-1])
            grouped[key] = stacked
        if cursor != len(features):
            raise ValueError("BP visual features were not fully consumed")
        if tokens_per_image is None:
            tokens_per_image = self._tokens_per_image_from_prompt(behavior_prompt, skipped)
        hidden = int(reference.shape[-1])
        for key in skipped:
            pixel = behavior_prompt["bp_pixel_values"][key]
            batch = 1 if pixel.ndim == 3 else int(pixel.shape[0])
            chunks = int(pixel.shape[0] if pixel.ndim == 3 else pixel.shape[1])
            grouped[key] = reference.new_zeros(batch, chunks, tokens_per_image, hidden)
        self.bp_obs_encoder.chunk_encoder._tokens_per_image = tokens_per_image
        return grouped

    def _tokens_per_image_from_prompt(self, behavior_prompt: dict, skipped: list[str]) -> int:
        merge = int(self.core.backbone.vlm_visual.spatial_merge_size)
        key = skipped[0]
        grid = behavior_prompt["bp_image_grid_thw"][key].reshape(-1, 3)[0]
        count = int(grid.prod().item()) // merge**2
        if count <= 0:
            raise ValueError(f"BP grid for {key} produced no visual tokens")
        return count

    def _scatter_current_prefix(
        self,
        current_features: list[torch.Tensor],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        image_embeddings = torch.cat(current_features, dim=0)
        vlm = self.core.backbone.vlm
        image_token_id = vlm.config.image_token_id
        if self.config.use_language:
            embeddings = vlm.get_input_embeddings()(input_ids)
        else:
            embeddings = image_embeddings.new_zeros(
                *input_ids.shape,
                image_embeddings.shape[-1],
            )
        flat = embeddings.reshape(-1, embeddings.shape[-1])
        flat_ids = input_ids.reshape(-1)
        image_positions = flat_ids == image_token_id
        if int(image_positions.sum()) != image_embeddings.shape[0]:
            raise ValueError(
                "Current image tokens do not match the shared ViT output: "
                f"tokens={int(image_positions.sum())}, features={image_embeddings.shape[0]}"
            )
        flat[image_positions] = image_embeddings.to(flat.dtype)
        return flat.reshape_as(embeddings)

    @dynamo_disable
    def embed_prefix_with_bp(
        self,
        *,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        behavior_prompt: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_pixels = pixel_values.reshape(-1, pixel_values.shape[-1])
        current_grids = image_grid_thw.reshape(-1, 3)
        bp_pixels, bp_grids, layout, skipped = self._pack_bp_images(
            behavior_prompt,
            device=current_pixels.device,
        )
        if bp_pixels is None:
            features = self._visual_features(current_pixels, current_grids)
            current_features = features
            bp_features: list[torch.Tensor] = []
        else:
            features = self._visual_features(
                torch.cat([current_pixels, bp_pixels], dim=0),
                torch.cat([current_grids, bp_grids], dim=0),
            )
            current_features = features[: current_grids.shape[0]]
            bp_features = features[current_grids.shape[0] :]
        prefix = self._scatter_current_prefix(current_features, input_ids)
        reference = current_features[0]
        visual_by_key = self._group_bp_features(
            bp_features,
            layout,
            skipped,
            behavior_prompt,
            reference,
        )
        chunk_encoder = self.bp_obs_encoder.chunk_encoder
        chunk_encoder._visual_tokens = visual_by_key
        try:
            encoded = self.bp_obs_encoder(behavior_prompt, visual=None, return_dict=True)
        finally:
            chunk_encoder._visual_tokens = None
        bp_embs = encoded.chunk_tokens
        if bp_embs.ndim == 2:
            bp_embs = bp_embs.unsqueeze(0)
        bp_embs = bp_embs.to(device=prefix.device, dtype=prefix.dtype)
        bp_mask = encoded.mask
        if bp_mask is None:
            bp_mask = torch.ones(bp_embs.shape[:2], dtype=torch.bool, device=prefix.device)
        elif bp_mask.ndim == 1:
            bp_mask = bp_mask.unsqueeze(0)
        bp_mask = bp_mask.to(device=prefix.device, dtype=torch.bool)
        if bp_mask.shape != bp_embs.shape[:2]:
            raise ValueError(
                f"BP mask shape {tuple(bp_mask.shape)} does not match BP embeddings {tuple(bp_embs.shape[:2])}"
            )
        prefix_mask = attention_mask.to(device=prefix.device, dtype=torch.bool)
        if prefix_mask.ndim == 1:
            prefix_mask = prefix_mask.unsqueeze(0)
        hidden = torch.cat([prefix, bp_embs], dim=1)
        valid_mask = torch.cat([prefix_mask, bp_mask], dim=1)
        return hidden, valid_mask

    @dynamo_disable
    def build_position_ids(
        self,
        input_ids,
        mm_token_type_ids,
        image_grid_thw,
        attention_mask,
        extra_tokens: int,
        bp_tokens: int,
        align_tokens: int = 0,
    ):
        prefix_ids = super().build_position_ids(
            input_ids,
            mm_token_type_ids,
            image_grid_thw,
            attention_mask,
            extra_tokens=0,
        )
        if bp_tokens < 0 or align_tokens < 0:
            raise ValueError("bp_tokens and align_tokens must be non-negative")
        pieces = [prefix_ids]
        start = prefix_ids.max(dim=-1).values + 1
        if bp_tokens:
            bp_positions = start[:, :, None] + torch.arange(bp_tokens, device=input_ids.device)
            pieces.append(bp_positions)
            start = bp_positions.max(dim=-1).values + 1
        if align_tokens:
            align_positions = start[:, :, None] + torch.arange(align_tokens, device=input_ids.device)
            pieces.append(align_positions)
            start = align_positions.max(dim=-1).values + 1
        if extra_tokens:
            extra = start[:, :, None] + torch.arange(extra_tokens, device=input_ids.device)
            pieces.append(extra)
        return torch.cat(pieces, dim=-1)

    def encode_bp_prompt(self, prompt: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """单独跑一条 BP 轨迹的 ViT + compressor。future target 由调用方包 no_grad。"""
        pixels = prompt.get("bp_pixel_values")
        if not isinstance(pixels, dict) or not pixels:
            raise KeyError("BP prompt requires bp_pixel_values")
        device = next(iter(pixels.values())).device
        bp_pixels, bp_grids, layout, skipped = self._pack_bp_images(prompt, device=device)
        if bp_pixels is None:
            raise ValueError("BP prompt has no active camera to encode")
        features = self._visual_features(bp_pixels, bp_grids)
        visual_by_key = self._group_bp_features(features, layout, skipped, prompt, features[0])
        chunk_encoder = self.bp_obs_encoder.chunk_encoder
        chunk_encoder._visual_tokens = visual_by_key
        try:
            encoded = self.bp_obs_encoder(prompt, visual=None, return_dict=True)
        finally:
            chunk_encoder._visual_tokens = None
        tokens = encoded.chunk_tokens
        token_mask = encoded.mask
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        if token_mask is None:
            token_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        elif token_mask.ndim == 1:
            token_mask = token_mask.unsqueeze(0)
        return tokens, token_mask.to(dtype=torch.bool)

    def _append_align_queries(
        self,
        vlm_hidden: torch.Tensor,
        vlm_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries = self.align_queries.to(device=vlm_hidden.device, dtype=vlm_hidden.dtype)
        queries = queries.unsqueeze(0).expand(vlm_hidden.shape[0], -1, -1)
        hidden = torch.cat([vlm_hidden, queries], dim=1)
        query_valid = torch.ones(
            vlm_hidden.shape[0],
            queries.shape[1],
            dtype=torch.bool,
            device=vlm_hidden.device,
        )
        return hidden, torch.cat([vlm_valid_mask, query_valid], dim=1)

    def _encode_future_target(self, future_prompt: dict) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                tokens, token_mask = self.encode_bp_prompt(future_prompt)
        finally:
            self.train(was_training)
        return tokens.detach(), token_mask.detach()

    def _forward_flow_and_align(
        self,
        vlm_hidden: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        vlm_valid_mask: torch.Tensor,
        n_align: int,
        target_tokens: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        core = self.core
        batch = actions.shape[0]
        noise = torch.randn_like(actions)
        beta = torch.distributions.Beta(
            torch.tensor(core.config.time_sampling_beta_alpha, device=actions.device),
            torch.tensor(core.config.time_sampling_beta_beta, device=actions.device),
        )
        time = beta.sample((batch,)) * core.config.time_sampling_scale + core.config.time_sampling_offset
        time_expanded = time[:, None, None]
        noisy_action = time_expanded * noise + (1 - time_expanded) * actions
        flow_target = noise - actions
        state_hidden = core.state_projection(state.to(core.state_projection.weight.dtype)).to(vlm_hidden.dtype)[
            :, None
        ]
        action_hidden = core._action_inputs(noisy_action, time).to(vlm_hidden.dtype)
        layout = Q35TokenLayout.for_step(
            "robot",
            expert_type="action",
            vlm=vlm_hidden.shape[1],
            state=1,
            action=action_hidden.shape[1],
        )
        valid_mask = torch.cat(
            (
                vlm_valid_mask.bool(),
                torch.ones(
                    batch,
                    1 + action_hidden.shape[1],
                    dtype=torch.bool,
                    device=vlm_hidden.device,
                ),
            ),
            dim=1,
        )
        attention_mask = layout.build_attention_mask(
            valid_mask=valid_mask,
            causal_vlm=getattr(core.config, "causal_vlm", False),
        )
        n_prefix = vlm_hidden.shape[1] - n_align
        attention_mask = block_align_query_attention(attention_mask, n_prefix, n_align)
        vlm_out, _, action_out = core._forward_joint(
            vlm_hidden,
            state_hidden,
            action_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        predicted_flow = core.action_head(action_out.to(core.action_head.weight.dtype)).float()
        align_hidden = vlm_out[:, -n_align:]
        align_loss, align_token_count = bp_align_loss(
            align_hidden,
            target_tokens.to(device=align_hidden.device),
            target_mask.to(device=align_hidden.device),
            loss_type=self.config.bp_align_loss_type,
        )
        queries_per_chunk = int(self.config.bp_num_query_tokens)
        valid_chunks = target_mask.to(device=align_hidden.device).reshape(
            target_mask.shape[0], -1, queries_per_chunk
        ).any(dim=-1).sum()
        result = {
            "flow": F.mse_loss(predicted_flow, flow_target.float(), reduction="none"),
            "align": align_loss,
            "align_valid_chunks": valid_chunks,
            "align_valid_tokens": align_token_count,
        }
        debug_check_finite("robot.losses", {"flow": result["flow"], "align": align_loss})
        return result

    def forward_robot(
        self,
        *,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        behavior_prompt: dict,
        future_prompt: dict | None = None,
        vlm_hidden: torch.Tensor | None = None,
        vlm_valid_mask: torch.Tensor | None = None,
    ):
        if vlm_hidden is None:
            vlm_hidden, vlm_valid_mask = self.embed_prefix_with_bp(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                input_ids=input_ids,
                attention_mask=attention_mask,
                behavior_prompt=behavior_prompt,
            )
        if vlm_valid_mask is None:
            raise ValueError("vlm_valid_mask is required when vlm_hidden is provided")
        bp_tokens = vlm_hidden.shape[1] - input_ids.shape[1]
        bp_task_embed = None
        if hasattr(self, "bp_task_head") and bp_tokens > 0:
            bp_hidden = vlm_hidden[:, -bp_tokens:].float()
            bp_valid = vlm_valid_mask[:, -bp_tokens:].to(bp_hidden.dtype).unsqueeze(-1)
            pooled = (bp_hidden * bp_valid).sum(dim=1) / bp_valid.sum(dim=1).clamp_min(1.0)
            head_dtype = next(self.bp_task_head.parameters()).dtype
            bp_task_embed = self.bp_task_head(pooled.to(head_dtype))
        n_align = 0
        target_tokens = None
        target_mask = None
        if float(self.config.lambda_bp_encoder) != 0.0:
            if future_prompt is None:
                raise KeyError("lambda_bp_encoder != 0 requires future_prompt")
            target_tokens, target_mask = self._encode_future_target(future_prompt)
            n_align = int(self.align_queries.shape[0])
            if target_tokens.shape[1] != n_align:
                raise ValueError(
                    f"future_prompt encoded {target_tokens.shape[1]} tokens, expected {n_align}"
                )
            vlm_hidden, vlm_valid_mask = self._append_align_queries(vlm_hidden, vlm_valid_mask)
        position_ids = self.build_position_ids(
            input_ids,
            mm_token_type_ids,
            image_grid_thw,
            attention_mask,
            1 + actions.shape[1],
            bp_tokens,
            align_tokens=n_align,
        )
        if n_align == 0:
            result = self.core.forward_embedded(
                vlm_hidden,
                state,
                actions,
                position_ids=position_ids,
                vlm_valid_mask=vlm_valid_mask,
            )
        else:
            result = self._forward_flow_and_align(
                vlm_hidden,
                state,
                actions,
                position_ids=position_ids,
                vlm_valid_mask=vlm_valid_mask,
                n_align=n_align,
                target_tokens=target_tokens,
                target_mask=target_mask,
            )
        if bp_task_embed is not None:
            result["bp_task_embed"] = bp_task_embed
        return result

    @torch.no_grad()
    def sample_actions(
        self,
        *,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state: torch.Tensor,
        behavior_prompt: dict,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        vlm_hidden, vlm_valid_mask = self.embed_prefix_with_bp(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            input_ids=input_ids,
            attention_mask=attention_mask,
            behavior_prompt=behavior_prompt,
        )
        bp_tokens = vlm_hidden.shape[1] - input_ids.shape[1]
        position_ids = self.build_position_ids(
            input_ids,
            mm_token_type_ids,
            image_grid_thw,
            attention_mask,
            1 + self.config.chunk_size,
            bp_tokens,
        )
        prefix_cache, action_attention_mask, action_position_ids = self.core.prepare_inference_cache(
            vlm_hidden,
            state,
            vlm_valid_mask=vlm_valid_mask,
            position_ids=position_ids,
        )
        if noise is None:
            noise = torch.randn(
                state.shape[0],
                self.config.chunk_size,
                self.config.max_action_dim,
                device=state.device,
                dtype=state.dtype,
            )
        resolved_steps = num_steps or self.config.num_inference_steps
        if resolved_steps == self.config.num_inference_steps:
            return self._sample_action_flow(
                noise,
                prefix_cache,
                action_attention_mask,
                action_position_ids,
            )
        return flow_euler_sample(
            noise,
            lambda actions, time: self.core.denoise_cached(
                actions,
                time,
                prefix_cache,
                action_attention_mask,
                action_position_ids,
            ),
            num_steps=resolved_steps,
        )


class BPVAv3Policy(PreTrainedPolicy):
    config_class = BPVAv3Config
    name = "bpvav3"

    def __init__(self, config: BPVAv3Config) -> None:
        super().__init__(config)
        config.validate_features()
        self.model = BPVAv3Model(config)
        actual_fingerprint = self.model.core.backbone.fingerprint.name
        if config.qwen3_5_fingerprint not in {None, actual_fingerprint}:
            raise ValueError(
                "BPVAv3 Qwen3.5 fingerprint mismatch: "
                f"expected {config.qwen3_5_fingerprint}, loaded {actual_fingerprint}"
            )
        config.qwen3_5_fingerprint = actual_fingerprint
        if config.compile_model:
            self.forward = torch.compile(self.forward, mode=config.compile_mode)
        self.reset()

    def to(self, *args, **kwargs):
        policy = super().to(*args, **kwargs)
        if not getattr(policy, "_qwen35_training_kernels_enabled", False):
            policy._qwen35_training_kernels_enabled = kernelize_qwen35_training_backbone(
                policy.model.core.backbone
            )
        return policy

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def classify_model_loading_keys(
        self, missing_keys: list[str], unexpected_keys: list[str]
    ) -> tuple[list[str], list[str], list[str]]:
        """Q35 / vlwa_q35 action-only ckpt 没有 BP encoder，这些 key 随机初始化。"""
        expected_missing = [
            key
            for key in missing_keys
            if key.startswith(("model.bp_obs_encoder.", "model.bp_task_head.")) or key == "model.align_queries"
        ]
        filtered_missing = [key for key in missing_keys if key not in expected_missing]
        return filtered_missing, unexpected_keys, expected_missing

    def _behavior_prompt(self, batch) -> dict:
        if "behavior_prompt" not in batch:
            raise KeyError("BPVAv3 requires batch['behavior_prompt']")
        return batch["behavior_prompt"]

    def forward(self, batch, *, validation_mode: str | None = None):
        if validation_mode not in {None, "flow"}:
            raise ValueError(f"Unsupported robot validation mode: {validation_mode!r}")
        raw_actions = batch[ACTION]
        actions = F.pad(raw_actions, (0, self.config.max_action_dim - raw_actions.shape[-1]))
        state = F.pad(
            batch[OBS_STATE],
            (0, self.config.max_state_dim - batch[OBS_STATE].shape[-1]),
        )
        use_align = float(self.config.lambda_bp_encoder) != 0.0
        future_prompt = batch.get("future_prompt") if use_align else None
        if use_align and future_prompt is None:
            raise KeyError("lambda_bp_encoder != 0 requires batch['future_prompt']")
        losses = self.model.forward_robot(
            pixel_values=batch[f"{OBS_PREFIX}pixel_values"],
            image_grid_thw=batch[f"{OBS_PREFIX}image_grid_thw"],
            input_ids=batch[f"{OBS_PREFIX}input_ids"],
            mm_token_type_ids=batch[f"{OBS_PREFIX}mm_token_type_ids"],
            attention_mask=batch[f"{OBS_PREFIX}attention_mask"],
            state=state,
            actions=actions,
            behavior_prompt=self._behavior_prompt(batch),
            future_prompt=future_prompt,
        )
        flow = losses["flow"][..., : raw_actions.shape[-1]]
        flow_loss = masked_flow_loss(
            flow,
            action_valid_mask=batch.get(ACTION_VALID_MASK),
            sample_mask=batch.get(SAMPLE_ACTION_LOSS_MASK),
        )
        lambda_bp_encoder = float(self.config.lambda_bp_encoder)
        if use_align:
            align_loss = losses["align"]
            valid_chunks = losses["align_valid_chunks"]
            align_loss_weighted = lambda_bp_encoder * align_loss
            total_loss = flow_loss + align_loss_weighted
        else:
            align_loss = flow_loss.new_zeros(())
            align_loss_weighted = flow_loss.new_zeros(())
            valid_chunks = flow_loss.new_zeros(())
            total_loss = flow_loss
        extra_logs = {}
        lambda_bp_task = float(getattr(self.config, "lambda_bp_task", 0.0))
        if lambda_bp_task != 0.0:
            if "bp_task_id" not in batch:
                raise KeyError("lambda_bp_task != 0 requires batch['bp_task_id'] (dataset.bp_attach_task_id=true)")
            task_loss, task_anchors = bp_task_supcon_loss(
                losses["bp_task_embed"], batch["bp_task_id"], float(self.config.bp_task_tau)
            )
            total_loss = total_loss + lambda_bp_task * task_loss.to(total_loss.dtype)
            extra_logs = {
                "loss_bp_task": float(task_loss.detach()),
                "bp_task_anchors": float(task_anchors),
            }
        debug_check_finite("robot.final_loss", total_loss)
        return total_loss, {
            "step_type": "robot",
            "loss_flow": float(flow_loss.detach()),
            "loss_action": float(flow_loss.detach()),
            "loss_align": float(align_loss.detach()),
            "loss_align_w": float(align_loss_weighted.detach()),
            "align_valid_chunks": float(valid_chunks.detach()),
            **extra_logs,
        }

    def predict_action_chunk(self, batch, **kwargs):
        self.eval()
        state = F.pad(
            batch[OBS_STATE],
            (0, self.config.max_state_dim - batch[OBS_STATE].shape[-1]),
        )
        actions = self.model.sample_actions(
            pixel_values=batch[f"{OBS_PREFIX}pixel_values"],
            image_grid_thw=batch[f"{OBS_PREFIX}image_grid_thw"],
            input_ids=batch[f"{OBS_PREFIX}input_ids"],
            mm_token_type_ids=batch[f"{OBS_PREFIX}mm_token_type_ids"],
            attention_mask=batch[f"{OBS_PREFIX}attention_mask"],
            state=state,
            behavior_prompt=self._behavior_prompt(batch),
            noise=kwargs.get("noise"),
            num_steps=kwargs.get("num_steps"),
        )
        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[..., :original_action_dim]

    def select_action(self, batch, **kwargs):
        if not self._action_queue:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()


__all__ = [
    "BPVAv3Model",
    "BPVAv3Policy",
]
