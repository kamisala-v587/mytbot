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

from .bp_encoder import build_bp_obs_encoder, sinusoidal_progress_encoding
from .configuration_bpvav31 import BPVAv31Config


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


def next_keyframe_index(progress: torch.Tensor, time_ratio: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """每个样本：归一化时间上“当前进度之后的下一个关键帧”下标。"""
    progress = progress.reshape(-1).float()
    # time_ratio: [B, C]
    greater = time_ratio.float() > progress[:, None]
    if mask is not None:
        greater = greater & mask.bool()
    # 若没有任何更大的关键帧，落到最后一个有效 chunk。
    none = ~greater.any(dim=1)
    idx = greater.float().argmax(dim=1)
    if mask is not None:
        last = mask.long().cumsum(dim=1).argmax(dim=1)
    else:
        last = torch.full_like(idx, time_ratio.shape[1] - 1)
    return torch.where(none, last, idx)


def bp_retrieve_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """logits [B, C]，target [B]。无效样本（mask 全假）跳过。"""
    if mask is not None:
        valid = mask.bool().any(dim=1)
        if not bool(valid.any()):
            return logits.sum() * 0.0, valid.sum()
        logits = logits[valid]
        target = target[valid]
        if mask is not None:
            logits = logits.masked_fill(~mask.bool()[valid], float("-inf"))
    loss = F.cross_entropy(logits, target.long())
    return loss, torch.tensor(logits.shape[0], device=logits.device)


def bp_temporal_supcon_loss(
    z: torch.Tensor,
    task_ids: torch.Tensor,
    progress: torch.Tensor,
    mask: torch.Tensor,
    tau: float,
    progress_tol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """逐 chunk 时序对比。z [B, C, D]；同任务且 |Δprogress|<tol 为正样本。"""
    b, c, d = z.shape
    flat = F.normalize(z.float().reshape(b * c, d), dim=-1)
    task = task_ids.reshape(b, 1).expand(b, c).reshape(-1)
    prog = progress.float().reshape(b, c).reshape(-1)
    valid = mask.bool().reshape(-1)
    self_mask = torch.eye(b * c, dtype=torch.bool, device=z.device)
    row_mask = self_mask | ~valid[None, :] | ~valid[:, None]
    logits = (flat @ flat.T / tau).masked_fill(row_mask, float("-inf"))
    # 无效行全是 -inf，logsumexp→-inf 会得到 nan；只在有效行上算 log_prob。
    lse = torch.logsumexp(logits, dim=1, keepdim=True)
    log_prob = torch.zeros_like(logits)
    log_prob = torch.where(valid[:, None], logits - lse, log_prob)
    same_task = task[:, None] == task[None, :]
    close = (prog[:, None] - prog[None, :]).abs() <= float(progress_tol)
    # 同任务且进度接近为正；同示教内进度差过大的自然不是正样本，从而成为负样本。
    positive = same_task & close & ~self_mask & valid[:, None] & valid[None, :]
    num_positive = positive.sum(dim=1)
    anchors = valid & (num_positive > 0)
    if not bool(anchors.any()):
        return flat.sum() * 0.0, anchors.sum()
    per_anchor = log_prob.masked_fill(~positive, 0.0).sum(dim=1)[anchors] / num_positive[anchors]
    return -per_anchor.mean(), anchors.sum()


def soft_phase_target(progress: torch.Tensor, ratios: torch.Tensor, mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """以当前进度为中心、按关键帧归一化时间的高斯软标签 [B, C]；无效 chunk 为 0。"""
    logits = -((ratios.float() - progress.reshape(-1, 1).float()) ** 2) / (2.0 * sigma**2)
    logits = logits.masked_fill(~mask.bool(), float("-inf"))
    return torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)


def bp_soft_retrieve_loss(
    logits: torch.Tensor, target: torch.Tensor, rows: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """软标签 CE，只在 rows 为真的样本上平均。logits 中无效 chunk 应已是 -inf。"""
    log_prob = torch.log_softmax(logits.masked_fill(~rows[:, None], 0.0), dim=-1)
    per_row = -(target * torch.nan_to_num(log_prob, neginf=0.0)).sum(-1)
    n = rows.sum()
    loss = (per_row * rows).sum() / n.clamp_min(1)
    return loss, n


def _gather_rows(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """跨卡拼接第 0 维；本卡部分保留梯度，其它卡的部分不回传。返回 (拼接结果, 本卡起始下标)。"""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return x, 0
    world, rank = dist.get_world_size(), dist.get_rank()
    n = torch.tensor([x.shape[0]], device=x.device, dtype=torch.long)
    sizes = [torch.zeros_like(n) for _ in range(world)]
    dist.all_gather(sizes, n)
    sizes = [int(s) for s in sizes]
    padded = torch.cat([x.detach(), x.new_zeros((max(sizes) - x.shape[0], *x.shape[1:]))])
    out = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(out, padded)
    parts = [o[:s] for o, s in zip(out, sizes)]
    parts[rank] = x
    return torch.cat(parts), sum(sizes[:rank])


def bp_tcn_loss(
    z: torch.Tensor,
    task_ids: torch.Tensor,
    arm_ids: torch.Tensor,
    episode_keys: torch.Tensor,
    ratios: torch.Tensor,
    mask: torch.Tensor,
    tau: float,
    progress_tol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """跨示教时序对比。z [B, C, D]，在加时间编码前的 chunk 内容上算。

    正样本：同任务、同手、不同示教、归一化时间差 <= tol。其余有效 chunk（含同示教其它相位）都是负样本。
    跨卡汇总后再算，本卡 chunk 为 anchor。
    """
    b, c, d = z.shape
    flat = F.normalize(z.float().reshape(b * c, d), dim=-1)
    meta = torch.stack(
        [
            task_ids.reshape(b, 1).expand(b, c).reshape(-1).double(),
            arm_ids.reshape(b, 1).expand(b, c).reshape(-1).double(),
            episode_keys.reshape(b, 1).expand(b, c).reshape(-1).double(),
            ratios.float().reshape(-1).double(),
            mask.bool().reshape(-1).double(),
        ],
        dim=-1,
    )
    all_z, offset = _gather_rows(flat)
    all_meta, _ = _gather_rows(meta)
    n_local = flat.shape[0]
    local = slice(offset, offset + n_local)
    valid_all = all_meta[:, 4] > 0.5
    logits = flat @ all_z.T / tau  # [n_local, N]
    self_mask = torch.zeros_like(logits, dtype=torch.bool)
    self_mask[torch.arange(n_local, device=z.device), torch.arange(offset, offset + n_local, device=z.device)] = True
    logits = logits.masked_fill(self_mask | ~valid_all[None, :], float("-inf"))
    lm = all_meta[local]
    valid_local = lm[:, 4] > 0.5
    lse = torch.logsumexp(logits, dim=1, keepdim=True)
    log_prob = torch.where(valid_local[:, None], logits - lse, torch.zeros_like(logits))
    positive = (
        (lm[:, None, 0] == all_meta[None, :, 0])
        & (lm[:, None, 1] == all_meta[None, :, 1])
        & (lm[:, None, 2] != all_meta[None, :, 2])
        & ((lm[:, None, 3] - all_meta[None, :, 3]).abs() <= float(progress_tol))
        & valid_local[:, None]
        & valid_all[None, :]
        & ~self_mask
    )
    num_positive = positive.sum(dim=1)
    anchors = valid_local & (num_positive > 0)
    if not bool(anchors.any()):
        return flat.sum() * 0.0, anchors.sum()
    per_anchor = torch.nan_to_num(log_prob, neginf=0.0).masked_fill(~positive, 0.0).sum(dim=1)[anchors] / num_positive[anchors]
    return -per_anchor.mean(), anchors.sum()


def bp_align_loss(pred, target, token_mask, loss_type="cosine"):
    if loss_type == "cosine":
        return bp_align_cosine_loss(pred, target, token_mask)
    if loss_type == "mse":
        return bp_align_mse_loss(pred, target, token_mask)
    raise ValueError(f"Unknown BP alignment loss type: {loss_type!r}")


class BPVAv31Model(Q35Model):
    """Q35 前向，BP 图像和当前观测共用一次 Qwen3.5 ViT。"""

    def __init__(self, config: BPVAv31Config) -> None:
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
        need_retrieve = float(getattr(config, "lambda_bp_retrieve", 0.0)) != 0.0 or float(
            getattr(config, "lambda_bp_next_state", 0.0)
        ) != 0.0
        content_mode = getattr(config, "bp_retrieve_mode", "timestamp") == "content"
        if need_retrieve and not content_mode:
            # query: pool(current prefix) + state + progress_encoding
            self.bp_retrieve_query = torch.nn.Sequential(
                torch.nn.Linear(text_hidden * 2 + config.max_state_dim, text_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(text_hidden, text_hidden),
            )
            self.bp_retrieve_key = torch.nn.Linear(text_hidden, text_hidden)
        if need_retrieve and content_mode:
            # query: 当前各相机 ViT 均值 + state；key: 加时间编码前的 BP chunk 内容。两边都不含进度。
            n_cam = int(config.num_camera_views)
            self.bp_content_query = torch.nn.Sequential(
                torch.nn.Linear(n_cam * visual_out + config.max_state_dim, text_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(text_hidden, text_hidden),
            )
            self.bp_content_key = torch.nn.Linear(text_hidden, text_hidden)
        if float(getattr(config, "lambda_bp_next_state", 0.0)) != 0.0:
            if getattr(config, "bp_next_state_mode", "absolute") == "delta":
                self.bp_delta_state_head = torch.nn.Sequential(
                    torch.nn.Linear(text_hidden + config.max_state_dim, text_hidden),
                    torch.nn.GELU(),
                    torch.nn.Linear(text_hidden, config.max_state_dim),
                )
            else:
                self.bp_next_state_head = torch.nn.Sequential(
                    torch.nn.Linear(text_hidden, text_hidden),
                    torch.nn.GELU(),
                    torch.nn.Linear(text_hidden, config.max_state_dim),
                )
        if float(getattr(config, "lambda_bp_temporal", 0.0)) != 0.0:
            head = torch.nn.Sequential(
                torch.nn.Linear(text_hidden, text_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(text_hidden, int(config.bp_task_proj_dim)),
            )
            if getattr(config, "bp_temporal_mode", "post_vlm") == "content":
                self.bp_content_temporal_head = head
            else:
                self.bp_temporal_head = head
        self._pending_state_extra: torch.Tensor | None = None
        if bool(getattr(config, "bp_subgoal_token", False)):
            action_hidden = int(self.core.state_projection.out_features)
            self.bp_subgoal_proj = torch.nn.Sequential(
                torch.nn.Linear(text_hidden + config.max_state_dim + 1, text_hidden),
                torch.nn.GELU(),
                torch.nn.Linear(text_hidden, action_hidden),
            )
            # 末层置零：初始时 token 为 0，动作专家行为与加载的 ckpt 一致。
            torch.nn.init.zeros_(self.bp_subgoal_proj[-1].weight)
            torch.nn.init.zeros_(self.bp_subgoal_proj[-1].bias)
            self.bp_subgoal_gate = torch.nn.Sequential(
                torch.nn.Linear(2 * text_hidden + 2, 256),
                torch.nn.GELU(),
                torch.nn.Linear(256, 1),
            )
            self.core.state_projection.register_forward_hook(self._add_state_extra)
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
    def _visual_features(
        self,
        pixels: torch.Tensor,
        grids: torch.Tensor,
        *,
        no_grad: bool = False,
    ) -> list[torch.Tensor]:
        vlm = self.core.backbone.vlm
        vision_dtype = next(self.core.backbone.vlm_visual.parameters()).dtype
        flat_pixels = pixels.reshape(-1, pixels.shape[-1]).to(dtype=vision_dtype)
        flat_grids = grids.reshape(-1, 3).to(device=flat_pixels.device)
        context = torch.no_grad() if (no_grad or self.config.frozen_vit) else nullcontext()
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
        elif self.config.frozen_bp_vit and not self.config.frozen_vit:
            current_features = self._visual_features(current_pixels, current_grids)
            bp_features = self._visual_features(bp_pixels, bp_grids, no_grad=True)
        else:
            features = self._visual_features(
                torch.cat([current_pixels, bp_pixels], dim=0),
                torch.cat([current_grids, bp_grids], dim=0),
            )
            current_features = features[: current_grids.shape[0]]
            bp_features = features[current_grids.shape[0] :]
        prefix = self._scatter_current_prefix(current_features, input_ids)
        self._last_current_visual = current_features
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
        features = self._visual_features(bp_pixels, bp_grids, no_grad=self.config.frozen_bp_vit)
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

    def _chunk_pool(
        self,
        bp_hidden: torch.Tensor,
        bp_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, C*Q, D] → chunk 均值 [B, C, D] 与 chunk mask [B, C]。"""
        queries = int(self.config.bp_num_query_tokens)
        batch, n_bp, dim = bp_hidden.shape
        if n_bp % queries:
            raise ValueError(f"BP token count {n_bp} is not divisible by bp_num_query_tokens={queries}")
        chunks = n_bp // queries
        hidden = bp_hidden.reshape(batch, chunks, queries, dim)
        valid = bp_valid.reshape(batch, chunks, queries)
        chunk_mask = valid.any(dim=-1)
        weights = valid.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        pooled = pooled.masked_fill(~chunk_mask[..., None], 0)
        return pooled, chunk_mask

    def _content_query_key(
        self, *, batch_size: int, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """读刚跑完的 embed_prefix_with_bp 缓存：返回 (query [B, D], key [B, C, D], chunk 内容 [B, C, D], chunk mask [B, C])。

        query 只用当前各相机 ViT 均值 + state，key 只用加时间编码前的 BP 内容，两边都没有进度信息。
        """
        tokens, cmask = self.bp_obs_encoder.last_content
        content = tokens.float().mean(dim=2)
        cur = torch.stack([f.float().mean(dim=0) for f in self._last_current_visual])
        n_cam = int(self.config.num_camera_views)
        if cur.shape[0] != batch_size * n_cam:
            raise ValueError(f"expected {batch_size}x{n_cam} current images, got {cur.shape[0]}")
        dtype = next(self.bp_content_query.parameters()).dtype
        q_in = torch.cat([cur.reshape(batch_size, -1), state.float()], dim=-1)
        query = F.normalize(self.bp_content_query(q_in.to(dtype)).float(), dim=-1)
        key = F.normalize(self.bp_content_key(content.to(dtype)).float(), dim=-1)
        return query, key, content, cmask

    def content_retrieval(
        self, *, batch_size: int, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (logits [B, C], chunk 内容 [B, C, D], chunk mask [B, C])。"""
        query, key, content, cmask = self._content_query_key(batch_size=batch_size, state=state)
        logits = torch.einsum("bd,bcd->bc", query, key) / float(self.config.bp_retrieve_tau)
        return logits.masked_fill(~cmask, float("-inf")), content, cmask

    def _add_state_extra(self, module, args, output):
        extra = self._pending_state_extra
        return output if extra is None else output + extra.to(device=output.device, dtype=output.dtype)

    def subgoal_extra(
        self,
        *,
        batch_size: int,
        state: torch.Tensor,
        behavior_prompt: dict,
        dropout: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """门控后的子目标向量 [B, H_action] 与门值 [B]。必须在 embed_prefix_with_bp 之后调用。"""
        query, key, content, cmask = self._content_query_key(batch_size=batch_size, state=state)
        logits = torch.einsum("bd,bcd->bc", query, key) / float(self.config.bp_retrieve_tau)
        attn = torch.nan_to_num(torch.softmax(logits.masked_fill(~cmask, float("-inf")), dim=-1), nan=0.0)
        retrieved = torch.einsum("bc,bcd->bd", attn, content)
        dtype = next(self.bp_delta_state_head.parameters()).dtype
        delta = self.bp_delta_state_head(torch.cat([retrieved, state.float()], -1).to(dtype)).float()
        ratios = behavior_prompt.get("source_time_ratio")
        if ratios is None:
            n = attn.shape[1]
            ratios = torch.arange(n, device=attn.device, dtype=torch.float32) / max(1, n - 1)
        ratios = torch.as_tensor(ratios, device=attn.device, dtype=torch.float32).expand_as(attn)
        phase = (attn * ratios).sum(-1, keepdim=True)
        token = self.bp_subgoal_proj(torch.cat([retrieved, delta, phase], -1).to(dtype)).float()
        max_p = attn.max(-1, keepdim=True).values
        entropy = -(attn * torch.log(attn.clamp_min(1e-8))).sum(-1, keepdim=True)
        matched_key = torch.einsum("bc,bcd->bd", attn, key)
        gate_in = torch.cat([query, matched_key, max_p, entropy], -1)
        gate = torch.sigmoid(self.bp_subgoal_gate(gate_in.to(dtype)).float()).squeeze(-1)
        effective = gate
        if dropout > 0.0:
            keep = (torch.rand(batch_size, device=gate.device) >= dropout).to(gate.dtype)
            effective = gate * keep
        return effective[:, None] * token, gate

    def _content_aux_losses(
        self,
        *,
        batch_size: int,
        state: torch.Tensor,
        behavior_prompt: dict,
        current_progress: torch.Tensor | None,
        bp_origin_state: torch.Tensor | None,
        bp_origin_time_ratio: torch.Tensor | None,
        bp_task_id: torch.Tensor | None,
        bp_meta: dict,
    ) -> dict[str, torch.Tensor]:
        extras: dict[str, torch.Tensor] = {}
        if current_progress is None:
            raise KeyError("content retrieval requires batch['current_progress']")
        logits, content, cmask = self.content_retrieval(batch_size=batch_size, state=state)
        device = logits.device
        ratios = torch.as_tensor(behavior_prompt["source_time_ratio"], device=device, dtype=torch.float32)
        ratios = ratios.reshape(batch_size, -1)
        progress = current_progress.to(device=device, dtype=torch.float32).reshape(-1)
        rows = cmask.any(dim=1)
        if bool(self.config.bp_align_require_arm_match) and "bp_arm_match" in bp_meta:
            rows = rows & bp_meta["bp_arm_match"].to(device).bool().reshape(-1)
        if bool(self.config.bp_align_skip_self) and "bp_is_self" in bp_meta:
            rows = rows & ~bp_meta["bp_is_self"].to(device).bool().reshape(-1)
        target = soft_phase_target(progress, ratios, cmask, float(self.config.bp_retrieve_sigma))
        extras["bp_align_frac"] = rows.float().mean()
        if float(self.config.lambda_bp_retrieve) != 0.0:
            loss, _ = bp_soft_retrieve_loss(logits, target, rows)
            extras["bp_retrieve_loss"] = loss
            with torch.no_grad():
                hit = (logits.argmax(-1) - target.argmax(-1)).abs() <= 0
                extras["bp_retrieve_acc"] = (hit & rows).float().sum() / rows.sum().clamp_min(1)
        if hasattr(self, "bp_delta_state_head"):
            if bp_origin_state is None or bp_origin_time_ratio is None:
                raise KeyError("bp_next_state_mode=delta requires bp_origin_state / bp_origin_time_ratio")
            attn = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
            retrieved = torch.einsum("bc,bcd->bd", attn, content)
            dtype = next(self.bp_delta_state_head.parameters()).dtype
            delta = self.bp_delta_state_head(torch.cat([retrieved, state.float()], -1).to(dtype)).float()
            origin = bp_origin_state.to(device=device, dtype=torch.float32).reshape(batch_size, -1, state.shape[-1])
            oratio = torch.as_tensor(bp_origin_time_ratio, device=device, dtype=torch.float32).reshape(batch_size, -1)
            idx = next_keyframe_index(progress, oratio, None)
            target_state = origin.gather(1, idx.view(-1, 1, 1).expand(-1, 1, origin.shape[-1])).squeeze(1)
            target_delta = target_state - state.float()
            per_row = (delta - target_delta).square().mean(-1)
            n = rows.sum().clamp_min(1)
            extras["bp_next_state_loss"] = (per_row * rows).sum() / n
            with torch.no_grad():
                extras["bp_next_state_copy_mse"] = (target_delta.square().mean(-1) * rows).sum() / n
        if hasattr(self, "bp_content_temporal_head"):
            if bp_task_id is None or "bp_arm_id" not in bp_meta or "bp_episode_key" not in bp_meta:
                raise KeyError("bp_temporal_mode=content requires bp_task_id / bp_arm_id / bp_episode_key")
            dtype = next(self.bp_content_temporal_head.parameters()).dtype
            z = self.bp_content_temporal_head(content.to(dtype))
            loss, anchors = bp_tcn_loss(
                z,
                bp_task_id.to(device),
                bp_meta["bp_arm_id"].to(device),
                bp_meta["bp_episode_key"].to(device),
                ratios,
                cmask,
                float(self.config.bp_temporal_tau),
                float(self.config.bp_temporal_progress_tol),
            )
            extras["bp_temporal_loss"] = loss
            extras["bp_temporal_anchors"] = anchors.to(dtype=loss.dtype)
        return extras

    def _retrieval_aux_losses(
        self,
        *,
        vlm_hidden: torch.Tensor,
        vlm_valid_mask: torch.Tensor,
        input_ids: torch.Tensor,
        state: torch.Tensor,
        behavior_prompt: dict,
        current_progress: torch.Tensor | None,
        bp_origin_state: torch.Tensor | None,
        bp_origin_time_ratio: torch.Tensor | None,
        bp_task_id: torch.Tensor | None,
        bp_meta: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        extras: dict[str, torch.Tensor] = {}
        bp_tokens = vlm_hidden.shape[1] - input_ids.shape[1]
        if bp_tokens <= 0:
            return extras
        bp_hidden = vlm_hidden[:, -bp_tokens:]
        bp_valid = vlm_valid_mask[:, -bp_tokens:]
        chunk, chunk_mask = self._chunk_pool(bp_hidden.float(), bp_valid)
        # 全局 task id：只对 8 个 chunk 池化，避免逐 token/逐 chunk 压成同一个 id。
        if hasattr(self, "bp_task_head"):
            weights = chunk_mask.to(chunk.dtype).unsqueeze(-1)
            pooled = (chunk * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            head_dtype = next(self.bp_task_head.parameters()).dtype
            extras["bp_task_embed"] = self.bp_task_head(pooled.to(head_dtype))
        if hasattr(self, "bp_temporal_head"):
            if bp_task_id is None:
                raise KeyError("lambda_bp_temporal != 0 requires bp_task_id")
            ratios = behavior_prompt.get("source_time_ratio")
            if ratios is None:
                ratios = (
                    torch.arange(chunk.shape[1], device=chunk.device, dtype=torch.float32)
                    .view(1, -1)
                    .expand(chunk.shape[0], -1)
                    / max(1, chunk.shape[1] - 1)
                )
            else:
                ratios = torch.as_tensor(ratios, device=chunk.device, dtype=torch.float32)
                if ratios.ndim == 1:
                    ratios = ratios.unsqueeze(0)
            head_dtype = next(self.bp_temporal_head.parameters()).dtype
            z = self.bp_temporal_head(chunk.to(head_dtype))
            temporal_loss, temporal_anchors = bp_temporal_supcon_loss(
                z,
                bp_task_id,
                ratios,
                chunk_mask,
                float(self.config.bp_temporal_tau),
                float(self.config.bp_temporal_progress_tol),
            )
            extras["bp_temporal_loss"] = temporal_loss
            extras["bp_temporal_anchors"] = temporal_anchors.to(dtype=temporal_loss.dtype)
        if hasattr(self, "bp_content_query"):
            extras.update(
                self._content_aux_losses(
                    batch_size=input_ids.shape[0],
                    state=state,
                    behavior_prompt=behavior_prompt,
                    current_progress=current_progress,
                    bp_origin_state=bp_origin_state,
                    bp_origin_time_ratio=bp_origin_time_ratio,
                    bp_task_id=bp_task_id,
                    bp_meta=bp_meta or {},
                )
            )
            return extras
        need_retrieve = hasattr(self, "bp_retrieve_query")
        if not need_retrieve:
            return extras
        if current_progress is None:
            raise KeyError("retrieval losses require batch['current_progress']")
        ratios = behavior_prompt.get("source_time_ratio")
        if ratios is None:
            raise KeyError("retrieval losses require behavior_prompt['source_time_ratio']")
        ratios = torch.as_tensor(ratios, device=chunk.device, dtype=torch.float32)
        if ratios.ndim == 1:
            ratios = ratios.unsqueeze(0)
        progress = current_progress.to(device=chunk.device, dtype=torch.float32).reshape(-1)
        target = next_keyframe_index(progress, ratios, chunk_mask)

        prefix = vlm_hidden[:, :-bp_tokens].float()
        prefix_valid = vlm_valid_mask[:, :-bp_tokens].to(prefix.dtype).unsqueeze(-1)
        pooled_cur = (prefix * prefix_valid).sum(dim=1) / prefix_valid.sum(dim=1).clamp_min(1.0)
        prog_enc = sinusoidal_progress_encoding(progress, prefix.shape[-1]).to(dtype=pooled_cur.dtype)
        state_feat = state.float()
        query_in = torch.cat([pooled_cur, prog_enc, state_feat], dim=-1)
        q_dtype = next(self.bp_retrieve_query.parameters()).dtype
        query = self.bp_retrieve_query(query_in.to(q_dtype)).float()
        key = self.bp_retrieve_key(chunk.to(q_dtype)).float()
        tau = float(self.config.bp_retrieve_tau)
        logits = torch.einsum("bd,bcd->bc", query, key) / tau
        logits = logits.masked_fill(~chunk_mask, float("-inf"))
        if float(self.config.lambda_bp_retrieve) != 0.0:
            retrieve_loss, retrieve_n = bp_retrieve_ce_loss(logits, target, chunk_mask)
            extras["bp_retrieve_loss"] = retrieve_loss
            extras["bp_retrieve_n"] = retrieve_n.to(dtype=retrieve_loss.dtype)
        if float(self.config.lambda_bp_next_state) != 0.0:
            if bp_origin_state is None:
                raise KeyError("lambda_bp_next_state != 0 requires batch['bp_origin_state']")
            # 全 masked 行 softmax 会 nan；用 mask 归一化。
            attn = torch.softmax(logits.masked_fill(~chunk_mask, float("-inf")), dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            retrieved = torch.einsum("bc,bcd->bd", attn, chunk)
            pred = self.bp_next_state_head(retrieved.to(next(self.bp_next_state_head.parameters()).dtype)).float()
            origin = bp_origin_state.to(device=pred.device, dtype=torch.float32)
            if origin.ndim == 2:
                origin = origin.unsqueeze(0)
            # 目标：当前轨迹在“下一关键帧”处的 state（origin 与 BP 共用归一化时间对齐）。
            origin_ratio = bp_origin_time_ratio
            if origin_ratio is None:
                origin_target = target
            else:
                origin_ratio = torch.as_tensor(origin_ratio, device=pred.device, dtype=torch.float32)
                if origin_ratio.ndim == 1:
                    origin_ratio = origin_ratio.unsqueeze(0)
                origin_target = next_keyframe_index(progress, origin_ratio, None)
            gather_idx = origin_target.view(-1, 1, 1).expand(-1, 1, origin.shape[-1])
            target_state = origin.gather(1, gather_idx).squeeze(1)
            extras["bp_next_state_loss"] = F.mse_loss(pred, target_state)
        return extras

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
        current_progress: torch.Tensor | None = None,
        bp_origin_state: torch.Tensor | None = None,
        bp_origin_time_ratio: torch.Tensor | None = None,
        bp_task_id: torch.Tensor | None = None,
        bp_meta: dict | None = None,
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
        aux = self._retrieval_aux_losses(
            vlm_hidden=vlm_hidden,
            vlm_valid_mask=vlm_valid_mask,
            input_ids=input_ids,
            state=state,
            behavior_prompt=behavior_prompt,
            current_progress=current_progress,
            bp_origin_state=bp_origin_state,
            bp_origin_time_ratio=bp_origin_time_ratio,
            bp_task_id=bp_task_id,
            bp_meta=bp_meta,
        )
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
        if hasattr(self, "bp_subgoal_proj"):
            extra, gate = self.subgoal_extra(
                batch_size=input_ids.shape[0],
                state=state,
                behavior_prompt=behavior_prompt,
                dropout=float(self.config.bp_subgoal_dropout) if self.training else 0.0,
            )
            self._pending_state_extra = extra
            with torch.no_grad():
                aux["bp_gate"] = gate.mean()
                match = (bp_meta or {}).get("bp_arm_match")
                if match is not None:
                    match = match.to(gate.device).bool().reshape(-1)
                    aux["bp_gate_same"] = (gate * match).sum() / match.sum().clamp_min(1)
                    aux["bp_gate_cross"] = (gate * ~match).sum() / (~match).sum().clamp_min(1)
        try:
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
        finally:
            self._pending_state_extra = None
        result.update(aux)
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
        if hasattr(self, "bp_subgoal_proj"):
            self._pending_state_extra, _ = self.subgoal_extra(
                batch_size=input_ids.shape[0], state=state, behavior_prompt=behavior_prompt
            )
        try:
            prefix_cache, action_attention_mask, action_position_ids = self.core.prepare_inference_cache(
                vlm_hidden,
                state,
                vlm_valid_mask=vlm_valid_mask,
                position_ids=position_ids,
            )
        finally:
            self._pending_state_extra = None
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


class BPVAv31Policy(PreTrainedPolicy):
    config_class = BPVAv31Config
    name = "bpvav31"

    def __init__(self, config: BPVAv31Config) -> None:
        super().__init__(config)
        config.validate_features()
        self.model = BPVAv31Model(config)
        actual_fingerprint = self.model.core.backbone.fingerprint.name
        if config.qwen3_5_fingerprint not in {None, actual_fingerprint}:
            raise ValueError(
                "BPVAv31 Qwen3.5 fingerprint mismatch: "
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
            if key.startswith(
                (
                    "model.bp_obs_encoder.",
                    "model.bp_task_head.",
                    "model.bp_retrieve_query.",
                    "model.bp_retrieve_key.",
                    "model.bp_next_state_head.",
                    "model.bp_temporal_head.",
                    "model.bp_content_query.",
                    "model.bp_content_key.",
                    "model.bp_delta_state_head.",
                    "model.bp_content_temporal_head.",
                    "model.bp_subgoal_proj.",
                    "model.bp_subgoal_gate.",
                )
            )
            or key == "model.align_queries"
        ]
        filtered_missing = [key for key in missing_keys if key not in expected_missing]
        # v3 → v3.1：进度编码替代可学习 chunk_position_embedding，旧权重可忽略。
        filtered_unexpected = [
            key for key in unexpected_keys if "chunk_position_embedding" not in key
        ]
        return filtered_missing, filtered_unexpected, expected_missing

    def _behavior_prompt(self, batch) -> dict:
        if "behavior_prompt" not in batch:
            raise KeyError("BPVAv31 requires batch['behavior_prompt']")
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
            current_progress=batch.get("current_progress"),
            bp_origin_state=batch.get("bp_origin_state"),
            bp_origin_time_ratio=batch.get("bp_origin_time_ratio"),
            bp_task_id=batch.get("bp_task_id"),
            bp_meta={k: batch[k] for k in ("bp_is_self", "bp_arm_match", "bp_arm_id", "bp_episode_key") if k in batch},
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
            extra_logs["loss_bp_task"] = float(task_loss.detach())
            extra_logs["bp_task_anchors"] = float(task_anchors)
        for name, key in (
            ("lambda_bp_retrieve", "bp_retrieve_loss"),
            ("lambda_bp_next_state", "bp_next_state_loss"),
            ("lambda_bp_temporal", "bp_temporal_loss"),
        ):
            weight = float(getattr(self.config, name, 0.0))
            if weight == 0.0:
                continue
            if key not in losses:
                raise KeyError(f"{name} != 0 but forward did not return {key}")
            part = losses[key].to(total_loss.dtype)
            total_loss = total_loss + weight * part
            extra_logs[f"loss_{key}"] = float(part.detach())
        if "bp_retrieve_n" in losses:
            extra_logs["bp_retrieve_n"] = float(losses["bp_retrieve_n"].detach())
        for key in (
            "bp_temporal_anchors", "bp_retrieve_acc", "bp_align_frac", "bp_next_state_copy_mse",
            "bp_gate", "bp_gate_same", "bp_gate_cross",
        ):
            if key in losses:
                extra_logs[key] = float(losses[key].detach())
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
    "BPVAv31Model",
    "BPVAv31Policy",
]
