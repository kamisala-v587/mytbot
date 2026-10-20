"""bpv4_bpen：BPv4 第一阶段的 BP encoder 训练模型（不是完整的 v4）。

prefix = 语言 token + 1 个进度 token + M 个 BP token（无当前图像）；suffix = state + noisy action（沿用 Q35 主干）。
BP token 来自缓存的关键帧 ViT 特征：chunk compressor（每帧 3 路 → Q 个 token）+ 时间编码 + 轨迹重采样器（可见关键帧 → M 个 token）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.q35.modeling_q35 import Q35Model, Q35Policy, debug_check_finite, masked_flow_loss
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_bpv4_bpen import BPV4BPEnConfig

BP_NEW_MODULE_PREFIXES = ("bp_obs_encoder.", "bp_resampler.", "bp_progress.")


def sinusoidal_encoding(x: torch.Tensor, dim: int, max_octaves: float = 8.0) -> torch.Tensor:
    """x in [0, 1] -> [..., dim]; frequencies pi * 2^(0..max_octaves) (bounded, no overflow for large dim)."""
    half = dim // 2
    i = torch.arange(half, device=x.device, dtype=torch.float32)
    freq = math.pi * (2.0 ** (i * (max_octaves / max(1, half - 1))))
    angles = x.float().clamp(0.0, 1.0)[..., None] * freq
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class _CrossBlock(nn.Module):
    """queries <- cross-attn(tokens) -> self-attn(queries) -> FFN, pre-norm."""

    def __init__(self, dim: int, heads: int, ff_mult: int, self_attn: bool = True) -> None:
        super().__init__()
        self.cross_norm_q, self.cross_norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True) if self_attn else None
        self.self_norm = nn.LayerNorm(dim) if self_attn else None
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Linear(dim * ff_mult, dim))

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_padding: torch.Tensor | None = None) -> torch.Tensor:
        kvn = self.cross_norm_kv(kv)
        q = q + self.cross(self.cross_norm_q(q), kvn, kvn, key_padding_mask=kv_padding, need_weights=False)[0]
        if self.self_attn is not None:
            qn = self.self_norm(q)
            q = q + self.self_attn(qn, qn, qn, need_weights=False)[0]
        return q + self.ff(self.ff_norm(q))


class BPChunkCompressor(nn.Module):
    """Per keyframe: C cameras x T visual tokens -> Q tokens (image only, no state / action)."""

    def __init__(self, cfg: BPV4BPEnConfig, out_dim: int) -> None:
        super().__init__()
        d = int(cfg.bp_compressor_dim)
        self.input_proj = nn.Linear(int(cfg.bp_visual_dim), d)
        self.camera_embedding = nn.Parameter(torch.zeros(int(cfg.bp_num_cameras), d))
        self.queries = nn.Parameter(torch.randn(int(cfg.bp_num_query_tokens), d) * 0.02)
        self.layers = nn.ModuleList(
            _CrossBlock(d, int(cfg.bp_compressor_num_heads), int(cfg.bp_compressor_ff_mult))
            for _ in range(int(cfg.bp_compressor_num_layers))
        )
        self.output_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, out_dim))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats [B, K, C, T, Dv] -> [B, K, Q, D]."""
        b, k, c, t, _ = feats.shape
        x = self.input_proj(feats.to(self.input_proj.weight.dtype)) + self.camera_embedding[None, None, :, None]
        x = x.reshape(b * k, c * t, -1)
        q = self.queries[None].expand(b * k, -1, -1)
        for layer in self.layers:
            q = layer(q, x)
        return self.output_proj(q).reshape(b, k, q.shape[1], -1)


class BPTrajectoryResampler(nn.Module):
    """Visible keyframe tokens [N, K*Q, D] -> M tokens; hidden keyframes are excluded via key padding."""

    def __init__(self, cfg: BPV4BPEnConfig, dim: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(int(cfg.bp_resampler_num_tokens), dim) * 0.02)
        self.layers = nn.ModuleList(
            _CrossBlock(dim, int(cfg.bp_resampler_num_heads), 4) for _ in range(int(cfg.bp_resampler_num_layers))
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        q = self.queries[None].expand(tokens.shape[0], -1, -1)
        for layer in self.layers:
            q = layer(q, tokens, kv_padding=~visible)
        return self.norm(q)


class BPProgressToken(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.null = nn.Parameter(torch.zeros(dim))

    def forward(self, progress: torch.Tensor, dropped: torch.Tensor) -> torch.Tensor:
        enc = self.mlp(sinusoidal_encoding(progress, self.dim).to(self.null.dtype))
        return torch.where(dropped[:, None], self.null[None].expand_as(enc), enc)[:, None]


class BPV4BPEnModel(Q35Model):
    def __init__(self, config: BPV4BPEnConfig) -> None:
        super().__init__(config)
        hidden = int(self.core.backbone.vlm.config.text_config.hidden_size)
        self.bp_obs_encoder = BPChunkCompressor(config, hidden)
        self.bp_resampler = BPTrajectoryResampler(config, hidden)
        self.bp_progress = BPProgressToken(hidden)
        # No current images in this stage: the ViT and the LM head are never used.
        for module in (self.core.backbone.vlm_visual, getattr(self.core.backbone.vlm, "lm_head", None)):
            if module is not None:
                for p in module.parameters():
                    p.requires_grad_(False)
        self.register_buffer("train_step", torch.zeros((), dtype=torch.long), persistent=True)

    def mask_prob(self) -> float:
        cfg = self.config
        ramp = max(1, int(cfg.mask_prob_ramp_steps))
        frac = min(1.0, float(self.train_step) / ramp)
        return float(cfg.mask_prob_start) + (float(cfg.mask_prob_end) - float(cfg.mask_prob_start)) * frac

    def bp_tokens(self, batch: dict, visible: torch.Tensor) -> torch.Tensor:
        """Compressor runs once per trajectory; the resampler runs per query. visible [B, S, K] -> [B*S, M, D]."""
        feats = batch["bp.feats"]  # [B, K, C, T, Dv]
        b, k = feats.shape[:2]
        s = visible.shape[1]
        content = self.bp_obs_encoder(feats)  # [B, K, Q, D]
        content = content + sinusoidal_encoding(batch["bp.time"], content.shape[-1]).to(content.dtype)[:, :, None]
        q = content.shape[2]
        tokens = content.reshape(b, k * q, -1).repeat_interleave(s, dim=0)  # [B*S, K*Q, D]
        key_visible = visible.reshape(b * s, k)[:, :, None].expand(-1, -1, q).reshape(b * s, k * q)
        return self.bp_resampler(tokens, key_visible)

    def visibility(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Per query: [B, S, K] visible keyframes and mask type (0 none, 1 event segment, 2 ratio window)."""
        kf_valid = batch["bp.kf_valid"][:, None]  # [B, 1, K]
        u_mask, u_type = batch["query.u_mask"], batch["query.u_type"]
        masked = u_mask < (self.mask_prob() if self.training else 0.0)
        use_event = (u_type < float(self.config.mask_event_segment_prob)) & batch["bp.event_ok"]
        hide = torch.where(use_event[..., None], ~batch["bp.vis_event"], ~batch["bp.vis_window"])
        visible = kf_valid & ~(hide & masked[..., None])
        mask_type = torch.where(masked, torch.where(use_event, 1, 2), 0)
        return visible, mask_type

    def forward_queries(
        self,
        batch: dict,
        visible: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, s = batch["query.progress"].shape
        n = b * s
        vlm = self.core.backbone.vlm
        input_ids = batch["observation.input_ids"].reshape(n, -1)
        text_valid = batch["observation.attention_mask"].reshape(n, -1).bool()
        text = vlm.get_input_embeddings()(input_ids)
        dtype = text.dtype
        progress = self.bp_progress(batch["query.progress"].reshape(n), batch["query.progress_drop"].reshape(n).bool())
        bp = self.bp_tokens(batch, visible)
        prefix = torch.cat([text, progress.to(dtype), bp.to(dtype)], dim=1)
        valid = torch.cat(
            [text_valid, torch.ones(n, 1 + bp.shape[1], dtype=torch.bool, device=text.device)], dim=1
        )
        state = batch[OBS_STATE].reshape(n, -1)
        state = F.pad(state, (0, self.config.max_state_dim - state.shape[-1]))
        actions = batch[ACTION].reshape(n, batch[ACTION].shape[2], -1)
        actions = F.pad(actions, (0, self.config.max_action_dim - actions.shape[-1]))
        return self.core.forward_embedded(
            prefix, state, actions, noise=noise, time=time, position_ids=None, vlm_valid_mask=valid
        )


class BPV4BPEnPolicy(Q35Policy):
    config_class = BPV4BPEnConfig
    name = "bpv4_bpen"

    def __init__(self, config: BPV4BPEnConfig) -> None:
        super(Q35Policy, self).__init__(config)
        config.validate_features()
        self.model = BPV4BPEnModel(config)
        config.qwen3_5_fingerprint = self.model.core.backbone.fingerprint.name
        self.reset()

    def classify_model_loading_keys(self, missing_keys: list[str], unexpected_keys: list[str]):
        expected = [k for k in missing_keys if k.startswith(tuple(f"model.{p}" for p in BP_NEW_MODULE_PREFIXES))
                    or k == "model.train_step"]
        return [k for k in missing_keys if k not in expected], list(unexpected_keys), expected

    def get_optim_params(self):
        cfg = self.config
        tail = tuple(f"model.core.backbone.vlm.model.language_model.layers.{i}." for i in cfg.vlm_tail_layers)
        groups = {"bp_new": [], "act": [], "vlm_tail": [], "vlm_rest": []}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(tuple(f"model.{x}" for x in BP_NEW_MODULE_PREFIXES)):
                groups["bp_new"].append(p)
            elif name.startswith("model.core.backbone.vlm."):
                groups["vlm_tail" if name.startswith(tail) else "vlm_rest"].append(p)
            else:
                groups["act"].append(p)
        vlm_extra = {
            "lr_zero_steps": int(cfg.vlm_zero_steps),
            "lr_ramp_steps": int(cfg.vlm_ramp_steps),
            "reset_state_at_unfreeze": bool(cfg.vlm_reset_optimizer_state_at_unfreeze),
        }
        return [
            {"name": "bp_new", "params": groups["bp_new"], "lr": float(cfg.bp_new_modules_lr)},
            {"name": "act", "params": groups["act"], "lr": float(cfg.act_expert_lr or cfg.optimizer_lr)},
            {"name": "vlm_tail", "params": groups["vlm_tail"], "lr": float(cfg.optimizer_lr) * float(cfg.vlm_tail_lr_mult), **vlm_extra},
            {"name": "vlm_rest", "params": groups["vlm_rest"], "lr": float(cfg.optimizer_lr) * float(cfg.vlm_lr_mult), **vlm_extra},
        ]

    def forward(self, batch, *, validation_mode: str | None = None):
        model = self.model
        if self.training:
            with torch.no_grad():
                model.train_step += 1
        visible, mask_type = model.visibility(batch)
        losses = model.forward_queries(batch, visible)
        raw_dim = batch[ACTION].shape[-1]
        flow = losses["flow"][..., :raw_dim]  # [N, H, A]
        valid = batch[ACTION_VALID_MASK].reshape(flow.shape[0], -1)
        loss = masked_flow_loss(flow, action_valid_mask=valid)
        debug_check_finite("bpv4_bpen.loss", loss)
        with torch.no_grad():
            w = valid[..., None].to(flow.dtype)
            per_query = (flow * w).sum((1, 2)) / (w.sum((1, 2)) * flow.shape[-1]).clamp_min(1)
            mt = mask_type.reshape(-1)
            logs = {"loss_action": float(loss.detach())}
            for code, label in ((0, "nomask"), (1, "event"), (2, "window")):
                sel = mt == code
                logs[f"bp_loss_{label}"] = float(per_query[sel].mean()) if bool(sel.any()) else 0.0
                logs[f"bp_frac_{label}"] = float(sel.float().mean())
            logs["bp_visible_kf"] = float(visible.sum(-1).float().mean())
            logs["bp_mask_prob"] = model.mask_prob()
        return loss, logs

    def predict_action_chunk(self, batch, **kwargs):
        raise NotImplementedError("bpv4_bpen is an encoder-training stage; closed-loop inference is not supported")

    def select_action(self, batch, **kwargs):
        raise NotImplementedError("bpv4_bpen is an encoder-training stage; closed-loop inference is not supported")
