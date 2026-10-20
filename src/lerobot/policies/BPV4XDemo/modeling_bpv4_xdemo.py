"""bpv4_xdemo：BPv4 第二阶段（跨示教）。

prefix = 当前三路图像 token（Qwen 原生 mrope 位置）+ 示教 D 的 M 个 BP token（顺序位置，接在图像之后）；
suffix = 当前 state + noisy action（沿用 Q35 主干）。没有语言和进度 token。
BP 编码器（bp_obs_encoder / bp_resampler）与第一阶段同名同结构，直接从 bpv4_bpen checkpoint 加载。
"""

from __future__ import annotations

import logging
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.BPV4BPEn.modeling_bpv4_bpen import BPChunkCompressor, BPTrajectoryResampler, sinusoidal_encoding
from lerobot.policies.q35.modeling_q35 import Q35Model, Q35Policy, debug_check_finite, masked_flow_loss
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_bpv4_xdemo import BPV4XDemoConfig

BP_ENCODER_PREFIXES = ("bp_obs_encoder.", "bp_resampler.")
NEW_MODULE_PREFIXES = ("bp_type_embedding",)
# Stage-1 modules that no longer exist in stage 2.
DROPPED_STAGE1_PREFIXES = ("model.bp_progress.", "model.train_step")


class BPV4XDemoModel(Q35Model):
    def __init__(self, config: BPV4XDemoConfig) -> None:
        super().__init__(config)
        hidden = int(self.core.backbone.vlm.config.text_config.hidden_size)
        self.bp_obs_encoder = BPChunkCompressor(config, hidden)
        self.bp_resampler = BPTrajectoryResampler(config, hidden)
        # 0: current-image tokens, 1: BP tokens. Zero init keeps the stage-1 representation untouched at the start.
        self.bp_type_embedding = nn.Parameter(torch.zeros(2, hidden))
        for module in (self.core.backbone.vlm_visual, getattr(self.core.backbone.vlm, "lm_head", None)):
            if module is not None:
                for p in module.parameters():
                    p.requires_grad_(False)

    def bp_tokens(self, batch: dict, visible: torch.Tensor | None = None) -> torch.Tensor:
        """One demonstration per pair: [B, K, C, T, Dv] -> [B, M, D]."""
        feats = batch["bp.feats"]
        visible = batch["bp.visible"] if visible is None else visible
        b, k = feats.shape[:2]
        content = self.bp_obs_encoder(feats)  # [B, K, Q, D]
        content = content + sinusoidal_encoding(batch["bp.time"], content.shape[-1]).to(content.dtype)[:, :, None]
        q = content.shape[2]
        key_visible = visible[:, :, None].expand(-1, -1, q).reshape(b, k * q)
        return self.bp_resampler(content.reshape(b, k * q, -1), key_visible)

    def build_prefix(
        self,
        batch: dict,
        *,
        bp: torch.Tensor | None = None,
        bp_valid: torch.Tensor | None = None,
        image_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns prefix hidden [N, L+M, D], prefix valid mask [N, L+M], and the input ids / grids for positions.

        bp_valid / image_valid: optional per-query bool [N]; False hides that part of the prefix (ablations).
        """
        b, s = batch["query.progress"].shape
        n = b * s
        input_ids = batch["observation.input_ids"].reshape(n, -1)
        attn = batch["observation.attention_mask"].reshape(n, -1).bool()
        grid = batch["observation.image_grid_thw"].reshape(-1, 3)
        image = self.embed_vlm(batch["observation.pixel_values"], grid, input_ids)  # [N, L, D]
        dtype = image.dtype
        bp = self.bp_tokens(batch) if bp is None else bp
        bp = bp.to(dtype).repeat_interleave(s, dim=0)  # [N, M, D]
        image = image + self.bp_type_embedding[0].to(dtype)
        bp = bp + self.bp_type_embedding[1].to(dtype)
        image_mask = attn if image_valid is None else attn & image_valid[:, None]
        bp_mask = torch.ones(n, bp.shape[1], dtype=torch.bool, device=bp.device)
        if bp_valid is not None:
            bp_mask = bp_mask & bp_valid[:, None]
        return torch.cat([image, bp], dim=1), torch.cat([image_mask, bp_mask], dim=1), attn

    def position_ids(self, batch: dict, attn: torch.Tensor, extra_tokens: int) -> torch.Tensor:
        b, s = batch["query.progress"].shape
        n = b * s
        return self.build_position_ids(
            batch["observation.input_ids"].reshape(n, -1),
            batch["observation.mm_token_type_ids"].reshape(n, -1),
            batch["observation.image_grid_thw"].reshape(-1, 3),
            attn,
            extra_tokens,
        )

    def forward_queries(
        self,
        batch: dict,
        *,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        bp: torch.Tensor | None = None,
        bp_valid: torch.Tensor | None = None,
        image_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, s = batch["query.progress"].shape
        n = b * s
        prefix, valid, attn = self.build_prefix(batch, bp=bp, bp_valid=bp_valid, image_valid=image_valid)
        m = prefix.shape[1] - attn.shape[1]
        horizon = batch[ACTION].shape[2]
        position_ids = self.position_ids(batch, attn, m + 1 + horizon)
        state = F.pad(batch[OBS_STATE].reshape(n, -1), (0, self.config.max_state_dim - batch[OBS_STATE].shape[-1]))
        actions = batch[ACTION].reshape(n, horizon, -1)
        actions = F.pad(actions, (0, self.config.max_action_dim - actions.shape[-1]))
        return self.core.forward_embedded(
            prefix, state, actions, noise=noise, time=time, position_ids=position_ids, vlm_valid_mask=valid
        )

    @torch.no_grad()
    def sample_queries(self, batch: dict, *, noise: torch.Tensor | None = None, num_steps: int | None = None):
        b, s = batch["query.progress"].shape
        n = b * s
        prefix, valid, attn = self.build_prefix(batch)
        m = prefix.shape[1] - attn.shape[1]
        position_ids = self.position_ids(batch, attn, m + 1 + self.config.chunk_size)
        state = F.pad(batch[OBS_STATE].reshape(n, -1), (0, self.config.max_state_dim - batch[OBS_STATE].shape[-1]))
        cache, action_mask, action_pos = self.core.prepare_inference_cache(
            prefix, state, vlm_valid_mask=valid, position_ids=position_ids
        )
        if noise is None:
            noise = torch.randn(n, self.config.chunk_size, self.config.max_action_dim, device=state.device,
                                dtype=state.dtype)
        if num_steps is None or num_steps == self.config.num_inference_steps:
            return self._sample_action_flow(noise, cache, action_mask, action_pos)
        actions, step = noise.clone(), -1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((n,), 1.0 - i / num_steps, device=noise.device, dtype=noise.dtype)
            actions = actions + step * self.core.denoise_cached(actions, t, cache, action_mask, action_pos)
        return actions


class BPV4XDemoPolicy(Q35Policy):
    config_class = BPV4XDemoConfig
    name = "bpv4_xdemo"

    def __init__(self, config: BPV4XDemoConfig) -> None:
        super(Q35Policy, self).__init__(config)
        config.validate_features()
        self.model = BPV4XDemoModel(config)
        config.qwen3_5_fingerprint = self.model.core.backbone.fingerprint.name
        self.reset()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def classify_model_loading_keys(self, missing_keys: list[str], unexpected_keys: list[str]):
        expected = [k for k in missing_keys if k.startswith(tuple(f"model.{p}" for p in NEW_MODULE_PREFIXES))]
        dropped = [k for k in unexpected_keys if k.startswith(DROPPED_STAGE1_PREFIXES)]
        if dropped:
            logging.info("bpv4_xdemo: ignoring %d stage-1 key(s) without a stage-2 counterpart (%s, ...)",
                         len(dropped), dropped[0])
        return ([k for k in missing_keys if k not in expected],
                [k for k in unexpected_keys if k not in dropped], expected)

    def get_optim_params(self):
        cfg = self.config
        tail = tuple(f"model.core.backbone.vlm.model.language_model.layers.{i}." for i in cfg.vlm_tail_layers)
        groups = {"bp_encoder": [], "new": [], "act": [], "vlm_tail": [], "vlm_rest": []}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(tuple(f"model.{x}" for x in BP_ENCODER_PREFIXES)):
                groups["bp_encoder"].append(p)
            elif name.startswith(tuple(f"model.{x}" for x in NEW_MODULE_PREFIXES)):
                groups["new"].append(p)
            elif name.startswith("model.core.backbone.vlm."):
                groups["vlm_tail" if name.startswith(tail) else "vlm_rest"].append(p)
            else:
                groups["act"].append(p)
        vlm_extra = {"lr_ramp_steps": int(cfg.vlm_ramp_steps)}
        return [
            {"name": "bp_encoder", "params": groups["bp_encoder"], "lr": float(cfg.bp_encoder_lr),
             "lr_zero_steps": int(cfg.bp_encoder_zero_steps), "lr_ramp_steps": int(cfg.bp_encoder_ramp_steps),
             "reset_state_at_unfreeze": bool(cfg.bp_encoder_reset_optimizer_state_at_unfreeze)},
            {"name": "new", "params": groups["new"], "lr": float(cfg.new_modules_lr)},
            {"name": "act", "params": groups["act"], "lr": float(cfg.act_expert_lr)},
            {"name": "vlm_tail", "params": groups["vlm_tail"], "lr": float(cfg.optimizer_lr) * float(cfg.vlm_tail_lr_mult),
             **vlm_extra},
            {"name": "vlm_rest", "params": groups["vlm_rest"], "lr": float(cfg.optimizer_lr) * float(cfg.vlm_lr_mult),
             **vlm_extra},
        ]

    def forward(self, batch, *, validation_mode: str | None = None):
        losses = self.model.forward_queries(batch)
        raw_dim = batch[ACTION].shape[-1]
        flow = losses["flow"][..., :raw_dim]  # [N, H, A]
        valid = batch[ACTION_VALID_MASK].reshape(flow.shape[0], -1)
        loss = masked_flow_loss(flow, action_valid_mask=valid)
        debug_check_finite("bpv4_xdemo.loss", loss)
        with torch.no_grad():
            s = batch["query.progress"].shape[1]
            w = valid[..., None].to(flow.dtype)
            per_query = (flow * w).sum((1, 2)) / (w.sum((1, 2)) * flow.shape[-1]).clamp_min(1)
            same_arm = batch["pair.same_arm"].bool().repeat_interleave(s)
            logs = {"loss_action": float(loss.detach())}
            for label, sel in (("same_arm", same_arm), ("diff_arm", ~same_arm)):
                logs[f"bp_loss_{label}"] = float(per_query[sel].mean()) if bool(sel.any()) else 0.0
            logs["bp_frac_same_arm"] = float(same_arm.float().mean())
            logs["bp_visible_kf"] = float(batch["bp.visible"].sum(-1).float().mean())
        return loss, logs

    def predict_action_chunk(self, batch, **kwargs):
        """batch laid out like the dataset (B pairs x S queries); returns [B*S, chunk, action_dim]."""
        self.eval()
        actions = self.model.sample_queries(batch, noise=kwargs.get("noise"), num_steps=kwargs.get("num_steps"))
        return actions[..., : self.config.output_features[ACTION].shape[0]]

    def select_action(self, batch, **kwargs):
        if not self._action_queue:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
