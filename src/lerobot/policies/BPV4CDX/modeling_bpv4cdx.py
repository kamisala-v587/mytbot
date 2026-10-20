"""bpv4cdx Stage A-AE model.

demo video (32 slots x 3 cameras x P cached ViT tokens) -> Temporal Video Q-Former -> Z_demo [B, 128, hidden]
prefix  = text tokens + [demo_start] + Z_demo + [demo_end] + segment_query        (Qwen3.5, frozen)
suffix  = state slot + noisy action (q35 action expert, flow matching)
The state slot carries a constant zero (main branch) or the segment-start state (AE-S0). With
expert_context="segment_query" the suffix sees only the segment_query position of the prefix, so everything the
expert gets from the demo passes through that token (task book 5.6).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.BPV4BPEn.modeling_bpv4_bpen import _CrossBlock, sinusoidal_encoding
from lerobot.policies.q35.modeling_q35 import Q35Model, Q35Policy, debug_check_finite, masked_flow_loss
from lerobot.policies.q35.token_layout import Q35TokenLayout
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_bpv4cdx import BPV4CDXAEConfig

NEW_MODULE_PREFIXES = ("demo_encoder.", "segment_query.", "demo_start", "demo_end")


class TemporalVideoQFormer(nn.Module):
    """[B, K, C, P, Dv] slot features -> [B, N, out_dim] time-ordered tokens (N = K x per_slot)."""

    def __init__(self, cfg: BPV4CDXAEConfig, out_dim: int) -> None:
        super().__init__()
        d = int(cfg.qformer_dim)
        self.dim = d
        self.per_slot = int(cfg.num_demo_tokens) // int(cfg.num_slots)
        self.input_proj = nn.Linear(int(cfg.visual_dim), d)
        self.camera_embedding = nn.Parameter(torch.zeros(int(cfg.num_cameras), d))
        self.spatial_embedding = nn.Parameter(torch.zeros(int(cfg.spatial_tokens), d))
        self.valid_embedding = nn.Parameter(torch.zeros(2, d))
        self.time_proj = nn.Linear(d, d)
        self.queries = nn.Parameter(torch.randn(self.per_slot, d) * 0.02)
        self.query_time_proj = nn.Linear(d, d)
        self.layers = nn.ModuleList(
            _CrossBlock(d, int(cfg.qformer_num_heads), int(cfg.qformer_ff_mult)) for _ in range(int(cfg.qformer_num_layers))
        )
        self.output_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, out_dim))

    def forward(self, feats: torch.Tensor, time: torch.Tensor, camera_valid: torch.Tensor) -> torch.Tensor:
        b, k, c, p, _ = feats.shape
        dtype = self.input_proj.weight.dtype
        t_emb = self.time_proj(sinusoidal_encoding(time, self.dim).to(dtype))  # [B, K, d]
        x = (
            self.input_proj(feats.to(dtype))
            + self.camera_embedding[None, None, :, None]
            + self.spatial_embedding[None, None, None, :]
            + self.valid_embedding[camera_valid.long()][:, :, :, None]
            + t_emb[:, :, None, None]
        ).reshape(b, k * c * p, -1)
        padding = (~camera_valid)[:, :, :, None].expand(-1, -1, -1, p).reshape(b, k * c * p)
        q_time = self.query_time_proj(sinusoidal_encoding(time, self.dim).to(dtype))  # [B, K, d]
        q = (self.queries[None, None] + q_time[:, :, None]).reshape(b, k * self.per_slot, -1)
        for layer in self.layers:
            q = layer(q, x, kv_padding=padding)
        return self.output_proj(q)


class SegmentQuery(nn.Module):
    """[tau_start, tau_end, steps / H_max] -> one prefix token."""

    def __init__(self, hidden: int, enc_dim: int = 256) -> None:
        super().__init__()
        self.enc_dim = enc_dim
        self.mlp = nn.Sequential(nn.Linear(2 * enc_dim + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, tau: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        dtype = self.mlp[0].weight.dtype
        x = torch.cat([sinusoidal_encoding(tau[:, 0], self.enc_dim), sinusoidal_encoding(tau[:, 1], self.enc_dim),
                       steps[:, None].float()], dim=-1)
        return self.mlp(x.to(dtype))[:, None]


class BPV4CDXAEModel(Q35Model):
    def __init__(self, config: BPV4CDXAEConfig) -> None:
        super().__init__(config)
        hidden = int(self.core.backbone.vlm.config.text_config.hidden_size)
        self.demo_encoder = TemporalVideoQFormer(config, hidden) if config.demo_input == "video" else None
        self.segment_query = SegmentQuery(hidden)
        self.demo_start = nn.Parameter(torch.randn(hidden) * 0.02)
        self.demo_end = nn.Parameter(torch.randn(hidden) * 0.02)
        # Qwen (language model, ViT, LM head) is frozen; gradients still flow through it to the new modules.
        for p in self.core.backbone.vlm.parameters():
            p.requires_grad_(False)

    # --- prefix -------------------------------------------------------------------------------------------------
    def encode_demo(self, batch: dict) -> torch.Tensor | None:
        """[B, N, hidden] Z_demo, or None for the text-only control."""
        if self.demo_encoder is None:
            return None
        return self.demo_encoder(batch["demo.feats"], batch["demo.time"], batch["demo.camera_valid"])

    def build_prefix(self, batch: dict, z_demo: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """-> prefix [B*S, L, hidden], valid [B*S, L]; the segment_query token is the last prefix position."""
        b, s = batch["seg.tau"].shape[:2]
        n = b * s
        vlm = self.core.backbone.vlm
        text = vlm.get_input_embeddings()(batch["text.input_ids"].reshape(n, -1))
        dtype, device = text.dtype, text.device
        parts = [text]
        valid = [batch["text.attention_mask"].reshape(n, -1).bool()]
        if z_demo is not None:
            z = z_demo.to(dtype).repeat_interleave(s, dim=0)
            parts += [self.demo_start.to(dtype).expand(n, 1, -1), z, self.demo_end.to(dtype).expand(n, 1, -1)]
            valid.append(torch.ones(n, z.shape[1] + 2, dtype=torch.bool, device=device))
        segq = self.segment_query(batch["seg.tau"].reshape(n, 2), batch["seg.steps"].reshape(n)).to(dtype)
        parts.append(segq)
        valid.append(torch.ones(n, 1, dtype=torch.bool, device=device))
        return torch.cat(parts, dim=1), torch.cat(valid, dim=1)

    def state_slot(self, batch: dict, n: int, device, dtype) -> torch.Tensor:
        dim = int(self.config.max_state_dim)
        if not self.config.use_segment_start_state:
            return torch.zeros(n, dim, device=device, dtype=dtype)
        state = batch[OBS_STATE].reshape(n, -1)
        state = F.pad(state, (0, dim - state.shape[-1])).to(dtype)
        if self.training:
            keep = (torch.rand(n, device=device) >= float(self.config.segment_start_state_mask_prob)).to(dtype)
            state = state * keep[:, None]
        return state

    def _restrict_expert(self, mask: torch.Tensor, vlm_len: int) -> torch.Tensor:
        """Suffix rows may see only the last prefix position (segment_query) when expert_context=segment_query."""
        if self.config.expert_context != "segment_query":
            return mask
        mask = mask.clone()
        mask[:, :, vlm_len:, : vlm_len - 1] = False
        return mask

    # --- training -----------------------------------------------------------------------------------------------
    def forward_segments(self, batch: dict, z_demo: torch.Tensor | None = None, *, noise=None, time=None,
                         use_demo: bool = True) -> dict[str, torch.Tensor]:
        core, cfg = self.core, self.config
        z = (z_demo if z_demo is not None else self.encode_demo(batch)) if use_demo else None
        prefix, valid = self.build_prefix(batch, z)
        n = prefix.shape[0]
        actions = batch[ACTION].reshape(n, batch[ACTION].shape[2], -1)
        actions = F.pad(actions, (0, cfg.max_action_dim - actions.shape[-1])).float()
        noise = torch.randn_like(actions) if noise is None else noise
        if time is None:
            beta = torch.distributions.Beta(torch.tensor(cfg.time_sampling_beta_alpha, device=actions.device),
                                            torch.tensor(cfg.time_sampling_beta_beta, device=actions.device))
            time = beta.sample((n,)) * cfg.time_sampling_scale + cfg.time_sampling_offset
        t = time[:, None, None]
        noisy = t * noise + (1 - t) * actions
        flow_target = noise - actions
        state_hidden = core.state_projection(
            self.state_slot(batch, n, prefix.device, core.state_projection.weight.dtype)).to(prefix.dtype)[:, None]
        action_hidden = core._action_inputs(noisy, time).to(prefix.dtype)
        layout = Q35TokenLayout.for_step("robot", expert_type="action", vlm=prefix.shape[1], state=1,
                                         action=action_hidden.shape[1])
        full_valid = torch.cat([valid, torch.ones(n, 1 + action_hidden.shape[1], dtype=torch.bool, device=prefix.device)], 1)
        mask = layout.build_attention_mask(valid_mask=full_valid, causal_vlm=getattr(cfg, "causal_vlm", False))
        mask = self._restrict_expert(mask, prefix.shape[1])
        positions = torch.arange(layout.total_tokens, device=prefix.device)[None, None].expand(3, n, -1)
        _, _, action_out = core._forward_joint(prefix, state_hidden, action_hidden, attention_mask=mask,
                                               position_ids=positions)
        predicted = core.action_head(action_out.to(core.action_head.weight.dtype)).float()
        result = {"flow": F.mse_loss(predicted, flow_target.float(), reduction="none")}
        debug_check_finite("bpv4cdx.flow", result)
        return result

    # --- inference ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def sample_segments(self, batch: dict, z_demo: torch.Tensor | None = None, *, noise=None, num_steps: int | None = None,
                        use_demo: bool = True) -> torch.Tensor:
        """Flow sampling of the segment actions -> [B*S, H, max_action_dim] (normalized delta space)."""
        core, cfg = self.core, self.config
        z = (z_demo if z_demo is not None else self.encode_demo(batch)) if use_demo else None
        prefix, valid = self.build_prefix(batch, z)
        n, horizon = prefix.shape[0], int(cfg.chunk_size)
        state_hidden = core.state_projection(
            self.state_slot(batch, n, prefix.device, core.state_projection.weight.dtype)).to(prefix.dtype)[:, None]
        layout = Q35TokenLayout.for_step("robot", expert_type="action", vlm=prefix.shape[1], state=1, action=horizon)
        full_valid = torch.cat([valid, torch.ones(n, 1 + horizon, dtype=torch.bool, device=prefix.device)], 1)
        mask = layout.build_attention_mask(valid_mask=full_valid, causal_vlm=getattr(cfg, "causal_vlm", False))
        mask = self._restrict_expert(mask, prefix.shape[1])
        positions = torch.arange(layout.total_tokens, device=prefix.device)[None, None].expand(3, n, -1)
        vlm_end = layout.span("vlm").stop
        _, _, cache = core.backbone.prepare_inference(prefix, state_hidden, attention_mask=mask[..., :vlm_end, :vlm_end],
                                                      position_ids=positions[..., :vlm_end])
        suffix = slice(layout.span("state").start, layout.total_tokens)
        act_mask, act_pos = mask[..., suffix, :], positions[..., suffix]
        if noise is None:
            noise = torch.randn(n, horizon, int(cfg.max_action_dim), device=prefix.device)
        actions = noise.clone().float()
        steps = int(num_steps or cfg.num_inference_steps)
        for step in range(steps):
            time = torch.full((n,), 1.0 - step / steps, device=prefix.device)
            actions = actions - core.denoise_cached(actions, time, cache, act_mask, act_pos).float() / steps
        return actions


class BPV4CDXAEPolicy(Q35Policy):
    config_class = BPV4CDXAEConfig
    name = "bpv4cdx_ae"

    def __init__(self, config: BPV4CDXAEConfig) -> None:
        super(Q35Policy, self).__init__(config)
        config.validate_features()
        self.model = BPV4CDXAEModel(config)
        config.qwen3_5_fingerprint = self.model.core.backbone.fingerprint.name
        self.reset()

    def classify_model_loading_keys(self, missing_keys: list[str], unexpected_keys: list[str]):
        expected = [k for k in missing_keys if k.startswith(tuple(f"model.{p}" for p in NEW_MODULE_PREFIXES))]
        return [k for k in missing_keys if k not in expected], list(unexpected_keys), expected

    def get_optim_params(self):
        cfg = self.config
        groups = {"cdx_new": [], "act": []}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            groups["cdx_new" if name.startswith(tuple(f"model.{x}" for x in NEW_MODULE_PREFIXES)) else "act"].append(p)
        return [
            {"name": "cdx_new", "params": groups["cdx_new"], "lr": float(cfg.new_modules_lr)},
            {"name": "act", "params": groups["act"], "lr": float(cfg.act_expert_lr)},
        ]

    def forward(self, batch, *, validation_mode: str | None = None):
        losses = self.model.forward_segments(batch)
        raw_dim = batch[ACTION].shape[-1]
        flow = losses["flow"][..., :raw_dim]
        valid = batch[ACTION_VALID_MASK].reshape(flow.shape[0], -1)
        loss = masked_flow_loss(flow, action_valid_mask=valid)
        debug_check_finite("bpv4cdx.loss", loss)
        with torch.no_grad():
            w = valid[..., None].to(flow.dtype)
            per_seg = (flow * w).sum((1, 2)) / (w.sum((1, 2)) * flow.shape[-1]).clamp_min(1)
            generic = batch["text.is_generic"].reshape(-1)
            logs = {"loss_action": float(loss.detach()), "bp_generic_frac": float(generic.float().mean())}
            for sel, label in ((generic, "generic"), (~generic, "canonical")):
                logs[f"bp_loss_{label}"] = float(per_seg[sel].mean()) if bool(sel.any()) else 0.0
        return loss, logs

    def predict_action_chunk(self, batch, **kwargs):
        raise NotImplementedError("bpv4cdx_ae reconstructs demo segments; use model.sample_segments")

    def select_action(self, batch, **kwargs):
        raise NotImplementedError("bpv4cdx_ae reconstructs demo segments; use model.sample_segments")
