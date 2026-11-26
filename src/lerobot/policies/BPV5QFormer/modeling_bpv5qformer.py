"""BPv5 Stage A model (wall 4B VLM + Temporal Q-Former).

demo P5 cache [32,3,5,2560] -> Q-Former -> Z_demo [B,128,2560]
prefix (causal) = [bos] + demo_start + Z_demo + demo_end + text(48) + segment_query
suffix = state slot + noisy action (Q35 action expert, flow matching); expert reads segment_query only.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.BPV4CDX.modeling_bpv4cdx import (
    LORA_KEY,
    NEW_MODULE_PREFIXES,
    SegmentQuery,
    TemporalVideoQFormer,
    add_qwen_lora,
)
from lerobot.policies.q35.modeling_q35 import Q35Model, Q35Policy, debug_check_finite
from lerobot.policies.q35.token_layout import Q35TokenLayout
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_bpv5qformer import BPV5QFormerConfig

# per-dimension validity for 80-D actions (broadcast over batch / segments / horizon)
ACTION_DIM_MASK = "action.dim_mask"


def _flow_weights(
    flow: torch.Tensor,
    *,
    step_mask: torch.Tensor,
    dim_mask: torch.Tensor,
) -> torch.Tensor:
    if dim_mask.ndim == 1:
        dim_mask = dim_mask[None, :].expand(flow.shape[0], -1)
    return step_mask.to(flow.dtype)[..., None] * dim_mask.to(flow.dtype)[:, None, :]


def masked_flow_loss_dims(
    flow: torch.Tensor,
    *,
    step_mask: torch.Tensor,
    dim_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean MSE over valid steps and valid action dims. flow [N,H,A]; step_mask [N,H]; dim_mask [A] or [N,A]."""
    w = _flow_weights(flow, step_mask=step_mask, dim_mask=dim_mask)
    return (flow * w).sum() / w.sum().clamp_min(1)


def per_trajectory_flow_loss(
    flow: torch.Tensor,
    *,
    step_mask: torch.Tensor,
    dim_mask: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean flow MSE per trajectory, plus how many action dims are in that mask."""
    w = _flow_weights(flow, step_mask=step_mask, dim_mask=dim_mask)
    per_row = (flow * w).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp_min(1)
    if flow.shape[0] % batch_size:
        raise ValueError(f"flow rows {flow.shape[0]} not divisible by batch {batch_size}")
    segments = flow.shape[0] // batch_size
    per_traj = per_row.view(batch_size, segments).mean(dim=1)
    if dim_mask.ndim == 1:
        n_dims = dim_mask.to(flow.dtype).sum().expand(batch_size)
    else:
        n_dims = dim_mask.reshape(batch_size, segments, -1)[:, 0].to(flow.dtype).sum(dim=1)
    return per_traj, n_dims


def ae3_flow_loss(flow: torch.Tensor, *, step_mask: torch.Tensor) -> torch.Tensor:
    """AE-3 reduction: step mask only; denominator = n_valid_steps * A (pads dilute the mean)."""
    w = step_mask.to(flow.dtype)[..., None]
    denom = w.sum() * flow.shape[-1]
    return (flow * w).sum() / denom.clamp_min(1)


class BPV5QFormerModel(Q35Model):
    def __init__(self, config: BPV5QFormerConfig) -> None:
        super().__init__(config)
        hidden = int(self.core.backbone.vlm.config.text_config.hidden_size)
        self.demo_encoder = TemporalVideoQFormer(config, hidden) if config.demo_input == "video" else None
        self.segment_query = SegmentQuery(hidden)
        self.demo_start = nn.Parameter(torch.randn(hidden) * 0.02)
        self.demo_end = nn.Parameter(torch.randn(hidden) * 0.02)
        self.bos = nn.Parameter(torch.randn(hidden) * 0.02)
        for p in self.core.backbone.vlm.parameters():
            p.requires_grad_(False)
        lora_ids = list(getattr(config, "qwen_lora_layer_ids", []) or [])
        self.lora_modules = (
            add_qwen_lora(
                self.core.backbone.vlm,
                int(config.qwen_lora_layers),
                int(config.qwen_lora_rank),
                float(config.qwen_lora_alpha),
                layer_ids=lora_ids or None,
            )
            if (config.qwen_lora_layers or lora_ids)
            else []
        )

    def encode_demo(self, batch: dict) -> torch.Tensor | None:
        if self.demo_encoder is None:
            return None
        return self.demo_encoder(batch["demo.feats"], batch["demo.time"], batch["demo.camera_valid"])

    def build_prefix(self, batch: dict, z_demo: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Image-then-language (wall-aligned): bos, demo*, text, segment_query."""
        b, s = batch["seg.tau"].shape[:2]
        n = b * s
        vlm = self.core.backbone.vlm
        text = vlm.get_input_embeddings()(batch["text.input_ids"].reshape(n, -1))
        dtype, device = text.dtype, text.device
        parts: list[torch.Tensor] = []
        valid: list[torch.Tensor] = []
        if self.config.prefix_bos:
            parts.append(self.bos.to(dtype).expand(n, 1, -1))
            valid.append(torch.ones(n, 1, dtype=torch.bool, device=device))
        if z_demo is not None:
            z = z_demo.to(dtype).repeat_interleave(s, dim=0)
            parts += [
                self.demo_start.to(dtype).expand(n, 1, -1),
                z,
                self.demo_end.to(dtype).expand(n, 1, -1),
            ]
            valid.append(torch.ones(n, z.shape[1] + 2, dtype=torch.bool, device=device))
        parts.append(text)
        valid.append(batch["text.attention_mask"].reshape(n, -1).bool())
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
        if self.config.expert_context != "segment_query":
            return mask
        mask = mask.clone()
        mask[:, :, vlm_len:, : vlm_len - 1] = False
        return mask

    def forward_segments(
        self,
        batch: dict,
        z_demo: torch.Tensor | None = None,
        *,
        noise=None,
        time=None,
        use_demo: bool = True,
    ) -> dict[str, torch.Tensor]:
        core, cfg = self.core, self.config
        z = (z_demo if z_demo is not None else self.encode_demo(batch)) if use_demo else None
        prefix, valid = self.build_prefix(batch, z)
        n = prefix.shape[0]
        actions = batch[ACTION].reshape(n, batch[ACTION].shape[2], -1)
        actions = F.pad(actions, (0, cfg.max_action_dim - actions.shape[-1])).float()
        noise = torch.randn_like(actions) if noise is None else noise
        if time is None:
            beta = torch.distributions.Beta(
                torch.tensor(cfg.time_sampling_beta_alpha, device=actions.device),
                torch.tensor(cfg.time_sampling_beta_beta, device=actions.device),
            )
            time = beta.sample((n,)) * cfg.time_sampling_scale + cfg.time_sampling_offset
        t = time[:, None, None]
        noisy = t * noise + (1 - t) * actions
        flow_target = noise - actions
        state_hidden = core.state_projection(
            self.state_slot(batch, n, prefix.device, core.state_projection.weight.dtype)
        ).to(prefix.dtype)[:, None]
        action_hidden = core._action_inputs(noisy, time).to(prefix.dtype)
        layout = Q35TokenLayout.for_step(
            "robot", expert_type="action", vlm=prefix.shape[1], state=1, action=action_hidden.shape[1]
        )
        full_valid = torch.cat(
            [valid, torch.ones(n, 1 + action_hidden.shape[1], dtype=torch.bool, device=prefix.device)], 1
        )
        mask = layout.build_attention_mask(valid_mask=full_valid, causal_vlm=bool(cfg.causal_vlm))
        mask = self._restrict_expert(mask, prefix.shape[1])
        positions = torch.arange(layout.total_tokens, device=prefix.device)[None, None].expand(3, n, -1)
        _, _, action_out = core._forward_joint(
            prefix, state_hidden, action_hidden, attention_mask=mask, position_ids=positions
        )
        predicted = core.action_head(action_out.to(core.action_head.weight.dtype)).float()
        result = {"flow": F.mse_loss(predicted, flow_target.float(), reduction="none")}
        debug_check_finite("bpv5.flow", result)
        return result

    @torch.no_grad()
    def sample_segments(
        self,
        batch: dict,
        z_demo: torch.Tensor | None = None,
        *,
        noise=None,
        num_steps: int | None = None,
        use_demo: bool = True,
        blank_demo: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate action segments. ``blank_demo`` [B] True zeros Z_demo (G9-style blank condition)."""
        core, cfg = self.core, self.config
        z = (z_demo if z_demo is not None else self.encode_demo(batch)) if use_demo else None
        if z is not None and blank_demo is not None:
            keep = (~blank_demo.to(device=z.device)).to(dtype=z.dtype)[:, None, None]
            z = z * keep
        prefix, valid = self.build_prefix(batch, z)
        n, horizon = prefix.shape[0], int(cfg.chunk_size)
        state_hidden = core.state_projection(
            self.state_slot(batch, n, prefix.device, core.state_projection.weight.dtype)
        ).to(prefix.dtype)[:, None]
        layout = Q35TokenLayout.for_step(
            "robot", expert_type="action", vlm=prefix.shape[1], state=1, action=horizon
        )
        full_valid = torch.cat(
            [valid, torch.ones(n, 1 + horizon, dtype=torch.bool, device=prefix.device)], 1
        )
        mask = layout.build_attention_mask(valid_mask=full_valid, causal_vlm=bool(cfg.causal_vlm))
        mask = self._restrict_expert(mask, prefix.shape[1])
        positions = torch.arange(layout.total_tokens, device=prefix.device)[None, None].expand(3, n, -1)
        vlm_end = layout.span("vlm").stop
        _, _, cache = core.backbone.prepare_inference(
            prefix,
            state_hidden,
            attention_mask=mask[..., :vlm_end, :vlm_end],
            position_ids=positions[..., :vlm_end],
        )
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


class BPV5QFormerPolicy(Q35Policy):
    config_class = BPV5QFormerConfig
    name = "bpv5_qformer"

    def __init__(self, config: BPV5QFormerConfig) -> None:
        super(Q35Policy, self).__init__(config)
        config.validate_features()
        self.model = BPV5QFormerModel(config)
        config.qwen3_5_fingerprint = self.model.core.backbone.fingerprint.name
        self.reset()

    def classify_model_loading_keys(self, missing_keys: list[str], unexpected_keys: list[str]):
        expected = [
            k
            for k in missing_keys
            if k.startswith(tuple(f"model.{p}" for p in NEW_MODULE_PREFIXES)) or LORA_KEY in k
        ]
        return [k for k in missing_keys if k not in expected], list(unexpected_keys), expected

    def get_optim_params(self):
        cfg = self.config
        groups = {"cdx_new": [], "act": [], "lora": []}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if LORA_KEY in name:
                groups["lora"].append(p)
            elif name.startswith(tuple(f"model.{x}" for x in NEW_MODULE_PREFIXES)):
                groups["cdx_new"].append(p)
            else:
                groups["act"].append(p)
        out = [
            {"name": "cdx_new", "params": groups["cdx_new"], "lr": float(cfg.new_modules_lr)},
            {"name": "act", "params": groups["act"], "lr": float(cfg.act_expert_lr)},
        ]
        if groups["lora"]:
            out.append({"name": "lora", "params": groups["lora"], "lr": float(cfg.qwen_lora_lr)})
        return out

    def _resolve_dim_mask(self, batch, flow: torch.Tensor) -> torch.Tensor:
        raw_dim = flow.shape[-1]
        if ACTION_DIM_MASK in batch:
            dim_mask = batch[ACTION_DIM_MASK]
            if dim_mask.ndim == 1:
                dim_mask = dim_mask[None, :].expand(flow.shape[0], -1)
            elif dim_mask.ndim == 2:
                # [B, A] -> [B*S, A] (S segments per trajectory)
                b = int(dim_mask.shape[0])
                if flow.shape[0] % b:
                    raise ValueError(f"action.dim_mask batch {b} does not divide flow rows {flow.shape[0]}")
                dim_mask = dim_mask.repeat_interleave(flow.shape[0] // b, dim=0)
            else:
                raise ValueError(f"action.dim_mask must be [A] or [B,A], got {tuple(dim_mask.shape)}")
            return dim_mask
        return torch.ones(flow.shape[0], raw_dim, dtype=torch.bool, device=flow.device)

    def forward(self, batch, *, validation_mode: str | None = None):
        losses = self.model.forward_segments(batch)
        raw_dim = batch[ACTION].shape[-1]
        flow = losses["flow"][..., :raw_dim]
        step_mask = batch[ACTION_VALID_MASK].reshape(flow.shape[0], -1)
        dim_mask = self._resolve_dim_mask(batch, flow)
        mode = str(getattr(self.config, "action_dim_mask_mode", "stats") or "stats")
        if mode == "ae3":
            loss = ae3_flow_loss(flow, step_mask=step_mask)
        elif mode == "all":
            ones = torch.ones_like(dim_mask)
            loss = masked_flow_loss_dims(flow, step_mask=step_mask, dim_mask=ones)
        else:
            loss = masked_flow_loss_dims(flow, step_mask=step_mask, dim_mask=dim_mask)
        debug_check_finite("bpv5.loss", loss)
        with torch.no_grad():
            generic = batch["text.is_generic"].reshape(-1)
            dropped = (
                batch["text.is_dropped"].reshape(-1)
                if "text.is_dropped" in batch
                else torch.zeros_like(generic)
            )
            loss_stats = masked_flow_loss_dims(flow, step_mask=step_mask, dim_mask=dim_mask)
            loss_all = masked_flow_loss_dims(
                flow, step_mask=step_mask, dim_mask=torch.ones_like(dim_mask)
            )
            loss_ae3 = ae3_flow_loss(flow, step_mask=step_mask)
            # which masked dims dominate residual (mean over batch/horizon of flow MSE)
            w = step_mask.to(flow.dtype)[..., None] * dim_mask.to(flow.dtype)[:, None, :]
            per_dim = (flow * w).sum(dim=(0, 1)) / w.sum(dim=(0, 1)).clamp_min(1)
            top_v, top_i = torch.topk(per_dim, k=min(3, per_dim.numel()))
            logs = {
                "loss_action": float(loss.detach()),
                # prefixes loss_action_dim* / bp_* are forwarded into train logs
                "loss_action_dim_stats": float(loss_stats),
                "loss_action_dim_all": float(loss_all),
                "loss_action_dim_ae3": float(loss_ae3),
                "bp_dim_mask_frac": float(dim_mask.float().mean()),
                "bp_top_dim0": float(top_i[0]) if top_i.numel() else -1.0,
                "bp_top_dim0_mse": float(top_v[0]) if top_v.numel() else float("nan"),
                "bp_generic_frac": float(generic.float().mean()),
                "bp_text_drop_frac": float(dropped.float().mean()),
            }
            if self.training and getattr(self.config, "log_sample_loss", False) and "demo.repo" in batch:
                # Same reduction as the optimized loss, so the CSV explains loss.log.
                logged_mask = torch.ones_like(dim_mask) if mode in {"all", "ae3"} else dim_mask
                self._append_sample_loss(batch, flow, step_mask, logged_mask)
        return loss, logs

    def _append_sample_loss(self, batch, flow, step_mask, dim_mask) -> None:
        log_dir = os.environ.get("BPV5_SAMPLE_LOSS_DIR")
        if not log_dir:
            return
        batch_size = len(batch["demo.repo"])
        per_traj, n_dims = per_trajectory_flow_loss(
            flow.detach(), step_mask=step_mask, dim_mask=dim_mask, batch_size=batch_size
        )
        step = int(getattr(self, "_sample_log_step", 0)) + 1
        self._sample_log_step = step
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        path = Path(log_dir) / f"sample_loss_rank{rank}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        episodes = batch["demo.episode"].detach().cpu().tolist()
        losses = per_traj.detach().float().cpu().tolist()
        dims = n_dims.detach().float().cpu().tolist()
        with path.open("a", encoding="utf-8") as fh:
            if new:
                fh.write("step,rank,loss,repo,robot,episode,n_dims\n")
            for i, repo in enumerate(batch["demo.repo"]):
                fh.write(
                    f"{step},{rank},{losses[i]:.6f},{repo},{batch['demo.robot'][i]},{int(episodes[i])},{dims[i]:.0f}\n"
                )

    def predict_action_chunk(self, batch, **kwargs):
        raise NotImplementedError("bpv5_qformer reconstructs demo segments; use model.sample_segments")

    def select_action(self, batch, **kwargs):
        raise NotImplementedError("bpv5_qformer reconstructs demo segments; use model.sample_segments")
