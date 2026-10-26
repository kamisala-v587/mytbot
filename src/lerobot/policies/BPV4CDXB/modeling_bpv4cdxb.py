"""bpv4cdx Stage B model (`bpv4cdx Stage B 开发文档 v1.md`, section 2).

prefix = [demo_start] + Z_demo + [demo_end] + [obs_start] + 3 x (<vision_start> + 64 image tokens + <vision_end>)
         + [obs_end]; everything goes straight into the frozen Qwen (bidirectional prefix). demo_start / demo_end come
         from the frozen Stage A-AE delivery; Z_demo is precomputed by the frozen encoder (dataset key `demo.z`).
suffix = current state slot + 50 noisy action tokens (q35 Action Expert, randomly initialised).
The dataset lays out `obs.input_ids` with plain-text placeholders at the demo / marker positions, so Qwen's rope index
gives them sequential positions and the images their native M-RoPE positions; the placeholders' embeddings are then
replaced here.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.q35.modeling_q35 import Q35Model, Q35Policy, debug_check_finite, masked_flow_loss
from lerobot.policies.q35.token_layout import Q35TokenLayout
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_bpv4cdxb import BPV4CDXBConfig

NEW_MODULE_PREFIXES = ("obs_start", "obs_end")
FROZEN_BUFFERS = ("demo_start", "demo_end")


def prefix_spans(config: BPV4CDXBConfig, total: int) -> dict[str, slice]:
    """Positions of the prefix parts for a prefix of length `total` (fixed layout, 3 valid cameras)."""
    demo = 2 + int(config.num_demo_tokens) if config.demo_input == "video" else 0
    marker = 1 if config.obs_marker_tokens else 0
    return {"demo": slice(0, demo), "obs_start": slice(demo, demo + marker),
            "images": slice(demo + marker, total - marker), "obs_end": slice(total - marker, total)}


class BPV4CDXBModel(Q35Model):
    def __init__(self, config: BPV4CDXBConfig) -> None:
        super().__init__(config)
        hidden = int(self.core.backbone.vlm.config.text_config.hidden_size)
        for p in self.core.backbone.vlm.parameters():
            p.requires_grad_(False)
        # frozen demo boundary tokens of the Stage A-AE delivery (filled by load_frozen_tokens; saved with the model)
        self.register_buffer("demo_start", torch.zeros(hidden))
        self.register_buffer("demo_end", torch.zeros(hidden))
        if config.obs_marker_tokens:
            self.obs_start = nn.Parameter(torch.randn(hidden) * 0.02)
            self.obs_end = nn.Parameter(torch.randn(hidden) * 0.02)
        if config.frozen_encoder:
            self.load_frozen_tokens(config.frozen_encoder)

    def load_frozen_tokens(self, frozen_dir: str) -> None:
        from .frozen import load_frozen

        frozen = load_frozen(frozen_dir)
        self.demo_start.copy_(frozen.demo_start.float())
        self.demo_end.copy_(frozen.demo_end.float())
        self.frozen_sha256 = frozen.ckpt_sha256

    # --- prefix -------------------------------------------------------------------------------------------------
    def build_prefix(self, batch: dict, *, blank_demo: torch.Tensor | None = None):
        """-> prefix [N, L, D], input ids [N, L], valid [N, L]; N = B x S (pairs x query time steps)."""
        cfg = self.config
        ids = batch["obs.input_ids"]
        b, s = ids.shape[:2]
        n = b * s
        ids = ids.reshape(n, -1)
        emb = self.embed_vlm(batch["obs.pixel_values"], batch["obs.image_grid_thw"].reshape(-1, 3), ids)
        dtype = emb.dtype
        sp = prefix_spans(cfg, ids.shape[1])
        parts = []
        if cfg.demo_input == "video":
            z = batch["demo.z"].to(dtype)  # [B, M, D]
            if blank_demo is not None:
                z = z * (~blank_demo.to(z.device))[:, None, None].to(dtype)
            z = z.repeat_interleave(s, dim=0)
            parts += [self.demo_start.to(dtype).expand(n, 1, -1), z, self.demo_end.to(dtype).expand(n, 1, -1)]
        if cfg.obs_marker_tokens:
            parts.append(self.obs_start.to(dtype).expand(n, 1, -1))
        parts.append(emb[:, sp["images"]])
        if cfg.obs_marker_tokens:
            parts.append(self.obs_end.to(dtype).expand(n, 1, -1))
        prefix = torch.cat(parts, dim=1)
        if prefix.shape[1] != ids.shape[1]:
            raise ValueError(f"prefix length {prefix.shape[1]} != input_ids length {ids.shape[1]}")
        valid = batch["obs.attention_mask"].reshape(n, -1).bool()
        return prefix, ids, valid

    def _positions(self, batch: dict, ids: torch.Tensor, valid: torch.Tensor, extra: int) -> torch.Tensor:
        n = ids.shape[0]
        return self.build_position_ids(ids, batch["obs.mm_token_type_ids"].reshape(n, -1),
                                       batch["obs.image_grid_thw"].reshape(-1, 3), valid, extra)

    def _mask(self, valid: torch.Tensor, horizon: int) -> torch.Tensor:
        n, length = valid.shape
        layout = Q35TokenLayout.for_step("robot", expert_type="action", vlm=length, state=1, action=horizon)
        full = torch.cat([valid, torch.ones(n, 1 + horizon, dtype=torch.bool, device=valid.device)], 1)
        mask = layout.build_attention_mask(valid_mask=full, causal_vlm=getattr(self.config, "causal_vlm", False))
        if self.config.expert_visibility == "obs_only":
            images = prefix_spans(self.config, length)["images"]
            mask = mask.clone()
            mask[:, :, length:, : images.start] = False
            mask[:, :, length:, images.stop : length] = False
        return mask

    def _state(self, batch: dict, n: int, device, dtype) -> torch.Tensor:
        cfg = self.config
        if not cfg.use_state:
            return torch.zeros(n, cfg.max_state_dim, device=device, dtype=dtype)
        state = batch[OBS_STATE].reshape(n, -1)
        state = F.pad(state, (0, cfg.max_state_dim - state.shape[-1])).to(dtype)
        if self.training and cfg.state_drop_prob > 0:
            keep = (torch.rand(n, device=device) >= float(cfg.state_drop_prob)).to(dtype)
            state = state * keep[:, None]
        return state

    # --- training -----------------------------------------------------------------------------------------------
    def forward_queries(self, batch: dict, *, noise=None, time=None, blank_demo=None) -> dict[str, torch.Tensor]:
        core, cfg = self.core, self.config
        prefix, ids, valid = self.build_prefix(batch, blank_demo=blank_demo)
        n = prefix.shape[0]
        actions = batch[ACTION].reshape(n, batch[ACTION].shape[-2], -1)
        horizon = actions.shape[1]
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
            self._state(batch, n, prefix.device, core.state_projection.weight.dtype)).to(prefix.dtype)[:, None]
        action_hidden = core._action_inputs(noisy, time).to(prefix.dtype)
        mask = self._mask(valid, horizon)
        positions = self._positions(batch, ids, valid, 1 + horizon)
        _, _, action_out = core._forward_joint(prefix, state_hidden, action_hidden, attention_mask=mask,
                                               position_ids=positions)
        predicted = core.action_head(action_out.to(core.action_head.weight.dtype)).float()
        result = {"flow": F.mse_loss(predicted, flow_target.float(), reduction="none")}
        debug_check_finite("bpv4cdxb.flow", result)
        return result

    # --- inference ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(self, batch: dict, *, noise=None, num_steps: int | None = None, blank_demo=None) -> torch.Tensor:
        """Flow sampling -> [N, H, max_action_dim] (normalized delta space)."""
        core, cfg = self.core, self.config
        prefix, ids, valid = self.build_prefix(batch, blank_demo=blank_demo)
        n, horizon = prefix.shape[0], int(cfg.chunk_size)
        state_hidden = core.state_projection(
            self._state(batch, n, prefix.device, core.state_projection.weight.dtype)).to(prefix.dtype)[:, None]
        mask = self._mask(valid, horizon)
        positions = self._positions(batch, ids, valid, 1 + horizon)
        vlm_end = prefix.shape[1]
        _, _, cache = core.backbone.prepare_inference(prefix, state_hidden, attention_mask=mask[..., :vlm_end, :vlm_end],
                                                      position_ids=positions[..., :vlm_end])
        act_mask, act_pos = mask[..., vlm_end:, :], positions[..., vlm_end:]
        if noise is None:
            noise = torch.randn(n, horizon, int(cfg.max_action_dim), device=prefix.device)
        actions = noise.clone().float()
        steps = int(num_steps or cfg.num_inference_steps)
        for step in range(steps):
            time = torch.full((n,), 1.0 - step / steps, device=prefix.device)
            actions = actions - core.denoise_cached(actions, time, cache, act_mask, act_pos).float() / steps
        return actions


class BPV4CDXBPolicy(Q35Policy):
    config_class = BPV4CDXBConfig
    name = "bpv4cdx_b"

    def __init__(self, config: BPV4CDXBConfig) -> None:
        super(Q35Policy, self).__init__(config)
        config.validate_features()
        self.model = BPV4CDXBModel(config)
        config.qwen3_5_fingerprint = self.model.core.backbone.fingerprint.name
        self.reset()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        cfg = self.config
        groups = {"b_new": [], "act": []}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            groups["b_new" if name.startswith(tuple(f"model.{x}" for x in NEW_MODULE_PREFIXES)) else "act"].append(p)
        return [{"name": "b_new", "params": groups["b_new"], "lr": float(cfg.new_modules_lr)},
                {"name": "act", "params": groups["act"], "lr": float(cfg.act_expert_lr)}]

    def forward(self, batch, *, validation_mode: str | None = None):
        losses = self.model.forward_queries(batch)
        raw_dim = batch[ACTION].shape[-1]
        flow = losses["flow"][..., :raw_dim]
        valid = batch[ACTION_VALID_MASK].reshape(flow.shape[0], -1)
        loss = masked_flow_loss(flow, action_valid_mask=valid)
        debug_check_finite("bpv4cdxb.loss", loss)
        logs = {"loss_action": float(loss.detach())}
        if "pair.contrast" in batch:
            logs["b_contrast_frac"] = float(batch["pair.contrast"].float().mean())
        return loss, logs

    def predict_action_chunk(self, batch, **kwargs):
        """batch laid out like the dataset (B pairs x S query steps) -> [B*S, chunk, action_dim]."""
        self.eval()
        actions = self.model.sample_actions(batch, noise=kwargs.get("noise"), num_steps=kwargs.get("num_steps"),
                                            blank_demo=kwargs.get("blank_demo"))
        return actions[..., : self.config.output_features[ACTION].shape[0]]

    def select_action(self, batch, **kwargs):
        if not self._action_queue:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
