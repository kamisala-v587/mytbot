"""BPv5 Stage B model.

Loads Stage A's VLM base, LoRA (layers 0–7), Q-Former and demo boundary tokens.
The Action Expert is the wider Q35 expert (ratio 0.4) and is randomly initialised.
History frames are not part of this version.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.BPV4CDX.modeling_bpv4cdx import LORA_KEY, TemporalVideoQFormer, add_qwen_lora
from lerobot.policies.BPV5QFormer.modeling_bpv5qformer import ae3_flow_loss, masked_flow_loss_dims
from lerobot.policies.BPV5StageB.configuration_bpv5stageb import (
    BPV5StageBConfig,
    phase_trainable,
    stage_a_keep_key,
)
from lerobot.policies.q35.modeling_q35 import Q35Model, Q35Policy, debug_check_finite
from lerobot.policies.q35.token_layout import Q35TokenLayout
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION, OBS_STATE

ACTION_DIM_MASK = "action.dim_mask"
LABEL_DEMO = "The above is a demonstration."
BRIDGE = "Now perform the current task."
NEW_MODULE_PREFIXES = ("obs_start", "obs_end")
_LORA_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _lora_layer_id(param_name: str) -> int | None:
    m = _LORA_LAYER_RE.search(param_name)
    return int(m.group(1)) if m else None


def _checkpoint_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    for candidate in (path / "model.safetensors", path / "pretrained_model" / "model.safetensors"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no model.safetensors under {path}")


def load_stage_a_weights(module: nn.Module, path: str | Path) -> dict[str, int]:
    """Copy VLM, LoRA, Q-Former and demo boundary tokens. Skip the Action Expert."""
    from safetensors.torch import load_file

    tensors = load_file(str(_checkpoint_file(path)))
    own = dict(module.named_parameters())
    copied = skipped = shape = 0
    with torch.no_grad():
        for key, value in tensors.items():
            if not stage_a_keep_key(key):
                skipped += 1
                continue
            target = key if key in own else key.split("model.", 1)[-1]
            if target not in own:
                skipped += 1
                continue
            if tuple(own[target].shape) != tuple(value.shape):
                shape += 1
                continue
            own[target].copy_(value.to(dtype=own[target].dtype, device=own[target].device))
            copied += 1
    logging.info("stage B loaded Stage A weights: copied=%d skipped=%d shape_mismatch=%d", copied, skipped, shape)
    if copied == 0:
        raise RuntimeError(f"Stage A checkpoint {path} matched no Stage B tensors")
    return {"copied": copied, "skipped": skipped, "shape_mismatch": shape}


class BPV5StageBModel(Q35Model):
    def __init__(self, config: BPV5StageBConfig) -> None:
        super().__init__(config)
        hidden = int(self.core.backbone.vlm.config.text_config.hidden_size)
        vlm = self.core.backbone.vlm
        for p in vlm.parameters():
            p.requires_grad_(False)
        self.demo_encoder = TemporalVideoQFormer(config, hidden)
        self.demo_start = nn.Parameter(torch.randn(hidden) * 0.02)
        self.demo_end = nn.Parameter(torch.randn(hidden) * 0.02)
        self.bos = nn.Parameter(torch.randn(hidden) * 0.02)
        self.obs_start = nn.Parameter(torch.randn(hidden) * 0.02)
        self.obs_end = nn.Parameter(torch.randn(hidden) * 0.02)
        lora_ids = list(config.qwen_lora_layer_ids or [])
        self.lora_modules = add_qwen_lora(
            vlm,
            int(config.qwen_lora_layers),
            int(config.qwen_lora_rank),
            float(config.qwen_lora_alpha),
            layer_ids=lora_ids or None,
        )
        self._pack_fixed_text(config)
        if config.stage_a_checkpoint:
            load_stage_a_weights(self, config.stage_a_checkpoint)
        self.apply_phase()

    def _pack_fixed_text(self, config: BPV5StageBConfig) -> None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(config.qwen3_5_processor_path or config.qwen3_5_pretrained_path)
        for name, text in (("label_demo", LABEL_DEMO), ("bridge", BRIDGE)):
            out = tok(text, add_special_tokens=False, return_tensors="pt")
            self.register_buffer(f"{name}_ids", out["input_ids"][0].long(), persistent=False)

    def apply_phase(self) -> None:
        flags = phase_trainable(self.config.phase)
        vlm = self.core.backbone.vlm
        # B-2/B-3: all installed LoRA. B-1: only lora_train_layer_ids (empty => freeze all).
        train_ids = {int(i) for i in (self.config.lora_train_layer_ids or [])}
        for name, param in vlm.named_parameters():
            if LORA_KEY not in name:
                param.requires_grad_(False)
                continue
            if flags["lora"]:
                param.requires_grad_(True)
                continue
            layer = _lora_layer_id(name)
            param.requires_grad_(layer is not None and layer in train_ids)
        for param in self.demo_encoder.parameters():
            param.requires_grad_(flags["qformer"])
        self.bos.requires_grad_(False)
        self.demo_start.requires_grad_(False)
        self.demo_end.requires_grad_(False)
        obs_on = bool(flags.get("obs_markers", True))
        self.obs_start.requires_grad_(obs_on)
        self.obs_end.requires_grad_(obs_on)
        if self.core.backbone.vlm_visual is not None:
            self.core.backbone.vlm_visual.requires_grad_(False)
        n_lora = sum(p.numel() for n, p in vlm.named_parameters() if p.requires_grad and LORA_KEY in n)
        # Log once per process — forward may re-call apply_phase every step.
        key = (self.config.phase, tuple(sorted(train_ids)), n_lora, obs_on)
        if getattr(self, "_apply_phase_logged", None) != key:
            self._apply_phase_logged = key
            logging.info(
                "stage B apply_phase=%s lora_flag=%s lora_train_layer_ids=%s lora_trainable_params=%d obs_markers=%s",
                self.config.phase,
                flags["lora"],
                sorted(train_ids),
                n_lora,
                obs_on,
            )

    def _text(self, ids: torch.Tensor) -> torch.Tensor:
        return self.core.backbone.vlm.get_input_embeddings()(ids)

    def _fixed(self, name: str, n: int, dtype: torch.dtype) -> torch.Tensor:
        ids = getattr(self, f"{name}_ids").to(device=self.bos.device)
        emb = self._text(ids[None].expand(n, -1))
        return emb.to(dtype)

    def _vision(self, batch: dict, n: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-camera tokens [n, 3, T, H], ids [n, 3, T], and the grid for those images."""
        pixels = batch["obs.pixel_values"]
        grid = batch["obs.image_grid_thw"].reshape(-1, 3)
        vlm = self.core.backbone.vlm
        merge = int(self.core.backbone.vlm_visual.spatial_merge_size)
        counts = [int(v) for v in (grid.prod(-1) // merge**2).tolist()]
        if len(set(counts)) != 1:
            raise ValueError(f"current-image token counts differ across the batch: {sorted(set(counts))}")
        tok = counts[0]
        ids = torch.full(
            (1, sum(counts)),
            int(vlm.config.image_token_id),
            dtype=torch.long,
            device=pixels.device,
        )
        flat = self.embed_vlm(pixels, grid, ids)[0]
        images = flat.view(-1, tok, flat.shape[-1])
        b = batch["obs.camera_valid"].shape[0]
        s = n // b
        images = images.view(b, s, 3, tok, -1).reshape(n, 3, tok, -1).to(dtype)
        image_ids = torch.full((n, 3, tok), int(vlm.config.image_token_id), dtype=torch.long, device=pixels.device)
        return images, image_ids, grid

    def build_prefix(
        self,
        batch: dict,
        z_demo: torch.Tensor | None,
        *,
        blank_demo: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """prefix, valid, index of obs_start, rope input ids, image grid."""
        b, s = batch["demo_text.input_ids"].shape[:2]
        n = b * s
        vlm = self.core.backbone.vlm
        embed = vlm.get_input_embeddings()
        demo_ids = batch["demo_text.input_ids"].reshape(n, -1)
        cur_ids = batch["current_text.input_ids"].reshape(n, -1)
        demo_txt = embed(demo_ids)
        cur_txt = embed(cur_ids)
        dtype, device = demo_txt.dtype, demo_txt.device
        if z_demo is None:
            raise RuntimeError("Stage B requires Z_demo")
        if blank_demo is not None:
            z_demo = z_demo * (~blank_demo.to(device=z_demo.device)).to(dtype=z_demo.dtype)[:, None, None]
        z = z_demo.to(dtype).repeat_interleave(s, dim=0)
        images, image_ids, grid = self._vision(batch, n, dtype)
        weight = embed.weight
        v_start = weight[int(vlm.config.vision_start_token_id)].to(dtype)
        v_end = weight[int(vlm.config.vision_end_token_id)].to(dtype)
        label = self._fixed("label_demo", n, dtype)
        bridge = self._fixed("bridge", n, dtype)
        pad_id = int(getattr(vlm.config, "pad_token_id", 0) or 0)

        def marker(param: nn.Parameter, token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
            return param.to(dtype).expand(n, 1, -1), torch.full((n, 1), token_id, dtype=torch.long, device=device)

        parts: list[torch.Tensor] = []
        ids: list[torch.Tensor] = []
        valid: list[torch.Tensor] = []
        types: list[torch.Tensor] = []

        def add(emb: torch.Tensor, tok: torch.Tensor, ok: torch.Tensor, mm: int) -> None:
            parts.append(emb)
            ids.append(tok)
            valid.append(ok)
            types.append(torch.full(tok.shape, mm, dtype=torch.long, device=device))

        bos, bos_id = marker(self.bos, pad_id)
        add(bos, bos_id, torch.ones(n, 1, dtype=torch.bool, device=device), 0)
        ds, ds_id = marker(self.demo_start, pad_id)
        add(ds, ds_id, torch.ones(n, 1, dtype=torch.bool, device=device), 0)
        add(z, torch.full((n, z.shape[1]), pad_id, dtype=torch.long, device=device), torch.ones(n, z.shape[1], dtype=torch.bool, device=device), 0)
        add(demo_txt, demo_ids, batch["demo_text.attention_mask"].reshape(n, -1).bool(), 0)
        de, de_id = marker(self.demo_end, pad_id)
        add(de, de_id, torch.ones(n, 1, dtype=torch.bool, device=device), 0)
        add(label, self.label_demo_ids.to(device)[None].expand(n, -1), torch.ones(n, label.shape[1], dtype=torch.bool, device=device), 0)
        add(bridge, self.bridge_ids.to(device)[None].expand(n, -1), torch.ones(n, bridge.shape[1], dtype=torch.bool, device=device), 0)
        obs_lo = sum(p.shape[1] for p in parts)
        os_, os_id = marker(self.obs_start, pad_id)
        add(os_, os_id, torch.ones(n, 1, dtype=torch.bool, device=device), 0)
        cam_ok = batch["obs.camera_valid"].reshape(n, 3)
        vs_id = int(vlm.config.vision_start_token_id)
        ve_id = int(vlm.config.vision_end_token_id)
        for c in range(3):
            span = 1 + images.shape[2] + 1
            block = torch.cat(
                [v_start.expand(n, 1, -1), images[:, c], v_end.expand(n, 1, -1)],
                dim=1,
            )
            tok = torch.cat(
                [
                    torch.full((n, 1), vs_id, dtype=torch.long, device=device),
                    image_ids[:, c],
                    torch.full((n, 1), ve_id, dtype=torch.long, device=device),
                ],
                dim=1,
            )
            ok = cam_ok[:, c:c + 1].expand(n, span)
            add(block, tok, ok, 0)
            types[-1][:, 1:-1] = 1
        oe, oe_id = marker(self.obs_end, pad_id)
        add(oe, oe_id, torch.ones(n, 1, dtype=torch.bool, device=device), 0)
        add(cur_txt, cur_ids, batch["current_text.attention_mask"].reshape(n, -1).bool(), 0)
        return (
            torch.cat(parts, dim=1),
            torch.cat(valid, dim=1),
            int(obs_lo),
            torch.cat(ids, dim=1),
            grid.reshape(n, 3, 3),
        )

    def _restrict_expert(self, mask: torch.Tensor, vlm_len: int, obs_lo: int) -> torch.Tensor:
        # V-obs (default): AE / state rows cannot attend tokens before obs_start.
        # expert_v_obs=False: keep the full causal prefix visible (incl. Z_demo).
        if not bool(getattr(self.config, "expert_v_obs", True)):
            return mask
        mask = mask.clone()
        mask[:, :, vlm_len:, :obs_lo] = False
        return mask

    def _run(self, batch, *, noise, time, blank_demo, sample_steps: int | None):
        core, cfg = self.core, self.config
        z = self.demo_encoder(batch["demo.feats"], batch["demo.time"], batch["demo.camera_valid"])
        prefix, valid, obs_lo, ids, grid = self.build_prefix(batch, z, blank_demo=blank_demo)
        n = prefix.shape[0]
        actions = batch[ACTION].reshape(n, batch[ACTION].shape[2], -1)
        actions = F.pad(actions, (0, cfg.max_action_dim - actions.shape[-1])).float()
        horizon = actions.shape[1]
        state_hidden = core.state_projection(
            self._state(batch, n, prefix.device, core.state_projection.weight.dtype)
        ).to(prefix.dtype)[:, None]
        if sample_steps is None:
            noise = torch.randn_like(actions) if noise is None else noise
            if time is None:
                beta = torch.distributions.Beta(
                    torch.tensor(cfg.time_sampling_beta_alpha, device=actions.device),
                    torch.tensor(cfg.time_sampling_beta_beta, device=actions.device),
                )
                time = beta.sample((n,)) * cfg.time_sampling_scale + cfg.time_sampling_offset
            t = time[:, None, None]
            noisy = t * noise + (1 - t) * actions
            target = noise - actions
            action_hidden = core._action_inputs(noisy, time).to(prefix.dtype)
        else:
            action_hidden = prefix.new_zeros(n, horizon, core.hidden_size)
        layout = Q35TokenLayout.for_step(
            "robot", expert_type="action", vlm=prefix.shape[1], state=1, action=horizon
        )
        full_valid = torch.cat(
            [valid, torch.ones(n, 1 + horizon, dtype=torch.bool, device=prefix.device)], 1
        )
        mask = self._restrict_expert(
            layout.build_attention_mask(valid_mask=full_valid, causal_vlm=True),
            prefix.shape[1],
            obs_lo,
        )
        positions = self.build_position_ids(ids, self._mm_types(batch, ids, obs_lo), grid.reshape(-1, 3), valid, 1 + horizon)
        if positions.shape[-1] != layout.total_tokens:
            raise RuntimeError(f"rope length {positions.shape[-1]} != tokens {layout.total_tokens}")
        if sample_steps is None:
            _, _, action_out = core._forward_joint(
                prefix, state_hidden, action_hidden, attention_mask=mask, position_ids=positions
            )
            predicted = core.action_head(action_out.to(core.action_head.weight.dtype)).float()
            flow = F.mse_loss(predicted, target.float(), reduction="none")
            return flow, noise, time
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
        steps = int(sample_steps)
        sampled = noise.clone().float()
        for step in range(steps):
            t = torch.full((n,), 1.0 - step / steps, device=prefix.device)
            sampled = sampled - core.denoise_cached(sampled, t, cache, act_mask, act_pos).float() / steps
        return sampled

    def _mm_types(self, batch: dict, ids: torch.Tensor, obs_lo: int) -> torch.Tensor:
        """1 on current-image tokens, 0 elsewhere. Image ids were written only inside the camera spans."""
        image_id = int(self.core.backbone.vlm.config.image_token_id)
        return (ids == image_id).long()

    def _state(self, batch, n, device, dtype) -> torch.Tensor:
        state = batch[OBS_STATE].reshape(n, -1)
        state = F.pad(state, (0, int(self.config.max_state_dim) - state.shape[-1])).to(dtype)
        return state

    def forward_segments(self, batch, *, noise=None, time=None, blank_demo=None):
        flow, noise, time = self._run(batch, noise=noise, time=time, blank_demo=blank_demo, sample_steps=None)
        result = {"flow": flow, "noise": noise, "time": time}
        debug_check_finite("bpv5b.flow", result)
        return result

    @torch.no_grad()
    def sample_segments(self, batch, *, noise=None, num_steps=None, blank_demo=None, **_kwargs):
        steps = int(num_steps or self.config.num_inference_steps)
        return self._run(batch, noise=noise, time=None, blank_demo=blank_demo, sample_steps=steps)


class BPV5StageBPolicy(Q35Policy):
    config_class = BPV5StageBConfig
    name = "bpv5_stageb"

    def __init__(self, config: BPV5StageBConfig) -> None:
        super(Q35Policy, self).__init__(config)
        config.validate_features()
        self.model = BPV5StageBModel(config)
        config.qwen3_5_fingerprint = self.model.core.backbone.fingerprint.name
        self.reset()

    def train(self, mode: bool = True):
        super().train(mode)
        # Phase flags live on requires_grad. Re-apply so a later load cannot leave LoRA trainable in B-1.
        self.model.apply_phase()
        return self

    def get_optim_params(self):
        cfg = self.config
        groups = {"obs": [], "act": [], "lora": [], "qformer": []}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if LORA_KEY in name:
                groups["lora"].append(param)
            elif "demo_encoder" in name:
                groups["qformer"].append(param)
            elif name.startswith(tuple(f"model.{p}" for p in NEW_MODULE_PREFIXES)):
                groups["obs"].append(param)
            else:
                groups["act"].append(param)
        out = []
        if groups["obs"]:
            out.append({"name": "obs", "params": groups["obs"], "lr": float(cfg.new_modules_lr)})
        if groups["act"]:
            out.append({"name": "act", "params": groups["act"], "lr": float(cfg.act_expert_lr)})
        if groups["lora"]:
            out.append({"name": "lora", "params": groups["lora"], "lr": float(cfg.qwen_lora_lr)})
        if groups["qformer"]:
            out.append({"name": "qformer", "params": groups["qformer"], "lr": float(cfg.qformer_lr)})
        if not out:
            raise RuntimeError(f"phase {cfg.phase} produced no trainable parameters")
        return out

    def _resolve_dim_mask(self, batch, flow):
        if ACTION_DIM_MASK not in batch:
            return torch.ones(flow.shape[0], flow.shape[-1], dtype=torch.bool, device=flow.device)
        dim_mask = batch[ACTION_DIM_MASK]
        if dim_mask.ndim == 1:
            return dim_mask[None, :].expand(flow.shape[0], -1)
        b = int(dim_mask.shape[0])
        return dim_mask.repeat_interleave(flow.shape[0] // b, dim=0)

    def _reduce_flow(self, batch, flow: torch.Tensor) -> torch.Tensor:
        raw_dim = batch[ACTION].shape[-1]
        flow = flow[..., :raw_dim]
        step_mask = batch[ACTION_VALID_MASK].reshape(flow.shape[0], -1)
        dim_mask = self._resolve_dim_mask(batch, flow)
        mode = str(getattr(self.config, "action_dim_mask_mode", "stats") or "stats")
        if mode == "ae3":
            return ae3_flow_loss(flow, step_mask=step_mask)
        if mode == "all":
            return masked_flow_loss_dims(flow, step_mask=step_mask, dim_mask=torch.ones_like(dim_mask))
        return masked_flow_loss_dims(flow, step_mask=step_mask, dim_mask=dim_mask)

    def predict_action_chunk(self, batch, **kwargs):
        """Online / server path: one segment per item (S=1), reuse sample_segments."""
        del kwargs  # ignore decode_image / server extras
        self.eval()
        # sample_segments reads ACTION only for horizon; inject zeros when serving.
        if ACTION not in batch:
            b, s = batch["demo_text.input_ids"].shape[:2]
            h = int(self.config.chunk_size)
            a = int(self.config.max_action_dim)
            device = batch["demo.feats"].device
            batch = {
                **batch,
                ACTION: torch.zeros(b, s, h, a, device=device, dtype=torch.float32),
            }
        pred = self.model.sample_segments(batch)
        # sample_segments returns [B*S, H, A]; with S=1 that is [B, H, A]
        return pred[..., : int(self.config.max_action_dim)]

    def forward(self, batch, *, validation_mode: str | None = None):
        losses = self.model.forward_segments(batch)
        loss_match = self._reduce_flow(batch, losses["flow"])
        loss = loss_match
        loss_blank = None
        blank_margin = None
        blank_gap = None
        m = None
        lam = float(getattr(self.config, "blank_margin_lambda", 0.0) or 0.0)
        every_n = int(getattr(self.config, "blank_margin_every_n_steps", 1) or 1)
        gate_thr = float(getattr(self.config, "blank_margin_gate_blank_over_match", 0.0) or 0.0)
        # Gate=0 → always armed (legacy). Gate>0 → wait for HOM to set _blank_margin_armed.
        armed = True if gate_thr <= 0 else bool(getattr(self, "_blank_margin_armed", False))
        if self.training:
            self._c1_train_step = int(getattr(self, "_c1_train_step", 0)) + 1
        step_ok = every_n <= 1 or (int(getattr(self, "_c1_train_step", 0)) % every_n == 0)
        if lam > 0 and self.training and armed and step_ok:
            # C1: shared noise/time; blank zeros Z_demo. Demo batch dim before segment expand.
            b = int(batch["demo.feats"].shape[0])
            blank_flag = torch.ones(b, dtype=torch.bool, device=losses["flow"].device)
            losses_blank = self.model.forward_segments(
                batch,
                noise=losses["noise"],
                time=losses["time"],
                blank_demo=blank_flag,
            )
            loss_blank = self._reduce_flow(batch, losses_blank["flow"])
            m_min = float(self.config.blank_margin_m)
            m_ratio = float(getattr(self.config, "blank_margin_m_ratio", 0.0) or 0.0)
            m_ratio_max = float(getattr(self.config, "blank_margin_m_ratio_max", 2.0) or 2.0)
            m_fixed = getattr(self, "_blank_margin_m_fixed", None)
            # Detach so the target gap does not create a trivial "inflate L_match" path.
            lm = loss_match.detach()
            if m_fixed is not None:
                # HOM-fixed m (e.g. max(blank, 2*match)); do not re-floor with blank_margin_m.
                m = loss_match.new_tensor(max(float(m_fixed), 0.0))
            elif m_ratio > 0:
                m = torch.clamp(m_ratio * lm, min=m_min, max=m_ratio_max * lm.clamp_min(m_min))
            else:
                m = loss_match.new_tensor(m_min)
            blank_gap = loss_blank - loss_match
            blank_margin = F.relu(m - blank_gap)
            loss = loss_match + lam * blank_margin
        debug_check_finite("bpv5b.loss", loss)
        with torch.no_grad():
            n_lora = sum(
                p.numel()
                for n, p in self.model.core.backbone.vlm.named_parameters()
                if p.requires_grad and LORA_KEY in n
            )
            logs = {
                "loss_action": float(loss_match.detach()),
                "bp_phase": float({"ae": 0, "b1": 1, "b2": 2, "b3": 3}[self.config.phase]),
                "bp_expert_v_obs": float(bool(getattr(self.config, "expert_v_obs", True))),
                "bp_lora_trainable": float(n_lora > 0),
                "bp_lora_trainable_params": float(n_lora),
                "bp_qformer_trainable": float(phase_trainable(self.config.phase)["qformer"]),
                "bp_blank_margin_armed": float(armed) if lam > 0 else 0.0,
            }
            if loss_blank is not None:
                logs["bp_loss_blank"] = float(loss_blank.detach())
                logs["bp_blank_gap"] = float(blank_gap.detach())
                logs["bp_blank_margin"] = float(blank_margin.detach())
                logs["bp_blank_margin_m"] = float(m.detach()) if torch.is_tensor(m) else float(m)
                logs["bp_blank_margin_lambda"] = lam
        return loss, logs