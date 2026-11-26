"""Stage B v2.

VLM and ViT load from wall-oss. The Q-Former loads from a Stage A checkpoint.
demo_start and demo_end are new parameters. The action expert is the q35 stack and
does not receive a state token. Flow matches wall-oss: t = 0 is noise, the target is
action - noise.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.BPV4CDX.modeling_bpv4cdx import TemporalVideoQFormer
from lerobot.policies.BPV5QFormer.modeling_bpv5qformer import ae3_flow_loss, masked_flow_loss_dims
from lerobot.policies.BPV5StageB.configuration_bpv5stageb_v2 import BPV5StageBV2Config
from lerobot.policies.BPV5StageB.modeling_bpv5stageb import BPV5StageBPolicy, _checkpoint_file
from lerobot.policies.q35.modeling_q35 import Q35Model, Q35Policy, debug_check_finite
from lerobot.policies.q35.token_layout import Q35TokenLayout
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION, OBS_STATE

ACTION_DIM_MASK = "action.dim_mask"
CAMERAS = ("face_view", "left_wrist_view", "right_wrist_view")
HEAD = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nObservation:"
AFTER_CURRENT = "\nPredict the next action in robot action.\nProprioception: "
AFTER_PROPRIO = "\nDemonstration:\n"
AFTER_Z = "\nDemonstrated task: "
TAIL = "\n<|im_end|>\n<|im_start|>assistant\n"


def proprio_bins(state: torch.Tensor, *, clip: float, bins: int) -> torch.Tensor:
    """Normalized state -> integer bins. ``state`` is already (x - mean) / std."""
    scaled = state.clamp(-clip, clip) / clip
    return torch.clamp(torch.floor((scaled + 1) / 2 * bins), 0, bins - 1).long()


def load_qformer_weights(module: nn.Module, path: str | Path) -> int:
    """Copy only TemporalVideoQFormer tensors from a Stage A checkpoint."""
    from safetensors.torch import load_file

    tensors = load_file(str(_checkpoint_file(path)))
    own = dict(module.named_parameters())
    copied = 0
    with torch.no_grad():
        for key, value in tensors.items():
            if "demo_encoder." not in key:
                continue
            target = key if key in own else "demo_encoder." + key.split("demo_encoder.", 1)[-1]
            if target not in own or tuple(own[target].shape) != tuple(value.shape):
                continue
            own[target].copy_(value.to(dtype=own[target].dtype, device=own[target].device))
            copied += 1
    if copied == 0:
        raise RuntimeError(f"Stage A checkpoint {path} has no matching demo_encoder tensors")
    logging.info("stage B v2 loaded Q-Former tensors: %d", copied)
    return copied


class BPV5StageBV2Model(Q35Model):
    def __init__(self, config: BPV5StageBV2Config) -> None:
        super().__init__(config)
        hidden = int(self.core.backbone.vlm.config.text_config.hidden_size)
        self.demo_encoder = TemporalVideoQFormer(config, hidden)
        self.demo_start = nn.Parameter(torch.randn(hidden) * 0.02)
        self.demo_end = nn.Parameter(torch.randn(hidden) * 0.02)
        self._tokenizer = None
        self._fixed_ids: dict[str, torch.Tensor] = {}
        for param in self.core.state_projection.parameters():
            param.requires_grad_(False)
        # A resumed checkpoint already contains the Q-Former. Load Stage A only for a fresh run.
        if config.stage_a_checkpoint and not config.pretrained_path:
            load_qformer_weights(self, config.stage_a_checkpoint)
        self.apply_phase()

    def _tokenizer_for(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            path = self.config.qwen3_5_processor_path or self.config.qwen3_5_pretrained_path
            self._tokenizer = AutoTokenizer.from_pretrained(path)
        return self._tokenizer

    def _fixed(self, text: str, n: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if text not in self._fixed_ids:
            ids = self._tokenizer_for()(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
            self._fixed_ids[text] = ids.long()
        ids = self._fixed_ids[text].to(device)
        emb = self.core.backbone.vlm.get_input_embeddings()(ids[None].expand(n, -1)).to(dtype)
        return emb, ids[None].expand(n, -1)

    def apply_phase(self) -> None:
        full = self.config.phase == "full"
        backbone = self.core.backbone
        vlm = backbone.vlm
        for param in vlm.parameters():
            param.requires_grad_(False)
        # After enable_fsdp_joint_pairs, decoder layers live under backbone.joint_pairs.
        if getattr(backbone, "_fsdp_joint_pairs", False):
            for pair in backbone.joint_pairs:
                for param in pair.vlm_layer.parameters():
                    param.requires_grad_(False)
        if full:
            for param in backbone.vlm_language_model.parameters():
                param.requires_grad_(True)
            if getattr(backbone, "_fsdp_joint_pairs", False):
                for pair in backbone.joint_pairs:
                    for param in pair.vlm_layer.parameters():
                        param.requires_grad_(True)
        for param in self.demo_encoder.parameters():
            param.requires_grad_(full)
        self.demo_start.requires_grad_(True)
        self.demo_end.requires_grad_(True)

    def _vision(self, batch: dict, n: int, dtype: torch.dtype):
        pixels = batch["obs.pixel_values"]
        grid = batch["obs.image_grid_thw"].reshape(-1, 3)
        vlm = self.core.backbone.vlm
        merge = int(self.core.backbone.vlm_visual.spatial_merge_size)
        counts = [int(v) for v in (grid.prod(-1) // merge**2).tolist()]
        if len(set(counts)) != 1:
            raise ValueError(f"current-image token counts differ across the batch: {sorted(set(counts))}")
        tok = counts[0]
        ids = torch.full((1, sum(counts)), int(vlm.config.image_token_id), dtype=torch.long, device=pixels.device)
        flat = self.embed_vlm(pixels, grid, ids)[0]
        images = flat.view(-1, tok, flat.shape[-1])
        b = batch["obs.camera_valid"].shape[0]
        s = n // b
        images = images.view(b, s, 3, tok, -1).reshape(n, 3, tok, -1).to(dtype)
        image_ids = torch.full((n, 3, tok), int(vlm.config.image_token_id), dtype=torch.long, device=pixels.device)
        return images, image_ids, grid

    def _proprio(self, batch, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        state = batch[OBS_STATE].reshape(n, -1).float()
        width = int(self.config.max_state_dim)
        if state.shape[-1] < width:
            state = F.pad(state, (0, width - state.shape[-1]))
        bins = proprio_bins(state, clip=float(self.config.proprio_clip), bins=int(self.config.state_bins))
        strings = [" ".join(str(int(v)) for v in row) for row in bins.tolist()]
        out = self._tokenizer_for()(strings, add_special_tokens=False, padding=True, return_tensors="pt")
        ids = out["input_ids"].to(device)
        attn = out["attention_mask"].to(device).bool()
        emb = self.core.backbone.vlm.get_input_embeddings()(ids)
        return emb, ids, attn

    def build_prefix(self, batch, z_demo, *, blank_demo=None):
        b, s = batch["demo_text.input_ids"].shape[:2]
        n = b * s
        vlm = self.core.backbone.vlm
        embed = vlm.get_input_embeddings()
        cur_ids = batch["current_text.input_ids"].reshape(n, -1)
        demo_ids = batch["demo_text.input_ids"].reshape(n, -1)
        cur = embed(cur_ids)
        demo = embed(demo_ids)
        dtype, device = cur.dtype, cur.device
        if z_demo is None:
            raise RuntimeError("Stage B v2 requires Z_demo")
        if blank_demo is not None:
            z_demo = z_demo * (~blank_demo.to(device=z_demo.device)).to(dtype=z_demo.dtype)[:, None, None]
        z = z_demo.to(dtype).repeat_interleave(s, dim=0)
        images, image_ids, grid = self._vision(batch, n, dtype)
        weight = embed.weight
        v_start = weight[int(vlm.config.vision_start_token_id)].to(dtype)
        v_end = weight[int(vlm.config.vision_end_token_id)].to(dtype)
        pad_id = int(getattr(vlm.config, "pad_token_id", 0) or 0)
        proprio, proprio_ids, proprio_ok = self._proprio(batch, n, device)
        proprio = proprio.to(dtype)

        parts, ids, valid, types = [], [], [], []

        def add(emb, tok, ok):
            parts.append(emb)
            ids.append(tok)
            valid.append(ok)
            types.append(torch.zeros(tok.shape, dtype=torch.long, device=device))

        def text(value: str):
            emb, tok = self._fixed(value, n, device, dtype)
            add(emb, tok, torch.ones(tok.shape, dtype=torch.bool, device=device))

        text(HEAD)
        vs_id = int(vlm.config.vision_start_token_id)
        ve_id = int(vlm.config.vision_end_token_id)
        cam_ok = batch["obs.camera_valid"].reshape(n, 3)
        for c, name in enumerate(CAMERAS):
            text(f" {name}: ")
            span = 1 + images.shape[2] + 1
            block = torch.cat([v_start.expand(n, 1, -1), images[:, c], v_end.expand(n, 1, -1)], dim=1)
            tok = torch.cat(
                [
                    torch.full((n, 1), vs_id, dtype=torch.long, device=device),
                    image_ids[:, c],
                    torch.full((n, 1), ve_id, dtype=torch.long, device=device),
                ],
                dim=1,
            )
            add(block, tok, cam_ok[:, c:c + 1].expand(n, span))
            types[-1][:, 1:-1] = 1
        text("\nInstruction: ")
        add(cur, cur_ids, batch["current_text.attention_mask"].reshape(n, -1).bool())
        text(AFTER_CURRENT)
        add(proprio, proprio_ids, proprio_ok)
        text(AFTER_PROPRIO)
        add(self.demo_start.to(dtype).expand(n, 1, -1), torch.full((n, 1), pad_id, dtype=torch.long, device=device), torch.ones(n, 1, dtype=torch.bool, device=device))
        add(z, torch.full((n, z.shape[1]), pad_id, dtype=torch.long, device=device), torch.ones(n, z.shape[1], dtype=torch.bool, device=device))
        add(self.demo_end.to(dtype).expand(n, 1, -1), torch.full((n, 1), pad_id, dtype=torch.long, device=device), torch.ones(n, 1, dtype=torch.bool, device=device))
        text(AFTER_Z)
        add(demo, demo_ids, batch["demo_text.attention_mask"].reshape(n, -1).bool())
        text(TAIL)
        return torch.cat(parts, 1), torch.cat(valid, 1), torch.cat(ids, 1), torch.cat(types, 1), grid.reshape(n, 3, 3)

    def _run(self, batch, *, noise, time, blank_demo, sample_steps: int | None):
        core, cfg = self.core, self.config
        z = self.demo_encoder(batch["demo.feats"], batch["demo.time"], batch["demo.camera_valid"])
        prefix, valid, ids, mm, grid = self.build_prefix(batch, z, blank_demo=blank_demo)
        n = prefix.shape[0]
        actions = batch[ACTION].reshape(n, batch[ACTION].shape[2], -1)
        actions = F.pad(actions, (0, cfg.max_action_dim - actions.shape[-1])).float()
        horizon = actions.shape[1]
        empty = prefix.new_zeros(n, 0, core.hidden_size)
        if sample_steps is None:
            noise = torch.randn_like(actions) if noise is None else noise
            if time is None:
                beta = torch.distributions.Beta(
                    torch.tensor(cfg.time_sampling_beta_alpha, device=actions.device),
                    torch.tensor(cfg.time_sampling_beta_beta, device=actions.device),
                )
                time = (1 - beta.sample((n,))) * cfg.time_sampling_scale
            t = time[:, None, None]
            noisy = (1 - t) * noise + t * actions
            target = actions - noise
            action_hidden = core._action_inputs(noisy, time).to(prefix.dtype)
        else:
            action_hidden = prefix.new_zeros(n, horizon, core.hidden_size)
            target = None
        layout = Q35TokenLayout.for_step("robot", expert_type="action", vlm=prefix.shape[1], state=0, action=horizon)
        if layout.total_tokens > int(self.config.max_length):
            raise RuntimeError(
                f"prefix+actions is {layout.total_tokens} tokens, above max_length {self.config.max_length}"
            )
        full_valid = torch.cat([valid, torch.ones(n, horizon, dtype=torch.bool, device=prefix.device)], 1)
        mask = layout.build_attention_mask(valid_mask=full_valid, causal_vlm=True)
        positions = self.build_position_ids(ids, mm, grid.reshape(-1, 3), valid, horizon)
        if positions.shape[-1] != layout.total_tokens:
            raise RuntimeError(f"rope length {positions.shape[-1]} != tokens {layout.total_tokens}")
        if sample_steps is None:
            _, _, action_out = core._forward_joint(prefix, empty, action_hidden, attention_mask=mask, position_ids=positions)
            predicted = core.action_head(action_out.to(core.action_head.weight.dtype)).float()
            flow = F.mse_loss(predicted, target.float(), reduction="none")
            return flow, noise, time
        vlm_end = layout.span("vlm").stop
        _, _, cache = core.backbone.prepare_inference(
            prefix, empty, attention_mask=mask[..., :vlm_end, :vlm_end], position_ids=positions[..., :vlm_end]
        )
        suffix = slice(layout.span("action").start, layout.total_tokens)
        act_mask, act_pos = mask[..., suffix, :], positions[..., suffix]
        if noise is None:
            noise = torch.randn(n, horizon, int(cfg.max_action_dim), device=prefix.device)
        steps = int(sample_steps)
        sampled = noise.clone().float()
        for step in range(steps):
            t = torch.full((n,), step / steps, device=prefix.device)
            sampled = sampled + core.denoise_cached(sampled, t, cache, act_mask, act_pos).float() / steps
        return sampled

    def forward_segments(self, batch, *, noise=None, time=None, blank_demo=None):
        flow, noise, time = self._run(batch, noise=noise, time=time, blank_demo=blank_demo, sample_steps=None)
        result = {"flow": flow, "noise": noise, "time": time}
        debug_check_finite("bpv5b2.flow", result)
        return result

    @torch.no_grad()
    def sample_segments(self, batch, *, noise=None, num_steps=None, blank_demo=None, **_kwargs):
        steps = int(num_steps or self.config.num_inference_steps)
        return self._run(batch, noise=noise, time=None, blank_demo=blank_demo, sample_steps=steps)


class BPV5StageBV2Policy(BPV5StageBPolicy):
    config_class = BPV5StageBV2Config
    name = "bpv5_stageb_v2"

    def __init__(self, config: BPV5StageBV2Config) -> None:
        super(Q35Policy, self).__init__(config)
        config.validate_features()
        self.model = BPV5StageBV2Model(config)
        config.qwen3_5_fingerprint = self.model.core.backbone.fingerprint.name
        self.reset()

    def train(self, mode: bool = True):
        super(BPV5StageBPolicy, self).train(mode)
        self.model.apply_phase()
        return self

    def get_optim_params(self):
        cfg = self.config
        groups = {"vlm": [], "qformer": [], "marker": [], "act": []}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "demo_encoder" in name:
                groups["qformer"].append(param)
            elif name.endswith("demo_start") or name.endswith("demo_end"):
                groups["marker"].append(param)
            elif (
                "vlm_language_model" in name
                or ".language_model." in name
                or (".joint_pairs." in name and ".vlm_layer." in name)
            ):
                groups["vlm"].append(param)
            else:
                groups["act"].append(param)
        spec = (
            ("vlm", cfg.vlm_lr),
            ("qformer", cfg.qformer_lr),
            ("marker", cfg.marker_lr),
            ("act", cfg.act_expert_lr),
        )
        out = [{"name": name, "params": groups[name], "lr": float(lr)} for name, lr in spec if groups[name]]
        if not out:
            raise RuntimeError(f"phase {cfg.phase} produced no trainable parameters")
        return out

    def _reduce_flow(self, batch, flow: torch.Tensor) -> torch.Tensor:
        raw_dim = batch[ACTION].shape[-1]
        flow = flow[..., :raw_dim]
        step_mask = batch[ACTION_VALID_MASK].reshape(flow.shape[0], -1)
        if ACTION_DIM_MASK not in batch:
            dim_mask = torch.ones(flow.shape[0], flow.shape[-1], dtype=torch.bool, device=flow.device)
        else:
            dim_mask = batch[ACTION_DIM_MASK]
            if dim_mask.ndim == 1:
                dim_mask = dim_mask[None, :].expand(flow.shape[0], -1)
            else:
                dim_mask = dim_mask.repeat_interleave(flow.shape[0] // int(dim_mask.shape[0]), dim=0)
        mode = str(getattr(self.config, "action_dim_mask_mode", "stats") or "stats")
        if mode == "ae3":
            return ae3_flow_loss(flow, step_mask=step_mask)
        if mode == "all":
            return masked_flow_loss_dims(flow, step_mask=step_mask, dim_mask=torch.ones_like(dim_mask))
        return masked_flow_loss_dims(flow, step_mask=step_mask, dim_mask=dim_mask)

    def forward(self, batch, *, validation_mode: str | None = None):
        del validation_mode
        losses = self.model.forward_segments(batch)
        loss = self._reduce_flow(batch, losses["flow"])
        debug_check_finite("bpv5b2.loss", loss)
        logs = {
            "loss_action": float(loss.detach()),
            "bp_phase": float(self.config.phase == "full"),
            "bp_qformer_trainable": float(self.config.phase == "full"),
        }
        return loss, logs
