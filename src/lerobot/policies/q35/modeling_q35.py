from __future__ import annotations

import math
import os
from collections import deque

import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_PREFIX, OBS_STATE, SAMPLE_ACTION_LOSS_MASK

from .backbone import Q35Backbone, Q35PrefixCache
from .configuration_q35 import Q35Config
from .qwen3_5_backbone import kernelize_qwen35_training_backbone
from .token_layout import Q35TokenLayout
from .transform_q35 import ACTION_VALID_MASK


@dynamo.disable
def debug_check_finite(name: str, value) -> None:
    if os.environ.get("LEROBOT_Q35_DEBUG_NONFINITE", "0") != "1":
        return

    pending = [(name, value)]
    while pending:
        path, current = pending.pop()
        if isinstance(current, dict):
            pending.extend((f"{path}.{key}", item) for key, item in current.items())
            continue
        if isinstance(current, (tuple, list)):
            pending.extend((f"{path}[{index}]", item) for index, item in enumerate(current))
            continue
        if not isinstance(current, torch.Tensor) or not current.is_floating_point():
            continue

        finite = torch.isfinite(current)
        if bool(finite.all()):
            continue
        finite_values = current.detach()[finite]
        finite_min = float(finite_values.min()) if finite_values.numel() else None
        finite_max = float(finite_values.max()) if finite_values.numel() else None
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        nonfinite = int((~finite).sum())
        raise FloatingPointError(
            f"Q35 non-finite tensor: rank={rank} path={path} shape={tuple(current.shape)} "
            f"dtype={current.dtype} nonfinite={nonfinite}/{current.numel()} "
            f"finite_min={finite_min} finite_max={finite_max}"
        )


def sinusoidal_time_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
) -> torch.Tensor:
    if dimension % 2:
        raise ValueError("Time embedding dimension must be even")
    fraction = torch.linspace(0.0, 1.0, dimension // 2, device=time.device, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    phase = time.float()[:, None] * (2 * math.pi / period)[None]
    return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)


def flow_euler_sample(noise: torch.Tensor, velocity_fn, *, num_steps: int) -> torch.Tensor:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    actions = noise
    step_size = -1.0 / num_steps
    for step in range(num_steps):
        time = torch.full(
            (noise.shape[0],),
            1.0 - step / num_steps,
            device=noise.device,
            dtype=noise.dtype,
        )
        actions = actions + step_size * velocity_fn(actions, time)
    return actions


def masked_flow_loss(
    flow: torch.Tensor,
    *,
    action_valid_mask: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = torch.ones(flow.shape[:2], dtype=torch.bool, device=flow.device)
    if action_valid_mask is not None:
        action_valid_mask = action_valid_mask.to(device=flow.device, dtype=torch.bool)
        if action_valid_mask.shape != flow.shape[:2]:
            raise ValueError(
                f"action_valid_mask must have shape {tuple(flow.shape[:2])}, "
                f"got {tuple(action_valid_mask.shape)}"
            )
        valid &= action_valid_mask
    if sample_mask is not None:
        sample_mask = sample_mask.reshape(-1).to(device=flow.device) > 0.5
        if sample_mask.shape[0] != flow.shape[0]:
            raise ValueError("sample action loss mask must match the Flow loss batch dimension")
        valid &= sample_mask[:, None]
    weights = valid[..., None].to(flow.dtype)
    denominator = weights.sum() * flow.shape[-1]
    return (flow * weights).sum() / denominator.clamp_min(1)


class Q35Core(nn.Module):
    """Action-expert inputs, Flow head, and prefix-cache execution."""

    def __init__(
        self,
        config,
        *,
        hidden_size: int | None = None,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone = Q35Backbone(config) if backbone is None else backbone
        derived_hidden_size = int(self.backbone.expert_config.hidden_size)
        if hidden_size is not None and hidden_size != derived_hidden_size:
            raise ValueError(
                f"Q35 hidden_size must match the derived action-expert width "
                f"{derived_hidden_size}, got {hidden_size}"
            )
        self.hidden_size = derived_hidden_size
        hidden_size = derived_hidden_size
        self.state_projection = nn.Linear(config.max_state_dim, hidden_size)
        self.action_projection = nn.Linear(config.max_action_dim, hidden_size)
        self.action_time_in = nn.Linear(2 * hidden_size, hidden_size)
        self.action_time_out = nn.Linear(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, config.max_action_dim)

    def _forward_joint(
        self,
        vlm_hidden: torch.Tensor,
        condition_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        if getattr(self.config, "gradient_checkpointing", False) and self.training:
            return torch.utils.checkpoint.checkpoint(
                lambda vlm, condition, action: self.backbone.forward_joint(
                    vlm,
                    condition,
                    action,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                ),
                vlm_hidden,
                condition_hidden,
                action_hidden,
                use_reentrant=False,
            )
        return self.backbone.forward_joint(
            vlm_hidden,
            condition_hidden,
            action_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def _action_inputs(self, actions: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # denoise_cached 的速度输出是 float32，下一采样步会把动作提升成 float32。
        # 投影权重在 bfloat16 推理时必须先对齐，否则 Linear 会报 dtype 不一致。
        actions = actions.to(dtype=self.action_projection.weight.dtype)
        action = self.action_projection(actions)
        time_embedding = sinusoidal_time_embedding(
            time,
            self.hidden_size,
            self.config.min_period,
            self.config.max_period,
        ).to(dtype=action.dtype)
        fused = torch.cat((action, time_embedding[:, None].expand_as(action)), dim=-1)
        return self.action_time_out(F.silu(self.action_time_in(fused)))

    def _forward_action_embedded(
        self,
        vlm_hidden: torch.Tensor,
        state: torch.Tensor,
        noisy_action: torch.Tensor,
        time: torch.Tensor,
        flow_target: torch.Tensor,
        *,
        position_ids: torch.Tensor | None,
        vlm_valid_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        batch = noisy_action.shape[0]
        state_hidden = self.state_projection(
            state.to(self.state_projection.weight.dtype)
        ).to(vlm_hidden.dtype)[:, None]
        action_hidden = self._action_inputs(noisy_action, time).to(vlm_hidden.dtype)
        layout = Q35TokenLayout.for_step(
            "robot",
            expert_type="action",
            vlm=vlm_hidden.shape[1],
            state=1,
            action=action_hidden.shape[1],
        )
        if vlm_valid_mask is None:
            vlm_valid_mask = torch.ones(
                batch,
                vlm_hidden.shape[1],
                dtype=torch.bool,
                device=vlm_hidden.device,
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
            causal_vlm=getattr(self.config, "causal_vlm", False),
        )
        if position_ids is None:
            positions = torch.arange(layout.total_tokens, device=vlm_hidden.device)
            position_ids = positions[None, None].expand(3, batch, -1)
        _, state_out, action_out = self._forward_joint(
            vlm_hidden,
            state_hidden,
            action_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        debug_check_finite(
            "robot.action_expert_outputs",
            {"state_out": state_out, "action_out": action_out},
        )
        predicted_flow = self.action_head(action_out.to(self.action_head.weight.dtype)).float()
        result = {"flow": F.mse_loss(predicted_flow, flow_target.float(), reduction="none")}
        debug_check_finite("robot.losses", result)
        return result

    def forward_embedded(
        self,
        vlm_hidden: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        vlm_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = actions.shape[0]
        noise = torch.randn_like(actions) if noise is None else noise
        if time is None:
            beta = torch.distributions.Beta(
                torch.tensor(self.config.time_sampling_beta_alpha, device=actions.device),
                torch.tensor(self.config.time_sampling_beta_beta, device=actions.device),
            )
            time = (
                beta.sample((batch,)) * self.config.time_sampling_scale
                + self.config.time_sampling_offset
            )
        time_expanded = time[:, None, None]
        noisy_action = time_expanded * noise + (1 - time_expanded) * actions
        flow_target = noise - actions
        return self._forward_action_embedded(
            vlm_hidden,
            state,
            noisy_action,
            time,
            flow_target,
            position_ids=position_ids,
            vlm_valid_mask=vlm_valid_mask,
        )

    def prepare_inference_cache(
        self,
        vlm_hidden: torch.Tensor,
        state: torch.Tensor,
        *,
        vlm_valid_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[Q35PrefixCache, torch.Tensor, torch.Tensor]:
        batch = vlm_hidden.shape[0]
        if vlm_valid_mask is None:
            vlm_valid_mask = torch.ones(
                batch,
                vlm_hidden.shape[1],
                dtype=torch.bool,
                device=vlm_hidden.device,
            )
        state_hidden = self.state_projection(state.to(self.state_projection.weight.dtype)).to(
            vlm_hidden.dtype
        )[:, None]
        layout = Q35TokenLayout.for_step(
            "robot",
            expert_type="action",
            vlm=vlm_hidden.shape[1],
            state=1,
            action=self.config.chunk_size,
        )
        valid_mask = torch.cat(
            (
                vlm_valid_mask.bool(),
                torch.ones(
                    batch,
                    1 + self.config.chunk_size,
                    dtype=torch.bool,
                    device=vlm_hidden.device,
                ),
            ),
            dim=1,
        )
        full_attention = layout.build_attention_mask(
            valid_mask=valid_mask,
            causal_vlm=getattr(self.config, "causal_vlm", False),
        )
        if position_ids is None:
            positions = torch.arange(layout.total_tokens, device=vlm_hidden.device)
            position_ids = positions[None, None].expand(3, batch, -1)
        vlm_end = layout.span("vlm").stop
        _, _, prefix_cache = self.backbone.prepare_inference(
            vlm_hidden,
            state_hidden,
            attention_mask=full_attention[..., :vlm_end, :vlm_end],
            position_ids=position_ids[..., :vlm_end],
        )
        if len(prefix_cache.full_attention) != self.backbone.expert_depth:
            raise ValueError("prefix cache depth does not match ActionExpert")
        suffix = slice(layout.span("state").start, layout.total_tokens)
        return (
            prefix_cache,
            full_attention[..., suffix, :],
            position_ids[..., suffix],
        )

    def denoise_cached(
        self,
        noisy_action: torch.Tensor,
        time: torch.Tensor,
        prefix_cache: Q35PrefixCache,
        action_attention_mask: torch.Tensor,
        action_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        action_hidden = self._action_inputs(noisy_action, time).to(
            self.backbone.action_norm.weight.dtype
        )
        output = self.backbone.denoise(
            action_hidden,
            prefix_cache,
            attention_mask=action_attention_mask,
            position_ids=action_position_ids,
        )
        return self.action_head(output.to(self.action_head.weight.dtype)).float()


class Q35Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.core = Q35Core(config)
        if config.gradient_checkpointing:
            self.core.backbone.vlm_language_model.gradient_checkpointing = True
            self.core.backbone.vlm_visual.gradient_checkpointing = True
        self._install_compiled_callables(config)

    def _install_compiled_callables(self, config) -> None:
        self._compiled_denoise = None
        if not config.compile_model:
            return
        torch.set_float32_matmul_precision("high")
        self._compiled_denoise = torch.compile(
            self.core.denoise_cached,
            mode=config.compile_mode,
        )

    @dynamo.disable
    def embed_vlm(self, pixel_values, image_grid_thw, input_ids) -> torch.Tensor:
        vlm = self.core.backbone.vlm
        vision_dtype = next(self.core.backbone.vlm_visual.parameters()).dtype
        pixels = pixel_values.reshape(-1, pixel_values.shape[-1]).to(dtype=vision_dtype)
        grid = image_grid_thw.reshape(-1, 3)
        image_outputs = vlm.model.get_image_features(pixels, grid)
        image_chunks = image_outputs.pooler_output
        image_embeddings = (
            torch.cat(image_chunks, dim=0)
            if isinstance(image_chunks, (tuple, list))
            else image_chunks
        )
        embeddings = vlm.get_input_embeddings()(input_ids)
        flat = embeddings.reshape(-1, embeddings.shape[-1])
        flat_ids = input_ids.reshape(-1)
        flat[flat_ids == vlm.config.image_token_id] = image_embeddings.to(flat.dtype)
        return flat.reshape_as(embeddings)

    @dynamo.disable
    def build_position_ids(
        self,
        input_ids,
        mm_token_type_ids,
        image_grid_thw,
        attention_mask,
        extra_tokens: int,
    ):
        vlm = self.core.backbone.vlm
        prefix_ids, _ = vlm.model.get_rope_index(
            input_ids,
            mm_token_type_ids,
            image_grid_thw=image_grid_thw.reshape(-1, 3),
            attention_mask=torch.ones_like(attention_mask),
        )
        start = prefix_ids.max(dim=-1).values + 1
        extra = torch.arange(extra_tokens, device=input_ids.device)[None, None]
        extra = start[:, :, None] + extra
        return torch.cat((prefix_ids, extra), dim=-1)

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
        vlm_hidden: torch.Tensor | None = None,
    ):
        if vlm_hidden is None:
            vlm_hidden = self.embed_vlm(pixel_values, image_grid_thw, input_ids) # TODO 新增embed_vlm_with_bp
        position_ids = self.build_position_ids(
            input_ids,
            mm_token_type_ids,
            image_grid_thw,
            attention_mask,
            1 + actions.shape[1],
        )
        return self.core.forward_embedded(
            vlm_hidden,
            state,
            actions,
            position_ids=position_ids,
            vlm_valid_mask=attention_mask,
        )

    @torch.no_grad()
    def _sample_action_flow(
        self,
        noise: torch.Tensor,
        prefix_cache: Q35PrefixCache,
        action_attention_mask: torch.Tensor,
        action_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        actions = noise.clone()
        step_size = -1.0 / self.config.num_inference_steps
        denoise = self._compiled_denoise or self.core.denoise_cached
        for step in range(self.config.num_inference_steps):
            time = torch.full(
                (noise.shape[0],),
                1.0 - step / self.config.num_inference_steps,
                device=noise.device,
                dtype=noise.dtype,
            )
            velocity = denoise(
                actions,
                time,
                prefix_cache,
                action_attention_mask,
                action_position_ids,
            )
            actions = actions + step_size * velocity
        return actions

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
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        vlm_hidden = self.embed_vlm(pixel_values, image_grid_thw, input_ids)
        position_ids = self.build_position_ids(
            input_ids,
            mm_token_type_ids,
            image_grid_thw,
            attention_mask,
            1 + self.config.chunk_size,
        )
        prefix_cache, action_attention_mask, action_position_ids = (
            self.core.prepare_inference_cache(
                vlm_hidden,
                state,
                vlm_valid_mask=attention_mask,
                position_ids=position_ids,
            )
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


class Q35Policy(PreTrainedPolicy):
    config_class = Q35Config
    name = "q35"

    def __init__(self, config) -> None:
        super().__init__(config)
        config.validate_features()
        self.model = Q35Model(config)
        actual_fingerprint = self.model.core.backbone.fingerprint.name
        if config.qwen3_5_fingerprint not in {None, actual_fingerprint}:
            raise ValueError(
                "Q35 Qwen3.5 fingerprint mismatch: "
                f"expected {config.qwen3_5_fingerprint}, loaded {actual_fingerprint}"
            )
        config.qwen3_5_fingerprint = actual_fingerprint
        self._install_compiled_training_forward(config)
        self.reset()

    def to(self, *args, **kwargs):
        policy = super().to(*args, **kwargs)
        if not getattr(policy, "_qwen35_training_kernels_enabled", False):
            policy._qwen35_training_kernels_enabled = kernelize_qwen35_training_backbone(
                policy.model.core.backbone
            )
        return policy

    def _install_compiled_training_forward(self, config) -> None:
        if config.compile_model:
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        # Earlier Q35 experiments returned VLM, AE attention, and AE FFN/I/O
        # groups using vlm_lr, world_action_attention_lr, and action_ffn_io_lr.
        # TBot's optimizer preset uses one optimizer_lr for all trainable params.
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, batch, *, validation_mode: str | None = None):
        if validation_mode not in {None, "flow"}:
            raise ValueError(f"Unsupported robot validation mode: {validation_mode!r}")
        raw_actions = batch[ACTION]
        actions = F.pad(raw_actions, (0, self.config.max_action_dim - raw_actions.shape[-1]))
        state = F.pad(
            batch[OBS_STATE],
            (0, self.config.max_state_dim - batch[OBS_STATE].shape[-1]),
        )
        vlm_hidden = self.model.embed_vlm(
            batch[f"{OBS_PREFIX}pixel_values"],
            batch[f"{OBS_PREFIX}image_grid_thw"],
            batch[f"{OBS_PREFIX}input_ids"],
        )
        losses = self.model.forward_robot(
            pixel_values=batch[f"{OBS_PREFIX}pixel_values"],
            image_grid_thw=batch[f"{OBS_PREFIX}image_grid_thw"],
            input_ids=batch[f"{OBS_PREFIX}input_ids"],
            mm_token_type_ids=batch[f"{OBS_PREFIX}mm_token_type_ids"],
            attention_mask=batch[f"{OBS_PREFIX}attention_mask"],
            state=state,
            actions=actions,
            vlm_hidden=vlm_hidden,
        )
        flow = losses["flow"][..., : raw_actions.shape[-1]]
        flow_loss = masked_flow_loss(
            flow,
            action_valid_mask=batch.get(ACTION_VALID_MASK),
            sample_mask=batch.get(SAMPLE_ACTION_LOSS_MASK),
        )
        debug_check_finite("robot.final_loss", flow_loss)
        return flow_loss, {"step_type": "robot", "loss_flow": float(flow_loss.detach())}

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
    "ACTION_VALID_MASK",
    "Q35Core",
    "Q35Model",
    "Q35Policy",
    "flow_euler_sample",
    "masked_flow_loss",
    "sinusoidal_time_embedding",
]
