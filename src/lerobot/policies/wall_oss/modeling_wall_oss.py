"""wall-oss policy: the vendored wall-x Qwen3.5 MoT VLA (`wall_x.model.qact.qwen3_5`) behind the TBot policy API."""

from __future__ import annotations

import logging
import re
from collections import deque
from pathlib import Path

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.policies.wall_oss.configuration_wall_oss import WallOSSConfig
from lerobot.policies.wall_oss.transform_wall_oss import ACTION_SLOT, load_wall_processor
from lerobot.utils.constants import ACTION, OBS_PREFIX, OBS_STATE, SAMPLE_ACTION_LOSS_MASK

# the action expert of the MoT layers (`*_action` projections / norms, `mlp_action_expert`, `norm_action`) and the
# action input / output layers
ACTION_PARAM = re.compile(r"_action\b|_action\.|mlp_action_expert|norm_action|action_processor\.")


def param_group(name: str) -> str:
    if ACTION_PARAM.search(name):
        return "action"
    if name.startswith("model.visual."):
        return "vision"
    return "vlm"


class WallOSSPolicy(PreTrainedPolicy):
    config_class = WallOSSConfig
    name = "wall_oss"

    def __init__(self, config: WallOSSConfig, **kwargs) -> None:
        super().__init__(config)
        config.validate_features()
        from wall_x.model.qact.qwen3_5 import Qwen3_5_VLAForConditionalGeneration, Qwen3_5Config

        self.processor = load_wall_processor(config.processor_path)
        wall_cfg = Qwen3_5Config.from_pretrained(config.wall_pretrained_path)
        wall_cfg.update_model_config({
            "data": {"use_state_string_representation": True, "action_horizon_flow": config.chunk_size},
            "dof_config": {"action": config.action_dim},
            "agent_pos_config": {"state": config.state_dim},
            "attn_implementation": config.attention_implementation,
        })
        self.model = Qwen3_5_VLAForConditionalGeneration(wall_cfg, self.processor, None)
        self.action_token_id = self.processor.tokenizer.convert_tokens_to_ids(ACTION_SLOT)
        # a fresh run starts from the wall checkpoint; from_pretrained loads the TBot checkpoint over this module
        if not config.pretrained_path:
            self.load_wall_weights(config.wall_pretrained_path)
        if config.dtype == "bfloat16":
            for p in self.model.parameters():  # buffers (rope inv_freq) stay float32
                p.data = p.data.to(torch.bfloat16)
        train = {"action": config.train_action_expert, "vlm": config.train_vlm, "vision": config.train_vision}
        for name, p in self.model.named_parameters():
            p.requires_grad_(train[param_group(name)])
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.reset()

    def load_wall_weights(self, path: str) -> None:
        """Every tensor of the wall checkpoint whose shape matches; the action input / output layers are rebuilt
        when the action / state widths differ from the checkpoint."""
        from safetensors.torch import load_file

        state = {}
        for f in sorted(Path(path).glob("*.safetensors")):
            state.update(load_file(str(f)))
        own = self.model.state_dict()
        rebuilt = sorted(k for k, v in state.items() if k in own and own[k].shape != v.shape)
        for k in rebuilt:
            state.pop(k)
        res = self.model.load_state_dict(state, strict=False)
        missing = [k for k in res.missing_keys if k != "lm_head.weight" and k not in rebuilt]
        if missing or res.unexpected_keys:
            raise RuntimeError(f"wall checkpoint {path}: missing {missing[:5]}, unexpected {res.unexpected_keys[:5]}")
        logging.info("wall_oss: loaded %d tensors from %s; rebuilt (new shape): %s", len(state), path, rebuilt)

    def reset(self) -> None:
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        return (p for p in self.parameters() if p.requires_grad)

    def model_inputs(self, batch: dict) -> dict:
        ids = batch[f"{OBS_PREFIX}input_ids"]
        pixel_values = batch[f"{OBS_PREFIX}pixel_values"]
        param_dtype = next(self.model.parameters()).dtype
        state = batch[OBS_STATE].float().reshape(ids.shape[0], 1, -1)
        return {
            "input_ids": ids,
            "attention_mask": batch[f"{OBS_PREFIX}attention_mask"],
            "mm_token_type_ids": batch[f"{OBS_PREFIX}mm_token_type_ids"],
            "pixel_values": pixel_values.reshape(-1, pixel_values.shape[-1]).to(param_dtype),
            "image_grid_thw": batch[f"{OBS_PREFIX}image_grid_thw"].reshape(-1, 3),
            "proprioception": state,
            "agent_pos_mask": torch.ones_like(state),
            "dataset_names": ["tbot"] * ids.shape[0],
        }

    def sample_time(self, n: int, device) -> torch.Tensor:
        """wall-x `DataCollator.sample_time`: (1 - Beta(a, b)) * s, t = 0 is the noise."""
        beta = torch.distributions.Beta(torch.tensor(self.config.time_sampling_beta_alpha),
                                        torch.tensor(self.config.time_sampling_beta_beta))
        return ((1 - beta.sample((n,))) * self.config.time_sampling_scale).to(device)

    def forward(self, batch: dict, *, validation_mode: str | None = None, sample_time: torch.Tensor | None = None):
        if validation_mode not in {None, "flow"}:
            raise ValueError(f"Unsupported validation mode: {validation_mode!r}")
        actions = batch[ACTION].float()
        n, horizon = actions.shape[:2]
        if (batch[f"{OBS_PREFIX}input_ids"] == self.action_token_id).sum(1).ne(horizon).any():
            raise ValueError(f"every prompt needs {horizon} action slots (dataset.action_horizon = policy.chunk_size)")
        step_mask = torch.ones(n, horizon, device=actions.device)
        if ACTION_VALID_MASK in batch:
            step_mask = step_mask * batch[ACTION_VALID_MASK].float()
        if SAMPLE_ACTION_LOSS_MASK in batch:
            step_mask = step_mask * batch[SAMPLE_ACTION_LOSS_MASK].float().reshape(n, 1)
        out = self.model(
            **self.model_inputs(batch),
            action_chunk=actions,
            dof_mask=torch.ones_like(actions),
            flow_loss_mask=step_mask,
            sample_time=sample_time if sample_time is not None else self.sample_time(n, actions.device),
            mode="train",
        )
        return out.loss, {"step_type": "robot", "loss_flow": float(out.flow_loss.detach())}

    def predict_action_chunk(self, batch, **kwargs):
        raise NotImplementedError("wall_oss inference is not wired into TBot yet")

    def select_action(self, batch, **kwargs):
        raise NotImplementedError("wall_oss inference is not wired into TBot yet")
