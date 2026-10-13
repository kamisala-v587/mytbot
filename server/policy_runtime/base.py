"""各 policy 共用的 checkpoint 加载、归一化和动作后处理。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import (
    ACTION,
    OBS_STATE,
    FirstRequestDumper,
    ServerConfig,
    assign_existing_or_remap,
    batch_model_tensor,
    collate_nested,
    load_train_config_or_none,
    prepare_import_path,
    resolve_ckpt_dir,
    resolve_device,
    resolve_runtime_dtype,
    resolve_stats,
    resolve_task_text,
)
from .input_mask import apply_client_masks, apply_model_masks, log_mask_rules, parse_mask_rules

prepare_import_path()

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class  # noqa: E402
from lerobot.transforms.constants import get_mask_mapping  # noqa: E402
from lerobot.transforms.core import NormalizeTransformFn, ResizeImagesWithPadFn, UnNormalizeTransformFn  # noqa: E402


@dataclass
class PreparedSample:
    inputs: dict[str, Any]
    state: np.ndarray


class PolicyAdapter:
    """一种 policy 的服务端适配器。子类放在独立模块里，并导出为 ADAPTER。"""

    name = "policy"
    family = "vla"
    aliases: tuple[str, ...] = ()

    def __init__(self, args: ServerConfig) -> None:
        self.args = args
        self.ckpt_dir = resolve_ckpt_dir(args.ckpt_path)
        self.train_cfg = load_train_config_or_none(self.ckpt_dir)
        self.config = PreTrainedConfig.from_pretrained(self.ckpt_dir)
        self._check_type()
        self._apply_runtime_overrides()
        self.before_policy_init()

        self.device = resolve_device(args.device)
        self.load_device = resolve_device(args.load_device) if args.load_device else "cpu"
        self.cosmos_device = resolve_device(args.cosmos_device) if args.cosmos_device else self.device
        self.config.device = self.load_device
        setattr(self.config, "cosmos_device", self.cosmos_device)
        self.runtime_dtype = resolve_runtime_dtype(args.dtype, self.device)
        self.config.dtype = "float32" if self.runtime_dtype == torch.float32 else "bfloat16"

        chunk_size = int(getattr(self.config, "chunk_size", 50))
        n_action_steps = int(getattr(self.config, "n_action_steps", chunk_size))
        self.infer_horizon = max(1, min(int(args.infer_horizon or n_action_steps), chunk_size))

        policy_cls = get_policy_class(self.config.type)
        logging.info("加载 %s: %s", policy_cls.__name__, self.ckpt_dir)
        self.policy = policy_cls.from_pretrained(config=self.config, pretrained_name_or_path=self.ckpt_dir)
        if hasattr(self.policy, "model"):
            setattr(self.policy.model, "omit_visual_tokens_in_causal_inference", bool(args.omit_visual_tokens))
        self.policy.config.device = self.device
        setattr(self.policy.config, "cosmos_device", self.cosmos_device)
        self.policy.to(device=self.device, dtype=self.runtime_dtype).eval()
        self.policy.requires_grad_(False)

        self._init_action_transforms()
        self.input_mask_rules = parse_mask_rules(getattr(args, "input_masks", None))
        log_mask_rules(self.input_mask_rules)
        self._dumper = None
        self.setup_policy()
        if self.args.debug and self.family in {"vla", "bp"}:
            if not self.args.debug_run_dir:
                raise RuntimeError("DEBUG=True 时必须提供 debug 目录。")
            self._dumper = FirstRequestDumper(
                Path(self.args.debug_run_dir) / self.family,
                label=self.family.upper(),
            )
        self._metadata = self.build_metadata(chunk_size)
        logging.info(
            "Server 就绪 | adapter=%s | type=%s | device=%s | horizon=%d | action_mode=%s",
            self.name,
            self.config.type,
            self.device,
            self.infer_horizon,
            self.action_mode,
        )

    @classmethod
    def matches(cls, requested: str, ckpt_type: str) -> bool:
        aliases = {alias.lower() for alias in cls.aliases}
        requested_name = requested.strip().lower()
        if requested_name not in {"", "auto"}:
            return requested_name in aliases
        return ckpt_type.lower() in aliases or ckpt_type in cls.aliases

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _check_type(self) -> None:
        raise NotImplementedError

    def before_policy_init(self) -> None:
        return None

    def setup_policy(self) -> None:
        return None

    def prepare(self, obs: dict[str, Any]) -> PreparedSample:
        raise NotImplementedError

    def on_request(self, connection_id: str, obs: dict[str, Any]) -> None:
        if self._dumper is None:
            return
        self._dumper.dump(connection_id, obs, self.resolve_task(obs))

    def resolve_task(self, obs: dict[str, Any]) -> str:
        return resolve_task_text(obs, self.args.default_prompt)

    def _apply_runtime_overrides(self) -> None:
        path_map = {
            "qwen3_vl_pretrained_path": (self.args.qwen3_vl_path, "/vla/workspace/models/Qwen3-VL-2B-Instruct"),
            "qwen3_vl_processor_path": (self.args.qwen3_vl_path, "/vla/workspace/models/Qwen3-VL-2B-Instruct"),
            "qwen3_5_pretrained_path": (self.args.qwen3_5_path, "/vla/workspace/shiliangchao/models/Qwen35_0_8"),
            "qwen3_5_processor_path": (self.args.qwen3_5_path, "/vla/workspace/shiliangchao/models/Qwen35_0_8"),
            "cosmos_tokenizer_path_or_name": (
                self.args.cosmos_tokenizer_path,
                "/vla/workspace/models/Cosmos-Tokenizer-CI8x8",
            ),
            "da3_model_path_or_name": (self.args.da3_model_path, "/vla/workspace/models/DA3-LARGE-1.1"),
            "da3_code_root": (self.args.da3_code_root, str(Path(__file__).resolve().parents[2] / "third_party" / "Depth-Anything-3")),
        }
        for attr, (explicit, fallback) in path_map.items():
            assign_existing_or_remap(self.config, attr, explicit, fallback)
        if self.args.disable_3d_teacher and hasattr(self.config, "lambda_3d"):
            self.config.lambda_3d = 0.0
        if self.args.num_inference_steps is not None and hasattr(self.config, "num_inference_steps"):
            self.config.num_inference_steps = int(self.args.num_inference_steps)

    def _init_action_transforms(self) -> None:
        stats_path = Path(self.args.stats_path).expanduser() if self.args.stats_path else self.ckpt_dir / "stats.json"
        if not stats_path.is_file():
            raise FileNotFoundError(f"未找到 stats.json: {stats_path}")
        self.stats_key, stats = resolve_stats(stats_path, self.args.stats_key)
        stat_keys = ["min", "max", "mean", "std"]
        self.state_stats = {OBS_STATE: {key: np.asarray(stats[OBS_STATE][key]) for key in stat_keys}}
        self.action_stats = {ACTION: {key: np.asarray(stats[ACTION][key]) for key in stat_keys}}
        self.target_action_dim = int(np.asarray(self.action_stats[ACTION]["mean"]).shape[0])
        self.resize_fn = ResizeImagesWithPadFn(height=self.args.resize_size, width=self.args.resize_size)
        self.normalize_state_fn = NormalizeTransformFn(
            selected_keys=[OBS_STATE], mode="mean_std", norm_stats=self.state_stats
        )
        self.unnormalize_action_fn = UnNormalizeTransformFn(
            selected_keys=[ACTION], mode="mean_std", norm_stats=self.action_stats
        )
        train_action_mode = None
        if self.train_cfg is not None:
            train_action_mode = str(getattr(self.train_cfg.dataset, "action_mode", "") or "").lower() or None
        requested = None if self.args.action_mode is None else str(self.args.action_mode).lower()
        if requested and train_action_mode and requested != train_action_mode:
            raise RuntimeError(
                f"action_mode 与 checkpoint 训练配置不一致: 请求={requested!r}, checkpoint={train_action_mode!r}"
            )
        self.action_mode = requested or train_action_mode or "delta"
        self.delta_mask = None
        if self.action_mode == "delta":
            try:
                mask = get_mask_mapping(self.stats_key)
            except KeyError:
                mask = get_mask_mapping("aloha")
            self.delta_mask = mask.detach().cpu().numpy().astype(np.float32)

    def build_metadata(self, chunk_size: int) -> dict[str, Any]:
        return {
            "model_type": self.config.type,
            "adapter": self.name,
            "family": self.family,
            "deployment": "mytbot_policy_server",
            "checkpoint_dir": str(self.ckpt_dir),
            "stats_key": self.stats_key,
            "action_mode": self.action_mode,
            "device": self.device,
            "dtype": str(self.runtime_dtype),
            "infer_horizon": self.infer_horizon,
            "chunk_size": chunk_size,
            "target_action_dim": self.target_action_dim,
            "default_prompt": self.args.default_prompt,
            "protocol": "msgpack observation dict -> action dict",
            "input_masks": [rule.spec for rule in self.input_mask_rules],
        }

    def pack_inputs(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        inputs = {}
        for key, value in sample.items():
            if key == "task" or not isinstance(value, torch.Tensor):
                continue
            inputs[key] = batch_model_tensor(value, self.device, self.runtime_dtype)
        return inputs

    def predict(self, inputs: dict[str, Any]) -> torch.Tensor:
        output = self.policy.predict_action_chunk(inputs, decode_image=False)
        if isinstance(output, tuple):
            output = output[0]
        if output.ndim != 3:
            raise RuntimeError(f"策略输出 shape 异常: {tuple(output.shape)}，期望 (B, T, A)")
        return output

    def postprocess(self, action_pred: torch.Tensor, states: list[np.ndarray]) -> list[dict[str, Any]]:
        results = []
        for batch_idx, state in enumerate(states):
            width = min(self.target_action_dim, int(action_pred.shape[-1]))
            model_action = action_pred[batch_idx, : self.infer_horizon, :width]
            action = self.unnormalize_action_fn({ACTION: model_action})[ACTION]
            model_action_np = model_action.detach().cpu().numpy().astype(np.float32)
            action_np = action.detach().cpu().numpy().astype(np.float32)
            if self.action_mode == "delta" and self.delta_mask is not None:
                state_pad = np.zeros_like(self.delta_mask, dtype=np.float32)
                usable_dims = min(len(state_pad), len(state))
                state_pad[:usable_dims] = state[:usable_dims]
                action_dims = min(action_np.shape[-1], len(self.delta_mask))
                action_np[:, :action_dims] += state_pad[None, :action_dims] * self.delta_mask[None, :action_dims]
            results.append(
                {
                    "actions": action_np,
                    "action": action_np[0],
                    "model_actions": model_action_np,
                    "model_action": model_action_np[0],
                }
            )
        return results

    def infer_batch(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not observations:
            return []
        if self.input_mask_rules:
            observations = [apply_client_masks(obs, self.input_mask_rules) for obs in observations]
        prepared = [self.prepare(obs) for obs in observations]
        if self.input_mask_rules:
            for item in prepared:
                apply_model_masks(item.inputs, self.input_mask_rules)
        try:
            return self._infer_prepared(prepared)
        except RuntimeError as exc:
            if len(prepared) == 1:
                raise
            logging.warning("动态 batch 失败，改为逐条推理 | batch_size=%d | %s", len(prepared), exc)
            results = []
            for item in prepared:
                results.extend(self._infer_prepared([item]))
            return results

    def _infer_prepared(self, prepared: list[PreparedSample]) -> list[dict[str, Any]]:
        inputs = collate_nested([item.inputs for item in prepared])
        with torch.inference_mode():
            action_pred = self.predict(inputs)
        if action_pred.shape[0] != len(prepared):
            raise RuntimeError(
                f"策略输出 batch 维异常: {tuple(action_pred.shape)}，期望 {len(prepared)}"
            )
        return self.postprocess(action_pred, [item.state for item in prepared])
