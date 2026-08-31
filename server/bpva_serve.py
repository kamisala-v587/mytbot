#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BPVA / LeRobot WebSocket inference server.

The wire protocol and action post-processing match ``serve_lerobot_policy.py``.
Each request must additionally contain ``task_type``.  A YAML mapping resolves
that task to one LeRobot dataset; the first readable episode is converted to a
fixed behavior prompt and cached for the lifetime of the server.

cd /vla/workspace/my_tbot

conda activate bptbot
python server/bpva_serve.py \
  --ckpt_path /vla/workspace/my_tbot/outputs/ckpts/bpva/bpva-clean-robotwin-v1.1/035000/pretrained_model \
  --bp_mapping_path /vla/workspace/my_tbot/.config/bpva_task_bps.yml \
  --stats_path /vla/workspace/my_tbot/norm_stats/robotwin_delta/stats.json \
  --action_mode delta \
  --host 0.0.0.0 \
  --port 8000


PYTHONPATH=/vla/workspace/my_tbot/src \
/vla/.conda/miniconda3/envs/bptbot/bin/python \
server/bpva_serve.py \
  --ckpt_path /path/to/bpva/checkpoint \
  --bp_mapping_path /vla/workspace/my_tbot/.config/bpva_task_bps.yml \
  --stats_path /vla/workspace/my_tbot/norm_stats/robotwin_delta/stats.json \
  --action_mode delta \
  --host 0.0.0.0 \
  --port 8000

"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import socket
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
MYTBOT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = MYTBOT_ROOT / "src"
for search_path in (SCRIPT_DIR, SRC_ROOT, MYTBOT_ROOT):
    path_str = str(search_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from serve_lerobot_policy import (  # noqa: E402
    CAMERA_ALIASES,
    TBotSA1PolicyService,
    WebsocketPolicyServer,
    coerce_history,
    load_train_config_or_none,
    resolve_ckpt_dir,
    resolve_device,
    resolve_runtime_dtype,
    resolve_stats,
)
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import get_policy_class  # noqa: E402
from lerobot.policies.names import is_bpva  # noqa: E402
from lerobot.transforms.constants import get_image_mapping, get_mask_mapping  # noqa: E402
from lerobot.transforms.core import NormalizeTransformFn, ResizeImagesWithPadFn, UnNormalizeTransformFn  # noqa: E402
from lerobot.transforms.core_bp import (  # noqa: E402
    BPDeltaActionTransformFn,
    BPNormalizeTransformFn,
    BPPadOrSampleChunksFn,
    BPPadStateAndActionTransformFn,
    BPRemapImageKeyTransformFn,
    BPResizeImagesWithPadFn,
    ImgOnlyQwen3VLTransformFn,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: E402

DEFAULT_MAPPING_PATH = SCRIPT_DIR / "bpva_task_bps.yml"
DEFAULT_QWEN3_VL_PATH = Path("/vla/workspace/models/Qwen3-VL-2B-Instruct")
BP_PREFIX = "behavior_prompt"


@dataclass(frozen=True)
class BPTaskSource:
    task_type: str
    dataset_path: Path
    video_backend: str = "pyav"


@dataclass
class BPVAServeArgs:
    ckpt_path: str
    bp_mapping_path: str = str(DEFAULT_MAPPING_PATH)
    host: str = "0.0.0.0"
    port: int = 8000
    default_prompt: str = "Execute the task."
    device: str = "auto"
    dtype: str = "bfloat16"
    infer_horizon: int | None = None
    stats_key: str | None = None
    stats_path: str | None = None
    action_mode: str | None = None
    resize_size: int = 224
    request_image_height: int = 480
    request_image_width: int = 640
    qwen3_vl_processor_path: str | None = None
    qwen3_vl_pretrained_path: str | None = None
    cosmos_tokenizer_path_or_name: str | None = None
    da3_model_path_or_name: str | None = None
    da3_code_root: str | None = None
    load_device: str | None = None
    cosmos_device: str | None = None
    disable_3d_teacher_for_eval: bool = True
    omit_visual_tokens_in_causal_inference: bool = True
    num_inference_steps: int | None = None


def _env_fallback(value: str | None, name: str) -> str | None:
    return value if value is not None else os.environ.get(name)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be boolean-like, got {raw!r}")


def parse_args() -> BPVAServeArgs:
    parser = argparse.ArgumentParser(
        description="启动 mytbot BPVA checkpoint 的 WebSocket 推理服务。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--bp_mapping_path", default=str(DEFAULT_MAPPING_PATH))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default_prompt", default="Execute the task.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--infer_horizon", type=int, default=None)
    parser.add_argument("--stats_key", default=None)
    parser.add_argument("--stats_path", default=None)
    parser.add_argument("--action_mode", choices=["abs", "delta"], default=None)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--request_image_height", type=int, default=480)
    parser.add_argument("--request_image_width", type=int, default=640)
    parser.add_argument("--qwen3_vl_processor_path", default=None)
    parser.add_argument("--qwen3_vl_pretrained_path", default=None)
    parser.add_argument("--cosmos_tokenizer_path_or_name", default=None)
    parser.add_argument("--da3_model_path_or_name", default=None)
    parser.add_argument("--da3_code_root", default=None)
    parser.add_argument("--load_device", default=None)
    parser.add_argument("--cosmos_device", default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--disable_3d_teacher_for_eval", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--omit_visual_tokens_in_causal_inference", action=argparse.BooleanOptionalAction, default=None
    )
    parsed = parser.parse_args()
    disable_3d = parsed.disable_3d_teacher_for_eval
    if disable_3d is None:
        disable_3d = _bool_env("DISABLE_DA3_TEACHER_FOR_EVAL", True)
    omit_visual = parsed.omit_visual_tokens_in_causal_inference
    if omit_visual is None:
        omit_visual = _bool_env("OMIT_VISUAL_TOKENS_IN_CAUSAL_INFERENCE", True)
    return BPVAServeArgs(
        ckpt_path=parsed.ckpt_path,
        bp_mapping_path=parsed.bp_mapping_path,
        host=parsed.host,
        port=parsed.port,
        default_prompt=parsed.default_prompt,
        device=parsed.device,
        dtype=parsed.dtype,
        infer_horizon=parsed.infer_horizon,
        stats_key=_env_fallback(parsed.stats_key, "STATS_KEY"),
        stats_path=_env_fallback(parsed.stats_path, "STATS_PATH"),
        action_mode=_env_fallback(parsed.action_mode, "ACTION_MODE"),
        resize_size=parsed.resize_size,
        request_image_height=parsed.request_image_height,
        request_image_width=parsed.request_image_width,
        qwen3_vl_processor_path=_env_fallback(parsed.qwen3_vl_processor_path, "QWEN3_VL_PROCESSOR_PATH"),
        qwen3_vl_pretrained_path=_env_fallback(parsed.qwen3_vl_pretrained_path, "QWEN3_VL_PRETRAINED_PATH"),
        cosmos_tokenizer_path_or_name=_env_fallback(
            parsed.cosmos_tokenizer_path_or_name, "COSMOS_TOKENIZER_PATH_OR_NAME"
        ),
        da3_model_path_or_name=_env_fallback(parsed.da3_model_path_or_name, "DA3_MODEL_PATH_OR_NAME"),
        da3_code_root=_env_fallback(parsed.da3_code_root, "DA3_CODE_ROOT"),
        load_device=_env_fallback(parsed.load_device, "LOAD_DEVICE"),
        cosmos_device=_env_fallback(parsed.cosmos_device, "COSMOS_DEVICE"),
        disable_3d_teacher_for_eval=disable_3d,
        omit_visual_tokens_in_causal_inference=omit_visual,
        num_inference_steps=parsed.num_inference_steps,
    )


def load_bp_task_sources(mapping_path: str | Path) -> tuple[Path, dict[str, BPTaskSource]]:
    path = Path(mapping_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"未找到 BP 映射 YAML: {path}")
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError(f"{path} 必须是 version: 1 的映射配置。")
    tasks = raw.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError(f"{path} 的 tasks 必须是非空字典。")
    sources: dict[str, BPTaskSource] = {}
    for raw_task, entry in tasks.items():
        task = str(raw_task).strip()
        if not task or not isinstance(entry, dict):
            raise ValueError(f"非法 task 映射: {raw_task!r}: {entry!r}")
        configured = entry.get("dataset_path")
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError(f"task={task!r} 缺少非空 dataset_path。")
        dataset_path = Path(configured).expanduser()
        if not dataset_path.is_absolute():
            dataset_path = (path.parent / dataset_path).resolve()
        else:
            dataset_path = dataset_path.resolve()
        info_path = dataset_path / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"task={task!r} 的 LeRobot 数据集无 meta/info.json: {dataset_path}")
        video_backend = str(entry.get("video_backend", "pyav")).strip() or "pyav"
        sources[task] = BPTaskSource(task, dataset_path, video_backend)
    return path, sources


class BehaviorPromptCache:
    """Lazily build and cache one deterministic BP per task type."""

    def __init__(
        self,
        sources: dict[str, BPTaskSource],
        *,
        config: PreTrainedConfig,
        state_stats: dict[str, dict[str, np.ndarray]],
        action_stats: dict[str, dict[str, np.ndarray]],
        action_mode: str,
    ) -> None:
        self.sources = dict(sources)
        self.config = config
        self.norm_stats = {**state_stats, **action_stats}
        self.action_mode = action_mode
        self._cache: dict[str, dict[str, Any]] = {}
        self._source_episodes: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def cached_tasks(self) -> tuple[str, ...]:
        return tuple(sorted(self._cache))

    def get(self, task_type: str) -> dict[str, Any]:
        if task_type not in self.sources:
            configured = ", ".join(sorted(self.sources))
            raise KeyError(f"未知 task_type={task_type!r}；已配置任务: {configured}")
        with self._lock:
            if task_type not in self._cache:
                prompt, episode = self._build_first_readable(self.sources[task_type])
                self._validate(prompt, task_type)
                self._cache[task_type] = prompt
                self._source_episodes[task_type] = episode
                logging.info("BP 已缓存 | task_type=%s | episode=%d", task_type, episode)
            return self._cache[task_type]

    def _build_first_readable(self, source: BPTaskSource) -> tuple[dict[str, Any], int]:
        meta = LeRobotDatasetMetadata(str(source.dataset_path))
        errors: list[str] = []
        for episode_idx in range(meta.total_episodes):
            try:
                return self._build_episode(source, meta, episode_idx), episode_idx
            except Exception as exc:
                errors.append(f"episode={episode_idx}: {type(exc).__name__}: {exc}")
                logging.warning("跳过不可读取 BP episode | task=%s | %s", source.task_type, errors[-1])
        detail = "\n".join(errors[-5:])
        raise RuntimeError(f"task={source.task_type!r} 没有可读取 episode。最近错误:\n{detail}")

    def _build_episode(
        self, source: BPTaskSource, meta: LeRobotDatasetMetadata, episode_idx: int
    ) -> dict[str, Any]:
        action_steps = int(getattr(self.config, "bp_action_chunk_size"))
        delta_timestamps = {
            ACTION: [step / meta.fps for step in range(action_steps)]
        }
        dataset = LeRobotDataset(
            str(source.dataset_path),
            episodes=[episode_idx],
            delta_timestamps=delta_timestamps,
            video_backend=source.video_backend,
        )
        if len(dataset) <= 0:
            raise ValueError("episode 为空")
        num_chunks = min(
            int(getattr(self.config, "bp_num_chunks")),
            max(1, math.ceil(len(dataset) / action_steps)),
        )
        frame_indices = (
            torch.linspace(0, len(dataset) - 1, steps=num_chunks).round().to(torch.long).tolist()
        )
        samples = [dataset[index] for index in frame_indices]
        image_keys = list(dataset.meta.camera_keys)
        prompt = {
            "images": {key: torch.stack([sample[key] for sample in samples]) for key in image_keys},
            "state": torch.stack([sample[OBS_STATE] for sample in samples]),
            "action": torch.stack([sample[ACTION] for sample in samples]),
            "action_is_pad": torch.stack(
                [sample.get("action_is_pad", torch.zeros(action_steps, dtype=torch.bool)) for sample in samples]
            ),
            "mask": torch.ones(num_chunks, dtype=torch.bool),
            "chunk_indices": torch.arange(num_chunks, dtype=torch.long),
        }
        data = {BP_PREFIX: prompt}
        image_mapping = get_image_mapping(dataset.meta.robot_type, dataset.meta.features)
        data = BPRemapImageKeyTransformFn(
            mapping=image_mapping,
            bp_camera_keys=list(getattr(self.config, "bp_camera_keys")),
        )(data)
        data = BPPadOrSampleChunksFn(num_chunks=int(getattr(self.config, "bp_num_chunks")))(data)
        height, width = tuple(getattr(self.config, "image_resolution", (224, 224)))
        data = BPResizeImagesWithPadFn(height=int(height), width=int(width))(data)
        if self.action_mode == "delta":
            mask = get_mask_mapping(dataset.meta.robot_type, dataset.meta.features)
            data = BPDeltaActionTransformFn(mask=mask)(data)
        data = BPNormalizeTransformFn(
            selected_keys=[OBS_STATE, ACTION], norm_stats=self.norm_stats
        )(data)
        data = BPPadStateAndActionTransformFn(
            max_state_dim=int(getattr(self.config, "max_state_dim")),
            max_action_dim=int(getattr(self.config, "max_action_dim")),
        )(data)
        return data[BP_PREFIX]

    def _validate(self, prompt: dict[str, Any], task_type: str) -> None:
        k = int(getattr(self.config, "bp_num_chunks"))
        t = int(getattr(self.config, "bp_action_chunk_size"))
        state_dim = int(getattr(self.config, "max_state_dim"))
        action_dim = int(getattr(self.config, "max_action_dim"))
        expected_images = set(getattr(self.config, "bp_camera_keys"))
        if set(prompt.get("images", {})) != expected_images:
            raise ValueError(f"task={task_type!r} BP 相机不匹配: {sorted(prompt.get('images', {}))}")
        for key, image in prompt["images"].items():
            if image.ndim != 4 or image.shape[0] != k or image.shape[1] != 3:
                raise ValueError(f"task={task_type!r} BP 图像 {key} shape 非法: {tuple(image.shape)}")
        if tuple(prompt["state"].shape) != (k, state_dim):
            raise ValueError(f"task={task_type!r} BP state shape 非法: {tuple(prompt['state'].shape)}")
        if tuple(prompt["action"].shape) != (k, t, action_dim):
            raise ValueError(f"task={task_type!r} BP action shape 非法: {tuple(prompt['action'].shape)}")
        if tuple(prompt["action_is_pad"].shape) != (k, t):
            raise ValueError(f"task={task_type!r} BP action_is_pad shape 非法")
        if tuple(prompt["mask"].shape) != (k,):
            raise ValueError(f"task={task_type!r} BP mask shape 非法")


class BPVAPolicyService(TBotSA1PolicyService):
    """BPVA checkpoint plus server-side task-specific behavior prompts."""

    def __init__(self, args: BPVAServeArgs):
        self.args = args
        self.ckpt_dir = resolve_ckpt_dir(args.ckpt_path)
        self.train_cfg = load_train_config_or_none(self.ckpt_dir)
        self.config = PreTrainedConfig.from_pretrained(self.ckpt_dir)
        if not is_bpva(self.config.type):
            raise ValueError(f"当前脚本仅支持 BPVA，checkpoint type={self.config.type!r}。")
        self._apply_runtime_overrides()
        # 完整 BPVA checkpoint 已包含 BP ViT 权重。构造模型时禁止 timm 再从
        # Hugging Face 下载初始化权重；from_pretrained 随后会恢复 checkpoint 权重。
        if bool(getattr(self.config, "bp_vision_pretrained", False)):
            logging.info("离线加载 BPVA checkpoint：跳过 timm BP ViT 预训练权重下载。")
            self.config.bp_vision_pretrained = False
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
            setattr(
                self.policy.model,
                "omit_visual_tokens_in_causal_inference",
                bool(args.omit_visual_tokens_in_causal_inference),
            )
        self.policy.config.device = self.device
        setattr(self.policy.config, "cosmos_device", self.cosmos_device)
        self.policy.to(device=self.device, dtype=self.runtime_dtype).eval()
        self.policy.requires_grad_(False)

        stats_path = Path(args.stats_path).expanduser() if args.stats_path else self.ckpt_dir / "stats.json"
        if not stats_path.is_file():
            raise FileNotFoundError(f"未找到 stats.json: {stats_path}")
        self.stats_key, stats = resolve_stats(stats_path, args.stats_key)
        stat_keys = ["min", "max", "mean", "std"]
        self.state_stats = {OBS_STATE: {key: np.asarray(stats[OBS_STATE][key]) for key in stat_keys}}
        self.action_stats = {ACTION: {key: np.asarray(stats[ACTION][key]) for key in stat_keys}}
        self.action_mean = np.asarray(self.action_stats[ACTION]["mean"], dtype=np.float32)
        self.action_std = np.asarray(self.action_stats[ACTION]["std"], dtype=np.float32)
        self.target_action_dim = int(self.action_mean.shape[0])

        processor_path = (
            args.qwen3_vl_processor_path
            or getattr(self.config, "qwen3_vl_processor_path", None)
            or getattr(self.config, "qwen3_vl_pretrained_path", None)
            or str(DEFAULT_QWEN3_VL_PATH)
        )
        self.resize_fn = ResizeImagesWithPadFn(height=args.resize_size, width=args.resize_size)
        self.normalize_state_fn = NormalizeTransformFn(
            selected_keys=[OBS_STATE], mode="mean_std", norm_stats=self.state_stats
        )
        self.unnormalize_action_fn = UnNormalizeTransformFn(
            selected_keys=[ACTION], mode="mean_std", norm_stats=self.action_stats
        )
        self.processor_fn = ImgOnlyQwen3VLTransformFn(pretrained_model_name_or_path=processor_path)

        train_action_mode = None
        if self.train_cfg is not None:
            train_action_mode = str(getattr(self.train_cfg.dataset, "action_mode", "") or "").lower() or None
        requested_action_mode = str(args.action_mode).lower() if args.action_mode else None
        if requested_action_mode and train_action_mode and requested_action_mode != train_action_mode:
            raise RuntimeError(
                f"action_mode 与 checkpoint 训练配置不一致: 请求={requested_action_mode!r}, "
                f"checkpoint={train_action_mode!r}"
            )
        self.action_mode = requested_action_mode or train_action_mode or "delta"
        self.delta_mask = None
        if self.action_mode == "delta":
            try:
                mask = get_mask_mapping(self.stats_key)
            except KeyError:
                mask = get_mask_mapping("aloha")
            self.delta_mask = mask.detach().cpu().numpy().astype(np.float32)

        self.mapping_path, sources = load_bp_task_sources(args.bp_mapping_path)
        self.bp_cache = BehaviorPromptCache(
            sources,
            config=self.config,
            state_stats=self.state_stats,
            action_stats=self.action_stats,
            action_mode=self.action_mode,
        )
        self._metadata = {
            "model_type": self.config.type,
            "deployment": "mytbot_bpva_websocket_server",
            "checkpoint_dir": str(self.ckpt_dir),
            "stats_key": self.stats_key,
            "action_mode": self.action_mode,
            "device": self.device,
            "dtype": str(self.runtime_dtype),
            "infer_horizon": self.infer_horizon,
            "chunk_size": chunk_size,
            "expected_action_dim": self.target_action_dim,
            "bp_mapping_path": str(self.mapping_path),
            "bp_selection": "first_readable_episode",
            "bp_configured_tasks": sorted(sources),
            "bp_num_chunks": int(getattr(self.config, "bp_num_chunks")),
            "bp_action_chunk_size": int(getattr(self.config, "bp_action_chunk_size")),
            "client_payload": "current observation + task text + required task_type",
        }
        logging.info(
            "BPVA Server 就绪 | device=%s | horizon=%d | action_mode=%s | tasks=%d",
            self.device,
            self.infer_horizon,
            self.action_mode,
            len(sources),
        )

    def _resolve_task_type(self, obs: dict[str, Any]) -> str:
        task_type = obs.get("task_type")
        if not isinstance(task_type, str) or not task_type.strip():
            raise KeyError("BPVA 请求缺少非空字符串 task_type（例如 adjust_bottle）。")
        return task_type.strip()

    def _resolve_images(self, obs: dict[str, Any]) -> dict[str, Any]:
        nested = obs.get("images")
        if isinstance(nested, dict):
            return nested
        return obs

    def _obs_to_inputs(self, obs: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        images = self._resolve_images(obs)
        head_history, head_mask = self._resolve_image_history(images, f"{OBS_IMAGES}.image0")
        left_history, left_mask = self._resolve_image_history(images, f"{OBS_IMAGES}.image1")
        right_history, right_mask = self._resolve_image_history(images, f"{OBS_IMAGES}.image2")
        state = self._resolve_state(obs)
        sample = {
            f"{OBS_IMAGES}.image0": torch.from_numpy(head_history).permute(0, 3, 1, 2).float() / 255.0,
            f"{OBS_IMAGES}.image1": torch.from_numpy(left_history).permute(0, 3, 1, 2).float() / 255.0,
            f"{OBS_IMAGES}.image2": torch.from_numpy(right_history).permute(0, 3, 1, 2).float() / 255.0,
            OBS_STATE: torch.from_numpy(state),
        }
        sample = self.resize_fn(sample)
        sample[f"{OBS_IMAGES}.image0_mask"] = torch.tensor(head_mask)
        sample[f"{OBS_IMAGES}.image1_mask"] = torch.tensor(left_mask)
        sample[f"{OBS_IMAGES}.image2_mask"] = torch.tensor(right_mask)
        sample = self.processor_fn(sample)
        sample = self.normalize_state_fn(sample)
        inputs = {key: self._batch_tensor(value) for key, value in sample.items() if isinstance(value, torch.Tensor)}
        task_type = self._resolve_task_type(obs)
        inputs[BP_PREFIX] = self._batch_nested(self.bp_cache.get(task_type))
        return inputs, state

    def _batch_tensor(self, value: torch.Tensor) -> torch.Tensor:
        value = value.unsqueeze(0)
        if value.is_floating_point():
            return value.to(device=self.device, dtype=self.runtime_dtype)
        return value.to(self.device)

    def _batch_nested(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._batch_nested(item) for key, item in value.items()}
        if isinstance(value, torch.Tensor):
            return self._batch_tensor(value)
        raise TypeError(f"behavior_prompt 包含不支持的类型: {type(value).__name__}")


def main(args: BPVAServeArgs) -> None:
    logging.info("启动参数:\n%s", json.dumps(asdict(args), indent=2, ensure_ascii=False))
    policy = BPVAPolicyService(args)
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "unknown"
    logging.info("BPVA WebSocket 服务 | host=%s ip=%s port=%d", args.host, local_ip, args.port)
    logging.info("Server metadata:\n%s", json.dumps(policy.metadata, indent=2, ensure_ascii=False))
    WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    main(parse_args())
