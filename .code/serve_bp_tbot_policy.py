#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mytbot BP_TBot / LeRobot Policy WebSocket Server
================================================

协议尽量沿用 ``.code/serve_lerobot_policy.py``：客户端发送当前观测，服务端返回 action。
BP_TBot 额外需要 behavior prompt：

1. 如果客户端 payload 中包含 ``behavior_prompt``，优先使用客户端传入的 BP。
2. 如果客户端未提供，则从 ``--bp_dataset`` 指定的 LeRobot 3.0 数据集中随机采样一条。

客户端可传已经处理好的 BP 字段，也可传 raw BP 字段。raw BP 推荐结构::

    {
        "images": {"cam_high": [K,H,W,C], ...},
        "state": [K,D],
        "action": [K,T,D],
        "action_is_pad": [K,T],
        "mask": [K],
    }

启动示例::

    python .code/serve_bp_tbot_policy.py \
        --ckpt_path outputs/BP_TBot/pretrain_v1/.../checkpoints/011000 \
        --bp_dataset /vla/workspace/data/hanging_mug/aloha-agilex_randomized_500 \
        --host 0.0.0.0 --port 8000 \
        --default_prompt "hang the mug"
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http
import json
import logging
import os
import random
import socket
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames

SCRIPT_DIR = Path(__file__).resolve().parent
MYTBOT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = MYTBOT_ROOT / "src"

for path in (SCRIPT_DIR, SRC_ROOT, MYTBOT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets.behavior_prompt_dataset import BehaviorPromptLeRobotDataset  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.utils import load_json  # noqa: E402
from lerobot.policies.factory import get_policy_class  # noqa: E402
from lerobot.policies.names import is_bp_tbot  # noqa: E402
from lerobot.transforms.constants import get_mask_mapping  # noqa: E402
from lerobot.transforms.core import (  # noqa: E402
    ComposeFieldsTransform,
    DeltaActionTransformFn,
    InjectMissingStateActionTransformFn,
    NormalizeTransformFn,
    PadStateAndActionTransformFn,
    RemapImageKeyTransformFn,
    ResizeImagesWithPadFn,
    UnNormalizeTransformFn,
    hydrate_compose_field_transform,
    hydrate_delta_action_transform,
    hydrate_inject_missing_state_action_transform,
    hydrate_normalize_transform,
    hydrate_remap_image_key_transform,
)
from lerobot.transforms.core_bp import (  # noqa: E402
    BPComposeFieldsTransform,
    BPDeltaActionTransformFn,
    BPImgOnlyQwen3VLTransformFn,
    BPNormalizeTransformFn,
    BPPadOrSampleChunksFn,
    BPPadStateAndActionTransformFn,
    BPRemapImageKeyTransformFn,
    BPResizeImagesWithPadFn,
    ImgOnlyQwen3VLTransformFn,
    UnifyBPInputsTransformFn,
)
from lerobot.transforms.constants import get_feature_mapping, get_image_mapping, get_mask_mapping as get_robot_mask_mapping  # noqa: E402
from lerobot.transforms.core import compose  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_STR  # noqa: E402

DEFAULT_QWEN3_VL_PATH = Path("/home/jovyan/vla/workspace/models/Qwen3-vl-2b-instruct")

FIELD_ALIASES = {
    f"{OBS_IMAGES}.image0": (
        "cam_high",
        "head",
        "image0",
        "observation.images.cam_high",
        f"{OBS_IMAGES}.image0",
    ),
    f"{OBS_IMAGES}.image1": (
        "cam_left_wrist",
        "left_wrist",
        "left",
        "image1",
        "observation.images.cam_left_wrist",
        f"{OBS_IMAGES}.image1",
    ),
    f"{OBS_IMAGES}.image2": (
        "cam_right_wrist",
        "right_wrist",
        "right",
        "image2",
        "observation.images.cam_right_wrist",
        f"{OBS_IMAGES}.image2",
    ),
    OBS_STATE: ("state", "qpos", OBS_STATE),
}

CAMERA_ALIASES = {
    key: aliases for key, aliases in FIELD_ALIASES.items() if key.startswith(OBS_IMAGES)
}

RAW_BP_CAMERA_KEYS = {
    "cam_high": "observation.images.cam_high",
    "head": "observation.images.cam_high",
    "image0": "observation.images.cam_high",
    f"{OBS_IMAGES}.image0": "observation.images.cam_high",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "left_wrist": "observation.images.cam_left_wrist",
    "left": "observation.images.cam_left_wrist",
    "image1": "observation.images.cam_left_wrist",
    f"{OBS_IMAGES}.image1": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
    "right_wrist": "observation.images.cam_right_wrist",
    "right": "observation.images.cam_right_wrist",
    "image2": "observation.images.cam_right_wrist",
    f"{OBS_IMAGES}.image2": "observation.images.cam_right_wrist",
}

BP_TENSOR_KEYS = {
    "mask",
    "chunk_indices",
    "source_time_ratio",
    "state",
    "action",
    "action_is_pad",
    "pixel_values",
    "image_grid_thw",
    "image_token_counts",
    "image_chunk_indices",
    "image_camera_indices",
}


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype for msgpack: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: Any) -> Any:
    if isinstance(obj, dict) and b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if isinstance(obj, dict) and b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


MsgpackPacker = functools.partial(msgpack.Packer, default=_pack_array)
msgpack_unpack = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


@dataclass
class ServeArgs:
    ckpt_path: str
    bp_dataset: str
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
    bp_dataset_revision: str = "v3.0"
    bp_num_chunks: int | None = None
    bp_seed: int = 0
    bp_same_episode_policy: str = "avoid"
    bp_cache_size: int = 16
    disable_3d_teacher_for_eval: bool = True
    num_inference_steps: int | None = None


def _env_fallback(value: str | None, env_name: str) -> str | None:
    return value if value is not None else os.environ.get(env_name)


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


def parse_args() -> ServeArgs:
    parser = argparse.ArgumentParser(
        description="启动 mytbot BP_TBot checkpoint 的 WebSocket 推理服务。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt_path", required=True, help="checkpoint step 目录或 pretrained_model 目录。")
    parser.add_argument("--bp_dataset", required=True, help="LeRobot 3.0 数据集路径，用作客户端未传 BP 时的指导轨迹来源。")
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
    parser.add_argument("--bp_dataset_revision", default="v3.0")
    parser.add_argument("--bp_num_chunks", type=int, default=None)
    parser.add_argument("--bp_seed", type=int, default=0)
    parser.add_argument("--bp_same_episode_policy", choices=["avoid", "allow", "forbid"], default="avoid")
    parser.add_argument("--bp_cache_size", type=int, default=16, help="预处理并缓存的 dataset fallback BP 数量；0 表示每次随机现取。")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument(
        "--disable_3d_teacher_for_eval",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="推理时关闭 DA3 teacher（默认读 DISABLE_DA3_TEACHER_FOR_EVAL 环境变量）。",
    )
    parsed = parser.parse_args()
    disable_3d = parsed.disable_3d_teacher_for_eval
    if disable_3d is None:
        disable_3d = _bool_env("DISABLE_DA3_TEACHER_FOR_EVAL", True)
    return ServeArgs(
        ckpt_path=parsed.ckpt_path,
        bp_dataset=parsed.bp_dataset,
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
        cosmos_tokenizer_path_or_name=_env_fallback(parsed.cosmos_tokenizer_path_or_name, "COSMOS_TOKENIZER_PATH_OR_NAME"),
        da3_model_path_or_name=_env_fallback(parsed.da3_model_path_or_name, "DA3_MODEL_PATH_OR_NAME"),
        da3_code_root=_env_fallback(parsed.da3_code_root, "DA3_CODE_ROOT"),
        load_device=_env_fallback(parsed.load_device, "LOAD_DEVICE"),
        cosmos_device=_env_fallback(parsed.cosmos_device, "COSMOS_DEVICE"),
        bp_dataset_revision=parsed.bp_dataset_revision,
        bp_num_chunks=parsed.bp_num_chunks,
        bp_seed=parsed.bp_seed,
        bp_same_episode_policy=parsed.bp_same_episode_policy,
        bp_cache_size=parsed.bp_cache_size,
        disable_3d_teacher_for_eval=disable_3d,
        num_inference_steps=parsed.num_inference_steps,
    )


def resolve_ckpt_dir(ckpt_path: str | Path) -> Path:
    ckpt_dir = Path(ckpt_path).expanduser().resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint 路径不存在: {ckpt_dir}")
    if (ckpt_dir / "config.json").is_file():
        return ckpt_dir
    pretrained_dir = ckpt_dir / "pretrained_model"
    if (pretrained_dir / "config.json").is_file():
        return pretrained_dir
    raise FileNotFoundError(f"在 {ckpt_dir} 或 {pretrained_dir} 下未找到 config.json。")


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_runtime_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        if device == "cpu":
            logging.warning("CPU 上使用 float32 替代 bfloat16。")
            return torch.float32
        return torch.bfloat16
    raise ValueError(f"不支持的 dtype: {dtype_name}")


def to_hwc_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"期望 3 维图像，得到 shape={array.shape}")
    if array.shape[-1] == 3:
        hwc = array
    elif array.shape[0] == 3:
        hwc = np.transpose(array, (1, 2, 0))
    else:
        raise ValueError(f"不支持的图像 shape: {array.shape}")
    if np.issubdtype(hwc.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(hwc)) <= 1.5 else 1.0
        hwc = np.clip(hwc * scale, 0.0, 255.0)
    else:
        hwc = np.clip(hwc, 0, 255)
    return np.ascontiguousarray(hwc.astype(np.uint8))


def coerce_history(image_value: Any) -> np.ndarray:
    array = np.asarray(image_value)
    if array.ndim == 3:
        frame = to_hwc_uint8(array)
        return np.stack([frame, frame], axis=0)
    if array.ndim != 4:
        raise ValueError(f"不支持的时序图像 shape: {array.shape}")
    if array.shape[-1] == 3:
        frames = [to_hwc_uint8(array[idx]) for idx in range(array.shape[0])]
    elif array.shape[1] == 3:
        frames = [to_hwc_uint8(array[idx]) for idx in range(array.shape[0])]
    else:
        raise ValueError(f"不支持的时序图像 shape: {array.shape}")
    if len(frames) == 1:
        return np.stack([frames[0], frames[0]], axis=0)
    return np.stack([frames[0], frames[-1]], axis=0)


def resolve_stats(stats_path: Path, requested_key: str | None) -> tuple[str, dict[str, Any]]:
    stats_root = load_json(stats_path)
    if OBS_STATE in stats_root and ACTION in stats_root:
        return requested_key or "default", stats_root
    if requested_key is not None:
        if requested_key not in stats_root:
            raise KeyError(f"stats_key={requested_key!r} 不在 {stats_path} 中")
        return requested_key, stats_root[requested_key]
    if len(stats_root) == 1:
        key = next(iter(stats_root))
        return key, stats_root[key]
    if "aloha" in stats_root:
        return "aloha", stats_root["aloha"]
    raise ValueError(f"stats.json 含多个 key {list(stats_root.keys())}，请指定 --stats_key。")


def load_train_config_or_none(ckpt_dir: Path) -> TrainPipelineConfig | None:
    try:
        return TrainPipelineConfig.from_pretrained(ckpt_dir)
    except Exception as exc:
        logging.warning("无法从 %s 加载 train_config.json: %s", ckpt_dir, exc)
        return None


def _tensor_from_any(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.from_numpy(np.asarray(value))


def _add_batch_and_move(value: torch.Tensor, device: str, runtime_dtype: torch.dtype) -> torch.Tensor:
    if value.dtype == torch.bool:
        return value[None].to(device)
    if value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        return value[None].to(device)
    if value.is_floating_point():
        return value[None].to(device=device, dtype=runtime_dtype)
    return value[None].to(device)


def _move_nested_to_device(value: Any, device: str, runtime_dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            return value.to(device)
        if value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            return value.to(device)
        if value.is_floating_point():
            return value.to(device=device, dtype=runtime_dtype)
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_nested_to_device(item, device, runtime_dtype) for key, item in value.items()}
    return value


def _map_alias_fields(payload: dict[str, Any], alias_mapping: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for canonical_key, aliases in alias_mapping.items():
        for alias in aliases:
            if alias in payload:
                mapped[canonical_key] = payload[alias]
                break
    return mapped


def _add_behavior_prompt_batch_dim(prompt: dict[str, Any]) -> dict[str, Any]:
    batched: dict[str, Any] = {}
    for key, value in prompt.items():
        if isinstance(value, torch.Tensor) and key in BP_TENSOR_KEYS:
            batched[key] = value if _looks_batched_bp_tensor(key, value) else value.unsqueeze(0)
        elif key == "images" and isinstance(value, dict):
            batched[key] = {
                image_key: image_value if _looks_batched_bp_image(image_value) else image_value.unsqueeze(0)
                for image_key, image_value in value.items()
            }
        else:
            batched[key] = value
    return batched


def _looks_batched_bp_tensor(key: str, value: torch.Tensor) -> bool:
    expected_unbatched_ndims = {
        "mask": 1,
        "chunk_indices": 1,
        "source_time_ratio": 1,
        "state": 2,
        "action": 3,
        "action_is_pad": 2,
        "pixel_values": 2,
        "image_grid_thw": 2,
        "image_token_counts": 1,
        "image_chunk_indices": 1,
        "image_camera_indices": 1,
    }
    expected = expected_unbatched_ndims.get(key)
    return expected is not None and value.ndim > expected


def _looks_batched_bp_image(value: torch.Tensor) -> bool:
    return value.ndim > 4


class BPTBotPolicyService:
    """BP_TBot checkpoint -> infer(obs) -> actions。"""

    def __init__(self, args: ServeArgs):
        self.args = args
        self.ckpt_dir = resolve_ckpt_dir(args.ckpt_path)
        self.train_cfg = load_train_config_or_none(self.ckpt_dir)

        self.config = PreTrainedConfig.from_pretrained(self.ckpt_dir)
        if not is_bp_tbot(self.config.type):
            raise ValueError(f"当前脚本仅支持 BP_TBot，checkpoint type={self.config.type!r}。")

        self._apply_runtime_overrides()
        self.device = resolve_device(args.device)
        self.load_device = resolve_device(args.load_device) if args.load_device else "cpu"
        self.cosmos_device = resolve_device(args.cosmos_device) if args.cosmos_device else self.device
        self.config.device = self.load_device
        setattr(self.config, "cosmos_device", self.cosmos_device)
        self.runtime_dtype = resolve_runtime_dtype(args.dtype, self.device)
        self.config.dtype = "float32" if self.runtime_dtype == torch.float32 else "bfloat16"

        chunk_size = int(getattr(self.config, "chunk_size", 50))
        n_action_steps = int(getattr(self.config, "n_action_steps", chunk_size))
        self.infer_horizon = int(args.infer_horizon or n_action_steps)
        self.infer_horizon = max(1, min(self.infer_horizon, chunk_size))

        policy_cls = get_policy_class(self.config.type)
        logging.info("加载 %s: %s", policy_cls.__name__, self.ckpt_dir)
        self.policy = policy_cls.from_pretrained(config=self.config, pretrained_name_or_path=self.ckpt_dir)
        self.policy.config.device = self.device
        setattr(self.policy.config, "cosmos_device", self.cosmos_device)
        self.policy.to(device=self.device, dtype=self.runtime_dtype).eval()
        self.policy.requires_grad_(False)

        stats_path = Path(args.stats_path).expanduser() if args.stats_path else self.ckpt_dir / "stats.json"
        if not stats_path.is_file():
            raise FileNotFoundError(f"未找到 stats.json: {stats_path}，请通过 --stats_path 指定。")
        self.stats_key, stats = resolve_stats(stats_path, args.stats_key)
        stat_keys = ["min", "max", "mean", "std"]
        self.state_stats = {OBS_STATE: {k: np.asarray(stats[OBS_STATE][k]) for k in stat_keys}}
        self.action_stats = {ACTION: {k: np.asarray(stats[ACTION][k]) for k in stat_keys}}
        self.action_mean = np.asarray(self.action_stats[ACTION]["mean"], dtype=np.float32)
        self.target_action_dim = int(self.action_mean.shape[0])

        processor_path = (
            args.qwen3_vl_processor_path
            or getattr(self.config, "qwen3_vl_processor_path", None)
            or getattr(self.config, "qwen3_vl_pretrained_path", None)
            or str(DEFAULT_QWEN3_VL_PATH)
        )
        self.unnormalize_action_fn = UnNormalizeTransformFn(
            selected_keys=[ACTION], mode="mean_std", norm_stats=self.action_stats
        )

        train_action_mode = None
        if self.train_cfg is not None:
            train_action_mode = str(getattr(self.train_cfg.dataset, "action_mode", "") or "").lower() or None
        requested_action_mode = None if args.action_mode is None else str(args.action_mode).lower()
        if requested_action_mode is not None and train_action_mode is not None and requested_action_mode != train_action_mode:
            raise RuntimeError(
                f"action_mode 与 checkpoint 训练配置不一致: 请求={requested_action_mode!r}, checkpoint={train_action_mode!r}"
            )
        self.action_mode = requested_action_mode or train_action_mode or "delta"

        self.delta_mask = None
        if self.action_mode == "delta":
            try:
                self.delta_mask = get_mask_mapping(self.stats_key).detach().cpu().numpy().astype(np.float32)
            except KeyError:
                self.delta_mask = get_mask_mapping("aloha").detach().cpu().numpy().astype(np.float32)

        self.rng = random.Random(args.bp_seed)
        self.bp_dataset = self._build_bp_dataset()
        self.current_transforms, self.bp_transform = self._build_server_transforms()
        self.bp_prompt_cache = self._build_bp_prompt_cache()

        self._metadata = {
            "model_type": self.config.type,
            "deployment": "mytbot_bp_tbot_websocket_server",
            "checkpoint_dir": str(self.ckpt_dir),
            "bp_dataset": args.bp_dataset,
            "bp_source_priority": "client_behavior_prompt_then_cached_or_random_dataset_sample",
            "bp_cache_size": len(self.bp_prompt_cache),
            "stats_key": self.stats_key,
            "action_mode": self.action_mode,
            "device": self.device,
            "dtype": str(self.runtime_dtype),
            "infer_horizon": self.infer_horizon,
            "chunk_size": chunk_size,
            "bp_num_chunks": int(getattr(self.config, "bp_num_chunks", args.bp_num_chunks or 4)),
            "bp_action_chunk_size": int(getattr(self.config, "bp_action_chunk_size", chunk_size)),
            "num_inference_steps": int(getattr(self.config, "num_inference_steps", 10)),
            "default_prompt": args.default_prompt,
            "target_action_dim": self.target_action_dim,
            "camera_aliases": {k: list(v) for k, v in CAMERA_ALIASES.items()},
            "notes": {
                "client_payload": "images dict + state + prompt/task + optional behavior_prompt",
                "behavior_prompt_priority": "client behavior_prompt overrides dataset fallback",
            },
        }
        logging.info(
            "BP Server 就绪 | type=%s | device=%s | horizon=%d | action_mode=%s | action_dim=%d",
            self.config.type,
            self.device,
            self.infer_horizon,
            self.action_mode,
            self.target_action_dim,
        )

    def _apply_runtime_overrides(self) -> None:
        args = self.args
        if args.qwen3_vl_pretrained_path and hasattr(self.config, "qwen3_vl_pretrained_path"):
            self.config.qwen3_vl_pretrained_path = args.qwen3_vl_pretrained_path
        if args.qwen3_vl_processor_path and hasattr(self.config, "qwen3_vl_processor_path"):
            self.config.qwen3_vl_processor_path = args.qwen3_vl_processor_path
        if args.cosmos_tokenizer_path_or_name and hasattr(self.config, "cosmos_tokenizer_path_or_name"):
            self.config.cosmos_tokenizer_path_or_name = args.cosmos_tokenizer_path_or_name
        if args.da3_model_path_or_name and hasattr(self.config, "da3_model_path_or_name"):
            self.config.da3_model_path_or_name = args.da3_model_path_or_name
        if args.da3_code_root and hasattr(self.config, "da3_code_root"):
            self.config.da3_code_root = args.da3_code_root
        if args.disable_3d_teacher_for_eval and hasattr(self.config, "lambda_3d"):
            self.config.lambda_3d = 0.0
        if args.num_inference_steps is not None and hasattr(self.config, "num_inference_steps"):
            self.config.num_inference_steps = int(args.num_inference_steps)

    def _resolve_delta_timestamps(self, repo_id: str, root: str, revision: str) -> dict[str, list[float]]:
        ds_meta = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
        fps = float(ds_meta.fps)
        image_delta_indices = getattr(self.config, "image_delta_indices", [-15, 0, 15])
        action_delta_indices = getattr(self.config, "action_delta_indices", list(range(int(self.config.chunk_size))))
        delta_timestamps: dict[str, list[float]] = {}
        for key in ds_meta.features:
            if key == ACTION and action_delta_indices is not None:
                delta_timestamps[key] = [i / fps for i in action_delta_indices]
            if key in ds_meta.camera_keys and image_delta_indices is not None:
                delta_timestamps[key] = [i / fps for i in image_delta_indices]
        return delta_timestamps

    def _build_bp_dataset(self) -> BehaviorPromptLeRobotDataset:
        repo_id = str(Path(self.args.bp_dataset).expanduser())
        root = repo_id
        revision = self.args.bp_dataset_revision
        delta_timestamps = self._resolve_delta_timestamps(repo_id, root, revision)
        current_ds = LeRobotDataset(
            repo_id,
            root=root,
            delta_timestamps=delta_timestamps,
            revision=revision,
            video_backend="pyav",
        )
        frame_ds = LeRobotDataset(repo_id, root=root, revision=revision, video_backend="pyav")
        transforms = self._hydrate_server_transforms(self._training_input_transforms())
        return BehaviorPromptLeRobotDataset(
            current_ds=current_ds,
            frame_ds=frame_ds,
            prompt_cfg=self._prompt_cfg_from_training_config(),
            transform=compose(transforms),
        )

    def _training_input_transforms(self):
        if self.train_cfg is None:
            raise RuntimeError("BP_TBot server requires checkpoint train_config.json to derive data_transforms.")
        return list(self.train_cfg.dataset.data_transforms.inputs)

    def _prompt_cfg_from_training_config(self):
        from lerobot.datasets.behavior_prompt_dataset import BehaviorPromptConfig

        dataset_cfg = self.train_cfg.dataset if self.train_cfg is not None else None
        return BehaviorPromptConfig(
            prompt_action_chunk_size=int(getattr(self.config, "bp_action_chunk_size", self.config.chunk_size)),
            same_episode_policy=self.args.bp_same_episode_policy,
            seed=self.args.bp_seed,
            num_chunks=int(self.args.bp_num_chunks or getattr(dataset_cfg, "bp_num_chunks", getattr(self.config, "bp_num_chunks", 4))),
            height=int(getattr(dataset_cfg, "height", self.args.resize_size)),
            width=int(getattr(dataset_cfg, "width", self.args.resize_size)),
            max_state_dim=int(getattr(dataset_cfg, "max_state_dim", getattr(self.config, "max_state_dim", 32))),
            max_action_dim=int(getattr(dataset_cfg, "max_action_dim", getattr(self.config, "max_action_dim", 32))),
            qwen3_vl_processor_path=str(getattr(dataset_cfg, "qwen3_vl_processor_path", getattr(self.config, "qwen3_vl_processor_path", DEFAULT_QWEN3_VL_PATH))),
            action_mode=self.action_mode,
        )

    def _build_server_transforms(self):
        transforms = self._hydrate_server_transforms(self._training_input_transforms())
        current_transforms = []
        bp_transforms = []
        for transform in transforms:
            if isinstance(transform, UnifyBPInputsTransformFn):
                continue
            if self._is_bp_transform(transform):
                bp_transforms.append(transform)
            elif self._is_current_transform(transform):
                current_transforms.append(self._adapt_current_transform(transform))
        return current_transforms, compose(bp_transforms)

    def _hydrate_server_transforms(self, transforms):
        transforms = list(transforms)
        transforms = hydrate_inject_missing_state_action_transform(transforms, self.bp_dataset.current_ds if hasattr(self, "bp_dataset") else self._server_hydration_dataset())
        dataset = self.bp_dataset.current_ds if hasattr(self, "bp_dataset") else self._server_hydration_dataset()
        transforms = hydrate_normalize_transform(transforms, dataset)
        transforms = hydrate_compose_field_transform(transforms, dataset)
        transforms = hydrate_delta_action_transform(transforms, dataset)
        transforms = hydrate_remap_image_key_transform(transforms, dataset)
        transforms = self._hydrate_bp_transforms(transforms, dataset)
        return transforms

    def _server_hydration_dataset(self):
        repo_id = str(Path(self.args.bp_dataset).expanduser())
        root = repo_id
        revision = self.args.bp_dataset_revision
        delta_timestamps = self._resolve_delta_timestamps(repo_id, root, revision)
        return LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps, revision=revision, video_backend="pyav")

    def _hydrate_bp_transforms(self, transforms, dataset):
        from dataclasses import replace

        robot_type = dataset.meta.robot_type
        features = dataset.meta.features
        feature_mapping = get_feature_mapping(robot_type, features)
        image_mapping = get_image_mapping(robot_type, features)
        selected_keys = feature_mapping[OBS_STATE] + feature_mapping[ACTION]
        hydrated = []
        for transform in transforms:
            if isinstance(transform, BPNormalizeTransformFn):
                transform = replace(transform, norm_stats=dataset.meta.stats, selected_keys=selected_keys)
            elif isinstance(transform, BPComposeFieldsTransform):
                transform = replace(transform, mapping=feature_mapping)
            elif isinstance(transform, BPDeltaActionTransformFn):
                transform = replace(transform, mask=get_robot_mask_mapping(robot_type, features))
            elif isinstance(transform, BPRemapImageKeyTransformFn):
                transform = replace(transform, mapping=image_mapping)
            hydrated.append(transform)
        return hydrated

    @staticmethod
    def _adapt_current_transform(transform):
        if isinstance(transform, NormalizeTransformFn):
            from dataclasses import replace

            return replace(transform, selected_keys=[OBS_STATE])
        return transform

    def _build_bp_prompt_cache(self) -> list[dict[str, Any]]:
        cache_size = max(0, int(self.args.bp_cache_size))
        if cache_size <= 0:
            return []
        count = min(cache_size, len(self.bp_dataset))
        cache = []
        for _ in range(count):
            cache.append(self._sample_dataset_behavior_prompt_uncached())
        logging.info("已缓存 %d 条 dataset fallback behavior_prompt。", len(cache))
        return cache

    @staticmethod
    def _is_bp_transform(transform) -> bool:
        return isinstance(
            transform,
            (
                BPPadOrSampleChunksFn,
                BPResizeImagesWithPadFn,
                BPRemapImageKeyTransformFn,
                BPNormalizeTransformFn,
                BPComposeFieldsTransform,
                BPDeltaActionTransformFn,
                BPPadStateAndActionTransformFn,
                BPImgOnlyQwen3VLTransformFn,
            ),
        )

    @staticmethod
    def _is_current_transform(transform) -> bool:
        # Online observations are already mapped to canonical image keys and do
        # not include action, so only keep transforms that can operate on obs-only samples.
        return isinstance(transform, (ResizeImagesWithPadFn, NormalizeTransformFn, ImgOnlyQwen3VLTransformFn))

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _resolve_prompt(self, obs: dict[str, Any]) -> str:
        prompt = obs.get("prompt") or obs.get("task") or self.args.default_prompt
        if not isinstance(prompt, str) or not prompt.strip():
            return self.args.default_prompt
        return prompt

    def _resolve_state(self, mapped_obs: dict[str, Any]) -> np.ndarray:
        if OBS_STATE not in mapped_obs:
            raise KeyError(f"请求缺少 state 字段，可接受别名: {FIELD_ALIASES[OBS_STATE]}")
        state = mapped_obs[OBS_STATE]
        return np.ascontiguousarray(np.asarray(state, dtype=np.float32).reshape(-1))

    def _resolve_image_history(self, mapped_obs: dict[str, Any], standardized_key: str) -> tuple[np.ndarray, bool]:
        value = mapped_obs.get(standardized_key)
        if value is None:
            blank = np.zeros((2, self.args.request_image_height, self.args.request_image_width, 3), dtype=np.uint8)
            return blank, False
        return coerce_history(value), True

    def _obs_to_current_inputs(self, obs: dict[str, Any]) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        mapped_obs = _map_alias_fields(obs, FIELD_ALIASES)
        head_history, head_mask = self._resolve_image_history(mapped_obs, f"{OBS_IMAGES}.image0")
        left_history, left_mask = self._resolve_image_history(mapped_obs, f"{OBS_IMAGES}.image1")
        right_history, right_mask = self._resolve_image_history(mapped_obs, f"{OBS_IMAGES}.image2")
        state = self._resolve_state(mapped_obs)
        prompt = self._resolve_prompt(obs)
        sample = {
            f"{OBS_IMAGES}.image0": torch.from_numpy(head_history).permute(0, 3, 1, 2).float() / 255.0,
            f"{OBS_IMAGES}.image1": torch.from_numpy(left_history).permute(0, 3, 1, 2).float() / 255.0,
            f"{OBS_IMAGES}.image2": torch.from_numpy(right_history).permute(0, 3, 1, 2).float() / 255.0,
            OBS_STATE: torch.from_numpy(state),
            "task": prompt,
        }
        masks = {
            f"{OBS_IMAGES}.image0_mask": torch.tensor(head_mask),
            f"{OBS_IMAGES}.image1_mask": torch.tensor(left_mask),
            f"{OBS_IMAGES}.image2_mask": torch.tensor(right_mask),
        }
        for transform in self.current_transforms:
            if isinstance(transform, ImgOnlyQwen3VLTransformFn):
                sample.update(masks)
            sample = transform(sample)
        inputs: dict[str, torch.Tensor] = {}
        for key, value in sample.items():
            if key == "task" or not isinstance(value, torch.Tensor):
                continue
            inputs[key] = _add_batch_and_move(value, self.device, self.runtime_dtype)
        return inputs, state

    def _raw_client_bp_to_prompt(self, client_bp: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(client_bp, dict):
            raise TypeError("behavior_prompt 必须是 dict。")
        processed_required = {
            "pixel_values",
            "image_grid_thw",
            "image_token_counts",
            "image_chunk_indices",
            "image_camera_indices",
            "state",
            "action",
            "mask",
        }
        if any(key in client_bp for key in ("pixel_values", "image_grid_thw")):
            missing = processed_required.difference(client_bp)
            if missing:
                raise KeyError(f"已处理 behavior_prompt 缺少字段: {sorted(missing)}")
            prompt = {key: _tensor_from_any(value) for key, value in client_bp.items() if key in BP_TENSOR_KEYS}
            if "action_is_pad" not in prompt:
                prompt["action_is_pad"] = torch.zeros(prompt["action"].shape[:-1], dtype=torch.bool)
            if "chunk_indices" not in prompt:
                prompt["chunk_indices"] = torch.arange(int(prompt["mask"].shape[-1]), dtype=torch.long)
            if "source_time_ratio" not in prompt:
                prompt["source_time_ratio"] = torch.linspace(0, 1, steps=int(prompt["mask"].shape[-1]))
            return prompt

        raw_images = client_bp.get("images")
        if not isinstance(raw_images, dict):
            raise KeyError("raw behavior_prompt 缺少 images 字典。")
        prompt_images: dict[str, torch.Tensor] = {}
        for raw_key, value in raw_images.items():
            image_key = RAW_BP_CAMERA_KEYS.get(raw_key, raw_key)
            array = np.asarray(value)
            if array.ndim != 4:
                raise ValueError(f"behavior_prompt image {image_key} 期望 [K,H,W,C] 或 [K,C,H,W], got {array.shape}")
            if array.shape[-1] == 3:
                tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float() / 255.0
            elif array.shape[1] == 3:
                tensor = torch.from_numpy(array).float()
                if tensor.max() > 1.5:
                    tensor = tensor / 255.0
            else:
                raise ValueError(f"behavior_prompt image {image_key} 不支持 shape={array.shape}")
            prompt_images[image_key] = tensor
        state = _tensor_from_any(client_bp["state"]).float()
        action = _tensor_from_any(client_bp["action"]).float()
        num_chunks = int(state.shape[0])
        prompt = {
            "images": prompt_images,
            "state": state,
            "action": action,
            "action_is_pad": _tensor_from_any(client_bp.get("action_is_pad", np.zeros(action.shape[:-1], dtype=bool))).to(torch.bool),
            "mask": _tensor_from_any(client_bp.get("mask", np.ones(num_chunks, dtype=bool))).to(torch.bool),
            "chunk_indices": torch.arange(num_chunks, dtype=torch.long),
            "source_time_ratio": _tensor_from_any(client_bp.get("source_time_ratio", np.linspace(0, 1, num_chunks))).float(),
        }
        transformed = self.bp_transform({"behavior_prompt": prompt})["behavior_prompt"]
        return transformed

    def _sample_dataset_behavior_prompt_uncached(self) -> dict[str, Any]:
        idx = self.rng.randrange(len(self.bp_dataset))
        sample = self.bp_dataset[idx]
        return sample["behavior_prompt"]

    def _sample_dataset_behavior_prompt(self) -> dict[str, Any]:
        if self.bp_prompt_cache:
            return self.rng.choice(self.bp_prompt_cache)
        return self._sample_dataset_behavior_prompt_uncached()

    def _resolve_behavior_prompt(self, obs: dict[str, Any]) -> tuple[dict[str, Any], str]:
        client_bp = obs.get("behavior_prompt")
        if client_bp is not None:
            return self._raw_client_bp_to_prompt(client_bp), "client"
        return self._sample_dataset_behavior_prompt(), "dataset"

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if obs.get("reset") or obs.get("timestep") == 0:
            self.policy.reset()
        inputs, state = self._obs_to_current_inputs(obs)
        behavior_prompt, bp_source = self._resolve_behavior_prompt(obs)
        behavior_prompt = _add_behavior_prompt_batch_dim(behavior_prompt)
        inputs["behavior_prompt"] = _move_nested_to_device(behavior_prompt, self.device, self.runtime_dtype)
        with torch.inference_mode():
            action_pred, _ = self.policy.predict_action_chunk(inputs, decode_image=False)
        if action_pred.ndim != 3:
            raise RuntimeError(f"策略输出 shape 异常: {tuple(action_pred.shape)}，期望 (B,T,A)")
        model_action_pred = action_pred[0, : self.infer_horizon, : self.target_action_dim]
        action_pred = self.unnormalize_action_fn({ACTION: model_action_pred})[ACTION]
        model_action_np = model_action_pred.detach().cpu().numpy().astype(np.float32)
        action_np = action_pred.detach().cpu().numpy().astype(np.float32)
        if self.action_mode == "delta" and self.delta_mask is not None:
            state_pad = np.zeros_like(self.delta_mask, dtype=np.float32)
            usable_dims = min(len(state_pad), len(state))
            state_pad[:usable_dims] = state[:usable_dims]
            action_dims = min(action_np.shape[-1], len(self.delta_mask))
            action_np[:, :action_dims] += state_pad[None, :action_dims] * self.delta_mask[None, :action_dims]
        return {
            "actions": action_np,
            "action": action_np[0],
            "model_actions": model_action_np,
            "model_action": model_action_np[0],
            "behavior_prompt_source": bp_source,
        }


class WebsocketPolicyServer:
    def __init__(self, policy: BPTBotPolicyService, host: str = "0.0.0.0", port: int = 8000, metadata: dict[str, Any] | None = None) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websocket_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        logging.info("客户端已连接: %s", websocket.remote_address)
        packer = MsgpackPacker()
        await websocket.send(packer.pack(self._metadata))
        prev_total_time: float | None = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_unpack(await websocket.recv())
                infer_start = time.monotonic()
                action = self._policy.infer(obs)
                infer_ms = (time.monotonic() - infer_start) * 1000.0
                action["server_timing"] = {"infer_ms": infer_ms}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0
                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                logging.info("客户端断开: %s", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: websocket_server.ServerConnection, request: websocket_server.Request) -> websocket_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main(args: ServeArgs) -> None:
    logging.info("启动参数:\n%s", json.dumps(asdict(args), indent=2, ensure_ascii=False))
    policy = BPTBotPolicyService(args)
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError as exc:
        local_ip = "unknown"
        logging.warning("无法解析本机 IP (%s): %s", hostname, exc)
    logging.info("BP_TBot WebSocket 服务 | host=%s ip=%s port=%d", args.host, local_ip, args.port)
    logging.info("Server metadata:\n%s", json.dumps(policy.metadata, indent=2, ensure_ascii=False))
    server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=policy.metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    main(parse_args())
