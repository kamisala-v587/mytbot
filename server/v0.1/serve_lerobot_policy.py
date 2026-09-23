#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mytbot TBot-SA1 / LeRobot Policy WebSocket Server
=================================================

与 ``myvla/scripts/serve_lerobot_policy.py`` 使用相同的客户端协议：
连接后先收 metadata，再循环发送 obs dict、接收 action dict（msgpack + numpy）。

本脚本面向 **mytbot** 训练产物，默认优化 TBot_SA1（RoboTwin / LIBERO 微调 checkpoint）。
TBot_SA1 使用 checkpoint 内 ``stats.json`` + Qwen3-VL processor，不依赖 ``make_pre_post_processors``。

启动示例
--------

.. code-block:: bash

   cd /home/jovyan/vla/workspace/mytbot
   source /home/jovyan/.conda/envs/tbot/bin/activate
   export PYTHONPATH="${PWD}/src"
   export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
   export DISABLE_DA3_TEACHER_FOR_EVAL=true

   python server/serve_lerobot_policy.py \\
       --ckpt_path outputs/TBot_SA1/2026-06-26/06-19-44_TBot_SFT_robotwin_v0/checkpoints/110000 \\
       --host 0.0.0.0 --port 8000 \\
       --default_prompt "adjust the bottle" \\
       --infer_horizon 16 \\
       --action_mode delta

客户端 payload 推荐格式（与 evaluation Real_Lift2 / LIBERO 一致）::

    {
        "images": {"cam_high": ..., "cam_left_wrist": ..., "cam_right_wrist": ...},
        "state": [...],
        "prompt": "task description",
        "reset": false,
        "timestep": 42,
    }

每路相机可为 ``[H,W,C]`` 或时序 ``[T,H,W,C]``（至少 1 帧；仅 1 帧时会复制为 2 帧以匹配训练时序）。
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http
import json
import logging
import os
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

# ---------------------------------------------------------------------------
# mytbot 路径
# ---------------------------------------------------------------------------
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
from lerobot.datasets.utils import load_json  # noqa: E402
from lerobot.policies.TBot_SA1.transform_tbot_sa1 import Qwen3_VLProcessorTransformFn  # noqa: E402
from lerobot.policies.factory import get_policy_class  # noqa: E402
from lerobot.policies.names import is_tbot_sa1  # noqa: E402
from lerobot.transforms.constants import get_mask_mapping  # noqa: E402
from lerobot.transforms.core import NormalizeTransformFn, ResizeImagesWithPadFn, UnNormalizeTransformFn  # noqa: E402
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE  # noqa: E402

DEFAULT_QWEN3_VL_PATH = Path("/home/jovyan/vla/workspace/models/Qwen3-vl-2b-instruct")
CONTROL_KEYS = frozenset({"reset", "timestep", "prompt"})

CAMERA_ALIASES = {
    f"{OBS_IMAGES}.image0": ("cam_high", "head", "image0", "observation.images.cam_high"),
    f"{OBS_IMAGES}.image1": ("cam_left_wrist", "left_wrist", "left", "image1", "observation.images.cam_left_wrist"),
    f"{OBS_IMAGES}.image2": (
        "cam_right_wrist",
        "right_wrist",
        "right",
        "image2",
        "observation.images.cam_right_wrist",
    ),
}


# ---------------------------------------------------------------------------
# msgpack + numpy
# ---------------------------------------------------------------------------
def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype for msgpack: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj: Any) -> Any:
    if isinstance(obj, dict) and b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if isinstance(obj, dict) and b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


MsgpackPacker = functools.partial(msgpack.Packer, default=_pack_array)
msgpack_unpack = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@dataclass
class ServeArgs:
    ckpt_path: str
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
        description="启动 mytbot TBot-SA1 checkpoint 的 WebSocket 推理服务。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt_path", required=True, help="checkpoint step 目录或 pretrained_model 目录。")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default_prompt", default="Execute the task.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--infer_horizon", type=int, default=None, help="返回 action 步数，默认读 config.n_action_steps。")
    parser.add_argument("--stats_key", default=None, help="stats.json 内 embodiment key，如 aloha。")
    parser.add_argument("--stats_path", default=None, help="外部 stats.json；默认使用 checkpoint 目录下 stats.json。")
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
    parser.add_argument(
        "--disable_3d_teacher_for_eval",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="推理时关闭 DA3 teacher（默认读 DISABLE_DA3_TEACHER_FOR_EVAL 环境变量）。",
    )
    parser.add_argument(
        "--omit_visual_tokens_in_causal_inference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="causal 微调模型仅做 action 推理时省略 visual-gen token。",
    )
    parsed = parser.parse_args()

    disable_3d = parsed.disable_3d_teacher_for_eval
    if disable_3d is None:
        disable_3d = _bool_env("DISABLE_DA3_TEACHER_FOR_EVAL", True)
    omit_visual = parsed.omit_visual_tokens_in_causal_inference
    if omit_visual is None:
        omit_visual = _bool_env("OMIT_VISUAL_TOKENS_IN_CAUSAL_INFERENCE", True)

    return ServeArgs(
        ckpt_path=parsed.ckpt_path,
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


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
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
    if OBS_STATE in stats_root and "action" in stats_root:
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


# ---------------------------------------------------------------------------
# TBot-SA1 推理服务
# ---------------------------------------------------------------------------
class TBotSA1PolicyService:
    """TBot_SA1 checkpoint → infer(obs) → actions（与 myvla serve 返回字段一致）。"""

    def __init__(self, args: ServeArgs):
        self.args = args
        self.ckpt_dir = resolve_ckpt_dir(args.ckpt_path)
        self.train_cfg = load_train_config_or_none(self.ckpt_dir)

        self.config = PreTrainedConfig.from_pretrained(self.ckpt_dir)
        if not is_tbot_sa1(self.config.type):
            raise ValueError(
                f"当前脚本仅支持 TBot_SA1，checkpoint type={self.config.type!r}。"
                "其他策略请使用 myvla/scripts/serve_lerobot_policy.py。"
            )

        self._apply_runtime_overrides()

        self.device = resolve_device(args.device)
        self.load_device = resolve_device(args.load_device) if args.load_device else (
            "cpu" if self.device != "cpu" else "cpu"
        )
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
            raise FileNotFoundError(f"未找到 stats.json: {stats_path}，请通过 --stats_path 指定。")
        self.stats_key, stats = resolve_stats(stats_path, args.stats_key)
        stat_keys = ["min", "max", "mean", "std"]
        self.state_stats = {OBS_STATE: {k: np.asarray(stats[OBS_STATE][k]) for k in stat_keys}}
        self.action_stats = {"action": {k: np.asarray(stats["action"][k]) for k in stat_keys}}
        self.action_mean = np.asarray(self.action_stats["action"]["mean"], dtype=np.float32)
        self.action_std = np.asarray(self.action_stats["action"]["std"], dtype=np.float32)
        self.target_action_dim = int(self.action_mean.shape[0])

        processor_path = (
            args.qwen3_vl_processor_path
            or getattr(self.config, "qwen3_vl_processor_path", None)
            or getattr(self.config, "qwen3_vl_pretrained_path", None)
            or str(DEFAULT_QWEN3_VL_PATH)
        )
        self.resize_fn = ResizeImagesWithPadFn(height=args.resize_size, width=args.resize_size)
        self.normalize_state_fn = NormalizeTransformFn(
            selected_keys=[OBS_STATE],
            mode="mean_std",
            norm_stats=self.state_stats,
        )
        self.unnormalize_action_fn = UnNormalizeTransformFn(
            selected_keys=["action"],
            mode="mean_std",
            norm_stats=self.action_stats,
        )
        self.processor_fn = Qwen3_VLProcessorTransformFn(
            pretrained_model_name_or_path=processor_path,
            max_length=int(getattr(self.config, "tokenizer_max_length", 48)),
        )

        train_action_mode = None
        if self.train_cfg is not None:
            train_action_mode = str(getattr(self.train_cfg.dataset, "action_mode", "") or "").lower() or None
        requested_action_mode = None if args.action_mode is None else str(args.action_mode).lower()
        if (
            requested_action_mode is not None
            and train_action_mode is not None
            and requested_action_mode != train_action_mode
        ):
            raise RuntimeError(
                f"action_mode 与 checkpoint 训练配置不一致: "
                f"请求={requested_action_mode!r}, checkpoint={train_action_mode!r}"
            )
        self.action_mode = requested_action_mode or train_action_mode or "delta"

        self.delta_mask = None
        if self.action_mode == "delta":
            try:
                self.delta_mask = get_mask_mapping(self.stats_key).detach().cpu().numpy().astype(np.float32)
            except KeyError:
                self.delta_mask = get_mask_mapping("aloha").detach().cpu().numpy().astype(np.float32)

        self._metadata = {
            "model_type": self.config.type,
            "deployment": "mytbot_tbot_sa1_websocket_server",
            "checkpoint_dir": str(self.ckpt_dir),
            "stats_key": self.stats_key,
            "action_mode": self.action_mode,
            "device": self.device,
            "dtype": str(self.runtime_dtype),
            "infer_horizon": self.infer_horizon,
            "chunk_size": chunk_size,
            "num_inference_steps": int(getattr(self.config, "num_inference_steps", 10)),
            "default_prompt": args.default_prompt,
            "target_action_dim": self.target_action_dim,
            "omit_visual_tokens_in_causal_inference": bool(args.omit_visual_tokens_in_causal_inference),
            "camera_aliases": {k: list(v) for k, v in CAMERA_ALIASES.items()},
            "notes": {
                "client_payload": "images dict or flat observation.images.* keys + state + prompt/task",
                "protocol": "compatible with myvla serve_lerobot_policy.py",
            },
        }
        logging.info(
            "Server 就绪 | type=%s | device=%s | horizon=%d | action_mode=%s | action_dim=%d",
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

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _resolve_prompt(self, obs: dict[str, Any]) -> str:
        prompt = obs.get("prompt") or obs.get("task") or self.args.default_prompt
        if not isinstance(prompt, str) or not prompt.strip():
            return self.args.default_prompt
        return prompt

    def _resolve_state(self, obs: dict[str, Any]) -> np.ndarray:
        state = obs.get("state")
        if state is None:
            state = obs.get("qpos")
        if state is None and OBS_STATE in obs:
            state = obs[OBS_STATE]
        if state is None:
            raise KeyError("请求缺少 state / qpos / observation.state。")
        return np.ascontiguousarray(np.asarray(state, dtype=np.float32).reshape(-1))

    def _resolve_images(self, obs: dict[str, Any]) -> dict[str, Any]:
        images = obs.get("images")
        if isinstance(images, dict):
            return images

        # RoboTwin's My_LeRobot client sends camera frames as flat observation.images.* keys.
        flat_images: dict[str, Any] = {}
        for aliases in CAMERA_ALIASES.values():
            for alias in aliases:
                if alias in obs:
                    flat_images[alias] = obs[alias]
        if flat_images:
            return flat_images

        expected = sorted({alias for aliases in CAMERA_ALIASES.values() for alias in aliases})
        raise KeyError(f"请求缺少 images 字典或顶层图像键。可用图像键示例: {expected}")

    def _resolve_image_history(self, images: dict[str, Any], standardized_key: str) -> tuple[np.ndarray, bool]:
        aliases = CAMERA_ALIASES[standardized_key]
        value = None
        for alias in aliases:
            if alias in images:
                value = images[alias]
                break
        if value is None:
            blank = np.zeros(
                (2, self.args.request_image_height, self.args.request_image_width, 3),
                dtype=np.uint8,
            )
            return blank, False
        return coerce_history(value), True

    def _obs_to_inputs(self, obs: dict[str, Any]) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        images = self._resolve_images(obs)

        head_history, head_mask = self._resolve_image_history(images, f"{OBS_IMAGES}.image0")
        left_history, left_mask = self._resolve_image_history(images, f"{OBS_IMAGES}.image1")
        right_history, right_mask = self._resolve_image_history(images, f"{OBS_IMAGES}.image2")
        state = self._resolve_state(obs)
        prompt = self._resolve_prompt(obs)

        sample = {
            f"{OBS_IMAGES}.image0": torch.from_numpy(head_history).permute(0, 3, 1, 2).float() / 255.0,
            f"{OBS_IMAGES}.image1": torch.from_numpy(left_history).permute(0, 3, 1, 2).float() / 255.0,
            f"{OBS_IMAGES}.image2": torch.from_numpy(right_history).permute(0, 3, 1, 2).float() / 255.0,
            OBS_STATE: torch.from_numpy(state),
            "task": prompt,
        }
        sample = self.resize_fn(sample)
        sample[f"{OBS_IMAGES}.image0_mask"] = torch.tensor(head_mask)
        sample[f"{OBS_IMAGES}.image1_mask"] = torch.tensor(left_mask)
        sample[f"{OBS_IMAGES}.image2_mask"] = torch.tensor(right_mask)
        sample = self.processor_fn(sample)
        sample = self.normalize_state_fn(sample)

        inputs: dict[str, torch.Tensor] = {}
        for key, value in sample.items():
            if key == "task" or not isinstance(value, torch.Tensor):
                continue
            if value.dtype == torch.bool:
                inputs[key] = value.reshape(1).to(self.device)
            elif value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                inputs[key] = value[None].to(self.device)
            elif value.is_floating_point():
                inputs[key] = value[None].to(device=self.device, dtype=self.runtime_dtype)
            else:
                inputs[key] = value[None].to(self.device)
        return inputs, state

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if obs.get("reset") or obs.get("timestep") == 0:
            self.policy.reset()

        inputs, state = self._obs_to_inputs(obs)
        with torch.inference_mode():
            action_pred, _ = self.policy.predict_action_chunk(inputs, decode_image=False)

        if action_pred.ndim != 3:
            raise RuntimeError(f"策略输出 shape 异常: {tuple(action_pred.shape)}，期望 (B, T, A)")
        model_action_pred = action_pred[0, : self.infer_horizon, : self.target_action_dim]
        action_pred = self.unnormalize_action_fn({"action": model_action_pred})["action"]
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
        }


# ---------------------------------------------------------------------------
# WebSocket Server（协议与 myvla 一致）
# ---------------------------------------------------------------------------
class WebsocketPolicyServer:
    def __init__(
        self,
        policy: TBotSA1PolicyService,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict[str, Any] | None = None,
    ) -> None:
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


def _health_check(
    connection: websocket_server.ServerConnection,
    request: websocket_server.Request,
) -> websocket_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main(args: ServeArgs) -> None:
    logging.info("启动参数:\n%s", json.dumps(asdict(args), indent=2, ensure_ascii=False))
    policy = TBotSA1PolicyService(args)

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError as exc:
        local_ip = "unknown"
        logging.warning("无法解析本机 IP (%s): %s", hostname, exc)

    logging.info(
        "TBot-SA1 WebSocket 服务 | host=%s ip=%s port=%d",
        args.host,
        local_ip,
        args.port,
    )
    logging.info("Server metadata:\n%s", json.dumps(policy.metadata, indent=2, ensure_ascii=False))

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    main(parse_args())
