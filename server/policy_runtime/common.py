"""WebSocket 协议、观测解析和动态 batch 服务。"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import http
import json
import logging
import os
import re
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames

OBS_IMAGES = "observation.images"
OBS_STATE = "observation.state"
ACTION = "action"
CONTROL_KEYS = frozenset({"reset", "timestep", "prompt", "task", "task_type", "state", "qpos"})

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

SERVER_DIR = Path(__file__).resolve().parents[1]
MYTBOT_ROOT = SERVER_DIR.parent
SRC_ROOT = MYTBOT_ROOT / "src"


@dataclass
class ServerConfig:
    policy: str
    ckpt_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "auto"
    dtype: str = "bfloat16"
    debug: bool = False
    debug_dir: str = ""
    debug_run_dir: str = ""
    max_batch_size: int = 4
    batch_wait_ms: float = 10.0
    queue_size: int = 64
    default_prompt: str = "Execute the task."
    infer_horizon: int | None = None
    stats_key: str | None = None
    stats_path: str | None = None
    action_mode: str | None = None
    resize_size: int = 224
    request_image_height: int = 480
    request_image_width: int = 640
    strict_history: bool = True
    bp_mapping_path: str = ""
    num_inference_steps: int | None = None
    qwen3_vl_path: str | None = None
    qwen3_5_path: str | None = None
    cosmos_tokenizer_path: str | None = None
    da3_model_path: str | None = None
    da3_code_root: str | None = None
    load_device: str | None = None
    cosmos_device: str | None = None
    disable_3d_teacher: bool = True
    omit_visual_tokens: bool = True


def prepare_import_path() -> None:
    for path in (SERVER_DIR, SRC_ROOT, MYTBOT_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


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


def read_ckpt_type(ckpt_path: str | Path) -> str:
    config_path = resolve_ckpt_dir(ckpt_path) / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    policy_type = payload.get("type")
    if not isinstance(policy_type, str) or not policy_type:
        raise ValueError(f"{config_path} 缺少 type。")
    return policy_type


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
    if array.shape[-1] in (1, 3, 4):
        hwc = array
    elif array.shape[0] in (1, 3, 4):
        hwc = np.transpose(array, (1, 2, 0))
    else:
        raise ValueError(f"不支持的图像 shape: {array.shape}")
    if np.issubdtype(hwc.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(hwc)) <= 1.5 else 1.0
        hwc = np.clip(hwc * scale, 0.0, 255.0)
    else:
        hwc = np.clip(hwc, 0, 255)
    hwc = hwc.astype(np.uint8)
    if hwc.shape[-1] == 1:
        hwc = np.repeat(hwc, 3, axis=-1)
    if hwc.shape[-1] == 4:
        hwc = hwc[..., :3]
    return np.ascontiguousarray(hwc)


def _frame_at(array: np.ndarray, index: int) -> np.ndarray:
    frame = array[index]
    if frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    return to_hwc_uint8(frame)


def repeat_frame(frame: np.ndarray, frames: int) -> np.ndarray:
    """把一帧复制成所需历史长度，时间维在第 0 维。"""
    image = to_hwc_uint8(frame)
    return np.stack([image] * frames, axis=0)


def coerce_history(image_value: Any, frames: int = 2) -> np.ndarray:
    """整理成 [frames,H,W,3]。只有 1 帧时复制补齐；多帧保留首帧和末帧。"""
    if frames < 1:
        raise ValueError(f"frames 必须 >= 1，收到 {frames}")
    array = np.asarray(image_value)
    if array.ndim == 3:
        return repeat_frame(array, frames)
    if array.ndim != 4:
        raise ValueError(f"不支持的时序图像 shape: {array.shape}")
    if array.shape[-1] not in (1, 3, 4) and array.shape[1] not in (1, 3, 4):
        raise ValueError(f"不支持的时序图像 shape: {array.shape}")
    extracted = [_frame_at(array, idx) for idx in range(array.shape[0])]
    if len(extracted) == 1:
        return np.stack([extracted[0]] * frames, axis=0)
    if frames == 1:
        return np.stack([extracted[-1]], axis=0)
    return np.stack([extracted[0], extracted[-1]], axis=0)


def _blank_history(height: int, width: int, frames: int) -> np.ndarray:
    return np.zeros((frames, height, width, 3), dtype=np.uint8)


def resolve_image_history(
    images: dict[str, Any],
    standardized_key: str,
    *,
    strict: bool,
    frames: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, bool]:
    value = None
    for alias in CAMERA_ALIASES[standardized_key]:
        if alias in images:
            value = images[alias]
            break
    if value is None:
        return _blank_history(height, width, frames), False
    raw = np.asarray(value)
    single_frame = raw.ndim == 3 or (raw.ndim == 4 and raw.shape[0] == 1)
    if single_frame:
        frame = raw if raw.ndim == 3 else _frame_at(raw, 0)
        return repeat_frame(frame, frames), True
    if strict:
        if raw.ndim != 4 or raw.shape[0] != frames:
            raise ValueError(
                f"相机 {standardized_key} 需要 {frames} 帧历史，或只提供 1 帧用于自动填充，收到 shape={raw.shape}。"
                "单帧会复制补齐；多帧时时间维放在第 0 维，例如 [2,H,W,3]。"
            )
        history = coerce_history(raw, frames) if frames <= 2 else np.stack(
            [_frame_at(raw, idx) for idx in range(frames)],
            axis=0,
        )
        if history.shape[0] != frames:
            raise RuntimeError(f"相机 {standardized_key} 转换后帧数异常: {history.shape}")
        return history, True
    if frames == 2:
        return coerce_history(raw, frames), True
    if raw.ndim == 4:
        return np.stack([_frame_at(raw, -1)], axis=0), True
    raise ValueError(f"不支持的图像 shape: {raw.shape}")


def resolve_images(obs: dict[str, Any]) -> dict[str, Any]:
    images = obs.get("images")
    if isinstance(images, dict):
        return images
    flat_images: dict[str, Any] = {}
    for aliases in CAMERA_ALIASES.values():
        for alias in aliases:
            if alias in obs:
                flat_images[alias] = obs[alias]
    if flat_images:
        return flat_images
    expected = sorted({alias for aliases in CAMERA_ALIASES.values() for alias in aliases})
    raise KeyError(f"请求缺少 images 字典或顶层图像键。可用图像键示例: {expected}")


def resolve_state(obs: dict[str, Any]) -> np.ndarray:
    state = obs.get("state")
    if state is None:
        state = obs.get("qpos")
    if state is None and OBS_STATE in obs:
        state = obs[OBS_STATE]
    if state is None:
        raise KeyError("请求缺少 state / qpos / observation.state。")
    return np.ascontiguousarray(np.asarray(state, dtype=np.float32).reshape(-1))


def resolve_task_text(obs: dict[str, Any], default_prompt: str) -> str:
    prompt = obs.get("prompt") or obs.get("task") or default_prompt
    if not isinstance(prompt, str) or not prompt.strip():
        return default_prompt
    return prompt


def hwc_tensor(history_hwc: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(history_hwc)).permute(0, 3, 1, 2).float() / 255.0


def batch_model_tensor(value: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    if value.dtype == torch.bool:
        return value.reshape(1).to(device)
    value = value[None]
    if value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        return value.to(device)
    if value.is_floating_point():
        return value.to(device=device, dtype=dtype)
    return value.to(device)


def batch_nested(value: Any, device: str, dtype: torch.dtype) -> Any:
    if isinstance(value, dict):
        return {key: batch_nested(item, device, dtype) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        value = value.unsqueeze(0)
        if value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device)
    raise TypeError(f"无法合批的类型: {type(value).__name__}")


def collate_nested(values: list[Any]) -> Any:
    if not values:
        raise ValueError("不能合并空 batch。")
    first = values[0]
    if isinstance(first, dict):
        keys = tuple(first)
        if any(tuple(value) != keys for value in values):
            raise ValueError("batch 中的嵌套输入 key 不一致。")
        return {key: collate_nested([value[key] for value in values]) for key in keys}
    if isinstance(first, torch.Tensor):
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise TypeError("batch 中同一字段的类型不一致。")
        return torch.cat(values, dim=0)
    raise TypeError(f"不支持合批的输入类型: {type(first).__name__}")


def load_train_config_or_none(ckpt_dir: Path):
    prepare_import_path()
    from lerobot.configs.train import TrainPipelineConfig

    try:
        return TrainPipelineConfig.from_pretrained(ckpt_dir)
    except Exception as exc:
        logging.warning("无法从 %s 加载 train_config.json: %s", ckpt_dir, exc)
        return None


def resolve_stats(stats_path: Path, requested_key: str | None) -> tuple[str, dict[str, Any]]:
    prepare_import_path()
    from lerobot.datasets.utils import load_json

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
    raise ValueError(f"stats.json 含多个 key {list(stats_root.keys())}，请指定 STATS_KEY。")


def assign_existing_or_remap(model_config: Any, attr: str, explicit: str | None, fallback: str | None) -> None:
    """常量路径存在时覆盖 checkpoint；否则保留或改写到本机真实存在的路径。"""
    if not hasattr(model_config, attr):
        return
    if explicit and Path(explicit).expanduser().exists():
        setattr(model_config, attr, str(Path(explicit).expanduser()))
        return
    current = getattr(model_config, attr, None)
    if isinstance(current, str) and current and Path(current).expanduser().exists():
        return
    if isinstance(current, str) and current:
        remapped = _remap_known_local_path(current, fallback)
        if remapped is not None:
            setattr(model_config, attr, remapped)
            logging.info("运行时修正 %s: %s -> %s", attr, current, remapped)
            return
    if explicit:
        setattr(model_config, attr, str(Path(explicit).expanduser()))
        return
    if fallback and Path(fallback).expanduser().exists():
        setattr(model_config, attr, str(Path(fallback).expanduser()))


def _remap_known_local_path(value: str, fallback: str | None) -> str | None:
    prefixes = ("/home/jovyan/workspace/mytbot", "/home/jovyan/workspace")
    replacements = (str(MYTBOT_ROOT), "/vla/workspace")
    for prefix, replacement_prefix in zip(prefixes, replacements, strict=True):
        if value.startswith(prefix):
            candidate = replacement_prefix + value[len(prefix):]
            if Path(candidate).expanduser().exists():
                return candidate
    if fallback and Path(fallback).expanduser().exists():
        return str(Path(fallback).expanduser())
    return None


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"


def summarize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        summary: dict[str, Any] = {"ndarray_shape": list(value.shape), "dtype": str(value.dtype)}
        if value.ndim <= 2 and value.size <= 128:
            summary["value"] = np.nan_to_num(value.astype(np.float64), nan=0.0).tolist()
        return summary
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"bytes_len": len(value)}
    if isinstance(value, dict):
        return {str(key): summarize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        if isinstance(value[0], (str, int, float, bool)) or value[0] is None:
            return list(value)
        return {"length": len(value), "item0": summarize_value(value[0])}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_rgb_image(path: Path, frame_hwc: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = to_hwc_uint8(frame_hwc)
    try:
        from PIL import Image

        Image.fromarray(array).save(path)
        return path
    except Exception:
        ppm_path = path.with_suffix(".ppm")
        height, width = array.shape[:2]
        with ppm_path.open("wb") as stream:
            stream.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            stream.write(array.tobytes())
        return ppm_path


class FirstRequestDumper:
    """每个 client 只记录第一次请求：任务文本、字段摘要和图像文件。"""

    def __init__(self, root: Path, label: str = "VLA") -> None:
        self.root = root
        self.label = label
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def dump(self, connection_id: str, obs: dict[str, Any], task_text: str) -> None:
        with self._lock:
            if connection_id in self._seen:
                return
            self._seen.add(connection_id)
            index = len(self._seen)
        client_dir = self.root / f"{index:03d}_{safe_name(connection_id)}"
        image_dir = client_dir / "images"
        saved: list[str] = []
        try:
            saved = self._save_images(image_dir, obs)
            summary = {
                "connection_id": connection_id,
                "task": task_text,
                "saved_images": saved,
                "request": summarize_value(obs),
            }
            client_dir.mkdir(parents=True, exist_ok=True)
            (client_dir / "task.txt").write_text(task_text + "\n", encoding="utf-8")
            (client_dir / "request.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logging.exception("%s 首次请求落盘失败 | client=%s", self.label, connection_id)
            return
        logging.info("%s 首次请求 | client=%s | task=%s | images=%s", self.label, connection_id, task_text, saved)
        logging.info(
            "%s 首次请求内容 | client=%s | %s",
            self.label,
            connection_id,
            json.dumps(summarize_value(obs), ensure_ascii=False, default=str),
        )

    def _save_images(self, image_dir: Path, obs: dict[str, Any]) -> list[str]:
        saved: list[str] = []
        for name, value in _iter_image_payloads(obs):
            array = np.asarray(value)
            frames = _frames_from_array(array)
            for frame_index, frame in enumerate(frames):
                filename = f"{safe_name(name)}_t{frame_index}.png"
                output = save_rgb_image(image_dir / filename, frame)
                saved.append(str(output))
        return saved


def _iter_image_payloads(obs: dict[str, Any]) -> list[tuple[str, Any]]:
    images = obs.get("images")
    if isinstance(images, dict):
        return [(str(key), value) for key, value in images.items()]
    payloads = []
    for key, value in obs.items():
        if key in CONTROL_KEYS or not isinstance(value, np.ndarray):
            continue
        if value.ndim in (3, 4):
            payloads.append((str(key), value))
    return payloads


def _frames_from_array(array: np.ndarray) -> list[np.ndarray]:
    if array.ndim == 3:
        return [array]
    if array.ndim != 4:
        return []
    if array.shape[-1] in (1, 3, 4) or array.shape[1] in (1, 3, 4):
        return [array[index] for index in range(array.shape[0])]
    return []


@dataclass
class _PendingRequest:
    obs: dict[str, Any]
    future: asyncio.Future
    enqueued_at: float
    connection_id: str


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


class DynamicBatchServer:
    def __init__(self, policy: Any, config: ServerConfig) -> None:
        self._policy = policy
        self._config = config
        self._queue: asyncio.Queue[_PendingRequest | None] = asyncio.Queue(maxsize=config.queue_size)
        self._metadata = {
            **dict(policy.metadata),
            "dynamic_batching": True,
            "max_batch_size": config.max_batch_size,
            "batch_wait_ms": config.batch_wait_ms,
            "queue_size": config.queue_size,
            "debug": config.debug,
        }
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        worker = asyncio.create_task(self._batch_worker(), name="dynamic-batch-worker")
        try:
            async with websocket_server.serve(
                self._handler,
                self._config.host,
                self._config.port,
                compression=None,
                max_size=None,
                process_request=_health_check,
            ) as server:
                await server.serve_forever()
        finally:
            await self._queue.put(None)
            with contextlib.suppress(asyncio.CancelledError):
                await worker

    async def _handler(self, websocket) -> None:
        connection_id = f"{websocket.remote_address}:{id(websocket)}"
        logging.info("客户端已连接: %s", connection_id)
        packer = MsgpackPacker()
        await websocket.send(packer.pack(self._metadata))
        while True:
            future = None
            try:
                started_at = time.monotonic()
                obs = msgpack_unpack(await websocket.recv())
                if not isinstance(obs, dict):
                    raise TypeError("请求 payload 必须是字典。")
                try:
                    self._policy.on_request(connection_id, obs)
                except Exception:
                    logging.exception("请求调试记录失败 | client=%s", connection_id)
                future = asyncio.get_running_loop().create_future()
                request = _PendingRequest(obs, future, time.monotonic(), connection_id)
                try:
                    self._queue.put_nowait(request)
                except asyncio.QueueFull as exc:
                    raise RuntimeError("推理队列已满，请稍后重试。") from exc
                action = await future
                timing = action.setdefault("server_timing", {})
                timing["total_ms"] = (time.monotonic() - started_at) * 1000.0
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                if future is not None and not future.done():
                    future.cancel()
                logging.info("客户端断开: %s", connection_id)
                break
            except asyncio.CancelledError:
                if future is not None and not future.done():
                    future.cancel()
                raise
            except Exception as exc:
                logging.exception("请求处理失败 | client=%s", connection_id)
                if future is not None and not future.done():
                    future.cancel()
                try:
                    await websocket.send(str(exc))
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error",
                    )
                except websockets.ConnectionClosed:
                    pass
                break

    async def _batch_worker(self) -> None:
        wait_seconds = self._config.batch_wait_ms / 1000.0
        while True:
            first = await self._queue.get()
            if first is None:
                self._queue.task_done()
                break
            batch = [first]
            deadline = time.monotonic() + wait_seconds
            while len(batch) < self._config.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    self._queue.task_done()
                    await self._queue.put(None)
                    break
                batch.append(item)
            active = [item for item in batch if not item.future.cancelled()]
            if active:
                infer_started = time.monotonic()
                try:
                    results = await asyncio.to_thread(self._policy.infer_batch, [item.obs for item in active])
                    infer_ms = (time.monotonic() - infer_started) * 1000.0
                    for item, result in zip(active, results, strict=True):
                        if item.future.cancelled():
                            continue
                        result["server_timing"] = {
                            "queue_ms": (infer_started - item.enqueued_at) * 1000.0,
                            "infer_ms": infer_ms,
                            "batch_size": len(active),
                        }
                        item.future.set_result(result)
                    logging.info(
                        "动态批推理完成 | batch_size=%d | infer_ms=%.1f | queue=%d",
                        len(active),
                        infer_ms,
                        self._queue.qsize(),
                    )
                except Exception as exc:
                    logging.exception("动态批推理失败 | batch_size=%d", len(active))
                    for item in active:
                        if not item.future.done():
                            item.future.set_exception(exc)
            for _ in batch:
                self._queue.task_done()


def serve(policy: Any, config: ServerConfig) -> None:
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "unknown"
    logging.info(
        "Policy WebSocket 服务 | policy=%s | host=%s | ip=%s | port=%d | debug=%s",
        config.policy,
        config.host,
        local_ip,
        config.port,
        config.debug,
    )
    logging.info("Server metadata:\n%s", json.dumps(policy.metadata, indent=2, ensure_ascii=False, default=str))
    DynamicBatchServer(policy, config).serve_forever()
