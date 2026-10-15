"""BPVAv2：按 task_type 使用固定示教缓存，未命中时随机选数据集和 episode。"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .base import PolicyAdapter, PreparedSample
from .common import (
    ACTION,
    CAMERA_ALIASES,
    OBS_STATE,
    batch_nested,
    hwc_tensor,
    resolve_image_history,
    resolve_images,
    resolve_state,
    safe_name,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.names import is_bpvav2
from lerobot.transforms.constants import get_image_mapping, get_mask_mapping
from lerobot.transforms.core_bp import (
    BPDeltaActionTransformFn,
    BPNormalizeTransformFn,
    BPPadOrSampleChunksFn,
    BPPadStateAndActionTransformFn,
    BPResizeImagesWithPadFn,
    BPVAv2QwenImageTransformFn,
    BPVAv2RemapImageKeyTransformFn,
    ImgOnlyQwen3VLTransformFn,
)

BP_PREFIX = "behavior_prompt"
BP_VIDEO_CACHE_DIRNAME = "bp_video_cache"


@dataclass(frozen=True)
class BPTaskSource:
    task_type: str
    dataset_paths: tuple[Path, ...]
    video_backend: str = "pyav"


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
        configured = entry.get("dataset_paths", entry.get("dataset_path"))
        if isinstance(configured, str):
            configured = [configured]
        if not isinstance(configured, list) or not configured:
            raise ValueError(f"task={task!r} 需要 dataset_path 或非空 dataset_paths。")
        dataset_paths = []
        for item in configured:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"task={task!r} 包含空的数据集路径。")
            dataset_path = Path(item).expanduser()
            if not dataset_path.is_absolute():
                dataset_path = (path.parent / dataset_path).resolve()
            else:
                dataset_path = dataset_path.resolve()
            dataset_paths.append(dataset_path)
        video_backend = str(entry.get("video_backend", "pyav")).strip() or "pyav"
        sources[task] = BPTaskSource(task, tuple(dataset_paths), video_backend)
    return path, sources


def fixed_bp_video_cache_dir(debug_dir: str, adapter_name: str) -> Path:
    """示教视频的固定缓存。启动时创建目录，并报告已经存在的 mp4。"""
    root = Path(debug_dir).expanduser()
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    cache_dir = (root / BP_VIDEO_CACHE_DIRNAME / adapter_name).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(cache_dir.glob("*.mp4"))
    if videos:
        logging.info(
            "检测到固定 BP 示教缓存 | adapter=%s | path=%s | mp4=%d",
            adapter_name,
            cache_dir,
            len(videos),
        )
        for video in videos:
            logging.info("缓存示教视频 | %s", video.name)
    else:
        logging.info(
            "固定 BP 示教缓存为空，首次请求时写入 | adapter=%s | path=%s",
            adapter_name,
            cache_dir,
        )
    return cache_dir


def _dataset_candidates(path: Path) -> list[Path]:
    if (path / "meta" / "info.json").is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"BP 数据集路径不存在: {path}")
    children = sorted(child for child in path.iterdir() if (child / "meta" / "info.json").is_file())
    if not children:
        raise FileNotFoundError(f"BP 路径下没有 LeRobot 数据集: {path}")
    return children


class BehaviorPromptCache:
    """按 task_type 读取固定目录里的示教缓存；没有命中时随机选一个数据集和可解码 episode。"""

    adapter_name = "bpvav2"

    def __init__(
        self,
        sources: dict[str, BPTaskSource],
        *,
        config,
        state_stats: dict[str, dict[str, np.ndarray]],
        action_stats: dict[str, dict[str, np.ndarray]],
        action_mode: str,
        qwen_path: str,
        cache_dir: Path,
    ) -> None:
        self.sources = dict(sources)
        self.config = config
        self.norm_stats = {**state_stats, **action_stats}
        self.action_mode = action_mode
        self.qwen_path = qwen_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get(self, task_type: str) -> dict[str, Any]:
        if task_type not in self.sources:
            configured = ", ".join(sorted(self.sources))
            raise KeyError(f"未知 task_type={task_type!r}；已配置任务: {configured}")
        with self._lock:
            if task_type not in self._cache:
                prompt = self._try_load(task_type)
                if prompt is None:
                    prompt, episode, dataset_path = self._build_random(self.sources[task_type])
                    prompt = self._persistable_prompt(prompt)
                    self._validate(prompt, task_type)
                    self._store(task_type, prompt, episode, dataset_path)
                self._cache[task_type] = prompt
            return self._cache[task_type]

    def _paths_for_stem(self, stem: str) -> tuple[Path, Path, Path]:
        return (
            self.cache_dir / f"{stem}.json",
            self.cache_dir / f"{stem}.pt",
            self.cache_dir / f"{stem}.mp4",
        )

    def _source_digest(self, task_type: str) -> str:
        paths = "|".join(str(path.resolve()) for path in self.sources[task_type].dataset_paths)
        return hashlib.sha1(paths.encode()).hexdigest()[:8]

    def _record_matches_source(self, task_type: str, meta_path: Path) -> bool:
        if not meta_path.is_file():
            return False
        try:
            record = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        cached = record.get("dataset_path")
        if not isinstance(cached, str) or not cached.strip():
            return False
        cached_path = Path(cached).expanduser().resolve()
        for root in self.sources[task_type].dataset_paths:
            root = root.resolve()
            if cached_path == root or root in cached_path.parents:
                return True
        return False

    def _cache_paths(self, task_type: str) -> tuple[Path, Path, Path]:
        """正确映射继续用原来的 task 文件；数据集对不上时另写一份，避免盖掉。"""
        legacy = self._paths_for_stem(safe_name(task_type))
        hashed = self._paths_for_stem(f"{safe_name(task_type)}-{self._source_digest(task_type)}")
        if self._record_matches_source(task_type, legacy[0]) and not hashed[1].is_file():
            return legacy
        return hashed

    def _signature(self) -> dict[str, Any]:
        height, width = tuple(getattr(self.config, "image_resolution", (224, 224)))
        signature = {
            "adapter": self.adapter_name,
            "bp_num_chunks": int(getattr(self.config, "bp_num_chunks")),
            "bp_action_chunk_size": int(getattr(self.config, "bp_action_chunk_size")),
            "bp_camera_keys": list(getattr(self.config, "bp_camera_keys")),
            "max_state_dim": int(getattr(self.config, "max_state_dim")),
            "max_action_dim": int(getattr(self.config, "max_action_dim")),
            "image_resolution": [int(height), int(width)],
            "action_mode": self.action_mode,
            "qwen_path": str(self.qwen_path),
        }
        # 只在开启时写入签名，已有缓存（可变 K）保持命中。
        if getattr(self.config, "bp_fixed_num_chunks", False):
            signature["bp_fixed_num_chunks"] = True
        return signature

    def _try_load(self, task_type: str) -> dict[str, Any] | None:
        meta_path, tensor_path, video_path = self._cache_paths(task_type)
        if not (meta_path.is_file() and tensor_path.is_file() and video_path.is_file()):
            return None
        try:
            record = json.loads(meta_path.read_text(encoding="utf-8"))
            if record.get("signature") != self._signature():
                logging.info("BP 缓存签名变化，重新生成 | task_type=%s | path=%s", task_type, video_path)
                return None
            prompt = torch.load(tensor_path, map_location="cpu", weights_only=False)
            self._validate(prompt, task_type)
        except Exception:
            logging.exception("BP 缓存损坏，重新生成 | task_type=%s", task_type)
            return None
        logging.info("从固定缓存加载 BP 示教 | task_type=%s | mp4=%s", task_type, video_path)
        return prompt

    def _store(self, task_type: str, prompt: dict[str, Any], episode_idx: int, dataset_path: Path) -> None:
        meta_path, tensor_path, video_path = self._cache_paths(task_type)
        self._write_episode_video(dataset_path, self.sources[task_type].video_backend, episode_idx, video_path)
        torch.save(prompt, tensor_path)
        record = {
            "task_type": task_type,
            "dataset_path": str(dataset_path),
            "episode_index": int(episode_idx),
            "video": video_path.name,
            "signature": self._signature(),
        }
        meta_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logging.info(
            "BP 示教已写入固定缓存 | task_type=%s | dataset=%s | episode=%d | mp4=%s",
            task_type,
            dataset_path,
            episode_idx,
            video_path,
        )

    def _build_random(self, source: BPTaskSource) -> tuple[dict[str, Any], int, Path]:
        candidates: list[Path] = []
        for root in source.dataset_paths:
            candidates.extend(_dataset_candidates(root))
        order = list(dict.fromkeys(candidates))
        random.shuffle(order)
        errors: list[str] = []
        for dataset_path in order:
            try:
                meta = LeRobotDatasetMetadata(str(dataset_path))
            except Exception as exc:
                errors.append(f"{dataset_path}: {type(exc).__name__}: {exc}")
                logging.warning("跳过不可读取 BP 数据集 | task=%s | %s", source.task_type, errors[-1])
                continue
            episodes = list(range(meta.total_episodes))
            random.shuffle(episodes)
            for episode_idx in episodes:
                try:
                    prompt = self._build_episode(source, dataset_path, meta, episode_idx)
                    return prompt, episode_idx, dataset_path
                except Exception as exc:
                    errors.append(f"{dataset_path} episode={episode_idx}: {type(exc).__name__}: {exc}")
                    logging.warning("跳过不可读取 BP episode | task=%s | %s", source.task_type, errors[-1])
        detail = "\n".join(errors[-5:])
        raise RuntimeError(f"task={source.task_type!r} 没有可读取的示教视频。最近错误:\n{detail}")

    def _apply_image_transform(self, data: dict[str, Any]) -> dict[str, Any]:
        return BPVAv2QwenImageTransformFn(pretrained_model_name_or_path=self.qwen_path)(data)

    def _persistable_prompt(self, prompt: dict[str, Any]) -> dict[str, Any]:
        return prompt

    def _build_episode(
        self,
        source: BPTaskSource,
        dataset_path: Path,
        meta: LeRobotDatasetMetadata,
        episode_idx: int,
    ) -> dict[str, Any]:
        action_steps = int(getattr(self.config, "bp_action_chunk_size"))
        dataset = LeRobotDataset(
            str(dataset_path),
            episodes=[episode_idx],
            delta_timestamps={ACTION: [step / meta.fps for step in range(action_steps)]},
            video_backend=source.video_backend,
        )
        if len(dataset) <= 0:
            raise ValueError("episode 为空")
        num_chunks = int(getattr(self.config, "bp_num_chunks"))
        if not getattr(self.config, "bp_fixed_num_chunks", False):
            num_chunks = min(num_chunks, max(1, math.ceil(len(dataset) / action_steps)))
        frame_indices = torch.linspace(0, len(dataset) - 1, steps=num_chunks).round().to(torch.long).tolist()
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
        data = BPVAv2RemapImageKeyTransformFn(
            mapping=image_mapping,
            bp_camera_keys=list(getattr(self.config, "bp_camera_keys")),
        )(data)
        data = BPPadOrSampleChunksFn(num_chunks=int(getattr(self.config, "bp_num_chunks")))(data)
        height, width = tuple(getattr(self.config, "image_resolution", (224, 224)))
        data = BPResizeImagesWithPadFn(height=int(height), width=int(width))(data)
        data = self._apply_image_transform(data)
        if self.action_mode == "delta":
            mask = get_mask_mapping(dataset.meta.robot_type, dataset.meta.features)
            data = BPDeltaActionTransformFn(mask=mask)(data)
        data = BPNormalizeTransformFn(selected_keys=[OBS_STATE, ACTION], norm_stats=self.norm_stats)(data)
        data = BPPadStateAndActionTransformFn(
            max_state_dim=int(getattr(self.config, "max_state_dim")),
            max_action_dim=int(getattr(self.config, "max_action_dim")),
        )(data)
        return data[BP_PREFIX]

    def _validate(self, prompt: dict[str, Any], task_type: str) -> None:
        chunks = int(getattr(self.config, "bp_num_chunks"))
        action_steps = int(getattr(self.config, "bp_action_chunk_size"))
        state_dim = int(getattr(self.config, "max_state_dim"))
        action_dim = int(getattr(self.config, "max_action_dim"))
        expected_images = set(getattr(self.config, "bp_camera_keys"))
        if set(prompt.get("images", {})) != expected_images:
            raise ValueError(f"task={task_type!r} BP 相机不匹配: {sorted(prompt.get('images', {}))}")
        for key, image in prompt["images"].items():
            if image.ndim != 4 or image.shape[0] != chunks or image.shape[1] != 3:
                raise ValueError(f"task={task_type!r} BP 图像 {key} shape 非法: {tuple(image.shape)}")
        if tuple(prompt["state"].shape) != (chunks, state_dim):
            raise ValueError(f"task={task_type!r} BP state shape 非法: {tuple(prompt['state'].shape)}")
        if tuple(prompt["action"].shape) != (chunks, action_steps, action_dim):
            raise ValueError(f"task={task_type!r} BP action shape 非法: {tuple(prompt['action'].shape)}")
        if tuple(prompt["action_is_pad"].shape) != (chunks, action_steps):
            raise ValueError(f"task={task_type!r} BP action_is_pad shape 非法")
        if tuple(prompt["mask"].shape) != (chunks,):
            raise ValueError(f"task={task_type!r} BP mask shape 非法")

    def _write_episode_video(
        self,
        dataset_path: Path,
        video_backend: str,
        episode_idx: int,
        output: Path,
    ) -> Path:
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("导出 MP4 需要 PyAV。") from exc

        meta = LeRobotDatasetMetadata(str(dataset_path))
        camera_key = _select_main_camera(list(meta.camera_keys))
        dataset = LeRobotDataset(
            str(dataset_path),
            episodes=[episode_idx],
            video_backend=video_backend,
        )
        if len(dataset) == 0:
            raise ValueError("待记录的 BP episode 为空。")
        first_frame = _tensor_to_rgb_uint8(dataset[0][camera_key])
        height, width = first_frame.shape[:2]
        output.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(output), mode="w") as container:
            stream = container.add_stream("libx264", rate=max(1, round(float(meta.fps))))
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            for index in range(len(dataset)):
                frame = first_frame if index == 0 else _tensor_to_rgb_uint8(dataset[index][camera_key])
                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return output


def _select_main_camera(camera_keys: list[str]) -> str:
    if not camera_keys:
        raise ValueError("BP 数据集没有相机字段。")
    priorities = ("image0", "cam_high", "head", "main", "front")
    lowered = {key: key.lower() for key in camera_keys}
    for token in priorities:
        for key in camera_keys:
            if token in lowered[key]:
                return key
    return camera_keys[0]


def _tensor_to_rgb_uint8(image: Any) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"图像必须为 3 维，实际 shape={array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating) and array.size and float(np.nanmax(array)) <= 1.5:
        array = array * 255.0
    array = np.clip(np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return np.ascontiguousarray(array)


class BPVAv2Adapter(PolicyAdapter):
    name = "bpvav2"
    family = "bp"
    aliases = ("bpvav2",)

    def _check_type(self) -> None:
        if not is_bpvav2(self.config.type):
            raise ValueError(
                f"POLICY=bpvav2 但 checkpoint type={self.config.type!r}。请改 POLICY，或把 POLICY 设为 auto。"
            )

    def before_policy_init(self) -> None:
        if bool(getattr(self.config, "bp_vision_pretrained", False)):
            logging.info("离线加载 BPVAv2 checkpoint：跳过 timm BP ViT 预训练权重下载。")
            self.config.bp_vision_pretrained = False

    def setup_policy(self) -> None:
        processor_path = (
            getattr(self.config, "qwen3_vl_processor_path", None)
            or getattr(self.config, "qwen3_vl_pretrained_path", None)
            or self.args.qwen3_vl_path
            or "/vla/workspace/models/Qwen3-VL-2B-Instruct"
        )
        self.processor_fn = ImgOnlyQwen3VLTransformFn(pretrained_model_name_or_path=str(processor_path))
        if not self.args.bp_mapping_path:
            raise ValueError("BPVAv2 需要 BP_MAPPING_PATH。")
        self.mapping_path, sources = load_bp_task_sources(self.args.bp_mapping_path)
        cache_dir = fixed_bp_video_cache_dir(self.args.debug_dir, self.name)
        self.bp_cache = BehaviorPromptCache(
            sources,
            config=self.config,
            state_stats=self.state_stats,
            action_stats=self.action_stats,
            action_mode=self.action_mode,
            qwen_path=str(processor_path),
            cache_dir=cache_dir,
        )
        self._metadata_extra = {
            "bp_mapping_path": str(self.mapping_path),
            "bp_video_cache_dir": str(cache_dir),
            "bp_selection": "cached_or_random_dataset_episode",
            "bp_configured_tasks": sorted(sources),
            "bp_num_chunks": int(getattr(self.config, "bp_num_chunks")),
            "bp_action_chunk_size": int(getattr(self.config, "bp_action_chunk_size")),
            "required_image_history": 2 if self.args.strict_history else 1,
        }
        logging.info("BPVAv2 任务映射 | tasks=%d | path=%s", len(sources), self.mapping_path)

    def build_metadata(self, chunk_size: int) -> dict[str, Any]:
        metadata = super().build_metadata(chunk_size)
        metadata.update(getattr(self, "_metadata_extra", {}))
        return metadata

    def prepare(self, obs: dict[str, Any]) -> PreparedSample:
        task_type = obs.get("task_type")
        if not isinstance(task_type, str) or not task_type.strip():
            raise KeyError("BPVAv2 请求缺少非空字符串 task_type。")
        images = resolve_images(obs)
        state_np = resolve_state(obs)
        sample: dict[str, Any] = {OBS_STATE: torch.from_numpy(state_np)}
        masks: dict[str, bool] = {}
        for key in CAMERA_ALIASES:
            history, mask = resolve_image_history(
                images,
                key,
                strict=self.args.strict_history,
                frames=2,
                height=self.args.request_image_height,
                width=self.args.request_image_width,
            )
            sample[key] = hwc_tensor(history)
            masks[key] = mask
        sample = self.resize_fn(sample)
        for key, mask in masks.items():
            sample[f"{key}_mask"] = torch.tensor(mask)
        sample = self.processor_fn(sample)
        sample = self.normalize_state_fn(sample)
        inputs = self.pack_inputs(sample)
        inputs[BP_PREFIX] = batch_nested(self.bp_cache.get(task_type.strip()), self.device, self.runtime_dtype)
        return PreparedSample(inputs, state_np)


ADAPTER = BPVAv2Adapter
