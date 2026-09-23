"""BPVAv2：按 task_type 缓存首个可读示教，并支持两帧历史观测。"""

from __future__ import annotations

import logging
import math
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


@dataclass(frozen=True)
class BPTaskSource:
    task_type: str
    dataset_path: Path
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
        configured = entry.get("dataset_path")
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError(f"task={task!r} 缺少非空 dataset_path。")
        dataset_path = Path(configured).expanduser()
        if not dataset_path.is_absolute():
            dataset_path = (path.parent / dataset_path).resolve()
        else:
            dataset_path = dataset_path.resolve()
        if not (dataset_path / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"task={task!r} 的 LeRobot 数据集无 meta/info.json: {dataset_path}")
        video_backend = str(entry.get("video_backend", "pyav")).strip() or "pyav"
        sources[task] = BPTaskSource(task, dataset_path, video_backend)
    return path, sources


class BehaviorPromptCache:
    """每个 task_type 懒加载并缓存一条确定性的 behavior prompt。"""

    def __init__(
        self,
        sources: dict[str, BPTaskSource],
        *,
        config,
        state_stats: dict[str, dict[str, np.ndarray]],
        action_stats: dict[str, dict[str, np.ndarray]],
        action_mode: str,
        qwen_path: str,
    ) -> None:
        self.sources = dict(sources)
        self.config = config
        self.norm_stats = {**state_stats, **action_stats}
        self.action_mode = action_mode
        self.qwen_path = qwen_path
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get(self, task_type: str) -> dict[str, Any]:
        if task_type not in self.sources:
            configured = ", ".join(sorted(self.sources))
            raise KeyError(f"未知 task_type={task_type!r}；已配置任务: {configured}")
        with self._lock:
            if task_type not in self._cache:
                prompt, episode = self._build_first_readable(self.sources[task_type])
                self._validate(prompt, task_type)
                self._cache[task_type] = prompt
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

    def _build_episode(self, source: BPTaskSource, meta: LeRobotDatasetMetadata, episode_idx: int) -> dict[str, Any]:
        action_steps = int(getattr(self.config, "bp_action_chunk_size"))
        dataset = LeRobotDataset(
            str(source.dataset_path),
            episodes=[episode_idx],
            delta_timestamps={ACTION: [step / meta.fps for step in range(action_steps)]},
            video_backend=source.video_backend,
        )
        if len(dataset) <= 0:
            raise ValueError("episode 为空")
        num_chunks = min(
            int(getattr(self.config, "bp_num_chunks")),
            max(1, math.ceil(len(dataset) / action_steps)),
        )
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
        data = BPVAv2QwenImageTransformFn(pretrained_model_name_or_path=self.qwen_path)(data)
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


class LoggingBehaviorPromptCache(BehaviorPromptCache):
    """缓存示教时，把主相机轨迹写成 MP4。失败不影响推理。"""

    def __init__(self, *args: Any, run_log_dir: Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.run_log_dir = run_log_dir

    def _build_first_readable(self, source: BPTaskSource) -> tuple[dict[str, Any], int]:
        prompt, episode_idx = super()._build_first_readable(source)
        try:
            output = self._write_episode_video(source, episode_idx)
            logging.info(
                "BP 示教轨迹已保存 | task_type=%s | episode=%d | path=%s",
                source.task_type,
                episode_idx,
                output,
            )
        except Exception:
            logging.exception(
                "BP 示教轨迹保存失败，继续推理 | task_type=%s | episode=%d",
                source.task_type,
                episode_idx,
            )
        return prompt, episode_idx

    def _write_episode_video(self, source: BPTaskSource, episode_idx: int) -> Path:
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("导出 MP4 需要 PyAV。") from exc

        meta = LeRobotDatasetMetadata(str(source.dataset_path))
        camera_key = _select_main_camera(list(meta.camera_keys))
        dataset = LeRobotDataset(
            str(source.dataset_path),
            episodes=[episode_idx],
            video_backend=source.video_backend,
        )
        if len(dataset) == 0:
            raise ValueError("待记录的 BP episode 为空。")
        first_frame = _tensor_to_rgb_uint8(dataset[0][camera_key])
        height, width = first_frame.shape[:2]
        output = self.run_log_dir / (
            f"{safe_name(source.task_type)}_episode_{episode_idx:06d}_{safe_name(camera_key)}.mp4"
        )
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
        cache_kwargs = dict(
            sources=sources,
            config=self.config,
            state_stats=self.state_stats,
            action_stats=self.action_stats,
            action_mode=self.action_mode,
            qwen_path=str(processor_path),
        )
        if self.args.debug:
            if not self.args.debug_run_dir:
                raise RuntimeError("DEBUG=True 时必须提供 debug 目录。")
            run_log_dir = Path(self.args.debug_run_dir) / "bp"
            self.bp_cache = LoggingBehaviorPromptCache(**cache_kwargs, run_log_dir=run_log_dir)
            self._metadata_extra = {"bp_video_logging": True, "bplogs_run_dir": str(run_log_dir)}
        else:
            self.bp_cache = BehaviorPromptCache(**cache_kwargs)
            self._metadata_extra = {"bp_video_logging": False}
        self._metadata_extra.update(
            {
                "bp_mapping_path": str(self.mapping_path),
                "bp_selection": "first_readable_episode",
                "bp_configured_tasks": sorted(sources),
                "bp_num_chunks": int(getattr(self.config, "bp_num_chunks")),
                "bp_action_chunk_size": int(getattr(self.config, "bp_action_chunk_size")),
                "required_image_history": 2 if self.args.strict_history else 1,
            }
        )
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
