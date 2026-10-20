"""BPVAv3：当前帧走 Qwen3.5，示教视频走固定缓存。"""

from __future__ import annotations

import json
import logging
import random
import threading
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from .base import PolicyAdapter, PreparedSample
from .bp_dim_use import apply_bp_dim_use, resolve_bp_dim_use
from .bpvav2 import BP_PREFIX, BehaviorPromptCache, fixed_bp_video_cache_dir, load_bp_task_sources
from .common import (
    CAMERA_ALIASES,
    OBS_STATE,
    batch_nested,
    hwc_tensor,
    resolve_image_history,
    resolve_images,
    resolve_state,
)
from lerobot.policies.BPVAv3.configuration_bpvav3 import validate_bp_dim_use
from lerobot.policies.BPVAv3.transform_bpvav3 import (
    BPVAv3ProcessorTransformFn,
    BPVAv3Qwen35ImageTransformFn,
)
from lerobot.policies.names import is_bpvav3


class BPVAv3PromptCache(BehaviorPromptCache):
    """示教侧使用 Qwen3.5 image patches，缓存里只保留模型需要的字段。"""

    adapter_name = "bpvav3"

    def _apply_image_transform(self, data: dict[str, Any]) -> dict[str, Any]:
        return BPVAv3Qwen35ImageTransformFn(pretrained_model_name_or_path=self.qwen_path)(data)

    def _persistable_prompt(self, prompt: dict[str, Any]) -> dict[str, Any]:
        kept = {
            "mask": prompt["mask"],
            "chunk_indices": prompt["chunk_indices"],
            "state": prompt["state"],
            "action": prompt["action"],
            "action_is_pad": prompt["action_is_pad"],
            "bp_pixel_values": prompt["bp_pixel_values"],
            "bp_image_grid_thw": prompt["bp_image_grid_thw"],
        }
        if "state_is_available" in prompt:
            kept["state_is_available"] = prompt["state_is_available"]
        if "image_masks" in prompt:
            kept["image_masks"] = prompt["image_masks"]
        return kept

    def _validate(self, prompt: dict[str, Any], task_type: str) -> None:
        chunks = int(getattr(self.config, "bp_num_chunks"))
        action_steps = int(getattr(self.config, "bp_action_chunk_size"))
        state_dim = int(getattr(self.config, "max_state_dim"))
        action_dim = int(getattr(self.config, "max_action_dim"))
        cameras = set(getattr(self.config, "bp_camera_keys"))
        pixels = prompt.get("bp_pixel_values") or {}
        grids = prompt.get("bp_image_grid_thw") or {}
        if set(pixels) != cameras or set(grids) != cameras:
            raise ValueError(
                f"task={task_type!r} BP 相机不匹配: pixels={sorted(pixels)} grids={sorted(grids)}"
            )
        for key, pixel in pixels.items():
            if pixel.ndim != 3 or int(pixel.shape[0]) != chunks:
                raise ValueError(f"task={task_type!r} BP pixel {key} shape 非法: {tuple(pixel.shape)}")
            grid = grids[key]
            if grid.ndim != 2 or int(grid.shape[0]) != chunks or int(grid.shape[1]) != 3:
                raise ValueError(f"task={task_type!r} BP grid {key} shape 非法: {tuple(grid.shape)}")
        if tuple(prompt["state"].shape) != (chunks, state_dim):
            raise ValueError(f"task={task_type!r} BP state shape 非法: {tuple(prompt['state'].shape)}")
        if tuple(prompt["action"].shape) != (chunks, action_steps, action_dim):
            raise ValueError(f"task={task_type!r} BP action shape 非法: {tuple(prompt['action'].shape)}")
        if tuple(prompt["action_is_pad"].shape) != (chunks, action_steps):
            raise ValueError(f"task={task_type!r} BP action_is_pad shape 非法")
        if tuple(prompt["mask"].shape) != (chunks,):
            raise ValueError(f"task={task_type!r} BP mask shape 非法")


def image_thumb(image: torch.Tensor) -> torch.Tensor:
    """任意 [.., H, W, 3] / [.., 3, H, W] 图像 → 28x28 低分辨率 [3*28*28]，取值 [0, 1]。"""
    image = torch.as_tensor(image).float()
    while image.ndim > 3:
        image = image[0]
    if image.shape[0] != 3 and image.shape[-1] == 3:
        image = image.permute(2, 0, 1)
    if image.max() > 1.5:
        image = image / 255.0
    return F.adaptive_avg_pool2d(image[None], (28, 28))[0].flatten()


class BPVAv3DemoPool(BPVAv3PromptCache):
    """每个任务一组候选示教，附带第 0 个关键帧的头部相机缩略图，用于按当前画面挑示教。"""

    def __init__(self, *args, pool_size: int, seed: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pool_size = int(pool_size)
        self.seed = int(seed)
        self._pools: dict[str, list[tuple[int, dict[str, Any], torch.Tensor]]] = {}
        self._capture: torch.Tensor | None = None

    def _apply_image_transform(self, data: dict[str, Any]) -> dict[str, Any]:
        images = data[BP_PREFIX]["images"]
        self._capture = image_thumb(images[sorted(images)[0]][0])
        return super()._apply_image_transform(data)

    def pool(self, task_type: str) -> list[tuple[int, dict[str, Any], torch.Tensor]]:
        with self._lock:
            if task_type in self._pools:
                return self._pools[task_type]
            source = self.sources[task_type]
            path = self.cache_dir / f"{task_type}-pool{self.pool_size}-s{self.seed}.pt"
            if path.is_file():
                record = torch.load(path, map_location="cpu", weights_only=False)
                if record.get("signature") == self._signature():
                    self._pools[task_type] = record["pool"]
                    return record["pool"]
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

            from .bpvav2 import _dataset_candidates

            dataset_path = _dataset_candidates(source.dataset_paths[0])[0]
            meta = LeRobotDatasetMetadata(str(dataset_path))
            rng = random.Random(f"{self.seed}:{task_type}")
            episodes = rng.sample(range(meta.total_episodes), min(self.pool_size, meta.total_episodes))
            pool = []
            for episode in episodes:
                prompt = self._persistable_prompt(self._build_episode(source, dataset_path, meta, episode))
                self._validate(prompt, task_type)
                pool.append((int(episode), prompt, self._capture))
            torch.save({"signature": self._signature(), "pool": pool}, path)
            logging.info("BP 示教池已生成 | task=%s | episodes=%s | path=%s", task_type, episodes, path)
            self._pools[task_type] = pool
            return pool


class BPVAv3Adapter(PolicyAdapter):
    name = "bpvav3"
    family = "bp"
    aliases = ("bpvav3",)

    def _check_type(self) -> None:
        if not is_bpvav3(self.config.type):
            raise ValueError(
                f"POLICY=bpvav3 但 checkpoint type={self.config.type!r}。请改 POLICY，或把 POLICY 设为 auto。"
            )

    def setup_policy(self) -> None:
        processor_path = (
            getattr(self.config, "qwen3_5_processor_path", None)
            or getattr(self.config, "qwen3_5_pretrained_path", None)
            or self.args.qwen3_5_path
        )
        if not processor_path:
            raise ValueError("缺少 Qwen3.5 路径，请设置 QWEN3_5_PATH。")
        self.processor_fn = BPVAv3ProcessorTransformFn(
            pretrained_model_name_or_path=str(processor_path),
            max_length=int(getattr(self.config, "tokenizer_max_length", 48)),
            current_image_index=0,
            use_language=bool(getattr(self.config, "use_language", True)),
        )
        if not self.args.bp_mapping_path:
            raise ValueError("BPVAv3 需要 BP_MAPPING_PATH。")
        self.bp_action_use, self.bp_state_use, self.bp_dim_source = resolve_bp_dim_use(
            self.args.bp_dim_config,
            self.train_cfg,
        )
        validate_bp_dim_use(
            self.bp_action_use,
            name="bp_action_use",
            max_dim=int(getattr(self.config, "max_action_dim")),
        )
        validate_bp_dim_use(
            self.bp_state_use,
            name="bp_state_use",
            max_dim=int(getattr(self.config, "max_state_dim")),
        )
        self.mapping_path, sources = load_bp_task_sources(self.args.bp_mapping_path)
        cache_dir = fixed_bp_video_cache_dir(self.args.debug_dir, self.name)
        self.bp_cache = BPVAv3PromptCache(
            sources,
            config=self.config,
            state_stats=self.state_stats,
            action_stats=self.action_stats,
            action_mode=self.action_mode,
            qwen_path=str(processor_path),
            cache_dir=cache_dir,
        )
        selection = (yaml.safe_load(Path(self.mapping_path).read_text(encoding="utf-8")) or {}).get("selection") or {}
        self.bp_select_mode = str(selection.get("mode", "fixed"))
        if self.bp_select_mode not in {"fixed", "best", "worst"}:
            raise ValueError(f"selection.mode 必须是 fixed / best / worst，得到 {self.bp_select_mode!r}")
        self.bp_pool = None
        self._bp_choice: dict[tuple[str, str], int] = {}
        self._bp_choice_lock = threading.Lock()
        self._bp_select_log = Path(cache_dir) / "selection_log.jsonl"
        if self.bp_select_mode != "fixed":
            self.bp_pool = BPVAv3DemoPool(
                sources,
                config=self.config,
                state_stats=self.state_stats,
                action_stats=self.action_stats,
                action_mode=self.action_mode,
                qwen_path=str(processor_path),
                cache_dir=cache_dir,
                pool_size=int(selection.get("pool_size", 10)),
                seed=int(selection.get("seed", 0)),
            )
        self._metadata_extra = {
            "bp_mapping_path": str(self.mapping_path),
            "bp_video_cache_dir": str(cache_dir),
            "bp_selection": self.bp_select_mode if self.bp_pool else "cached_or_random_dataset_episode",
            "bp_configured_tasks": sorted(sources),
            "bp_num_chunks": int(getattr(self.config, "bp_num_chunks")),
            "bp_action_chunk_size": int(getattr(self.config, "bp_action_chunk_size")),
            "required_image_history": 1,
            "use_language": bool(getattr(self.config, "use_language", True)),
            "bp_action_use": self.bp_action_use,
            "bp_state_use": self.bp_state_use,
            "bp_dim_source": self.bp_dim_source,
        }
        logging.info("BPVAv3 任务映射 | tasks=%d | path=%s", len(sources), self.mapping_path)
        logging.info(
            "BP 示教维度 | action=%s | state=%s | source=%s",
            self.bp_action_use,
            self.bp_state_use,
            self.bp_dim_source,
        )

    def build_metadata(self, chunk_size: int) -> dict[str, Any]:
        metadata = super().build_metadata(chunk_size)
        metadata.update(getattr(self, "_metadata_extra", {}))
        return metadata

    def on_request(self, connection_id: str, obs: dict[str, Any]) -> None:
        obs["__bp_connection"] = connection_id
        super().on_request(connection_id, obs)

    def _select_prompt(self, obs: dict[str, Any], task_type: str, head_image: torch.Tensor) -> dict[str, Any]:
        """每个连接在 episode 开头（reset / 首次请求）按头部画面选一条示教，之后保持不变。"""
        pool = self.bp_pool.pool(task_type)
        key = (str(obs.get("__bp_connection", "")), task_type)
        with self._bp_choice_lock:
            if bool(obs.get("reset")) or key not in self._bp_choice:
                thumb = image_thumb(head_image)
                dist = torch.stack([(thumb - item[2]).square().mean() for item in pool])
                index = int(dist.argmin() if self.bp_select_mode == "best" else dist.argmax())
                self._bp_choice[key] = index
                record = {"task": task_type, "connection": key[0], "mode": self.bp_select_mode,
                          "episode": pool[index][0], "pool": [item[0] for item in pool],
                          "dist": [round(float(d), 5) for d in dist]}
                with self._bp_select_log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
            return pool[self._bp_choice[key]][1]

    def prepare(self, obs: dict[str, Any]) -> PreparedSample:
        task_type = obs.get("task_type")
        if not isinstance(task_type, str) or not task_type.strip():
            raise KeyError("BPVAv3 请求缺少非空字符串 task_type。")
        images = resolve_images(obs)
        state_np = resolve_state(obs)
        sample: dict[str, Any] = {"task": self.resolve_task(obs), OBS_STATE: torch.from_numpy(state_np)}
        masks: dict[str, bool] = {}
        for key in CAMERA_ALIASES:
            history, mask = resolve_image_history(
                images,
                key,
                strict=False,
                frames=1,
                height=self.args.request_image_height,
                width=self.args.request_image_width,
            )
            sample[key] = hwc_tensor(history)
            masks[key] = mask
        sample = self.resize_fn(sample)
        head_image = sample["observation.images.image0"]
        for key, mask in masks.items():
            sample[f"{key}_mask"] = torch.tensor(mask)
        sample = self.processor_fn(sample)
        sample = self.normalize_state_fn(sample)
        kept = {
            OBS_STATE: sample[OBS_STATE],
            "observation.pixel_values": sample["observation.pixel_values"],
            "observation.image_grid_thw": sample["observation.image_grid_thw"],
            "observation.input_ids": sample["observation.input_ids"],
            "observation.attention_mask": sample["observation.attention_mask"],
            "observation.mm_token_type_ids": sample["observation.mm_token_type_ids"],
        }
        inputs = self.pack_inputs(kept)
        if self.bp_pool is not None:
            raw_prompt = self._select_prompt(obs, task_type.strip(), head_image)
        else:
            raw_prompt = self.bp_cache.get(task_type.strip())
        prompt = batch_nested(raw_prompt, self.device, self.runtime_dtype)
        inputs[BP_PREFIX] = apply_bp_dim_use(prompt, self.bp_action_use, self.bp_state_use)
        return PreparedSample(inputs, state_np)


ADAPTER = BPVAv3Adapter
