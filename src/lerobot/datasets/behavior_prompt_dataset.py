from __future__ import annotations

import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transforms.constants import get_feature_mapping, get_image_mapping, get_mask_mapping
from lerobot.transforms.core import (
    ComposeFieldsTransform,
    DataDict,
    DataTransformFn,
    DeltaActionTransformFn,
    IdentityTransformFn,
    InjectMissingStateActionTransformFn,
    NormalizeTransformFn,
    PadStateAndActionTransformFn,
    RemapImageKeyTransformFn,
    ResizeImagesWithPadFn,
    compose,
    hydrate_compose_field_transform,
    hydrate_delta_action_transform,
    hydrate_inject_missing_state_action_transform,
    hydrate_normalize_transform,
    hydrate_remap_image_key_transform,
)
from lerobot.transforms.core_bp import (
    BPComposeFieldsTransform,
    BPDeltaActionTransformFn,
    BPNormalizeTransformFn,
    BPPadOrSampleChunksFn,
    BPPadStateAndActionTransformFn,
    BPRemapImageKeyTransformFn,
    BPResizeImagesWithPadFn,
    BPVAv2QwenImageTransformFn,
    BPVAv2RemapImageKeyTransformFn,
    ImgOnlyQwen3VLTransformFn,
    UnifyBPInputsTransformFn,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


BP_PREFIX = "behavior_prompt"

logger = logging.getLogger(__name__)


@dataclass
class BehaviorPromptConfig:
    """Configuration for sampling a behavior prompt trajectory.

    Prompt episodes are drawn within the current repo, optionally matching the full task name.
    The selected trajectory is sampled into at most K action chunks before video IO.
    A later transform only pads short trajectories to fixed K for batching.
    """

    prompt_action_chunk_size: int = 50
    max_prompt_chunks: int | None = None
    # true 时所有轨迹都取满 K 个关键帧，不再按 ceil(T / prompt_action_chunk_size) 截断。
    fixed_num_chunks: bool = False
    # true 时给样本加 bp_task_id（repo 上级目录名的稳定 hash），供 BP 任务对比损失使用。
    attach_task_id: bool = False
    seed: int = 0
    # random/avoid 默认同 repo；*_in_task 按完整任务描述匹配；avoid_in_task 无候选时 mask BP。
    bp_sampling_mode: str = "random"
    align_bp_encoder: bool = False
    future_filter: bool = True

    # BPVA 默认数据变换配置。
    num_chunks: int = 4
    height: int = 224
    width: int = 224
    max_state_dim: int = 32
    max_action_dim: int = 32
    qwen3_vl_processor_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    bp_camera_keys: list[str] | None = None
    action_mode: str = ""
    dynamic_bp_cameras: bool = False

    def __post_init__(self) -> None:
        if self.bp_camera_keys is None:
            self.bp_camera_keys = [
                f"{OBS_IMAGES}.image0",
                f"{OBS_IMAGES}.image1",
                f"{OBS_IMAGES}.image2",
            ]
        if self.bp_sampling_mode not in {"random", "origin", "avoid", "random_in_task", "random_in_repo", "avoid_in_task", "avoid_in_repo"}:
            raise ValueError(
                "Invalid bp_sampling_mode; expected origin, random[_in_task|_in_repo], or avoid[_in_task|_in_repo], "
                f"got {self.bp_sampling_mode!r}"
            )


def _make_bpva_transform_fns(prompt_cfg: BehaviorPromptConfig) -> list[DataTransformFn]:
    """构造 BPVA 数据变换链，不依赖 DatasetConfig 实例。"""
    transforms: list[DataTransformFn] = [
        # BP 分支：固定 K 个 chunk，并处理图像、状态和动作。
        (BPVAv2RemapImageKeyTransformFn(bp_camera_keys=list(prompt_cfg.bp_camera_keys or []))
         if prompt_cfg.dynamic_bp_cameras
         else BPRemapImageKeyTransformFn(bp_camera_keys=list(prompt_cfg.bp_camera_keys or []))),
        BPPadOrSampleChunksFn(num_chunks=prompt_cfg.num_chunks),
        BPResizeImagesWithPadFn(height=prompt_cfg.height, width=prompt_cfg.width),
        *( [BPVAv2QwenImageTransformFn(pretrained_model_name_or_path=prompt_cfg.qwen3_vl_processor_path)]
           if prompt_cfg.dynamic_bp_cameras else [] ),
        BPComposeFieldsTransform(),
        BPDeltaActionTransformFn(),
        BPNormalizeTransformFn(),
        BPPadStateAndActionTransformFn(
            max_state_dim=prompt_cfg.max_state_dim,
            max_action_dim=prompt_cfg.max_action_dim,
        ),
        # 当前观测分支：沿用 TBot 的 Qwen 图像输入。
        InjectMissingStateActionTransformFn(),
        ResizeImagesWithPadFn(height=prompt_cfg.height, width=prompt_cfg.width),
        RemapImageKeyTransformFn(),
        NormalizeTransformFn(),
        ComposeFieldsTransform(),
        PadStateAndActionTransformFn(
            max_state_dim=prompt_cfg.max_state_dim,
            max_action_dim=prompt_cfg.max_action_dim,
        ),
        ImgOnlyQwen3VLTransformFn(pretrained_model_name_or_path=prompt_cfg.qwen3_vl_processor_path),
        UnifyBPInputsTransformFn(),
    ]
    if str(prompt_cfg.action_mode).lower() == "obs":
        transforms = [t for t in transforms if not isinstance(t, (BPDeltaActionTransformFn, DeltaActionTransformFn))]
    elif not any(isinstance(t, DeltaActionTransformFn) for t in transforms):
        current_insert_idx = next((i for i, t in enumerate(transforms) if isinstance(t, ResizeImagesWithPadFn)), 0)
        transforms.insert(current_insert_idx, DeltaActionTransformFn())
    return transforms

class BehaviorPromptLeRobotDataset(Dataset):
    """Attach a behavior prompt trajectory to each current LeRobot sample.

    Args:
        current_ds: Dataset with delta timestamps. It provides the normal TBot
            current sample and action window.
        prompt_ds: Dataset whose images use only the current frame while action
            uses the configured future action window. One access returns a complete BP chunk.
        prompt_cfg: Prompt sampling configuration.
        transform: Optional transform applied after `behavior_prompt` is attached.

    Output:
        The returned sample contains all current sample fields plus a nested
        `behavior_prompt` dictionary with images/state/action/mask and source
        metadata needed by BP transforms.
    """

    def __init__(
        self,
        current_ds: LeRobotDataset,
        prompt_ds: LeRobotDataset,
        prompt_cfg: BehaviorPromptConfig,
        transform: DataTransformFn | None = None,
    ) -> None:
        self.current_ds = current_ds
        self.prompt_ds = prompt_ds
        self.prompt_cfg = prompt_cfg
        self.transform = transform or IdentityTransformFn()
        self.rng = random.Random(prompt_cfg.seed)

        started_at = time.perf_counter()
        self._episode_ranges = self._build_episode_ranges()
        logger.info(
            "Built %d behavior-prompt episode ranges in %.3fs",
            len(self._episode_ranges),
            time.perf_counter() - started_at,
        )
        if prompt_cfg.bp_sampling_mode in {"random_in_task", "avoid_in_task"}:
            self._current_task_names = self._build_task_index_to_name(current_ds)
            self._prompt_task_names = self._build_task_index_to_name(prompt_ds)
            self._task_to_episodes = self._build_task_to_episodes()
            if prompt_cfg.bp_sampling_mode == "avoid_in_task":
                rows = self._episode_metadata_rows()
                masked_frames = 0
                total_frames = 0
                for episode, (start, end, _) in self._episode_ranges.items():
                    task_ids = self._task_indices_from_metadata(rows[episode])
                    if not task_ids:
                        task_ids = {self._task_indices_at_episode_starts([episode])[episode]}
                    # Weight each task by the frames carrying that task (usually one per episode).
                    pools = [self._task_to_episodes.get(self._prompt_task_names[t], []) for t in task_ids]
                    if len(task_ids) == 1:
                        total_frames += end - start
                        if not any(other != episode for other in pools[0]):
                            masked_frames += end - start
                    else:
                        for idx in range(start, end):
                            task = self._to_int(self.prompt_ds.hf_dataset[idx]["task_index"])
                            total_frames += 1
                            if not any(other != episode for other in self._task_to_episodes.get(self._prompt_task_names[task], [])):
                                masked_frames += 1
                if masked_frames:
                    logger.warning("avoid_in_task: no alternative BP for %.2f%% of loaded frames in repo %s; these samples will have BP masked", 100 * masked_frames / max(total_frames, 1), self.repo_id)
                logger.info("Expected masked BP frame fraction for repo %s: %d/%d (%.2f%%)", self.repo_id, masked_frames, total_frames, 100 * masked_frames / max(total_frames, 1))


    @classmethod
    def with_default_transforms(
        cls,
        current_ds: LeRobotDataset,
        prompt_ds: LeRobotDataset,
        prompt_cfg: BehaviorPromptConfig | None = None,
    ) -> BehaviorPromptLeRobotDataset:
        """构造 BPVA 数据集及其完整数据变换链。"""
        if prompt_cfg is None:
            prompt_cfg = BehaviorPromptConfig()
        transforms = _make_bpva_transform_fns(prompt_cfg)
        # UnifyBPInputsTransformFn 会移除可变长采样元数据，只保留可稳定拼 batch 的模型输入。
        transforms = hydrate_inject_missing_state_action_transform(transforms, current_ds)
        transforms = hydrate_normalize_transform(transforms, current_ds)
        transforms = hydrate_compose_field_transform(transforms, current_ds)
        transforms = hydrate_delta_action_transform(transforms, current_ds)
        transforms = hydrate_remap_image_key_transform(transforms, current_ds)

        robot_type = prompt_ds.meta.robot_type
        feature_mapping = get_feature_mapping(robot_type, prompt_ds.meta.features)
        image_mapping = get_image_mapping(robot_type, prompt_ds.meta.features)
        for idx, transform in enumerate(transforms):
            if isinstance(transform, BPNormalizeTransformFn):
                transforms[idx] = replace(
                    transform,
                    norm_stats=prompt_ds.meta.stats,
                    selected_keys=[OBS_STATE, ACTION],
                    mapping=feature_mapping,
                )
            elif isinstance(transform, BPComposeFieldsTransform):
                transforms[idx] = replace(transform, mapping=feature_mapping)
            elif isinstance(transform, BPDeltaActionTransformFn):
                transforms[idx] = replace(transform, mask=get_mask_mapping(robot_type, current_ds.meta.features))
            elif isinstance(transform, (BPRemapImageKeyTransformFn, BPVAv2RemapImageKeyTransformFn)):
                transforms[idx] = replace(transform, mapping=image_mapping)

        return cls(
            current_ds=current_ds,
            prompt_ds=prompt_ds,
            prompt_cfg=prompt_cfg,
            transform=compose(transforms),
        )

    @property
    def repo_id(self) -> str:
        """Expose the underlying repo id for MultiLeRobotDataset logging."""
        return self.current_ds.repo_id

    @property
    def meta(self) -> Any:
        """Expose current dataset metadata for wrappers that expect LeRobot-like datasets."""
        return self.current_ds.meta

    @property
    def num_frames(self) -> int:
        """Return the number of frames from the current dataset."""
        return self.current_ds.num_frames

    @property
    def num_episodes(self) -> int:
        """Return the number of episodes from the current dataset."""
        return self.current_ds.num_episodes

    def __len__(self) -> int:
        """Return the number of current samples."""
        return len(self.current_ds)

    @staticmethod
    def _to_int(value: Any) -> int:
        """Convert tensor/numpy/singleton-container scalar metadata to int."""
        while isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"Expected a scalar or singleton sequence, got {value!r}")
            value = value[0]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"Expected a scalar tensor, got shape {tuple(value.shape)}")
            return int(value.item())
        item = getattr(value, "item", None)
        if callable(item):
            try:
                value = item()
            except ValueError:
                pass
        return int(value)

    @staticmethod
    def _row_get(row: Any, key: str, default: Any = None) -> Any:
        if isinstance(row, dict):
            return row.get(key, default)
        get = getattr(row, "get", None)
        if callable(get):
            return get(key, default)
        try:
            return row[key]
        except (KeyError, TypeError, IndexError):
            return default

    def _episode_metadata_rows(self) -> dict[int, Any]:
        """Normalize list/DataFrame/HF episode metadata into an episode lookup."""
        episodes_meta = self.prompt_ds.meta.episodes
        if episodes_meta is None:
            raise ValueError("prompt_ds.meta.episodes is required to build episode ranges")

        if hasattr(episodes_meta, "to_dict") and hasattr(episodes_meta, "iloc"):
            rows = (episodes_meta.iloc[pos] for pos in range(len(episodes_meta)))
        elif isinstance(episodes_meta, dict):
            lengths = [len(value) for value in episodes_meta.values() if hasattr(value, "__len__")]
            if not lengths or len(set(lengths)) != 1:
                raise ValueError("Column-oriented episode metadata has inconsistent lengths")
            rows = ({key: value[pos] for key, value in episodes_meta.items()} for pos in range(lengths[0]))
        else:
            rows = (episodes_meta[pos] for pos in range(len(episodes_meta)))

        result: dict[int, Any] = {}
        for position, row in enumerate(rows):
            raw_episode_idx = self._row_get(row, "episode_index", position)
            episode_idx = self._to_int(raw_episode_idx)
            if episode_idx in result:
                raise ValueError(f"Duplicate episode_index={episode_idx} in prompt metadata")
            result[episode_idx] = row
        return result

    def _build_episode_ranges(self) -> dict[int, tuple[int, int, int]]:
        """Build episode -> (local start, local end, absolute start) in O(episodes)."""
        metadata_rows = self._episode_metadata_rows()
        selected = (
            set(metadata_rows)
            if self.prompt_ds.episodes is None
            else {self._to_int(episode_idx) for episode_idx in self.prompt_ds.episodes}
        )
        missing = selected.difference(metadata_rows)
        if missing:
            raise ValueError(f"Selected prompt episodes missing from metadata: {sorted(missing)}")

        absolute_ranges: list[tuple[int, int, int]] = []
        dataset_total_frames = self._to_int(self.prompt_ds.meta.total_frames)
        for episode_idx in selected:
            row = metadata_rows[episode_idx]
            start = self._to_int(self._row_get(row, "dataset_from_index"))
            end = self._to_int(self._row_get(row, "dataset_to_index"))
            if start < 0 or end <= start:
                raise ValueError(f"Invalid range [{start}, {end}) for episode={episode_idx}")
            if end > dataset_total_frames:
                raise ValueError(
                    f"Episode={episode_idx} range end {end} exceeds metadata total_frames={dataset_total_frames}"
                )
            absolute_ranges.append((start, end, episode_idx))

        absolute_ranges.sort()
        previous_end = -1
        local_start = 0
        ranges: dict[int, tuple[int, int, int]] = {}
        for absolute_start, absolute_end, episode_idx in absolute_ranges:
            if absolute_start < previous_end:
                raise ValueError(
                    f"Overlapping prompt episode ranges near episode={episode_idx}: "
                    f"start={absolute_start}, previous_end={previous_end}"
                )
            length = absolute_end - absolute_start
            ranges[episode_idx] = (local_start, local_start + length, absolute_start)
            local_start += length
            previous_end = absolute_end

        if local_start != len(self.prompt_ds.hf_dataset):
            raise ValueError(
                "Prompt episode metadata covers "
                f"{local_start} rows, but loaded hf_dataset has {len(self.prompt_ds.hf_dataset)} rows"
            )
        return ranges

    @classmethod
    def _build_task_index_to_name(cls, dataset: LeRobotDataset) -> dict[int, str]:
        """Build stable task identities independent of per-dataset integer numbering."""
        task_table = getattr(dataset.meta, "tasks", None)
        if task_table is None or not hasattr(task_table, "iterrows"):
            raise ValueError(f"Dataset {getattr(dataset, 'repo_id', '<unknown>')!r} requires tabular task metadata")
        result: dict[int, str] = {}
        for raw_name, row in task_table.iterrows():
            task_idx = cls._to_int(cls._row_get(row, "task_index"))
            task_name = str(raw_name).strip()
            if not task_name:
                raise ValueError(f"Empty task name for task_index={task_idx}")
            if task_idx in result and result[task_idx] != task_name:
                raise ValueError(f"Duplicate task_index={task_idx} has inconsistent names")
            result[task_idx] = task_name
        if not result:
            raise ValueError("Task metadata must not be empty")
        return result

    def _current_task_identity(self, task_idx: int) -> str:
        try:
            return self._current_task_names[task_idx]
        except KeyError as exc:
            raise KeyError(f"Current task_index={task_idx} is absent from current task metadata") from exc

    def _task_indices_from_metadata(self, row: Any) -> set[int]:
        """Read task IDs from common episode metadata schemas."""
        for key in ("task_index", "task_indices"):
            value = self._row_get(row, key)
            if value is not None:
                values = value if isinstance(value, (list, tuple, set)) else [value]
                return {self._to_int(item) for item in values}

        tasks = self._row_get(row, "tasks")
        if tasks is None:
            return set()
        tasks = tasks if isinstance(tasks, (list, tuple, set)) else [tasks]
        task_table = getattr(self.prompt_ds.meta, "tasks", None)
        result: set[int] = set()
        for task in tasks:
            try:
                result.add(self._to_int(task))
                continue
            except (TypeError, ValueError):
                pass
            if task_table is not None:
                try:
                    task_row = task_table.loc[task]
                    result.add(self._to_int(self._row_get(task_row, "task_index")))
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue
        return result

    def _task_indices_at_episode_starts(self, episode_ids: list[int]) -> dict[int, int]:
        """Fallback task lookup using only one local HF row per episode."""
        starts = [self._episode_ranges[episode_idx][0] for episode_idx in episode_ids]
        try:
            values = self.prompt_ds.hf_dataset["task_index"][starts]
            return {episode_idx: self._to_int(value) for episode_idx, value in zip(episode_ids, values, strict=True)}
        except (KeyError, TypeError, IndexError, ValueError):
            return {
                episode_idx: self._to_int(self.prompt_ds.hf_dataset[local_start]["task_index"])
                for episode_idx, local_start in zip(episode_ids, starts, strict=True)
            }

    def _build_task_to_episodes(self) -> dict[str, list[int]]:
        """Build canonical task-name -> prompt episodes despite task index renumbering."""
        metadata_rows = self._episode_metadata_rows()
        task_to_episodes: dict[str, set[int]] = defaultdict(set)
        missing_task_episodes: list[int] = []
        for episode_idx in self._episode_ranges:
            task_indices = self._task_indices_from_metadata(metadata_rows[episode_idx])
            if not task_indices:
                missing_task_episodes.append(episode_idx)
                continue
            for task_idx in task_indices:
                try:
                    task_name = self._prompt_task_names[task_idx]
                except KeyError as exc:
                    raise KeyError(f"Prompt task_index={task_idx} is absent from prompt task metadata") from exc
                task_to_episodes[task_name].add(episode_idx)

        if missing_task_episodes:
            for episode_idx, task_idx in self._task_indices_at_episode_starts(missing_task_episodes).items():
                try:
                    task_name = self._prompt_task_names[task_idx]
                except KeyError as exc:
                    raise KeyError(f"Prompt task_index={task_idx} is absent from prompt task metadata") from exc
                task_to_episodes[task_name].add(episode_idx)
        return {task_name: sorted(episodes) for task_name, episodes in task_to_episodes.items()}

    def _bp_sampling_mode(self) -> str:
        prompt_cfg = getattr(self, "prompt_cfg", None)
        mode = "random" if prompt_cfg is None else str(getattr(prompt_cfg, "bp_sampling_mode", "random"))
        if mode not in {"random", "origin", "avoid", "random_in_task", "random_in_repo", "avoid_in_task", "avoid_in_repo"}:
            raise ValueError(
                "Invalid bp_sampling_mode; expected origin, random[_in_task|_in_repo], or avoid[_in_task|_in_repo], "
                f"got {mode!r}"
            )
        return mode

    def _sample_prompt_episode(self, current_episode_idx: int, current_task_identity: Any = None) -> int | None:
        """Task modes match full task names within this loaded repo; None means masked BP."""
        mode = self._bp_sampling_mode()
        if mode == "origin":
            return current_episode_idx
        candidates = (list(self._task_to_episodes.get(current_task_identity, []))
                      if mode.endswith("_in_task") else list(self._episode_ranges))
        if mode in {"avoid", "avoid_in_repo", "avoid_in_task"}:
            candidates = [episode for episode in candidates if episode != current_episode_idx]
        if not candidates:
            if mode == "avoid_in_task":
                return None
            raise RuntimeError(f"bp_sampling_mode={mode!r} has no eligible BP episode")
        return self.rng.choice(candidates)

    def _resolve_num_chunks(self, trajectory_len: int) -> int:
        """在视频读取前确定候选数，最多读取模型所需的 K 个 chunk。"""
        if self.prompt_cfg.prompt_action_chunk_size <= 0:
            raise ValueError("prompt_action_chunk_size must be positive")
        trajectory_chunks = max(1, math.ceil(trajectory_len / self.prompt_cfg.prompt_action_chunk_size))
        max_chunks = self.prompt_cfg.num_chunks
        if self.prompt_cfg.max_prompt_chunks is not None:
            max_chunks = min(max_chunks, int(self.prompt_cfg.max_prompt_chunks))
        if self.prompt_cfg.fixed_num_chunks:
            return max_chunks
        return min(trajectory_chunks, max_chunks)

    @staticmethod
    def _linspace_offsets(trajectory_len: int, num: int) -> list[int]:
        """Uniformly sample offsets while preserving torch.linspace().round() semantics."""
        if num <= 0:
            return []
        if trajectory_len <= 0:
            raise ValueError(f"trajectory_len must be positive, got {trajectory_len}")
        if trajectory_len == 1:
            return [0] * num
        return torch.linspace(0, trajectory_len - 1, steps=num).round().to(torch.long).tolist()

    def _missing_state_action_placeholder_dim(self, robot_type: str) -> int:
        injected_dim = int(getattr(self.current_ds, "_missing_state_action_placeholder_dim", 0) or 0)
        if injected_dim > 0:
            return injected_dim
        mask_dim = len(get_mask_mapping(robot_type, self.prompt_ds.meta.features))
        if mask_dim > 0:
            return mask_dim
        return max(1, min(self.prompt_cfg.max_state_dim, self.prompt_cfg.max_action_dim))

    def _compose_prompt_field(
        self,
        sample: DataDict,
        canonical_key: str,
        mapping: dict[str, list[str]],
        robot_type: str,
    ) -> torch.Tensor:
        expected_keys = list(mapping.get(canonical_key, [canonical_key]))
        missing_keys = [key for key in expected_keys if key not in sample]
        if missing_keys:
            repo_id = getattr(self.prompt_ds, "repo_id", "<unknown>")
            raise KeyError(
                "Behavior prompt schema is missing required fields: "
                f"repo_id={repo_id!r}, robot_type={robot_type!r}, "
                f"available_keys={sorted(sample.keys())}, expected_keys={expected_keys}, "
                f"missing_keys={missing_keys}"
            )
        values = [torch.as_tensor(sample[key]) for key in expected_keys]
        try:
            if len(values) == 1:
                return values[0]
            max_ndim = max(1, max(value.ndim for value in values))
            aligned_values = []
            for value in values:
                while value.ndim < max_ndim:
                    value = value.unsqueeze(-1)
                aligned_values.append(value)
            return torch.cat(aligned_values, dim=-1)
        except RuntimeError as exc:
            shapes = {key: tuple(value.shape) for key, value in zip(expected_keys, values, strict=True)}
            raise ValueError(
                f"Cannot compose {canonical_key!r} for repo_id={getattr(self.prompt_ds, 'repo_id', '<unknown>')!r}, "
                f"robot_type={robot_type!r}; expected_keys={expected_keys}, shapes={shapes}"
            ) from exc

    def _current_frame_offset(self, current_sample: DataDict, episode_idx: int) -> int:
        """当前样本在 episode 内的帧号，和 linspace 锚点用同一套下标。"""
        if "frame_index" in current_sample:
            return self._to_int(current_sample["frame_index"])
        local_start, local_end, absolute_start = self._episode_ranges[episode_idx]
        if "index" in current_sample:
            offset = self._to_int(current_sample["index"]) - int(absolute_start)
            trajectory_len = local_end - local_start
            if 0 <= offset < trajectory_len:
                return offset
        raise KeyError("future_prompt requires frame_index or an in-episode index")

    def _build_prompt(self, current_sample: DataDict, episode_idx: int | None = None) -> dict[str, Any]:
        """Build the raw behavior_prompt dict before BP transforms.

        每个关键帧只访问一次 prompt_ds，同时取得三路当前图像、state 和未来 action window。
        episode_idx 给定时不再采样，future_prompt 用它固定到当前 episode。
        """
        current_episode_idx = self._to_int(current_sample["episode_index"])
        current_task_idx = self._to_int(current_sample.get("task_index", 0))
        mask_missing_bp = False
        if episode_idx is None:
            identity = (self._current_task_identity(current_task_idx)
                        if self._bp_sampling_mode().endswith("_in_task") else current_task_idx)
            prompt_episode_idx = self._sample_prompt_episode(current_episode_idx, identity)
            mask_missing_bp = prompt_episode_idx is None
            if mask_missing_bp:
                # Use the already decoded current sample only as a shape-compatible placeholder.
                prompt_episode_idx = current_episode_idx
        else:
            prompt_episode_idx = int(episode_idx)
            if prompt_episode_idx not in self._episode_ranges:
                raise KeyError(f"Prompt episode {prompt_episode_idx} is not in this dataset")
        local_start, local_end, absolute_start = self._episode_ranges[prompt_episode_idx]
        trajectory_len = local_end - local_start
        prompt_num_chunks = 1 if mask_missing_bp else self._resolve_num_chunks(trajectory_len)

        prompt_frame_offsets = self._linspace_offsets(trajectory_len, prompt_num_chunks)
        prompt_local_indices = [local_start + offset for offset in prompt_frame_offsets]
        prompt_frame_indices = [absolute_start + offset for offset in prompt_frame_offsets]
        source_time_ratio = torch.tensor(
            [offset / max(1, trajectory_len - 1) for offset in prompt_frame_offsets],
            dtype=torch.float32,
        )

        # LeRobotDataset.__getitem__ indexes the loaded hf_dataset, so subset datasets require local indices.
        if mask_missing_bp:
            placeholder = dict(current_sample)
            state_keys = get_feature_mapping(self.prompt_ds.meta.robot_type, self.prompt_ds.meta.features)[OBS_STATE]
            for key in list(self.prompt_ds.active_visual_keys) + list(state_keys):
                if key in placeholder:
                    value = torch.as_tensor(placeholder[key])
                    expected_ndim = 3 if key in self.prompt_ds.active_visual_keys else 1
                    while value.ndim > expected_ndim:
                        value = value[0]
                    placeholder[key] = value
            prompt_samples = [placeholder]
        else:
            prompt_samples = [self.prompt_ds[idx] for idx in prompt_local_indices]
        image_keys = list(self.prompt_ds.active_visual_keys)
        prompt_images = {
            key: torch.stack([sample[key] for sample in prompt_samples], dim=0)
            for key in image_keys
        }

        robot_type = self.prompt_ds.meta.robot_type
        feature_mapping = get_feature_mapping(robot_type, self.prompt_ds.meta.features)
        if robot_type == "egodex_v":
            placeholder_dim = self._missing_state_action_placeholder_dim(robot_type)
            prompt_state = torch.zeros(prompt_num_chunks, placeholder_dim, dtype=torch.float32)
            prompt_action = torch.zeros(
                prompt_num_chunks,
                self.prompt_cfg.prompt_action_chunk_size,
                placeholder_dim,
                dtype=torch.float32,
            )
            prompt_action_is_pad = torch.ones(
                prompt_num_chunks, self.prompt_cfg.prompt_action_chunk_size, dtype=torch.bool
            )
            prompt_state_is_available = torch.zeros(prompt_num_chunks, dtype=torch.bool)
        else:
            prompt_state = torch.stack(
                [self._compose_prompt_field(sample, OBS_STATE, feature_mapping, robot_type) for sample in prompt_samples],
                dim=0,
            )
            prompt_action = torch.stack(
                [self._compose_prompt_field(sample, ACTION, feature_mapping, robot_type) for sample in prompt_samples],
                dim=0,
            )
            prompt_action_is_pad = torch.stack(
                [
                    sample.get(
                        "action_is_pad",
                        torch.zeros(sample_action.shape[-2], dtype=torch.bool, device=sample_action.device),
                    )
                    for sample, sample_action in zip(prompt_samples, prompt_action, strict=True)
                ],
                dim=0,
            )
            prompt_state_is_available = torch.ones(prompt_num_chunks, dtype=torch.bool)
        expected_action_steps = self.prompt_cfg.prompt_action_chunk_size
        if prompt_action.shape[-2] != expected_action_steps:
            raise ValueError(
                f"prompt_ds action window must contain {expected_action_steps} steps, "
                f"got shape {tuple(prompt_action.shape)}"
            )

        return {
            "images": prompt_images,
            "state": prompt_state,
            "action": prompt_action,
            "action_is_pad": prompt_action_is_pad,
            "state_is_available": prompt_state_is_available,
            "mask": torch.full((prompt_num_chunks,), not mask_missing_bp, dtype=torch.bool),
            "num_chunks": torch.tensor(prompt_num_chunks, dtype=torch.long),
            "chunk_indices": torch.arange(prompt_num_chunks, dtype=torch.long),
            "prompt_action_chunk_size": torch.tensor(self.prompt_cfg.prompt_action_chunk_size, dtype=torch.long),
            "source_episode_index": torch.tensor(prompt_episode_idx, dtype=torch.long),
            "source_episode_length": torch.tensor(trajectory_len, dtype=torch.long),
            "source_indices": torch.tensor(prompt_frame_indices, dtype=torch.long),
            "source_frame_offsets": torch.tensor(prompt_frame_offsets, dtype=torch.long),
            "source_time_ratio": source_time_ratio,
            "task_index": torch.tensor(current_task_idx, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> DataDict:
        """Return one current sample with an attached raw behavior_prompt."""
        current = dict(self.current_ds[idx])
        current[BP_PREFIX] = self._build_prompt(current)
        return self.transform(current)
