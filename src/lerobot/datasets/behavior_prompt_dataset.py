from __future__ import annotations

import math
import random
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
    ImgOnlyQwen3VLTransformFn,
    UnifyBPInputsTransformFn,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


BP_PREFIX = "behavior_prompt"




@dataclass
class BehaviorPromptConfig:
    """Configuration for sampling a behavior prompt trajectory.

    The prompt trajectory is sampled into at most K action chunks before video IO.
    A later transform only pads short trajectories to fixed K for batching.
    """

    prompt_action_chunk_size: int = 50
    max_prompt_chunks: int | None = None
    same_episode_policy: str = "avoid"  # avoid / allow / forbid
    seed: int = 0

    # BPVA 默认数据变换配置。
    num_chunks: int = 4
    height: int = 224
    width: int = 224
    max_state_dim: int = 32
    max_action_dim: int = 32
    qwen3_vl_processor_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    bp_camera_keys: list[str] | None = None
    action_mode: str = ""

    def __post_init__(self) -> None:
        if self.bp_camera_keys is None:
            self.bp_camera_keys = [
                f"{OBS_IMAGES}.image0",
                f"{OBS_IMAGES}.image1",
                f"{OBS_IMAGES}.image2",
            ]


def _make_bpva_transform_fns(prompt_cfg: BehaviorPromptConfig) -> list[DataTransformFn]:
    """构造 BPVA 数据变换链，不依赖 DatasetConfig 实例。"""
    transforms: list[DataTransformFn] = [
        # BP 分支：固定 K 个 chunk，并处理图像、状态和动作。
        BPRemapImageKeyTransformFn(bp_camera_keys=list(prompt_cfg.bp_camera_keys or [])),
        BPPadOrSampleChunksFn(num_chunks=prompt_cfg.num_chunks),
        BPResizeImagesWithPadFn(height=prompt_cfg.height, width=prompt_cfg.width),
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
        self._episode_to_indices = self._build_episode_to_indices()
        self._task_to_episodes = self._build_task_to_episodes()


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

        robot_type = current_ds.meta.robot_type
        feature_mapping = get_feature_mapping(robot_type, current_ds.meta.features)
        image_mapping = get_image_mapping(robot_type, current_ds.meta.features)
        for idx, transform in enumerate(transforms):
            if isinstance(transform, BPNormalizeTransformFn):
                transforms[idx] = replace(
                    transform,
                    norm_stats=current_ds.meta.stats,
                    selected_keys=feature_mapping[OBS_STATE] + feature_mapping[ACTION],
                )
            elif isinstance(transform, BPComposeFieldsTransform):
                transforms[idx] = replace(transform, mapping=feature_mapping)
            elif isinstance(transform, BPDeltaActionTransformFn):
                transforms[idx] = replace(transform, mask=get_mask_mapping(robot_type, current_ds.meta.features))
            elif isinstance(transform, BPRemapImageKeyTransformFn):
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
        if isinstance(value, torch.Tensor):
            return int(value.item())
        return int(value)

    def _build_episode_to_indices(self) -> dict[int, list[int]]:
        """Build episode -> absolute frame index list without decoding images."""
        episode_to_indices: dict[int, list[int]] = defaultdict(list)
        for row_idx in range(len(self.prompt_ds)):
            row = self.prompt_ds.hf_dataset[row_idx]
            episode_idx = self._to_int(row["episode_index"])
            absolute_idx = self._to_int(row["index"])
            episode_to_indices[episode_idx].append(absolute_idx)
        return dict(episode_to_indices)

    def _build_task_to_episodes(self) -> dict[int, list[int]]:
        """Build task -> episodes lookup so prompts prefer the same task."""
        task_to_episodes: dict[int, set[int]] = defaultdict(set)
        for episode_idx, indices in self._episode_to_indices.items():
            first = self.prompt_ds.hf_dataset[indices[0]]
            task_idx = self._to_int(first.get("task_index", 0))
            task_to_episodes[task_idx].add(episode_idx)
        return {task_idx: sorted(episodes) for task_idx, episodes in task_to_episodes.items()}

    def _sample_prompt_episode(self, current_episode_idx: int, current_task_idx: int) -> int:
        """Sample a prompt episode, preferring the same task and obeying same-episode policy."""
        candidates = list(self._task_to_episodes.get(current_task_idx, []))
        if not candidates:
            candidates = sorted(self._episode_to_indices.keys())

        if self.prompt_cfg.same_episode_policy in {"avoid", "forbid"}:
            different_episode_candidates = [ep for ep in candidates if ep != current_episode_idx]
            if different_episode_candidates:
                candidates = different_episode_candidates
            elif self.prompt_cfg.same_episode_policy == "forbid":
                raise RuntimeError(
                    f"No different prompt episode for episode={current_episode_idx}, task={current_task_idx}"
                )

        return self.rng.choice(candidates)

    def _resolve_num_chunks(self, trajectory_len: int) -> int:
        """在视频读取前确定候选数，最多读取模型所需的 K 个 chunk。"""
        if self.prompt_cfg.prompt_action_chunk_size <= 0:
            raise ValueError("prompt_action_chunk_size must be positive")
        trajectory_chunks = max(1, math.ceil(trajectory_len / self.prompt_cfg.prompt_action_chunk_size))
        max_chunks = self.prompt_cfg.num_chunks
        if self.prompt_cfg.max_prompt_chunks is not None:
            max_chunks = min(max_chunks, int(self.prompt_cfg.max_prompt_chunks))
        return min(trajectory_chunks, max_chunks)

    @staticmethod
    def _linspace_indices(indices: list[int], num: int) -> list[int]:
        """Uniformly sample `num` absolute indices from one episode."""
        if num <= 0:
            return []
        if len(indices) == 1:
            return [indices[0]] * num
        positions = torch.linspace(0, len(indices) - 1, steps=num).round().to(torch.long).tolist()
        return [indices[pos] for pos in positions]

    def _build_prompt(self, current_sample: DataDict) -> dict[str, Any]:
        """Build the raw behavior_prompt dict before BP transforms.

        每个关键帧只访问一次 prompt_ds，同时取得三路当前图像、state 和未来 action window。
        """
        current_episode_idx = self._to_int(current_sample["episode_index"])
        current_task_idx = self._to_int(current_sample.get("task_index", 0))
        prompt_episode_idx = self._sample_prompt_episode(current_episode_idx, current_task_idx)
        prompt_episode_indices = self._episode_to_indices[prompt_episode_idx]
        prompt_num_chunks = self._resolve_num_chunks(len(prompt_episode_indices))

        prompt_frame_indices = self._linspace_indices(prompt_episode_indices, prompt_num_chunks)
        index_to_offset = {absolute_idx: offset for offset, absolute_idx in enumerate(prompt_episode_indices)}
        prompt_frame_offsets = [index_to_offset[idx] for idx in prompt_frame_indices]
        source_time_ratio = torch.tensor(
            [offset / max(1, len(prompt_episode_indices) - 1) for offset in prompt_frame_offsets],
            dtype=torch.float32,
        )

        prompt_samples = [self.prompt_ds[idx] for idx in prompt_frame_indices]
        image_keys = list(self.prompt_ds.meta.camera_keys)
        prompt_images = {
            key: torch.stack([sample[key] for sample in prompt_samples], dim=0)
            for key in image_keys
        }
        prompt_state = torch.stack([sample[OBS_STATE] for sample in prompt_samples], dim=0)
        prompt_action = torch.stack([sample[ACTION] for sample in prompt_samples], dim=0)
        prompt_action_is_pad = torch.stack(
            [
                sample.get(
                    "action_is_pad",
                    torch.zeros(sample[ACTION].shape[0], dtype=torch.bool),
                )
                for sample in prompt_samples
            ],
            dim=0,
        )
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
            "mask": torch.ones(prompt_num_chunks, dtype=torch.bool),
            "num_chunks": torch.tensor(prompt_num_chunks, dtype=torch.long),
            "chunk_indices": torch.arange(prompt_num_chunks, dtype=torch.long),
            "prompt_action_chunk_size": torch.tensor(self.prompt_cfg.prompt_action_chunk_size, dtype=torch.long),
            "source_episode_index": torch.tensor(prompt_episode_idx, dtype=torch.long),
            "source_episode_length": torch.tensor(len(prompt_episode_indices), dtype=torch.long),
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
