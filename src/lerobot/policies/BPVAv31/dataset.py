from __future__ import annotations

import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.behavior_prompt_dataset import (
    BP_PREFIX,
    BehaviorPromptConfig,
    BehaviorPromptLeRobotDataset,
)
from lerobot.policies.BPVAv31.configuration_bpvav31 import BPVAv31DatasetConfig
from lerobot.policies.BPVAv31.transform_bpvav31 import (
    BPVAv31ProcessorTransformFn,
    BPVAv31Qwen35ImageTransformFn,
    BPVAv31SampleTransform,
    BPVAv31SelectPromptDimsFn,
    UnifyBPVAv31InputsTransformFn,
)
from lerobot.policies.q35.transform_q35 import (
    Q35ComposeFieldsTransformFn,
    Q35DeltaActionTransformFn,
    RemapQ35ImageKeyTransformFn,
)
from lerobot.transforms.core import (
    DeltaActionTransformFn,
    InjectMissingStateActionTransformFn,
    NormalizeTransformFn,
    PadStateAndActionTransformFn,
    Pi05ImageAugmentFn,
    ResizeImagesWithPadFn,
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
    BPVAv2RemapImageKeyTransformFn,
    BPResizeImagesWithPadFn,
)
from lerobot.transforms.constants import get_feature_mapping, get_image_mapping, get_mask_mapping
from lerobot.utils.constants import ACTION, OBS_STATE


ARM_LEFT, ARM_RIGHT, ARM_BOTH = 0, 1, 2


class BPVAv31BehaviorPromptDataset(BehaviorPromptLeRobotDataset):
    """BPVAv31 采样。可挂 future_prompt（align）与 origin_prompt（检索 / 下一 state）。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._arm_labels = self._compute_arm_labels(float(getattr(self.prompt_cfg, "arm_ratio", 3.0)))

    def _compute_arm_labels(self, ratio: float) -> dict[int, int]:
        """单臂 episode：一只手关节总运动量超过另一只手 ratio 倍。只认 left_* / right_* 各 7 维的 state。"""
        ds = self.prompt_ds
        keys = get_feature_mapping(ds.meta.robot_type, ds.meta.features).get(OBS_STATE, [OBS_STATE])
        names = (ds.meta.features.get(keys[0], {}) or {}).get("names") if len(keys) == 1 else None
        if isinstance(names, dict):
            names = next(iter(names.values()), None)
        labels = {int(ep): ARM_BOTH for ep in self._episode_ranges}
        if not names or len(names) < 14 or not str(names[0]).startswith("left") or not str(names[7]).startswith("right"):
            return labels
        hf = ds.hf_dataset.with_format(None).select_columns(["episode_index", keys[0]])
        episodes = np.asarray(hf["episode_index"])
        states = np.asarray(hf[keys[0]], dtype=np.float32)
        for ep in labels:
            s = states[episodes == ep]
            if len(s) < 2:
                continue
            motion = np.abs(np.diff(s, axis=0)).sum(0)
            left, right = float(motion[:7].sum()), float(motion[7:14].sum())
            labels[ep] = ARM_LEFT if left > ratio * right else ARM_RIGHT if right > ratio * left else ARM_BOTH
        return labels

    def _sample_prompt_episode(self, current_episode_idx: int, current_task_identity=None):
        """arm_match_prob=p：以概率 p 只在同手示教里抽，否则只在异手示教里抽；对应集合为空时退回全部候选。"""
        p = getattr(self.prompt_cfg, "arm_match_prob", None)
        mode = self._bp_sampling_mode()
        if p is None or mode == "origin" or mode.endswith("_in_task"):
            return super()._sample_prompt_episode(current_episode_idx, current_task_identity)
        candidates = list(self._episode_ranges)
        if mode in {"avoid", "avoid_in_repo"}:
            candidates = [episode for episode in candidates if episode != current_episode_idx]
        if not candidates:
            raise RuntimeError(f"bp_sampling_mode={mode!r} has no eligible BP episode")
        arm = self._arm_labels[int(current_episode_idx)]
        same = [e for e in candidates if self._arm_labels[e] == arm]
        cross = [e for e in candidates if self._arm_labels[e] != arm]
        pool = same if self.rng.random() < float(p) else cross
        return self.rng.choice(pool or candidates)

    def __getitem__(self, idx: int):
        current = dict(self.current_ds[idx])
        episode_idx = self._to_int(current["episode_index"])
        frame_offset = self._current_frame_offset(current, episode_idx)
        local_start, local_end, _ = self._episode_ranges[episode_idx]
        episode_len = max(1, int(local_end - local_start))
        current_progress = float(frame_offset) / float(max(1, episode_len - 1))

        current[BP_PREFIX] = self._build_prompt(current)
        bp_episode = self._to_int(current[BP_PREFIX]["source_episode_index"])
        if bool(getattr(self.prompt_cfg, "align_bp_encoder", False)):
            future = self._build_prompt(current, episode_idx=episode_idx)
            future["current_frame_offset"] = torch.tensor(frame_offset, dtype=torch.long)
            current["future_prompt"] = future
        if bool(getattr(self.prompt_cfg, "attach_retrieval", False)):
            if bool(getattr(self.prompt_cfg, "attach_origin", True)):
                # 当前轨迹自己的关键帧：经同一套 BP normalize/pad，供下一关键帧 state 监督。
                current["origin_prompt"] = self._build_prompt(current, episode_idx=episode_idx)
            current["current_progress"] = torch.tensor(current_progress, dtype=torch.float32)
        sample = self.transform(current)
        if bool(getattr(self.prompt_cfg, "attach_task_id", False)):
            sample["bp_task_id"] = torch.tensor(self._bp_task_id(), dtype=torch.long)
        bp_arm = self._arm_labels.get(bp_episode, ARM_BOTH)
        sample["bp_is_self"] = torch.tensor(bp_episode == episode_idx)
        sample["bp_arm_match"] = torch.tensor(bp_arm == self._arm_labels.get(episode_idx, ARM_BOTH))
        sample["bp_arm_id"] = torch.tensor(bp_arm, dtype=torch.long)
        repo_hash = zlib.crc32(str(self.current_ds.repo_id).encode("utf-8")) & 0x7FFFFFFF
        sample["bp_episode_key"] = torch.tensor(repo_hash * 100_000 + bp_episode, dtype=torch.long)
        return sample

    def _bp_task_id(self) -> int:
        # clean / randomized repo 都放在 <task>/<variant> 下，用上级目录名把同任务的不同 repo 归为一类。
        task = Path(str(self.current_ds.repo_id)).parent.name or str(self.current_ds.repo_id)
        return zlib.crc32(task.encode("utf-8")) & 0x7FFFFFFF


def make_bpvav31_behavior_dataset(
    current_ds,
    prompt_ds,
    dataset_cfg: BPVAv31DatasetConfig,
    *,
    bp_action_chunk_size: int,
) -> BehaviorPromptLeRobotDataset:
    """给每个当前样本挂一条同任务 BP 示教轨迹，并用 Qwen3.5 处理器打包。"""

    bp_transforms = [
        BPVAv2RemapImageKeyTransformFn(bp_camera_keys=list(dataset_cfg.bp_camera_keys)),
        BPPadOrSampleChunksFn(num_chunks=dataset_cfg.bp_num_chunks),
        BPResizeImagesWithPadFn(height=dataset_cfg.height, width=dataset_cfg.width),
        BPVAv31Qwen35ImageTransformFn(
            pretrained_model_name_or_path=dataset_cfg.qwen3_5_processor_path,
            revision=dataset_cfg.qwen3_5_processor_revision,
        ),
        BPComposeFieldsTransform(),
        BPDeltaActionTransformFn(),
        BPNormalizeTransformFn(),
        BPPadStateAndActionTransformFn(
            max_state_dim=dataset_cfg.max_state_dim,
            max_action_dim=dataset_cfg.max_action_dim,
        ),
        BPVAv31SelectPromptDimsFn(
            bp_action_use=dataset_cfg.bp_action_use,
            bp_state_use=dataset_cfg.bp_state_use,
        ),
    ]
    current_transforms = [
        InjectMissingStateActionTransformFn(),
        Q35DeltaActionTransformFn(),
        ResizeImagesWithPadFn(height=dataset_cfg.height, width=dataset_cfg.width),
        RemapQ35ImageKeyTransformFn(current_image_index=dataset_cfg.current_image_index),
        NormalizeTransformFn(),
        Q35ComposeFieldsTransformFn(),
        PadStateAndActionTransformFn(
            max_state_dim=dataset_cfg.max_state_dim,
            max_action_dim=dataset_cfg.max_action_dim,
        ),
        BPVAv31ProcessorTransformFn(
            pretrained_model_name_or_path=dataset_cfg.qwen3_5_processor_path,
            revision=dataset_cfg.qwen3_5_processor_revision,
            current_image_index=dataset_cfg.current_image_index,
            use_language=dataset_cfg.use_language,
        ),
        UnifyBPVAv31InputsTransformFn(),
    ]
    if dataset_cfg.action_mode != "delta":
        bp_transforms = [
            item for item in bp_transforms if not isinstance(item, (DeltaActionTransformFn, BPDeltaActionTransformFn))
        ]
        current_transforms = [
            item
            for item in current_transforms
            if not isinstance(item, (DeltaActionTransformFn, BPDeltaActionTransformFn))
        ]
    if (
        dataset_cfg.image_transforms.enable
        and dataset_cfg.image_transforms.preset in {"pi05", "pi0.5", "pi05_style"}
    ):
        insert_at = next(
            (
                index + 1
                for index, item in enumerate(current_transforms)
                if isinstance(item, ResizeImagesWithPadFn)
            ),
            0,
        )
        current_transforms.insert(insert_at, Pi05ImageAugmentFn())

    bp_count = len(bp_transforms)
    transforms = bp_transforms + current_transforms
    transforms = hydrate_inject_missing_state_action_transform(transforms, current_ds)
    transforms = hydrate_normalize_transform(transforms, current_ds)
    transforms = hydrate_compose_field_transform(transforms, current_ds)
    transforms = hydrate_delta_action_transform(transforms, current_ds)
    transforms = hydrate_remap_image_key_transform(transforms, current_ds)

    robot_type = prompt_ds.meta.robot_type
    feature_mapping = get_feature_mapping(robot_type, prompt_ds.meta.features)
    image_mapping = get_image_mapping(robot_type, prompt_ds.meta.features)
    for index, transform in enumerate(transforms):
        if isinstance(transform, BPNormalizeTransformFn):
            transforms[index] = replace(
                transform,
                norm_stats=prompt_ds.meta.stats,
                selected_keys=[OBS_STATE, ACTION],
                mapping=feature_mapping,
            )
        elif isinstance(transform, BPComposeFieldsTransform):
            transforms[index] = replace(transform, mapping=feature_mapping)
        elif isinstance(transform, BPDeltaActionTransformFn):
            transforms[index] = replace(
                transform,
                mask=get_mask_mapping(robot_type, current_ds.meta.features),
            )
        elif isinstance(transform, BPVAv2RemapImageKeyTransformFn):
            transforms[index] = replace(transform, mapping=image_mapping)
    bp_transforms = transforms[:bp_count]
    current_transforms = transforms[bp_count:]

    prompt_cfg = BehaviorPromptConfig(
        prompt_action_chunk_size=bp_action_chunk_size,
        seed=int(dataset_cfg.bp_seed),
        num_chunks=dataset_cfg.bp_num_chunks,
        fixed_num_chunks=bool(dataset_cfg.bp_fixed_num_chunks),
        attach_task_id=bool(dataset_cfg.bp_attach_task_id),
        height=dataset_cfg.height,
        width=dataset_cfg.width,
        max_state_dim=dataset_cfg.max_state_dim,
        max_action_dim=dataset_cfg.max_action_dim,
        qwen3_vl_processor_path=dataset_cfg.qwen3_5_processor_path,
        bp_camera_keys=list(dataset_cfg.bp_camera_keys),
        action_mode=dataset_cfg.action_mode,
        dynamic_bp_cameras=True,
        bp_sampling_mode=dataset_cfg.bp_sampling_mode,
        align_bp_encoder=bool(dataset_cfg.align_bp_encoder),
        future_filter=bool(dataset_cfg.future_filter),
    )
    # BehaviorPromptConfig 未声明的字段用 setattr，避免改公共 dataclass。
    setattr(prompt_cfg, "attach_retrieval", bool(getattr(dataset_cfg, "bp_attach_retrieval", True)))
    setattr(prompt_cfg, "attach_origin", bool(getattr(dataset_cfg, "bp_attach_origin", True)))
    setattr(prompt_cfg, "arm_match_prob", getattr(dataset_cfg, "bp_arm_match_prob", None))
    setattr(prompt_cfg, "arm_ratio", float(getattr(dataset_cfg, "bp_arm_ratio", 3.0)))
    return BPVAv31BehaviorPromptDataset(
        current_ds=current_ds,
        prompt_ds=prompt_ds,
        prompt_cfg=prompt_cfg,
        transform=BPVAv31SampleTransform(
            bp_transforms,
            current_transforms,
            future_filter=bool(dataset_cfg.future_filter),
        ),
    )
