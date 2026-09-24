from __future__ import annotations

from dataclasses import replace

import torch

from lerobot.datasets.behavior_prompt_dataset import (
    BP_PREFIX,
    BehaviorPromptConfig,
    BehaviorPromptLeRobotDataset,
)
from lerobot.policies.BPVAv3.configuration_bpvav3 import BPVAv3DatasetConfig
from lerobot.policies.BPVAv3.transform_bpvav3 import (
    BPVAv3ProcessorTransformFn,
    BPVAv3Qwen35ImageTransformFn,
    BPVAv3SampleTransform,
    UnifyBPVAv3InputsTransformFn,
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


class BPVAv3BehaviorPromptDataset(BehaviorPromptLeRobotDataset):
    """BPVAv3 采样。align_bp_encoder 打开时，额外挂当前 episode 的 future_prompt。"""

    def __getitem__(self, idx: int):
        current = dict(self.current_ds[idx])
        current[BP_PREFIX] = self._build_prompt(current)
        if bool(getattr(self.prompt_cfg, "align_bp_encoder", False)):
            episode_idx = self._to_int(current["episode_index"])
            future = self._build_prompt(current, episode_idx=episode_idx)
            future["current_frame_offset"] = torch.tensor(
                self._current_frame_offset(current, episode_idx),
                dtype=torch.long,
            )
            current["future_prompt"] = future
        return self.transform(current)


def make_bpvav3_behavior_dataset(
    current_ds,
    prompt_ds,
    dataset_cfg: BPVAv3DatasetConfig,
    *,
    bp_action_chunk_size: int,
) -> BehaviorPromptLeRobotDataset:
    """给每个当前样本挂一条同任务 BP 示教轨迹，并用 Qwen3.5 处理器打包。"""

    bp_transforms = [
        BPVAv2RemapImageKeyTransformFn(bp_camera_keys=list(dataset_cfg.bp_camera_keys)),
        BPPadOrSampleChunksFn(num_chunks=dataset_cfg.bp_num_chunks),
        BPResizeImagesWithPadFn(height=dataset_cfg.height, width=dataset_cfg.width),
        BPVAv3Qwen35ImageTransformFn(
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
        BPVAv3ProcessorTransformFn(
            pretrained_model_name_or_path=dataset_cfg.qwen3_5_processor_path,
            revision=dataset_cfg.qwen3_5_processor_revision,
            current_image_index=dataset_cfg.current_image_index,
            use_language=dataset_cfg.use_language,
        ),
        UnifyBPVAv3InputsTransformFn(),
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
    return BPVAv3BehaviorPromptDataset(
        current_ds=current_ds,
        prompt_ds=prompt_ds,
        prompt_cfg=prompt_cfg,
        transform=BPVAv3SampleTransform(
            bp_transforms,
            current_transforms,
            future_filter=bool(dataset_cfg.future_filter),
        ),
    )
