from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import ClassVar

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.q35.transform_q35 import (
    Q35ComposeFieldsTransformFn,
    Q35DeltaActionTransformFn,
    Qwen3_5ProcessorTransformFn,
    RemapQ35ImageKeyTransformFn,
    UnifyQ35InputsTransformFn,
)
from lerobot.transforms.core import (
    ComposeFieldsTransform,
    DeltaActionTransformFn,
    NormalizeTransformFn,
    PadStateAndActionTransformFn,
    Pi05ImageAugmentFn,
    RemapImageKeyTransformFn,
    ResizeImagesWithPadFn,
    TransformGroup,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


Q35 = "q35"


@DatasetConfig.register_subclass(Q35)
@dataclass
class Q35DatasetConfig(DatasetConfig):
    _canonical_type: ClassVar[str] = Q35
    height: int = 224
    width: int = 224
    max_state_dim: int = 32
    max_action_dim: int = 32
    current_image_index: int = 0
    qwen3_5_processor_path: str = "Qwen/Qwen3.5-0.8B-Base"
    qwen3_5_processor_revision: str | None = None
    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                Q35DeltaActionTransformFn(),
                ResizeImagesWithPadFn(height=224, width=224),
                RemapQ35ImageKeyTransformFn(),
                NormalizeTransformFn(),
                Q35ComposeFieldsTransformFn(),
                PadStateAndActionTransformFn(max_state_dim=32, max_action_dim=32),
                Qwen3_5ProcessorTransformFn(),
                UnifyQ35InputsTransformFn(),
            ],
            outputs=[],
        )
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        inputs = []
        uses_pi05_image_aug = (
            self.image_transforms.enable
            and self.image_transforms.preset in {"pi05", "pi0.5", "pi05_style"}
        )
        for transform in self.data_transforms.inputs:
            if isinstance(transform, Pi05ImageAugmentFn):
                continue
            if isinstance(transform, DeltaActionTransformFn):
                inputs.append(
                    Q35DeltaActionTransformFn(
                        mask=transform.mask,
                        mapping=transform.mapping,
                        skip=getattr(transform, "skip", False),
                    )
                )
            elif isinstance(transform, NormalizeTransformFn):
                inputs.append(transform)
            elif isinstance(transform, ComposeFieldsTransform):
                inputs.append(Q35ComposeFieldsTransformFn(mapping=transform.mapping))
            elif isinstance(transform, PadStateAndActionTransformFn):
                inputs.append(
                    PadStateAndActionTransformFn(
                        max_state_dim=transform.max_state_dim,
                        max_action_dim=transform.max_action_dim,
                    )
                )
            elif isinstance(transform, RemapImageKeyTransformFn):
                inputs.append(
                    RemapQ35ImageKeyTransformFn(
                        mapping=transform.mapping,
                        current_image_index=self.current_image_index,
                    )
                )
            elif isinstance(transform, Qwen3_5ProcessorTransformFn):
                inputs.append(
                    replace(
                        transform,
                        pretrained_model_name_or_path=self.qwen3_5_processor_path,
                        revision=self.qwen3_5_processor_revision,
                        current_image_index=self.current_image_index,
                    )
                )
            else:
                inputs.append(transform)
        if uses_pi05_image_aug:
            insert_at = next(
                (index + 1 for index, transform in enumerate(inputs) if isinstance(transform, ResizeImagesWithPadFn)),
                0,
            )
            inputs.insert(insert_at, Pi05ImageAugmentFn())
        has_delta = any(isinstance(transform, DeltaActionTransformFn) for transform in inputs)
        if self.action_mode == "delta" and not has_delta:
            inputs.insert(0, Q35DeltaActionTransformFn())
        elif self.action_mode != "delta" and has_delta:
            inputs = [transform for transform in inputs if not isinstance(transform, DeltaActionTransformFn)]
        self.data_transforms = replace(self.data_transforms, inputs=inputs)


@PreTrainedConfig.register_subclass(Q35)
@dataclass
class Q35Config(PreTrainedConfig):
    _canonical_type: ClassVar[str] = Q35

    # Extension point for additional expert implementations, such as a
    # world-aware ActionExpert. This handoff currently implements only "action".
    expert_type: str = "action"
    qwen3_5_pretrained_path: str = "Qwen/Qwen3.5-0.8B-Base"
    qwen3_5_model_revision: str | None = None
    qwen3_5_processor_path: str | None = None
    qwen3_5_processor_revision: str | None = None
    qwen3_5_fingerprint: str | None = None
    # The source Q35 implementation derives AE geometry from any finite positive
    # ratio. Existing experiments and Flow checkpoints use 0.5; 0.3 was covered
    # only by structure checks. Keep the checkpoint's original ratio when loading.
    expert_width_ratio: float = 0.5
    dtype: str = "bfloat16"

    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    image_resolution: tuple[int, int] = (224, 224)
    image_delta_indices: list[int] = field(default_factory=lambda: [0])
    empty_cameras: int = 0
    num_camera_views: int = 3
    gradient_checkpointing: bool = False
    attention_implementation: str = "sdpa"
    causal_vlm: bool = False
    compile_model: bool = False
    compile_mode: str = "default"

    optimizer_lr: float = 5.0e-5
    # Earlier Q35 experiments assigned separate learning rates to the VLM, AE
    # attention, and AE FFN/I/O. The handoff follows TBot's single optimizer_lr.
    # vlm_lr: float = 1.0e-5
    # world_action_attention_lr: float = 5.0e-5
    # action_ffn_io_lr: float = 5.0e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 1.0e-6

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.image_resolution = tuple(self.image_resolution)
        self.optimizer_betas = tuple(self.optimizer_betas)
        if self.qwen3_5_processor_path is None:
            self.qwen3_5_processor_path = self.qwen3_5_pretrained_path
        if self.expert_type != "action":
            raise ValueError("Q35 currently implements only expert_type='action'")
        if self.qwen3_5_fingerprint not in {None, "0.8B", "2B", "4B"}:
            raise ValueError("qwen3_5_fingerprint must be one of: 0.8B, 2B, 4B")
        if not math.isfinite(self.expert_width_ratio) or self.expert_width_ratio <= 0:
            raise ValueError("expert_width_ratio must be a finite positive number")
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be bfloat16 or float32")
        if self.attention_implementation not in {"eager", "sdpa"}:
            raise ValueError("attention_implementation must be eager or sdpa")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot be greater than chunk_size")
        if self.image_delta_indices != [0]:
            raise ValueError("Q35 action-only policies require image_delta_indices=[0]")
        if self.num_camera_views != 3:
            raise ValueError("Q35 currently requires exactly three camera slots")
        for name in ("chunk_size", "n_action_steps", "max_state_dim", "max_action_dim", "num_inference_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def validate_features(self) -> None:
        for index in range(self.empty_cameras):
            self.input_features[f"{OBS_IMAGES}.empty_camera_{index}"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
        self.input_features.setdefault(
            OBS_STATE,
            PolicyFeature(type=FeatureType.STATE, shape=(self.max_state_dim,)),
        )
        self.output_features.setdefault(
            ACTION,
            PolicyFeature(type=FeatureType.ACTION, shape=(self.max_action_dim,)),
        )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
