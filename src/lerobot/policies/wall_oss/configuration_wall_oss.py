"""wall-oss (X Square Robot, Qwen3.5 MoT VLA) as a TBot policy: dataset type `wall_oss`, policy type `wall_oss`.

The model code is the vendored `wall_x` package (src/wall_x, see VENDORED.md); this policy only adapts TBot data and
training to it. Inputs follow wall-x's LeRobot pipeline (`wall_x/data/backends/lerobot`): state and action min-max
normalized to [-1, 1], one chat prompt with every camera, the instruction, the state as text and `chunk_size` action
slots, flow time `(1 - Beta(1.5, 1)) * 0.999` with t = 0 the noise.
"""

from __future__ import annotations

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
    RemapQ35ImageKeyTransformFn,
)
from lerobot.policies.wall_oss.transform_wall_oss import (
    UnifyWallInputsTransformFn,
    WallNormalizeTransformFn,
    WallPromptTransformFn,
)
from lerobot.transforms.core import DeltaActionTransformFn, TransformGroup
from lerobot.utils.constants import ACTION, OBS_STATE

WALL_OSS = "wall_oss"


@DatasetConfig.register_subclass(WALL_OSS)
@dataclass
class WallOSSDatasetConfig(DatasetConfig):
    _canonical_type: ClassVar[str] = WALL_OSS
    # Qwen3.5 processor (tokenizer + image processor); the wall checkpoint ships none and uses Qwen3.5-4B's
    processor_path: str = "/vla/workspace/models/q35/base_800m"
    # prompt names of the canonical cameras image0 / image1 / image2 (wall-x RoboDojo naming)
    camera_names: list[str] = field(default_factory=lambda: ["face_view", "left_wrist_view", "right_wrist_view"])
    # longest image side before Qwen's smart resize, and the smart-resize settings (wall-x RoboDojo values)
    image_resolution: int = 256
    image_factor: int = 28
    min_pixels: int = 3136
    max_pixels: int = 1003520
    # every sample is padded to this many tokens; longer prompts are an error. Right padding keeps the real tokens
    # exactly as unpadded (left padding changes the flow loss by about 1%)
    max_length: int = 448
    padding_side: str = "right"
    system_prompt: str = "You are a helpful assistant."
    instruction_label: str = "Instruction"
    state_bins: int = 256
    # action slots in the prompt; must equal policy.chunk_size
    action_horizon: int = 32
    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                Q35DeltaActionTransformFn(),
                RemapQ35ImageKeyTransformFn(),
                WallNormalizeTransformFn(),
                Q35ComposeFieldsTransformFn(),
                WallPromptTransformFn(),
                UnifyWallInputsTransformFn(),
            ],
            outputs=[],
        )
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        inputs = []
        for transform in self.data_transforms.inputs:
            if isinstance(transform, WallPromptTransformFn):
                transform = replace(
                    transform,
                    processor_path=self.processor_path,
                    camera_names=list(self.camera_names),
                    image_resolution=self.image_resolution,
                    image_factor=self.image_factor,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                    max_length=self.max_length,
                    padding_side=self.padding_side,
                    system_prompt=self.system_prompt,
                    instruction_label=self.instruction_label,
                    state_bins=self.state_bins,
                    action_horizon=self.action_horizon,
                )
            inputs.append(transform)
        has_delta = any(isinstance(t, DeltaActionTransformFn) for t in inputs)
        if self.action_mode == "delta" and not has_delta:
            inputs.insert(0, Q35DeltaActionTransformFn())
        elif self.action_mode != "delta" and has_delta:
            inputs = [t for t in inputs if not isinstance(t, DeltaActionTransformFn)]
        self.data_transforms = replace(self.data_transforms, inputs=inputs)


@PreTrainedConfig.register_subclass(WALL_OSS)
@dataclass
class WallOSSConfig(PreTrainedConfig):
    _canonical_type: ClassVar[str] = WALL_OSS

    # wall checkpoint used when the run starts from scratch (pretrained_path unset): config.json + model.safetensors
    wall_pretrained_path: str = "/vla/workspace/models/wall-oss/wall-oss-1.0-lite-robodojo-base"
    processor_path: str = "/vla/workspace/models/q35/base_800m"
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"

    # action / state widths of the data; the wall action input / output layers are rebuilt when they differ from
    # the checkpoint (26-d RoboDojo actions)
    action_dim: int = 14
    state_dim: int = 14
    chunk_size: int = 32
    n_action_steps: int = 32
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999

    # trained parameter groups; everything else is frozen
    train_action_expert: bool = True
    train_vlm: bool = False
    train_vision: bool = False
    gradient_checkpointing: bool = False

    # wall-x RoboDojo fine-tuning used lr 3e-5 -> 3e-6 (cosine), AdamW betas (0.9, 0.999)
    optimizer_lr: float = 3.0e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 3.0e-6

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.optimizer_betas = tuple(self.optimizer_betas)
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be bfloat16 or float32")
        if self.attention_implementation not in {"eager", "sdpa"}:
            raise ValueError("attention_implementation must be eager or sdpa")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot be greater than chunk_size")
        if not (self.train_action_expert or self.train_vlm or self.train_vision):
            raise ValueError("nothing to train")

    def validate_features(self) -> None:
        self.input_features.setdefault(OBS_STATE, PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,)))
        self.output_features.setdefault(ACTION, PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,)))

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
