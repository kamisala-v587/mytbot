from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import ClassVar

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.TBot_SA1.configuration_tbot_sa1 import TBotSA1Config, TBotSA1DatasetConfig
from lerobot.policies.TBot_SA1.da3_teacher import resolve_da3_backbone_defaults
from lerobot.utils.constants import OBS_IMAGES
from lerobot.transforms.core import (
    ComposeFieldsTransform,
    DeltaActionTransformFn,
    InjectMissingStateActionTransformFn,
    NormalizeTransformFn,
    PadStateAndActionTransformFn,
    RemapImageKeyTransformFn,
    ResizeImagesWithPadFn,
    TransformGroup,
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

BPVA = "bpva"


@DatasetConfig.register_subclass("BP_TBot_v2")
@DatasetConfig.register_subclass("BPVA")
@DatasetConfig.register_subclass("bp_tbot_v2")
@DatasetConfig.register_subclass("tbot_bp")
@DatasetConfig.register_subclass(BPVA)
@dataclass
class BPVADatasetConfig(TBotSA1DatasetConfig):
    """Dataset config for Behavior Prompting for Vision-Action (BPVA).

    The current sample follows the TBot image-only input path, while a nested
    `behavior_prompt` branch is processed by BP-specific transforms.
    """

    _canonical_type: ClassVar[str] = BPVA
    bp_num_chunks: int = 4
    bp_same_episode_policy: str = "avoid"
    bp_seed: int = 0
    bp_camera_keys: list[str] = field(
        default_factory=lambda: [
            f"{OBS_IMAGES}.image0",
            f"{OBS_IMAGES}.image1",
            f"{OBS_IMAGES}.image2",
        ]
    )
    action_mode: str = "delta"

    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                # BP-only branch：只保留 BPObsEncoder 需要的图像、state、action，不再生成 Qwen pixel_values。
                BPRemapImageKeyTransformFn(),
                BPPadOrSampleChunksFn(num_chunks=BPVADatasetConfig.bp_num_chunks),
                BPResizeImagesWithPadFn(height=BPVADatasetConfig.height, width=BPVADatasetConfig.width),
                BPComposeFieldsTransform(),
                BPDeltaActionTransformFn(),
                BPNormalizeTransformFn(),
                BPPadStateAndActionTransformFn(
                    max_state_dim=BPVADatasetConfig.max_state_dim,
                    max_action_dim=BPVADatasetConfig.max_action_dim,
                ),
                # Current branch: mirror TBot transforms, but Qwen input is image-only.
                InjectMissingStateActionTransformFn(),
                ResizeImagesWithPadFn(height=BPVADatasetConfig.height, width=BPVADatasetConfig.width),
                RemapImageKeyTransformFn(),
                NormalizeTransformFn(),
                ComposeFieldsTransform(),
                PadStateAndActionTransformFn(
                    max_state_dim=BPVADatasetConfig.max_state_dim,
                    max_action_dim=BPVADatasetConfig.max_action_dim,
                ),
                ImgOnlyQwen3VLTransformFn(),
                UnifyBPInputsTransformFn(),
            ],
            outputs=[],
        )
    )

    def __post_init__(self):
        """Propagate local BP camera, Qwen processor, and delta-action settings to transforms."""
        original_action_mode = self.action_mode
        if str(original_action_mode).lower() == "obs":
            # Parent configs only validate abs/delta; BPVA treats obs as an input-only path.
            self.action_mode = "abs"
        super().__post_init__()
        self.action_mode = original_action_mode

        inputs = list(self.data_transforms.inputs)
        use_delta = str(self.action_mode).lower() != "obs"
        if use_delta and not any(isinstance(t, DeltaActionTransformFn) for t in inputs):
            current_insert_idx = next((i for i, t in enumerate(inputs) if isinstance(t, ResizeImagesWithPadFn)), len(inputs))
            inputs.insert(current_insert_idx, DeltaActionTransformFn())
        elif not use_delta:
            inputs = [t for t in inputs if not isinstance(t, (BPDeltaActionTransformFn, DeltaActionTransformFn))]

        for idx, transform in enumerate(inputs):
            if isinstance(transform, BPRemapImageKeyTransformFn):
                inputs[idx] = replace(transform, bp_camera_keys=list(self.bp_camera_keys))
            elif isinstance(transform, BPPadOrSampleChunksFn):
                inputs[idx] = replace(transform, num_chunks=self.bp_num_chunks)
            elif isinstance(transform, ImgOnlyQwen3VLTransformFn):
                inputs[idx] = replace(transform, pretrained_model_name_or_path=self.qwen3_vl_processor_path)
        self.data_transforms = replace(self.data_transforms, inputs=inputs)


@PreTrainedConfig.register_subclass("BP_TBot_v2")
@PreTrainedConfig.register_subclass("BPVA")
@PreTrainedConfig.register_subclass("bp_tbot_v2")
@PreTrainedConfig.register_subclass("tbot_bp")
@PreTrainedConfig.register_subclass(BPVA)
@dataclass
class BPVAConfig(TBotSA1Config):
    """Policy config for Behavior Prompting for Vision-Action (BPVA).

    BP-specific fields describe the prefix extension added on top of TBot's
    original middle/suffix and loss computation.
    """

    _canonical_type: ClassVar[str] = BPVA

    # Temporary BPVA bootstrap: reuse the released TBot-SA1 base checkpoint and
    # randomly initialize BP-only prefix modules that do not exist in that checkpoint.
    pretrained_path: str | None = None
    qwen3_vl_variant: str = "qwen3_vl_28l"
    action_expert_variant: str = "qwen3_28l"
    qwen3_vl_pretrained_path: str = "/vla/workspace/models/Qwen3-VL-2B-Instruct"
    cosmos_tokenizer_path_or_name: str = "/vla/workspace/models/Cosmos-Tokenizer-CI8x8"
    enable_3d_queries: bool = True
    num_3d_query_tokens: int = 432
    lambda_3d: float = 0.01
    da3_model_path_or_name: str = "/vla/.models/DA3-LARGE-1.1"
    log_da3_teacher_timing: bool = True

    bp_num_chunks: int = 4
    bp_action_chunk_size: int = 50
    bp_camera_keys: list[str] = field(
        default_factory=lambda: [
            f"{OBS_IMAGES}.image0",
            f"{OBS_IMAGES}.image1",
            f"{OBS_IMAGES}.image2",
        ]
    )
    bp_vision_model_name: str = "vit_base_patch16_clip_224.openai"
    bp_vision_pretrained: bool = True
    bp_vision_checkpoint_path: str | None = None
    bp_token_dim: int = 768
    bp_image_feature_aggregation: str = "cls"
    bp_share_rgb_model: bool = True
    bp_use_vision_norm: bool = True
    bp_freeze_vision_encoder: bool = True
    bp_use_action_step_embedding: bool = True
    bp_use_modality_type_embedding: bool = True
    bp_use_chunk_position_embedding: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.bp_num_chunks <= 0:
            raise ValueError("bp_num_chunks must be positive")
        if self.bp_action_chunk_size <= 0:
            raise ValueError("bp_action_chunk_size must be positive")
        if len(self.bp_camera_keys) != 3:
            raise ValueError("bp_camera_keys must contain exactly 3 camera keys")
        if self.bp_token_dim <= 0:
            raise ValueError("bp_token_dim must be positive")
        if self.bp_vision_checkpoint_path is not None and not isinstance(self.bp_vision_checkpoint_path, str):
            raise TypeError("bp_vision_checkpoint_path must be a string or None")
        if self.bp_image_feature_aggregation not in {"cls", "mean"}:
            raise ValueError("bp_image_feature_aggregation must be one of {'cls', 'mean'}")

    def resolve_da3_backbone_defaults(self) -> None:
        """Keep DA3 default resolution behavior identical to TBotSA1Config."""
        resolve_da3_backbone_defaults(self)
