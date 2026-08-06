from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import ClassVar

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.TBot_SA1.configuration_tbot_sa1 import TBotSA1Config, TBotSA1DatasetConfig
from lerobot.policies.TBot_SA1.da3_teacher import resolve_da3_backbone_defaults
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
    BPImgOnlyQwen3VLTransformFn,
    BPNormalizeTransformFn,
    BPPadOrSampleChunksFn,
    BPPadStateAndActionTransformFn,
    BPRemapImageKeyTransformFn,
    BPResizeImagesWithPadFn,
    ImgOnlyQwen3VLTransformFn,
    UnifyBPInputsTransformFn,
)

BP_TBOT = "BP_TBot"


@DatasetConfig.register_subclass(BP_TBOT)
@DatasetConfig.register_subclass("bp_tbot")
@dataclass
class BPTBotDatasetConfig(TBotSA1DatasetConfig):
    """Dataset config for Behavior-Prompted TBot.

    The current sample follows the TBot image-only input path, while a nested
    `behavior_prompt` branch is processed by BP-specific transforms.
    """

    _canonical_type: ClassVar[str] = BP_TBOT
    bp_num_chunks: int = 4
    bp_same_episode_policy: str = "avoid"
    bp_seed: int = 0
    action_mode: str = ""

    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                # BP-only branch: fixed K, image/state/action processing, Qwen image-only pixels.
                BPPadOrSampleChunksFn(num_chunks=BPTBotDatasetConfig.bp_num_chunks),
                BPResizeImagesWithPadFn(height=BPTBotDatasetConfig.height, width=BPTBotDatasetConfig.width),
                BPRemapImageKeyTransformFn(),
                BPComposeFieldsTransform(),
                BPNormalizeTransformFn(),
                BPPadStateAndActionTransformFn(
                    max_state_dim=BPTBotDatasetConfig.max_state_dim,
                    max_action_dim=BPTBotDatasetConfig.max_action_dim,
                ),
                BPImgOnlyQwen3VLTransformFn(),
                # Current branch: mirror TBot transforms, but Qwen input is image-only.
                InjectMissingStateActionTransformFn(),
                ResizeImagesWithPadFn(height=BPTBotDatasetConfig.height, width=BPTBotDatasetConfig.width),
                RemapImageKeyTransformFn(),
                NormalizeTransformFn(),
                ComposeFieldsTransform(),
                PadStateAndActionTransformFn(
                    max_state_dim=BPTBotDatasetConfig.max_state_dim,
                    max_action_dim=BPTBotDatasetConfig.max_action_dim,
                ),
                ImgOnlyQwen3VLTransformFn(),
                UnifyBPInputsTransformFn(),
            ],
            outputs=[],
        )
    )

    def __post_init__(self):
        """Propagate local Qwen processor path and delta-action setting to transforms."""
        super().__post_init__()
        inputs = list(self.data_transforms.inputs)
        for idx, transform in enumerate(inputs):
            if isinstance(transform, BPPadOrSampleChunksFn):
                inputs[idx] = replace(transform, num_chunks=self.bp_num_chunks)
            elif isinstance(transform, (BPImgOnlyQwen3VLTransformFn, ImgOnlyQwen3VLTransformFn)):
                inputs[idx] = replace(transform, pretrained_model_name_or_path=self.qwen3_vl_processor_path)
        inputs = [t for t in inputs if not isinstance(t, (BPDeltaActionTransformFn, DeltaActionTransformFn))]
        if self.action_mode == "delta":
            bp_insert_idx = next((i for i, t in enumerate(inputs) if isinstance(t, BPNormalizeTransformFn)), 0)
            inputs.insert(bp_insert_idx, BPDeltaActionTransformFn())
            current_insert_idx = next((i for i, t in enumerate(inputs) if isinstance(t, ResizeImagesWithPadFn)), 0)
            inputs.insert(current_insert_idx, DeltaActionTransformFn())
        self.data_transforms = replace(self.data_transforms, inputs=inputs)


@PreTrainedConfig.register_subclass(BP_TBOT)
@PreTrainedConfig.register_subclass("bp_tbot")
@dataclass
class BPTBotConfig(TBotSA1Config):
    """Policy config for Behavior-Prompted TBot.

    BP-specific fields describe the prefix extension added on top of TBot's
    original middle/suffix and loss computation.
    """

    _canonical_type: ClassVar[str] = BP_TBOT

    # Temporary BP_TBot bootstrap: reuse the released TBot-SA1 base checkpoint and
    # randomly initialize BP-only prefix modules that do not exist in that checkpoint.
    pretrained_path: str | None = "/vla/workspace/models/tbot_base"
    qwen3_vl_variant: str = "qwen3_vl_28l"
    action_expert_variant: str = "qwen3_28l"
    qwen3_vl_pretrained_path: str = "/vla/.models/Qwen3-VL-2B-Instruct"
    cosmos_tokenizer_path_or_name: str = "/vla/.models/Cosmos-Tokenizer-CI8x8"
    enable_3d_queries: bool = True
    num_3d_query_tokens: int = 432
    lambda_3d: float = 0.01
    da3_model_path_or_name: str = "/vla/.models/DA3-LARGE-1.1"
    log_da3_teacher_timing: bool = True

    bp_num_chunks: int = 4
    bp_action_chunk_size: int = 50
    bp_use_type_embedding: bool = True
    bp_use_chunk_embedding: bool = True
    bp_use_action_step_embedding: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.bp_num_chunks <= 0:
            raise ValueError("bp_num_chunks must be positive")
        if self.bp_action_chunk_size <= 0:
            raise ValueError("bp_action_chunk_size must be positive")

    def resolve_da3_backbone_defaults(self) -> None:
        """Keep DA3 default resolution behavior identical to TBotSA1Config."""
        resolve_da3_backbone_defaults(self)
