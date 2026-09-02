from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import ClassVar

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.TBot_SA1.configuration_tbot_sa1 import TBotSA1Config, TBotSA1DatasetConfig
from lerobot.policies.TBot_SA1.da3_teacher import resolve_da3_backbone_defaults
from lerobot.transforms.core import (
    ComposeFieldsTransform, DeltaActionTransformFn, InjectMissingStateActionTransformFn,
    NormalizeTransformFn, PadStateAndActionTransformFn, RemapImageKeyTransformFn,
    ResizeImagesWithPadFn, TransformGroup,
)
from lerobot.transforms.core_bp import (
    BPComposeFieldsTransform, BPDeltaActionTransformFn, BPNormalizeTransformFn,
    BPPadOrSampleChunksFn, BPPadStateAndActionTransformFn, BPResizeImagesWithPadFn,
    BPVAv2QwenImageTransformFn, BPVAv2RemapImageKeyTransformFn,
    ImgOnlyQwen3VLTransformFn, UnifyBPInputsTransformFn,
)
from lerobot.utils.constants import OBS_IMAGES

BPVAV2 = "bpvav2"


@DatasetConfig.register_subclass(BPVAV2)
@dataclass
class BPVAv2DatasetConfig(TBotSA1DatasetConfig):
    """Dataset configuration for BPVAv2's dynamic-camera behavior prompts."""

    _canonical_type: ClassVar[str] = BPVAV2
    bp_num_chunks: int = 4
    bp_same_episode_policy: str = "avoid"
    bp_seed: int = 0
    batch_prompt_video_decode: bool = False
    bp_camera_keys: list[str] = field(default_factory=lambda: [f"{OBS_IMAGES}.image0"])
    action_mode: str = "delta"

    data_transforms: TransformGroup = field(default_factory=lambda: TransformGroup(inputs=[
        BPVAv2RemapImageKeyTransformFn(),
        BPPadOrSampleChunksFn(num_chunks=BPVAv2DatasetConfig.bp_num_chunks),
        BPResizeImagesWithPadFn(height=BPVAv2DatasetConfig.height, width=BPVAv2DatasetConfig.width),
        BPVAv2QwenImageTransformFn(),
        BPComposeFieldsTransform(), BPDeltaActionTransformFn(), BPNormalizeTransformFn(),
        BPPadStateAndActionTransformFn(max_state_dim=BPVAv2DatasetConfig.max_state_dim,
                                      max_action_dim=BPVAv2DatasetConfig.max_action_dim),
        InjectMissingStateActionTransformFn(),
        ResizeImagesWithPadFn(height=BPVAv2DatasetConfig.height, width=BPVAv2DatasetConfig.width),
        RemapImageKeyTransformFn(), NormalizeTransformFn(), ComposeFieldsTransform(),
        PadStateAndActionTransformFn(max_state_dim=BPVAv2DatasetConfig.max_state_dim,
                                    max_action_dim=BPVAv2DatasetConfig.max_action_dim),
        ImgOnlyQwen3VLTransformFn(), UnifyBPInputsTransformFn(),
    ], outputs=[]))

    def __post_init__(self):
        original_action_mode = self.action_mode
        if str(original_action_mode).lower() == "obs":
            self.action_mode = "abs"
        super().__post_init__()
        self.action_mode = original_action_mode
        self._validate_bp_camera_keys()

        inputs = list(self.data_transforms.inputs)
        use_delta = str(self.action_mode).lower() != "obs"
        if use_delta and not any(isinstance(t, DeltaActionTransformFn) for t in inputs):
            insert_at = next((i for i, t in enumerate(inputs) if isinstance(t, ResizeImagesWithPadFn)), len(inputs))
            inputs.insert(insert_at, DeltaActionTransformFn())
        elif not use_delta:
            inputs = [t for t in inputs if not isinstance(t, (BPDeltaActionTransformFn, DeltaActionTransformFn))]
        for idx, transform in enumerate(inputs):
            if isinstance(transform, BPVAv2RemapImageKeyTransformFn):
                inputs[idx] = replace(transform, bp_camera_keys=list(self.bp_camera_keys))
            elif isinstance(transform, BPPadOrSampleChunksFn):
                inputs[idx] = replace(transform, num_chunks=self.bp_num_chunks)
            elif isinstance(transform, (BPVAv2QwenImageTransformFn, ImgOnlyQwen3VLTransformFn)):
                inputs[idx] = replace(transform, pretrained_model_name_or_path=self.qwen3_vl_processor_path)
        self.data_transforms = replace(self.data_transforms, inputs=inputs)

    def _validate_bp_camera_keys(self) -> None:
        if not 1 <= len(self.bp_camera_keys):
            raise ValueError("bp_camera_keys must contain 1 or more canonical image keys")
        if len(set(self.bp_camera_keys)) != len(self.bp_camera_keys):
            raise ValueError("bp_camera_keys must not contain duplicates")
        invalid = [key for key in self.bp_camera_keys if not key.startswith(f"{OBS_IMAGES}.")]
        if invalid:
            raise ValueError(f"bp_camera_keys must be canonical observation image keys: {invalid}")


@PreTrainedConfig.register_subclass(BPVAV2)
@dataclass
class BPVAv2Config(TBotSA1Config):
    """BPVAv2 policy config; BP images share the Qwen3-VL visual encoder."""

    _canonical_type: ClassVar[str] = BPVAV2
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
    bp_camera_keys: list[str] = field(default_factory=lambda: [f"{OBS_IMAGES}.image0"])
    bp_encoder_version: str = "query_compressor_v1"
    bp_freeze_shared_visual: bool = True
    bp_compressor_dim: int = 512
    bp_state_action_hidden_dim: int = 256
    bp_num_query_tokens: int = 5
    bp_compressor_num_layers: int = 2
    bp_compressor_num_heads: int = 8
    bp_compressor_ff_mult: int = 4
    bp_use_modality_type_embedding: bool = True
    bp_use_camera_embedding: bool = True
    bp_use_chunk_position_embedding: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.bp_num_chunks <= 0:
            raise ValueError("bp_num_chunks must be positive")
        if self.bp_action_chunk_size <= 0:
            raise ValueError("bp_action_chunk_size must be positive")
        if self.bp_encoder_version != "query_compressor_v1":
            raise ValueError("bp_encoder_version must be 'query_compressor_v1'")
        for name in (
            "bp_compressor_dim",
            "bp_state_action_hidden_dim",
            "bp_num_query_tokens",
            "bp_compressor_num_layers",
            "bp_compressor_num_heads",
            "bp_compressor_ff_mult",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.bp_compressor_dim % self.bp_compressor_num_heads:
            raise ValueError("bp_compressor_dim must be divisible by bp_compressor_num_heads")
        if not self.bp_camera_keys:
            raise ValueError("bp_camera_keys must contain 1 or more canonical image keys")
        if len(set(self.bp_camera_keys)) != len(self.bp_camera_keys):
            raise ValueError("bp_camera_keys must not contain duplicates")
        invalid = [key for key in self.bp_camera_keys if not key.startswith(f"{OBS_IMAGES}.")]
        if invalid:
            raise ValueError(f"bp_camera_keys must be canonical observation image keys: {invalid}")

    def resolve_da3_backbone_defaults(self) -> None:
        resolve_da3_backbone_defaults(self)
