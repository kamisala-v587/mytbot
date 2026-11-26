"""Stage B v2 config.

Prefix is the wall-oss chat template. The demonstration span sits after Proprioception and before
<|im_end|>. The action expert is the q35 stack at width ratio 0.4 and has no state token.
VLM weights come from wall-oss. Only the Q-Former is loaded from a Stage A checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.BPV4BPEn.configuration_bpv4_bpen import BPV4GroupedSchedulerConfig
from lerobot.policies.BPV5QFormer.configuration_bpv5qformer import BPV5QFormerConfig
from lerobot.policies.BPV5StageB.configuration_bpv5stageb import BPV5StageBDatasetConfig

BPV5_STAGEB_V2 = "bpv5_stageb_v2"
PHASES = ("short", "full")


@DatasetConfig.register_subclass(BPV5_STAGEB_V2)
@dataclass
class BPV5StageBV2DatasetConfig(BPV5StageBDatasetConfig):
    _canonical_type: ClassVar[str] = BPV5_STAGEB_V2
    image_size: int = 448


@PreTrainedConfig.register_subclass(BPV5_STAGEB_V2)
@dataclass
class BPV5StageBV2Config(BPV5QFormerConfig):
    _canonical_type: ClassVar[str] = BPV5_STAGEB_V2

    expert_width_ratio: float = 0.4
    # short: action expert and the two demonstration boundaries.
    # full: also the VLM text tower and the Q-Former, at lower learning rates.
    phase: str = "short"
    stage_a_checkpoint: str = ""
    vlm_lr: float = 2e-5
    qformer_lr: float = 5e-6
    act_expert_lr: float = 5e-5
    marker_lr: float = 5e-5
    qwen_lora_layers: int = 0
    qwen_lora_layer_ids: list[int] = field(default_factory=list)
    use_segment_start_state: bool = False
    proprio_clip: float = 3.0
    state_bins: int = 255
    # Three 448 views are about 600 vision tokens. 2048 leaves room for text, bins and Z_demo.
    max_length: int = 2048

    def __post_init__(self) -> None:
        self.expert_context = "full_prefix"
        super().__post_init__()
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        if self.proprio_clip <= 0:
            raise ValueError("proprio_clip must be > 0")
        if self.state_bins < 2:
            raise ValueError("state_bins must be >= 2")
        if self.max_length < 1:
            raise ValueError("max_length must be >= 1")

    def get_scheduler_preset(self) -> BPV4GroupedSchedulerConfig:
        return BPV4GroupedSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )
