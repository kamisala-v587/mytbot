"""bpv4cdx Stage B configs (`bpv4cdx Stage B 开发文档 v1.md`, sections 2 and 7).

prefix = [demo_start] + Z_demo(128, frozen Stage A-AE encoder, cached) + [demo_end]
         + [obs_start] + 3 x (<vision_start> + 64 current image tokens + <vision_end>) + [obs_end]
All of it goes straight into the frozen Qwen (main branch, no fusion module); the Action Expert is randomly
initialised and sees the whole prefix (expert_visibility="full") or only the current image tokens ("obs_only").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.BPV4BPEn.configuration_bpv4_bpen import BPV4GroupedSchedulerConfig
from lerobot.policies.q35.configuration_q35 import Q35Config

BPV4CDX_B = "bpv4cdx_b"
CAMERAS = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]


@DatasetConfig.register_subclass(BPV4CDX_B)
@dataclass
class BPV4CDXBDatasetConfig(DatasetConfig):
    """One item = one (query episode j, demo episode i) pair x `queries_per_item` time steps of j."""

    _canonical_type: ClassVar[str] = BPV4CDX_B
    action_mode: str = "delta"
    max_state_dim: int = 32
    max_action_dim: int = 32
    qwen3_5_processor_path: str = ""

    # pairing index (BPV4CDXB.pairs) and its data root; Z_demo cache dir of one frozen delivery (BPV4CDXB.zdemo_cache)
    index: str = ""
    data_root: str = ""
    zdemo_dir: str = ""
    camera_keys: list[str] = field(default_factory=lambda: list(CAMERAS))
    image_size: int = 224

    split: str = "train"  # train / val (anchor split of the index) / all
    queries_per_item: int = 4  # time steps per pair, one per time stratum
    # probability that a batch slot is filled with a contrast group (same anchor, every goal) instead of one pair;
    # implemented by the dataset's train sampler, which yields the group members consecutively
    contrast_prob: float = 0.5
    # small subsets (B-1): only these query episodes ("repo|episode"), and / or fixed (query, demo) pairs
    query_whitelist: list[str] = field(default_factory=list)
    fixed_pairs: list[list[str]] = field(default_factory=list)
    # all query time steps 0, 1, ... instead of stratified random ones (single-pair overfitting)
    all_frames: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.split not in {"train", "val", "all"}:
            raise ValueError("split must be train, val or all")
        if not 0.0 <= self.contrast_prob <= 1.0:
            raise ValueError("contrast_prob must be in [0, 1]")
        if not self.index or not self.zdemo_dir or not self.qwen3_5_processor_path:
            raise ValueError("dataset.index, dataset.zdemo_dir and dataset.qwen3_5_processor_path must be set")


@PreTrainedConfig.register_subclass(BPV4CDX_B)
@dataclass
class BPV4CDXBConfig(Q35Config):
    _canonical_type: ClassVar[str] = BPV4CDX_B

    # frozen Stage A-AE delivery (StageA_AE_frozen/<step>); demo_start / demo_end are taken from it
    frozen_encoder: str = ""
    num_demo_tokens: int = 128
    # [obs_start] / [obs_end] around the current images (learnable); False: no observation marker tokens
    obs_marker_tokens: bool = True
    # full: the Action Expert sees the whole prefix; obs_only: only the 3 x 66 current-image tokens
    expert_visibility: str = "full"
    # 0: main branch (direct concatenation). >0: bidirectional cross-attention fusion layer (side branch F-X)
    fusion_layers: int = 0
    # video: demo-conditioned; none: no demo tokens (blank-trained control)
    demo_input: str = "video"
    use_state: bool = True
    state_drop_prob: float = 0.0

    new_modules_lr: float = 1e-4
    act_expert_lr: float = 1e-4

    preflight_memory_check: bool = True
    preflight_memory_limit_frac: float = 0.9

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.expert_visibility not in {"full", "obs_only"}:
            raise ValueError("expert_visibility must be full or obs_only")
        if self.demo_input not in {"video", "none"}:
            raise ValueError("demo_input must be video or none")
        if self.fusion_layers:
            raise NotImplementedError("the fusion layer (side branch F-X) is not implemented yet; use fusion_layers=0")

    def get_scheduler_preset(self) -> BPV4GroupedSchedulerConfig:
        return BPV4GroupedSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )
