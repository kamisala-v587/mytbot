"""BPv5 Stage A (BPV5QFormer): wall 4B VLM + Temporal Q-Former; reconstruct demo action segments.

Prefix (causal, image then language): [bos] + demo_start + Z_demo + demo_end + text(48) + segment_query.
Shared-prefix isolation is off by default (`share_prefix=False`); enable later behind this flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.BPV4BPEn.configuration_bpv4_bpen import BPV4GroupedSchedulerConfig
from lerobot.policies.q35.configuration_q35 import Q35Config

BPV5_QFORMER = "bpv5_qformer"
DEMO_INPUTS = ("video", "none")
EXPERT_CONTEXTS = ("segment_query", "full_prefix")
WALL_VLM = "/vla/workspace/models/wall-oss/wall-oss-1.0-lite-vlm"


@DatasetConfig.register_subclass(BPV5_QFORMER)
@dataclass
class BPV5QFormerDatasetConfig(DatasetConfig):
    """One item = one 80-D trajectory + S target segments; demo ViT features from the wall P5 cache."""

    _canonical_type: ClassVar[str] = BPV5_QFORMER
    # DatasetConfig requires repo_id; BPv5 resolves real repos via sources / repo_ids
    repo_id: str = "bpv5_80d"
    action_mode: str = "delta"
    max_state_dim: int = 80
    max_action_dim: int = 80
    qwen3_5_processor_path: str = WALL_VLM

    generic_text_prob: float = 0.5
    text_drop_prob: float = 0.0
    language_max_length: int = 48

    num_slots: int = 32
    sampler: str = "su80"
    cache_dir: str = "/vla/workspace/cache/bpv5_cleaned_p5"
    spatial_tokens: int = 5

    segment_len: int = 50
    segments_per_trajectory: int = 4
    segment_mode: str = "random"
    length_weighted_sampling: bool = True

    # relative repo ids under the cleaned data root; empty = resolve via data80d.repos()
    repo_ids: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    # Optional overrides (E1/E3 fitting). Empty -> data80d defaults / env.
    data_root: str | None = None
    stats_root: str | None = None
    stats_file: str | None = None
    # action80d | native14_pad32 (E1 / AE-3 style) | native14_scatter80
    action_layout: str = "action80d"
    # bpv5_80d is the production rule from 80D_action_norm_stats_delta_abs.md.
    # The other modes are retained only for historical alignment experiments.
    delta_mode: str = "bpv5_80d"
    # Force these action dims out of ACTION_DIM_MASK (E2: drop gripper slots 16,56).
    drop_action_slots: list[int] = field(default_factory=list)
    # If non-empty, loss mask is the stats mask AND these slots (ablation A/B/C).
    keep_action_slots: list[int] = field(default_factory=list)
    max_trajectories: int = 0
    max_trajectories_seed: int = 42
    split: str = "train"
    load_threads: int = 16
    # last N cached episodes per repo -> val (excluded from train); 0 = no holdout split
    val_holdout_last_per_repo: int = 0

    # training-time held-out generation error at every checkpoint save.
    # Conditions: matched + blank + wrong by default; alt deferred until Stage B demo pairing is settled.
    heldout_monitor: bool = True
    heldout_monitor_max_episodes: int = 24
    heldout_monitor_alternate: bool = False
    heldout_monitor_wrong: bool = True
    heldout_monitor_batch: int = 4
    # also write notebook-friendly JSON under <output_dir>/g9/ (like Stage B G9 dumps)
    heldout_monitor_write_g9_json: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.split not in {"train", "val", "all"}:
            raise ValueError("split must be train, val or all")
        if self.spatial_tokens not in {5, 64}:
            raise ValueError("spatial_tokens must be 5 or 64")
        if self.segment_mode not in {"random", "fixed"}:
            raise ValueError("segment_mode must be random or fixed")
        if not 0.0 <= self.generic_text_prob <= 1.0 or not 0.0 <= self.text_drop_prob <= 1.0:
            raise ValueError("generic_text_prob and text_drop_prob must be in [0, 1]")
        if not self.cache_dir or not self.qwen3_5_processor_path:
            raise ValueError("cache_dir and qwen3_5_processor_path must be set explicitly")
        if self.val_holdout_last_per_repo < 0:
            raise ValueError("val_holdout_last_per_repo must be >= 0")
        if self.action_layout not in {"action80d", "native14_pad32", "native14_scatter80"}:
            raise ValueError("action_layout must be action80d, native14_pad32, or native14_scatter80")
        if self.delta_mode not in {"bpv5_80d", "state_mask", "tbot_aloha"}:
            raise ValueError("delta_mode must be bpv5_80d, state_mask, or tbot_aloha")
        if self.action_layout.startswith("native14_") and not self.stats_file:
            raise ValueError(f"{self.action_layout} requires dataset.stats_file (tbot 14-D stats.json)")
        if self.action_layout == "native14_pad32" and (
            int(self.max_action_dim) != 32 or int(self.max_state_dim) != 32
        ):
            raise ValueError("native14_pad32 requires max_action_dim=max_state_dim=32 (AE-3/tbot width)")
        if self.action_layout in {"action80d", "native14_scatter80"} and (
            int(self.max_action_dim) != 80 or int(self.max_state_dim) != 80
        ):
            raise ValueError(f"{self.action_layout} requires max_action_dim=max_state_dim=80")
        for d in self.drop_action_slots:
            if int(d) < 0 or int(d) >= int(self.max_action_dim):
                raise ValueError(f"drop_action_slots dim {d} out of range for max_action_dim={self.max_action_dim}")
        for d in self.keep_action_slots:
            if int(d) < 0 or int(d) >= int(self.max_action_dim):
                raise ValueError(f"keep_action_slots dim {d} out of range for max_action_dim={self.max_action_dim}")


@PreTrainedConfig.register_subclass(BPV5_QFORMER)
@dataclass
class BPV5QFormerConfig(Q35Config):
    _canonical_type: ClassVar[str] = BPV5_QFORMER

    qwen3_5_pretrained_path: str = WALL_VLM
    qwen3_5_processor_path: str = WALL_VLM
    expert_width_ratio: float = 0.25
    max_state_dim: int = 80
    max_action_dim: int = 80
    causal_vlm: bool = True
    freeze_vision_encoder: bool = True
    gradient_checkpointing: bool = True

    num_slots: int = 32
    num_cameras: int = 3
    visual_dim: int = 2560
    spatial_tokens: int = 5
    num_demo_tokens: int = 128
    qformer_dim: int = 1024
    qformer_num_layers: int = 2
    qformer_num_heads: int = 16
    qformer_ff_mult: int = 4

    demo_input: str = "video"
    prefix_bos: bool = True
    expert_context: str = "segment_query"
    # future: strict shared-prefix isolation (default off; first Stage A ship uses per-segment prefixes)
    share_prefix: bool = False

    use_segment_start_state: bool = False
    segment_start_state_mask_prob: float = 0.5

    # How action.dim_mask enters flow loss (diagnose 80D / norm / mask):
    #   stats  — default: mean over ACTION_DIM_MASK valid dims (per-repo group mask)
    #   all    — ignore dim_mask; mean over all A dims with step mask only (pads dilute like AE-3)
    #   ae3    — AE-3 formula: sum(flow*step) / (n_valid_steps * A); pads in denominator
    action_dim_mask_mode: str = "stats"
    # Write one CSV row per trajectory (repo / robot / episode / loss) on each rank.
    log_sample_loss: bool = False

    new_modules_lr: float = 1e-4
    act_expert_lr: float = 1e-4
    # LoRA on wall VLM: qwen_lora_layer_ids wins when non-empty; else last qwen_lora_layers.
    # Qwen3.5 block = 3 linear_attention + 1 full_attention (stride 4). First 2 blocks → [0..7].
    qwen_lora_layers: int = 0
    qwen_lora_layer_ids: list[int] = field(default_factory=list)
    qwen_lora_rank: int = 16
    qwen_lora_alpha: float = 32.0
    qwen_lora_lr: float = 1e-4

    optimizer_lr: float = 1e-4
    scheduler_warmup_steps: int = 2_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 5e-5

    preflight_memory_check: bool = True
    preflight_memory_limit_frac: float = 0.9

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.demo_input not in DEMO_INPUTS:
            raise ValueError(f"demo_input must be one of {DEMO_INPUTS}")
        if self.expert_context not in EXPERT_CONTEXTS:
            raise ValueError(f"expert_context must be one of {EXPERT_CONTEXTS}")
        if self.share_prefix:
            raise NotImplementedError("share_prefix=True is not implemented yet; use per-segment prefixes")
        if self.num_demo_tokens % self.num_slots:
            raise ValueError("num_demo_tokens must be a multiple of num_slots")
        if self.qformer_dim % self.qformer_num_heads:
            raise ValueError("qformer_dim must be divisible by qformer_num_heads")
        if self.spatial_tokens not in {5, 64}:
            raise ValueError("spatial_tokens must be 5 or 64")
        if self.qwen_lora_layers < 0 or self.qwen_lora_rank <= 0:
            raise ValueError("qwen_lora_layers must be >= 0 and qwen_lora_rank > 0")
        if any(int(i) < 0 for i in self.qwen_lora_layer_ids):
            raise ValueError("qwen_lora_layer_ids must be non-negative")
        if self.action_dim_mask_mode not in {"stats", "all", "ae3"}:
            raise ValueError("action_dim_mask_mode must be one of: stats, all, ae3")

    def get_scheduler_preset(self) -> BPV4GroupedSchedulerConfig:
        return BPV4GroupedSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )
