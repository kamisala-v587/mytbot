"""bpv4cdx Stage A-AE: full demo video (32 slots x 3 cameras, cached ViT features) -> Temporal Q-Former -> Z_demo;
prefix = [BOS] + canonical / generic text (48 tokens) + [demo_start] Z_demo [demo_end] + segment_query (180
positions); the q35 action expert reconstructs the demo's own 50-step action segment. Qwen is frozen; no current
observation, no state (main branch). Design: code/conclusion/bpv4CDX/bpv4cdx 实现细节Stage A-AE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.BPV4BPEn.configuration_bpv4_bpen import BPV4GroupedSchedulerConfig
from lerobot.policies.q35.configuration_q35 import Q35Config

BPV4CDX_AE = "bpv4cdx_ae"
DEMO_INPUTS = ("video", "none")
EXPERT_CONTEXTS = ("segment_query", "full_prefix")


@DatasetConfig.register_subclass(BPV4CDX_AE)
@dataclass
class BPV4CDXAEDatasetConfig(DatasetConfig):
    """One item = one demo trajectory (32-slot cached ViT features) + S target segments of that same trajectory."""

    _canonical_type: ClassVar[str] = BPV4CDX_AE
    action_mode: str = "delta"
    max_state_dim: int = 32
    max_action_dim: int = 32
    # repo_id_file lines may be relative `<task>/<variant>`; they are joined with data_root.
    data_root: str | None = None
    qwen3_5_processor_path: str = ""

    # text, per segment: first dropped with text_drop_prob (tokens kept, attention mask all zero; BOS stays visible),
    # otherwise the generic sentence with generic_text_prob, else the canonical text (from the task name)
    generic_text_prob: float = 0.5
    text_drop_prob: float = 0.0
    language_max_length: int = 48

    # demo slots (must match the cache)
    num_slots: int = 32
    sampler: str = "sa"
    cache_dir: str = ""
    # 5: 2x2 average pooling of the 8x8 ViT grid + global mean; 64: all tokens
    spatial_tokens: int = 5

    # target segments (segment_len == policy.chunk_size)
    segment_len: int = 50
    segments_per_trajectory: int = 4
    # random: grid offset in [0, L), starts offset + k L split into S time strata, one random start per stratum;
    # fixed: starts 0, L, 2L, ... (first S of them; evaluation and G3)
    segment_mode: str = "random"
    # training draws trajectories with probability proportional to their length, task-stratified (dataset sampler)
    length_weighted_sampling: bool = True

    # HOM (held-out monitor): at every checkpoint save, sample actions on a fixed, task-stratified set of held-out
    # episodes (fixed noise, fixed segment starts) and log their MSE to the true actions; 0 = all held-out episodes
    heldout_monitor: bool = True
    heldout_monitor_max_episodes: int = 50
    heldout_monitor_alternate: bool = True
    heldout_monitor_batch: int = 16
    # every N-th save HOM uses all held-out episodes instead of heldout_monitor_max_episodes (0 = never)
    heldout_monitor_full_every: int = 0
    # URDF for the end-effector (link6) position error in cm (RoboTwin aloha-agilex only); "auto" = the copy shipped
    # in the repo (assets/urdf/aloha-agilex), "" or a missing file skips it
    heldout_monitor_urdf: str = "auto"

    # --- AE-3 (multi-embodiment, full data) ---
    # split manifest (tools/bpv4cdx/ae3_manifest.py): repos by machine-independent relative id + held-out episodes;
    # when set it replaces repo_id_file / val_holdout_last. data_roots: local roots the relative ids are resolved under
    split_manifest: str = ""
    data_roots: list[str] = field(default_factory=list)
    manifest_sources: list[str] = field(default_factory=list)  # only these top-level sources (empty = all)
    # per-repo memory-mapped state / action arrays (BPV4CDX.ae3_data prepare-arrays); "" = read the parquet files
    array_store: str = ""
    # cache file layout: task_variant = <cache>/<task>/<variant>/epXXXXX.pt (AE-1 / AE-2, bpv4); rel = <cache>/<relative
    # repo id>/epXXXXX.pt (AE-3, unique across sources)
    cache_layout: str = "task_variant"
    # optional list of available cache files (BPV4CDX.ae3_data cache-index): only these trajectories are used, so
    # training can start on a partially built cache; "" = every trajectory must have its cache file
    cache_index: str = ""
    # threads for loading repo metadata / arrays at start-up
    load_threads: int = 16

    val_holdout_last: dict[str, int] = field(default_factory=lambda: {"clean_50": 5, "randomized_100": 10})
    split: str = "train"
    # small subsets (G3 / smoke): keep only these "relative_repo|episode" keys, and / or the first N trajectories
    episode_whitelist: list[str] = field(default_factory=list)
    max_trajectories: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.split not in {"train", "val", "all"}:
            raise ValueError("split must be train, val or all")
        if self.spatial_tokens not in {5, 64}:
            raise ValueError("spatial_tokens must be 5 or 64")
        if self.segment_mode not in {"random", "fixed"}:
            raise ValueError("segment_mode must be random or fixed")
        if self.cache_layout not in {"task_variant", "rel"}:
            raise ValueError("cache_layout must be task_variant or rel")
        if not 0.0 <= self.generic_text_prob <= 1.0 or not 0.0 <= self.text_drop_prob <= 1.0:
            raise ValueError("generic_text_prob and text_drop_prob must be in [0, 1]")
        if not self.cache_dir or not self.qwen3_5_processor_path:
            raise ValueError("cache_dir and qwen3_5_processor_path must be set explicitly")


@PreTrainedConfig.register_subclass(BPV4CDX_AE)
@dataclass
class BPV4CDXAEConfig(Q35Config):
    _canonical_type: ClassVar[str] = BPV4CDX_AE

    # demo video encoder
    num_slots: int = 32
    num_cameras: int = 3
    visual_dim: int = 1024
    spatial_tokens: int = 5
    num_demo_tokens: int = 128  # time-ordered, num_demo_tokens / num_slots queries per slot
    qformer_dim: int = 512
    qformer_num_layers: int = 2
    qformer_num_heads: int = 8
    qformer_ff_mult: int = 4

    # "none" = text-only control model (no Z_demo tokens in the prefix)
    demo_input: str = "video"
    # learnable BOS vector at prefix position 0; False only for checkpoints trained before it existed
    prefix_bos: bool = True
    # segment_query: action expert attends only to the segment_query position of the prefix (task book 5.6);
    # full_prefix: it attends to the whole prefix (ablation)
    expert_context: str = "segment_query"

    # AE-S0 (off in the main branch): segment-start state through the state token, masked with this probability
    use_segment_start_state: bool = False
    segment_start_state_mask_prob: float = 0.5

    # learning rates (the scheduler scales every group by the same warmup / cosine factor)
    new_modules_lr: float = 1e-4
    act_expert_lr: float = 1e-4

    # Qwen upper-layer LoRA (conditional side branch, off = 0): attention / linear-attention and MLP linears of the
    # last qwen_lora_layers decoder layers; the original Qwen weights stay frozen
    qwen_lora_layers: int = 0
    qwen_lora_rank: int = 16
    qwen_lora_alpha: float = 32.0
    qwen_lora_lr: float = 1e-4

    preflight_memory_check: bool = True
    preflight_memory_limit_frac: float = 0.9

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.demo_input not in DEMO_INPUTS:
            raise ValueError(f"demo_input must be one of {DEMO_INPUTS}")
        if self.expert_context not in EXPERT_CONTEXTS:
            raise ValueError(f"expert_context must be one of {EXPERT_CONTEXTS}")
        if self.num_demo_tokens % self.num_slots:
            raise ValueError("num_demo_tokens must be a multiple of num_slots")
        if self.qformer_dim % self.qformer_num_heads:
            raise ValueError("qformer_dim must be divisible by qformer_num_heads")
        if self.spatial_tokens not in {5, 64}:
            raise ValueError("spatial_tokens must be 5 or 64")
        if self.qwen_lora_layers < 0 or self.qwen_lora_rank <= 0:
            raise ValueError("qwen_lora_layers must be >= 0 and qwen_lora_rank > 0")

    def get_scheduler_preset(self) -> BPV4GroupedSchedulerConfig:
        return BPV4GroupedSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )
