"""BPv5 Stage B config.

Prefix (causal, no history, no segment_query):
  bos → demo_start → Z_demo → demo text → demo_end
    → label_demo → bridge
    → obs_start → 3×(vision_start + ViT + vision_end) → obs_end
    → current text
Action Expert default is V-obs (current-observation span + current text only).
Set expert_v_obs=false to let AE attend the full prefix (incl. Z_demo) — ablation only.

phase:
  ae  freeze VLM base, LoRA, Q-Former (BP), demo/obs markers; train AE only (random init)
  b1  freeze VLM base, Stage A LoRA, Q-Former, demo boundary tokens; train AE + obs markers
  b2  also train the same LoRA
  b3  also train the Q-Former (BP)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.BPV4BPEn.configuration_bpv4_bpen import BPV4GroupedSchedulerConfig
from lerobot.policies.BPV5QFormer.configuration_bpv5qformer import (
    BPV5QFormerConfig,
    BPV5QFormerDatasetConfig,
)

BPV5_STAGEB = "bpv5_stageb"
PHASES = ("ae", "b1", "b2", "b3")


@DatasetConfig.register_subclass(BPV5_STAGEB)
@dataclass
class BPV5StageBDatasetConfig(BPV5QFormerDatasetConfig):
    """Query episode j, demo episode i≠j in the same leaf. Current frames are decoded online."""

    _canonical_type: ClassVar[str] = BPV5_STAGEB

    language_max_length: int = 96
    # A leaf must have at least this many cached episodes so i≠j is possible (doc: bucket > 3).
    min_repo_episodes: int = 4
    # M1: L0 = same leaf; L1 = same leaf + exact task text. See pairing.SOURCE_TIER.
    # Optional per-source override, e.g. {"droid": "L1", "agibot": "L0"}.
    pairing_tiers: dict[str, str] | None = None
    # Doc rule: bucket size > 3 ⇒ keep queries whose M1 bucket (incl. self) has ≥ this many eps.
    pairing_min_bucket: int = 4
    # Legacy lists by default; shared_bucket stores one bucket plus lightweight query views.
    pairing_storage: str = "list"
    # Opt-in startup component timings only (no per-item diagnostics).
    init_timing: bool = False
    # M4: P(mask) and, given a mask, P(generic sentence) vs pure pad.
    demo_text_mask_prob: float = 0.25
    demo_text_generic_ratio: float = 0.8
    current_text_mask_prob: float = 0.40
    current_text_generic_ratio: float = 0.7
    image_size: int = 224

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.min_repo_episodes < 2:
            raise ValueError("min_repo_episodes must be >= 2 so a demo episode i≠j exists")
        if int(self.pairing_min_bucket) < 2:
            raise ValueError("pairing_min_bucket must be >= 2")
        if self.pairing_storage not in ("list", "shared_bucket"):
            raise ValueError("pairing_storage must be list or shared_bucket")
        if self.pairing_tiers:
            for src, tier in self.pairing_tiers.items():
                if tier not in ("L0", "L1"):
                    raise ValueError(f"pairing_tiers[{src!r}] must be L0 or L1, got {tier!r}")
        for name in (
            "demo_text_mask_prob",
            "demo_text_generic_ratio",
            "current_text_mask_prob",
            "current_text_generic_ratio",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if int(self.image_size) < 32:
            raise ValueError("image_size must be >= 32")


@PreTrainedConfig.register_subclass(BPV5_STAGEB)
@dataclass
class BPV5StageBConfig(BPV5QFormerConfig):
    _canonical_type: ClassVar[str] = BPV5_STAGEB

    expert_width_ratio: float = 0.4
    phase: str = "b1"
    # B-1 only. Directory of a Stage A pretrained_model (model.safetensors). Ignored when empty.
    stage_a_checkpoint: str = ""
    qformer_lr: float = 5e-6
    # Stage B always owns the Stage A adapter, even while B-1 freezes it.
    qwen_lora_layer_ids: list[int] = field(default_factory=lambda: list(range(8)))
    # B-1 only: which installed LoRA layers get gradients. Empty = freeze all LoRA (default B-1).
    # B-2 / B-3 ignore this and train every installed LoRA layer.
    lora_train_layer_ids: list[int] = field(default_factory=list)
    qwen_lora_lr: float = 2e-5
    new_modules_lr: float = 5e-5
    act_expert_lr: float = 5e-5
    use_segment_start_state: bool = True
    segment_start_state_mask_prob: float = 0.0
    # C1 blank-demo margin (HOM-aligned). 0 disables. When >0:
    #   L = L_match + λ * max(0, m - (L_blank - L_match))
    # with shared noise/time; blank zeros Z_demo.
    # Adaptive m (PB-2.1+): m = clamp(m_ratio * L_match.detach(), m_min, m_ratio_max * L_match.detach())
    # Fixed-m mode: set m_ratio=0 so m stays at blank_margin_m (= m_min).
    blank_margin_lambda: float = 0.0
    blank_margin_m: float = 0.05  # absolute floor for m (m_min)
    blank_margin_m_ratio: float = 0.0  # 0 = fixed m; else e.g. 1.0 → target gap ≈ 1× L_match
    blank_margin_m_ratio_max: float = 2.0  # upper clamp as multiple of L_match
    # Apply C1 only every N training forwards (1 = every step). Saves compute when λ>0.
    blank_margin_every_n_steps: int = 1
    # HOM gate: if >0, C1 is armed only when blank_over_match.ratio < this threshold
    # (updated each HOM / save). 0 = always armed when λ>0 (legacy).
    blank_margin_gate_blank_over_match: float = 0.0
    # If >0, after each HOM set fixed
    #   m = max(mse_blank, scale * mse_matched)
    # so the hinge is active at the current ckpt (gap < m). scale=2 → target blank≳2×match.
    blank_margin_m_hom_matched_scale: float = 0.0
    # True (default): AE cannot attend prefix tokens before obs_start (V-obs / M3.6).
    # False: AE attends the full causal prefix including Z_demo (diagnostic ablation).
    expert_v_obs: bool = True

    def __post_init__(self) -> None:
        # Parent rejects any expert_context other than segment_query / full_prefix.
        self.expert_context = "full_prefix"
        super().__post_init__()
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        if abs(float(self.expert_width_ratio) - 0.4) > 1e-6:
            # 2560 * 0.4 = 1024. Other widths are a different experiment.
            pass
        if self.qformer_lr <= 0:
            raise ValueError("qformer_lr must be > 0")
        if float(self.blank_margin_lambda) < 0:
            raise ValueError("blank_margin_lambda must be >= 0")
        if float(self.blank_margin_m) < 0:
            raise ValueError("blank_margin_m must be >= 0")
        if float(self.blank_margin_m_ratio) < 0:
            raise ValueError("blank_margin_m_ratio must be >= 0")
        if float(self.blank_margin_m_ratio_max) < float(self.blank_margin_m_ratio):
            raise ValueError("blank_margin_m_ratio_max must be >= blank_margin_m_ratio")
        if int(self.blank_margin_every_n_steps) < 1:
            raise ValueError("blank_margin_every_n_steps must be >= 1")
        if float(self.blank_margin_gate_blank_over_match) < 0:
            raise ValueError("blank_margin_gate_blank_over_match must be >= 0")
        if float(self.blank_margin_m_hom_matched_scale) < 0:
            raise ValueError("blank_margin_m_hom_matched_scale must be >= 0")
    def get_scheduler_preset(self) -> BPV4GroupedSchedulerConfig:
        return BPV4GroupedSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )


def phase_trainable(phase: str) -> dict[str, bool]:
    """Which Stage A / Stage B modules receive gradients (AE handled separately)."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    return {
        "lora": phase in {"b2", "b3"},
        "qformer": phase == "b3",
        "demo_tokens": False,
        "vlm_base": False,
        # obs_start/obs_end: Stage B markers; frozen in ae-only warmup
        "obs_markers": phase != "ae",
    }


def stage_a_keep_key(key: str) -> bool:
    """Tensors copied from a Stage A checkpoint. The Action Expert and segment_query are not copied."""
    if "segment_query" in key:
        return False
    if ".vlm." in key:
        return True
    tail = key.split("model.", 1)[-1]
    return tail == "bos" or tail in {"demo_start", "demo_end"} or tail.startswith("demo_encoder.")
