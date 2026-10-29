"""B-4: trainable demo encoder (Stage B 开发文档 2.7 / 5.4; StageB实施.md D7).

The Stage A-AE delivery's TemporalVideoQFormer (Q-Former, embeddings, projector) becomes a trainable submodule
`demo_encoder`; Z_demo is computed online from the demo's AE-3 P5 record (dataset `demo_features="p5"`) instead of
the Z_demo cache, so gradients reach the encoder. The ViT stays frozen (P5 features are cached ViT outputs). At the
start of B-4 the online Z_demo equals the cached one (same delivery weights), so the model output is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import torch


def checkpoint_flags(pretrained_dir: str) -> dict[str, bool]:
    """Which optional modules a saved Stage B checkpoint already contains (decides the build order before loading)."""
    from safetensors import safe_open

    with safe_open(str(Path(pretrained_dir) / "model.safetensors"), "pt") as f:
        keys = list(f.keys())
    return {"lora": any("lora_" in k for k in keys), "demo_encoder": any(k.startswith("model.demo_encoder.") for k in keys)}


def make_demo_encoder(frozen_dir: str) -> torch.nn.Module:
    from .frozen import load_frozen

    encoder = load_frozen(frozen_dir).encoder  # checksums verified against the delivery manifest
    encoder.train().requires_grad_(True)
    return encoder
