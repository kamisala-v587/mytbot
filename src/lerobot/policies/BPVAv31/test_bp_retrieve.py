"""BPVAv3.1 检索 / 时序损失的轻量单测。"""

from __future__ import annotations

import torch

from lerobot.policies.BPVAv31.bp_encoder import sinusoidal_progress_encoding
from lerobot.policies.BPVAv31.modeling_bpvav31 import (
    bp_retrieve_ce_loss,
    bp_temporal_supcon_loss,
    next_keyframe_index,
)


def test_next_keyframe_index():
    ratios = torch.tensor([[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
    assert int(next_keyframe_index(torch.tensor([0.15]), ratios)) == 1
    assert int(next_keyframe_index(torch.tensor([0.95]), ratios)) == 5
    assert int(next_keyframe_index(torch.tensor([1.0]), ratios)) == 5


def test_retrieve_ce_and_temporal():
    logits = torch.randn(4, 8)
    target = torch.tensor([0, 1, 2, 3])
    mask = torch.ones(4, 8, dtype=torch.bool)
    loss, n = bp_retrieve_ce_loss(logits, target, mask)
    assert loss.ndim == 0 and int(n) == 4

    z = torch.randn(4, 8, 16)
    task = torch.tensor([1, 1, 2, 2])
    progress = torch.linspace(0, 1, 8).view(1, 8).expand(4, 8)
    t_loss, anchors = bp_temporal_supcon_loss(z, task, progress, mask, tau=0.2, progress_tol=0.15)
    assert t_loss.ndim == 0
    assert int(anchors) >= 0


def test_progress_encoding_shape():
    for dim in (64, 256, 1024):
        enc = sinusoidal_progress_encoding(torch.tensor([0.0, 0.5, 1.0]), dim)
        assert enc.shape == (3, dim)
        assert torch.isfinite(enc).all(), f"non-finite progress encoding at dim={dim}"


if __name__ == "__main__":
    test_next_keyframe_index()
    test_retrieve_ce_and_temporal()
    test_progress_encoding_shape()
    print("ok")
