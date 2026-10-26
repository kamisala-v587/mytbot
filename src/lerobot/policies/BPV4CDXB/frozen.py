"""Load the Stage A-AE frozen delivery (`StageA_AE_frozen.ckpt`, BPV4CDX/freeze_ae.py) for Stage B.

The encoder checksums are recomputed from the delivered tensors and must match the manifest; Stage B refuses to run
with a modified or mismatching encoder (design doc 6.0, development doc 2.2).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from lerobot.policies.BPV4CDX.model_keys import ENCODER_PREFIXES


@dataclass
class FrozenDemoEncoder:
    encoder: torch.nn.Module  # TemporalVideoQFormer, eval mode, requires_grad False
    demo_start: torch.Tensor  # [hidden]
    demo_end: torch.Tensor  # [hidden]
    manifest: dict
    ckpt_sha256: str
    config: object  # BPV4CDXAEConfig of the delivery

    @property
    def tag(self) -> str:
        """Short id used in Z_demo cache paths."""
        return self.ckpt_sha256[:12]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_checksums(tensors: dict[str, torch.Tensor]) -> dict[str, dict]:
    """Same hashing as freeze_ae.encoder_checksums (sorted keys, key bytes + float32 bytes)."""
    out = {}
    for prefix in ENCODER_PREFIXES:
        h = hashlib.sha256()
        keys = sorted(k for k in tensors if k.startswith(prefix))
        for k in keys:
            h.update(k.encode())
            h.update(tensors[k].float().cpu().numpy().tobytes())
        out[prefix] = {"tensors": len(keys), "sha256": h.hexdigest()}
    return out


def load_frozen(frozen_dir: str | Path, device: str = "cpu") -> FrozenDemoEncoder:
    import lerobot.policies.BPV4CDX.configuration_bpv4cdx  # noqa: F401  registers the bpv4cdx_ae config
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.BPV4CDX.modeling_bpv4cdx import TemporalVideoQFormer

    frozen_dir = Path(frozen_dir)
    ckpt = frozen_dir / "StageA_AE_frozen.ckpt"
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    tensors, manifest = data["state_dict"], data["manifest"]
    if manifest.get("provisional"):
        raise RuntimeError(f"{ckpt} is a provisional delivery (missing gates {manifest.get('missing_gates')})")
    if tensor_checksums(tensors) != manifest["encoder_checksums"]:
        raise RuntimeError(f"encoder checksums of {ckpt} do not match its manifest")
    cfg = PreTrainedConfig.from_pretrained(str(frozen_dir / "pretrained_model"))
    hidden = int(tensors["model.demo_start"].shape[-1])
    encoder = TemporalVideoQFormer(cfg, hidden)
    prefix = "model.demo_encoder."
    missing, unexpected = encoder.load_state_dict({k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)},
                                                  strict=True), None
    del missing, unexpected
    encoder.eval().requires_grad_(False).to(device)
    return FrozenDemoEncoder(encoder, tensors["model.demo_start"].to(device), tensors["model.demo_end"].to(device),
                             manifest, sha256_file(ckpt), cfg)


def describe(frozen: FrozenDemoEncoder) -> dict:
    m = frozen.manifest
    return {"ckpt_sha256": frozen.ckpt_sha256, "source_checkpoint": m.get("source_checkpoint"),
            "sampler": m.get("sampler"), "spatial_tokens_used": m.get("vit", {}).get("spatial_tokens_used"),
            "waived_gates": m.get("waived_gates")}


if __name__ == "__main__":
    import sys

    f = load_frozen(sys.argv[1])
    print(json.dumps(describe(f), indent=1, ensure_ascii=False))
