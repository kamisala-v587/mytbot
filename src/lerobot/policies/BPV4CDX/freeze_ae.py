"""AE-4 frozen delivery (design doc 4.5): `<out>/StageA_AE_frozen.ckpt` (encoder, demo_start / demo_end,
segment_query tensors + manifest), `<out>/manifest.json` and a full copy of the checkpoint in `<out>/pretrained_model`.
The manifest records everything Stage B must reuse unchanged: ViT source and fingerprint, sampler and its version,
Q-Former / embeddings / projector checksums, normalizer (stats) hashes, camera order, time convention, code revision,
and the gate results that justify the delivery. Any change of these means a new encoder version.

Refuses to write unless the gates listed in --require passed (G7_seeds from a gate_g4_g7 run with --ckpt_seed2, or
from --seed2_gate); --force writes a provisional delivery, recorded in the manifest.

Usage: python -m lerobot.policies.BPV4CDX.freeze_ae --ckpt <pretrained_model> --out <dir>/StageA_AE_frozen \
         --gate g3=<g3.json> --gate g4_g7=<run.json> [--seed2_gate <run_seed2.json>] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open

from .common import code_revision, load_cfg
from .model_keys import ENCODER_PREFIXES
from .sampling import SAMPLER_VERSIONS

SAMPLING_FILES = ("sampling.py", "embodiment.py", "../BPV4BPEn/keyframes.py")
ALOHA_CAMERAS = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encoder_checksums(weights: Path) -> dict[str, str]:
    out = {}
    with safe_open(str(weights), "pt") as f:
        for prefix in ENCODER_PREFIXES:
            h = hashlib.sha256()
            keys = sorted(k for k in f.keys() if k.startswith(prefix))
            for k in keys:
                h.update(k.encode())
                h.update(f.get_tensor(k).float().numpy().tobytes())
            out[prefix] = {"tensors": len(keys), "sha256": h.hexdigest()}
    return out


def gate_status(path: Path) -> dict:
    r = json.loads(path.read_text())
    status = {"file": str(path), "pass": r.get("pass")}
    for g in ("G4", "G5", "G7_heldout_part", "G7_round_trip", "G7_seeds"):
        if g in r:
            status[g] = r[g].get("pass", r[g].get("G4_G5_heldout"))
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate", action="append", default=[], help="name=path to a gate result json")
    ap.add_argument("--seed2_gate", default=None,
                    help="G4-G7 result of the same variant trained with another seed (G7: >= 2 seeds, same direction)")
    ap.add_argument("--require", default="G3,G4,G5,G7_round_trip,G7_seeds")
    ap.add_argument("--waive", action="append", default=[],
                    help="gate=reason: a required gate formally replaced by other evidence (recorded in the manifest)")
    ap.add_argument("--external", action="append", default=[],
                    help="name=text: gate results run elsewhere (e.g. on the training machine), recorded as given")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    waived = dict(w.split("=", 1) for w in args.waive)
    ckpt, out = Path(args.ckpt), Path(args.out)
    cfg = load_cfg(ckpt)
    gates = {name: gate_status(Path(p)) for name, p in (g.split("=", 1) for g in args.gate)}
    flat = {}
    for st in gates.values():
        for k, v in st.items():
            if k in ("G4", "G5", "G7_round_trip", "G7_seeds"):
                flat[k] = v
        if "G4" not in st and "G5" not in st and st.get("pass") is not None:
            flat.setdefault("G3", st["pass"])
    if args.seed2_gate:
        gates["seed2"] = gate_status(Path(args.seed2_gate))
        flat["G7_seeds"] = bool(flat.get("G4") and flat.get("G5") and gates["seed2"].get("G4") and gates["seed2"].get("G5"))
    missing = [g for g in args.require.split(",") if g and not flat.get(g) and g not in waived]
    if missing and not args.force:
        raise SystemExit(f"gates not passed or not given: {missing} (use --force for a provisional delivery)")

    out.mkdir(parents=True, exist_ok=True)
    dst = out / "pretrained_model"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(ckpt, dst)
    pkg = Path(__file__).parent
    d, p = cfg.dataset, cfg.policy
    # AE-3 (split manifest): samplers on the MASK_MAPPING dims, cameras via IMAGE_MAPPING, stats per robot type
    multi = bool(getattr(d, "split_manifest", ""))
    version_key = {"su": "su_emb", "sa": "sa_emb"}.get(d.sampler, d.sampler) if multi else d.sampler
    stats_root = Path(d.external_stats_root)
    stats_files = sorted(stats_root.glob(f"*/{d.action_mode}/stats.json")) if multi else \
        [stats_root / "aloha" / d.action_mode / "stats.json"]
    manifest = {
        "provisional": bool(missing),
        "missing_gates": missing,
        "waived_gates": waived,
        "external_gates": dict(e.split("=", 1) for e in args.external),
        "source_checkpoint": str(ckpt),
        "code_revision": code_revision(),
        "vit": {"source": d.qwen3_5_processor_path, "fingerprint": p.qwen3_5_fingerprint, "image_size": 224,
                "tokens_per_image": 64, "spatial_tokens_used": d.spatial_tokens},
        "sampler": {"name": d.sampler, "version": SAMPLER_VERSIONS.get(version_key), "version_key": version_key,
                    "num_slots": d.num_slots, "alpha": 0.5 if d.sampler in ("sa", "sv") else None,
                    "code_sha256": {f: sha256_file((pkg / f).resolve()) for f in SAMPLING_FILES}},
        "embeddings": {"camera": "learned [3, qformer_dim], zero init", "spatial": f"learned [{d.spatial_tokens}, qformer_dim]",
                       "valid": "learned [2, qformer_dim]", "time": "Linear(sin_enc(t, qformer_dim)), separate for KV / queries",
                       "projector": "LayerNorm + Linear(qformer_dim -> hidden), inside demo_encoder.output_proj"},
        "camera_order": ("3 slots (image0 head, image1 left or single wrist, image2 right wrist) from the q35 "
                         "IMAGE_MAPPING of each robot type; a missing camera is zero with camera_valid False")
                        if multi else ALOHA_CAMERAS,
        "time_convention": "slot time = frame / (episode_length - 1); segment tau = start / (L - 1), end / (L - 1)",
        "normalizer": {"stats_root": str(stats_root), "action_mode": d.action_mode,
                       "stats_sha256": {str(f.relative_to(stats_root)): sha256_file(f) for f in stats_files if f.is_file()},
                       "checkpoint_stats_sha256": sha256_file(ckpt / "stats.json") if (ckpt / "stats.json").is_file() else None},
        "cache": {"spatial_stored": "P5 [32, 3, 5, 1024]" if multi else "P64 [32, 3, 64, 1024]",
                  "layout": getattr(d, "cache_layout", "task_variant"), "split_manifest": getattr(d, "split_manifest", "")},
        "encoder_checksums": encoder_checksums(ckpt / "model.safetensors"),
        "policy": {"num_demo_tokens": p.num_demo_tokens, "qformer_dim": p.qformer_dim,
                   "qformer_num_layers": p.qformer_num_layers, "expert_context": p.expert_context,
                   "demo_input": p.demo_input, "chunk_size": p.chunk_size},
        "gates": gates,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    # single-file delivery: encoder + special tokens (+ segment_query) tensors with the manifest
    with safe_open(str(ckpt / "model.safetensors"), "pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys() if k.startswith(ENCODER_PREFIXES)}
    torch.save({"state_dict": tensors, "manifest": manifest}, out / "StageA_AE_frozen.ckpt")
    print(out / "StageA_AE_frozen.ckpt", "provisional" if missing else "complete")


if __name__ == "__main__":
    main()
