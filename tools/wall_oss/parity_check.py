"""wall_oss parity: TBot `wall_oss` data path + policy vs wall-x's own LeRobot data path, on real frames.

For each sample the raw frame (after the delta transform, before normalization) goes through
  ours : the remaining `wall_oss` dataset transforms + WallOSSPolicy.forward
  wall : wall-x functions — PreprocessedDataset._vision_preprocess, get_wallx_normal_text, replace_action_token,
         DataCollator._normalize, preprocesser_call — and the same model called with wall-x's batch keys
and the script compares token ids, pixel values, image grids and the flow loss under the same noise and flow time.
wall-x's loader imports qwen_vl_utils; it is taken from --pydeps (test only, not a TBot dependency).

usage: source tools/bpv4cdxb/env.local.sh
       python tools/wall_oss/parity_check.py [--config configs/Pro6k/wall_oss/wall_bp_baseline.jsonc] [--n 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/Pro6k/wall_oss/wall_bp_baseline.jsonc")
ap.add_argument("--repo", default=None, help="default: first repo of dataset.repo_id_file")
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--pydeps", default="/vla/workspace/code/wallx_probe/pydeps")
ap.add_argument("--fp32", action="store_true", help="run the model in float32")
a = ap.parse_args()
sys.path.append(a.pydeps)
try:
    import numba  # noqa: F401
except ImportError:  # wall-x's loader imports numba for geometry helpers this check never calls
    numba = types.ModuleType("numba")
    numba.jit = lambda *args, **kw: args[0] if args and callable(args[0]) else (lambda f: f)
    numba.njit = numba.jit
    numba.prange = range
    sys.modules["numba"] = numba

import draccus  # noqa: E402

from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets.factory import make_dataset  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_policy  # noqa: E402
from lerobot.policies.wall_oss.transform_wall_oss import WallNormalizeTransformFn, WallPromptTransformFn  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_PREFIX, OBS_STATE  # noqa: E402
from lerobot.utils.jsonc import load_jsonc  # noqa: E402
from wall_x.data.backends.lerobot.loader import DataCollator, PreprocessedDataset  # noqa: E402
from wall_x.data.backends.lerobot.utils import (  # noqa: E402
    get_wallx_normal_text,
    preprocesser_call,
    replace_action_token,
)

raw = load_jsonc(a.config)
repo = a.repo or Path(raw["dataset"]["repo_id_file"]).read_text().split()[0]
tmp = Path(tempfile.mkdtemp())
(tmp / "repos.txt").write_text(repo + "\n")
raw["dataset"]["repo_id_file"] = str(tmp / "repos.txt")
(tmp / "cfg.json").write_text(json.dumps(raw))
cfg = draccus.parse(TrainPipelineConfig, config_path=str(tmp / "cfg.json"), args=[])
dataset, _ = make_dataset(cfg)
ds = dataset.datasets[0] if hasattr(dataset, "datasets") else dataset
transforms = ds._transform.transforms
cut = next(i for i, t in enumerate(transforms) if isinstance(t, WallNormalizeTransformFn))
prompt_t = next(t for t in transforms if isinstance(t, WallPromptTransformFn))
norm_t = transforms[cut]
print(f"repo {repo}: {len(ds)} frames; transforms {[type(t).__name__ for t in transforms]}")

policy = make_policy(cfg=cfg.policy).to(a.device).eval()
if a.fp32:  # separates bf16 rounding from real differences
    for p in policy.model.parameters():
        p.data = p.data.float()
processor = policy.processor
cams = [f"{OBS_IMAGES}.image{i}" for i in range(3)]
wall_ds = types.SimpleNamespace(
    hf_dataset=types.SimpleNamespace(meta=types.SimpleNamespace(camera_keys=cams)),
    data_config=types.SimpleNamespace(resolution=dict(zip(prompt_t.camera_names, [prompt_t.image_resolution] * 3)),
                                      image_factor=prompt_t.image_factor, min_pixels=prompt_t.min_pixels,
                                      max_pixels=prompt_t.max_pixels),
    _cam_key_mapping=dict(zip(cams, prompt_t.camera_names)),
)
state_key, action_key = norm_t.selected_keys[0], norm_t.selected_keys[-1]


def stat(key, name):
    return torch.as_tensor(np.asarray(norm_t.norm_stats[key][name]), dtype=torch.float32)


ok = True
rng = np.random.default_rng(0)
for idx in rng.choice(len(ds), a.n, replace=False).tolist():
    sample = LeRobotDataset.__getitem__(ds, idx)
    for t in transforms[:cut]:
        sample = t(sample)
    pre = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sample.items()}
    ours = sample
    for t in transforms[cut:]:
        ours = t(ours)

    # wall-x path on the same frame
    frames = {k: pre[k][0] for k in cams}
    images, *_ = PreprocessedDataset._vision_preprocess(wall_ds, frames)
    text, _ = get_wallx_normal_text({"instruction": pre["task"]}, policy.config.chunk_size, 0,
                                    cam_mapping=wall_ds._cam_key_mapping)
    text = replace_action_token([text], None, None, None)
    state = pre[state_key].float().reshape(1, 1, -1)
    action = pre[action_key].float().reshape(1, policy.config.chunk_size, -1)
    state_n = DataCollator._normalize(state, stat(state_key, "min"), stat(state_key, "max") - stat(state_key, "min"))
    action_n = DataCollator._normalize(action, stat(action_key, "min"),
                                       stat(action_key, "max") - stat(action_key, "min"))
    wall = preprocesser_call(processor, images=[images], text=text, padding=True, truncation=True,
                             max_length=768, norm_state=state_n, agent_pos_mask=torch.ones_like(state_n),
                             state_bins=prompt_t.state_bins)

    ids = ours[f"{OBS_PREFIX}input_ids"][ours[f"{OBS_PREFIX}attention_mask"].bool()]
    checks = {
        "input_ids": torch.equal(ids, wall["input_ids"][0]),
        "pixel_values": torch.equal(ours[f"{OBS_PREFIX}pixel_values"], wall["pixel_values"]),
        "image_grid_thw": torch.equal(ours[f"{OBS_PREFIX}image_grid_thw"], wall["image_grid_thw"]),
        "state": torch.allclose(ours[OBS_STATE].float().reshape(-1), state_n.reshape(-1), atol=1e-6),
        "action": torch.allclose(ours[ACTION].float(), action_n[0], atol=1e-6),
    }

    batch = {k: (v[None].to(a.device) if torch.is_tensor(v) else v) for k, v in ours.items()}
    t_flow = torch.tensor([0.37], device=a.device)
    with torch.no_grad():
        torch.manual_seed(0)
        loss_ours = float(policy.forward(batch, sample_time=t_flow)[0])
        torch.manual_seed(0)
        out = policy.model(
            input_ids=wall["input_ids"].to(a.device), attention_mask=wall["attention_mask"].to(a.device),
            pixel_values=wall["pixel_values"].to(a.device, next(policy.model.parameters()).dtype),
            image_grid_thw=wall["image_grid_thw"].to(a.device), action_chunk=action_n.to(a.device),
            dof_mask=torch.ones_like(action_n).to(a.device), proprioception=state_n.to(a.device),
            agent_pos_mask=torch.ones_like(state_n).to(a.device), dataset_names=["tbot"], sample_time=t_flow,
            flow_loss_mask=batch["action_is_valid"].float() if "action_is_valid" in batch else None,
            mode="train")
        loss_wall = float(out.loss)
        # where a loss difference comes from: our batch without left padding / without mm_token_type_ids
        keep = batch[f"{OBS_PREFIX}attention_mask"][0].bool()
        variants = {}
        for label, drop_pad, drop_mm in (("ours_unpadded", True, False), ("ours_no_mm_types", False, True),
                                         ("ours_unpadded_no_mm", True, True)):
            vb = dict(batch)
            if drop_pad:
                for k in ("input_ids", "attention_mask", "mm_token_type_ids"):
                    vb[f"{OBS_PREFIX}{k}"] = batch[f"{OBS_PREFIX}{k}"][:, keep]
            inputs = policy.model_inputs(vb)
            if drop_mm:
                inputs.pop("mm_token_type_ids")
            torch.manual_seed(0)
            acts = vb[ACTION].float()
            variants[label] = round(float(policy.model(
                **inputs, action_chunk=acts, dof_mask=torch.ones_like(acts), sample_time=t_flow,
                flow_loss_mask=vb["action_is_valid"].float() if "action_is_valid" in vb else None,
                mode="train").loss), 6)
    print("  variants:", variants)
    checks["loss"] = abs(loss_ours - loss_wall) <= 1e-3 * max(1.0, abs(loss_wall))
    ok &= all(checks.values())
    print(f"frame {idx}: {checks}  loss ours {loss_ours:.6f} wall {loss_wall:.6f}  tokens {len(ids)} "
          f"(padded {ours[f'{OBS_PREFIX}input_ids'].numel()}), task {pre['task'][:60]!r}")
    if idx == 0 or not checks["input_ids"]:
        print(processor.tokenizer.decode(ids[:400]).replace("<|image_pad|>", "")[:900])

print("PARITY OK" if ok else "PARITY MISMATCH")
