"""BPv5 Stage B server adapter for Damntwin / BP RoboTwin eval.

Checkpoint type: bpv5_stageb. Builds the Stage B prefix batch (demo P5 + current obs +
dual text), runs sample_segments, then maps the 80-D aloha layout back to Damntwin's 14-D qpos.

Demo cache: cleaned RoboTwin P5 under BPV5_DEMO_CACHE_DIR (default training cache), leaf
robotwin/<task>/aloha-agilex_clean_50. Candidate-set ids like place_shoe_bp__I0__L0__x map to
task place_shoe.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.policies.BPV5QFormer.data80d import NATIVE_ALOHA_SLOTS
from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision
from lerobot.transforms.core import NormalizeTransformFn, ResizeImagesWithPadFn, UnNormalizeTransformFn
from lerobot.utils.constants import ACTION, OBS_STATE

import lerobot.policies.BPV5StageB.configuration_bpv5stageb  # noqa: F401

from .base import PolicyAdapter, PreparedSample
from .bpvav2 import fixed_bp_video_cache_dir, load_bp_task_sources
from .common import (
    CAMERA_ALIASES,
    hwc_tensor,
    resolve_image_history,
    resolve_images,
    resolve_state,
)

CHOICE_CACHE = 512
DEFAULT_DEMO_CACHE = "/vla/workspace/cache/bpv5_cleaned_p5"
DEFAULT_STATS = "/vla/workspace/my_tbot/norm_stats_bpv5_pt/robotwin_clean_mixed_80d/aloha/stats.json"
SLOTS = tuple(NATIVE_ALOHA_SLOTS)  # 14 native dims -> 80-D indices


def _robotwin_leaf(task_or_cs: str) -> str:
    base = str(task_or_cs).split("__", 1)[0]
    if base.endswith("_bp"):
        base = base[:-3]
    return f"robotwin/{base}/aloha-agilex_clean_50"


def _pad_state14_to_80(state14: np.ndarray) -> np.ndarray:
    out = np.zeros(80, dtype=np.float32)
    s = np.asarray(state14, dtype=np.float32).reshape(-1)
    for i, slot in enumerate(SLOTS):
        if i < len(s):
            out[slot] = s[i]
    return out


def _slice_80_to_14(action80: np.ndarray) -> np.ndarray:
    a = np.asarray(action80, dtype=np.float32)
    if a.ndim == 1:
        return np.asarray([a[s] for s in SLOTS], dtype=np.float32)
    return np.stack([a[..., s] for s in SLOTS], axis=-1).astype(np.float32)


class BPV5StageBAdapter(PolicyAdapter):
    name = "bpv5_stageb"
    family = "bp"
    aliases = ("bpv5_stageb", "bpv5stageb")

    def _check_type(self) -> None:
        if self.config.type != "bpv5_stageb":
            raise ValueError(f"POLICY=bpv5_stageb but checkpoint type={self.config.type!r}")

    def before_policy_init(self) -> None:
        # Serving never continues Stage A load / LoRA train flags from the training JSON.
        if getattr(self.config, "stage_a_checkpoint", ""):
            logging.info("bpv5_stageb: clear stage_a_checkpoint (weights are in this ckpt)")
            self.config.stage_a_checkpoint = ""
        if getattr(self.config, "pretrained_path", None):
            self.config.pretrained_path = None
        # Sweep may pass BPv4's --qwen3_5_path; restore this checkpoint's wall-oss paths.
        ckpt_cfg = json.loads((self.ckpt_dir / "config.json").read_text())
        for attr in ("qwen3_5_pretrained_path", "qwen3_5_processor_path"):
            path = ckpt_cfg.get(attr)
            if isinstance(path, str) and path and Path(path).expanduser().exists():
                if getattr(self.config, attr, None) != path:
                    logging.info("bpv5_stageb: restore %s -> %s", attr, path)
                setattr(self.config, attr, path)

    def _init_action_transforms(self) -> None:
        stats_path = Path(self.args.stats_path).expanduser() if self.args.stats_path else None
        if stats_path is None or not stats_path.is_file():
            train_stats = None
            if self.train_cfg is not None:
                train_stats = getattr(self.train_cfg.dataset, "stats_file", None)
            stats_path = Path(train_stats or DEFAULT_STATS)
        if not stats_path.is_file():
            raise FileNotFoundError(f"bpv5_stageb stats not found: {stats_path}")
        raw = json.loads(stats_path.read_text())
        if "action_80D" not in raw or "state_80D" not in raw:
            raise ValueError(f"bpv5_stageb expects action_80D/state_80D in {stats_path}")
        meta = raw.get("metadata") or {}
        self.stats_key = "aloha_80d"
        self.state_stats = {
            OBS_STATE: {k: np.asarray(raw["state_80D"][k], dtype=np.float32) for k in ("min", "max", "mean", "std")}
        }
        self.action_stats = {
            ACTION: {k: np.asarray(raw["action_80D"][k], dtype=np.float32) for k in ("min", "max", "mean", "std")}
        }
        self.target_action_dim = 80  # unnormalize in 80-D; postprocess slices to 14
        self.resize_fn = ResizeImagesWithPadFn(height=self.args.resize_size, width=self.args.resize_size)
        self.normalize_state_fn = NormalizeTransformFn(
            selected_keys=[OBS_STATE], mode="mean_std", norm_stats=self.state_stats
        )
        self.unnormalize_action_fn = UnNormalizeTransformFn(
            selected_keys=[ACTION], mode="mean_std", norm_stats=self.action_stats
        )
        self.action_mode = "delta"
        delta = meta.get("delta_mask")
        self.delta_mask = np.asarray(delta, dtype=np.float32) if delta is not None else None
        self._stats_path = stats_path
        logging.info("bpv5_stageb stats=%s action_layout=80d->14 slots", stats_path)

    def setup_policy(self) -> None:
        dataset_cfg = getattr(self.train_cfg, "dataset", None)
        processor_path = (
            getattr(dataset_cfg, "qwen3_5_processor_path", None) if dataset_cfg is not None else None
        ) or self.config.qwen3_5_pretrained_path
        if not Path(str(processor_path)).exists():
            processor_path = self.config.qwen3_5_pretrained_path
        self.image_processor, self.token_ids = load_qwen35_vision(str(processor_path))
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(processor_path))
        self.lang_len = int(getattr(dataset_cfg, "language_max_length", 96) or 96) if dataset_cfg else 96
        demo_dir = os.environ.get("BPV5_DEMO_CACHE_DIR") or (
            str(dataset_cfg.cache_dir) if dataset_cfg is not None else DEFAULT_DEMO_CACHE
        )
        self.demo_root = Path(demo_dir)
        if not self.demo_root.is_dir():
            raise FileNotFoundError(f"bpv5_stageb demo cache missing: {self.demo_root}")
        if not self.args.bp_mapping_path:
            raise ValueError("bpv5_stageb needs bp_mapping_path (candidate-set mapping)")
        self.mapping_path, self.sources = load_bp_task_sources(self.args.bp_mapping_path)
        self.select_seed = int(os.environ.get("BPV5_DEMO_SEED", "0"))
        self.select_mode = os.environ.get("BPV5_DEMO_SELECT", "episode")
        if self.select_mode not in {"episode", "set"}:
            raise ValueError(f"BPV5_DEMO_SELECT must be episode or set, got {self.select_mode!r}")
        self.log_dir = fixed_bp_video_cache_dir(self.args.debug_dir, self.name)
        self.select_log = self.log_dir / "selection_log.jsonl"
        self._pools: dict[str, tuple[str, Path, list[int]]] = {}
        self._choice: OrderedDict[tuple[str, str], tuple[dict[str, torch.Tensor], dict[str, Any]]] = OrderedDict()
        self._episode_count: dict[str, int] = {}
        self._lock = threading.Lock()
        self._text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._metadata_extra = {
            "bp_mapping_path": str(self.mapping_path),
            "bp_demo_cache_dir": str(self.demo_root),
            "bp_demo_leaf": "robotwin/<task>/aloha-agilex_clean_50",
            "action_layout": "80d->14 via NATIVE_ALOHA_SLOTS",
            "expert_v_obs": bool(getattr(self.config, "expert_v_obs", True)),
            "required_image_history": 1,
            "use_language": True,
            "action_dim": 14,
        }
        logging.info(
            "bpv5_stageb | demo=%s | mapping=%s | expert_v_obs=%s",
            self.demo_root,
            self.mapping_path,
            self._metadata_extra["expert_v_obs"],
        )

    def build_metadata(self, chunk_size: int) -> dict[str, Any]:
        metadata = super().build_metadata(chunk_size)
        metadata.update(getattr(self, "_metadata_extra", {}))
        metadata["action_dim"] = 14
        metadata["target_action_dim"] = 14
        return metadata

    def on_request(self, connection_id: str, obs: dict[str, Any]) -> None:
        obs["__bp_connection"] = connection_id
        super().on_request(connection_id, obs)

    def _tokens(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        if text not in self._text_cache:
            out = self.tokenizer(
                text,
                max_length=self.lang_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self._text_cache[text] = (out["input_ids"][0], out["attention_mask"][0].bool())
        return self._text_cache[text]

    def _pool(self, task: str) -> tuple[str, Path, list[int]]:
        if task in self._pools:
            return self._pools[task]
        leaf = _robotwin_leaf(task)
        cache_dir = self.demo_root / leaf
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"bpv5_stageb: no P5 leaf for task={task!r} -> {cache_dir}")
        episodes = sorted(
            int(n[2:-3])
            for n in os.listdir(cache_dir)
            if n.startswith("ep") and n.endswith(".pt") and ".tmp" not in n
        )
        if not episodes:
            raise FileNotFoundError(f"bpv5_stageb: empty demo cache {cache_dir}")
        logging.info("bpv5_stageb demo pool | task=%s | eps=%d | %s", task, len(episodes), cache_dir)
        self._pools[task] = (leaf, cache_dir, episodes)
        return self._pools[task]

    def _load_demo(self, cache_dir: Path, episode: int) -> dict[str, torch.Tensor]:
        path = cache_dir / f"ep{episode:06d}.pt"
        if not path.is_file():
            path = cache_dir / f"ep{episode:05d}.pt"
        rec = torch.load(path, map_location="cpu", weights_only=True)
        feats = rec["feats"]
        valid = rec.get("camera_valid")
        return {
            "demo.feats": feats,
            "demo.time": rec["time_ratio"].float(),
            "demo.camera_valid": (
                valid.bool() if valid is not None else torch.ones(feats.shape[:2], dtype=torch.bool)
            ),
        }

    def _select(self, obs: dict[str, Any], task: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        uid = obs.get("episode_uid")
        uid = str(uid) if uid not in (None, "") else None
        if self.select_mode == "set":
            key = (task, "set")
        else:
            key = (task, uid) if uid is not None else (task, "conn:" + str(obs.get("__bp_connection", "")))
        with self._lock:
            if key in self._choice and (uid is not None or self.select_mode == "set" or not bool(obs.get("reset"))):
                self._choice.move_to_end(key)
                return self._choice[key]
            leaf, cache_dir, episodes = self._pool(task)
            if self.select_mode == "set":
                index = random.Random(f"{self.select_seed}:{task}").randrange(len(episodes))
            elif uid is not None:
                index = random.Random(f"{self.select_seed}:{task}:{uid}").randrange(len(episodes))
            else:
                count = self._episode_count.get(task, 0)
                self._episode_count[task] = count + 1
                index = random.Random(f"{self.select_seed}:{task}:{count}").randrange(len(episodes))
            demo = self._load_demo(cache_dir, episodes[index])
            info = {
                "bp_demo_episode": int(episodes[index]),
                "bp_demo_select": self.select_mode,
                "bp_demo_leaf": leaf,
            }
            with self.select_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"task": task, **info}) + "\n")
            self._choice[key] = (demo, info)
            while len(self._choice) > CHOICE_CACHE:
                self._choice.popitem(last=False)
            return self._choice[key]

    def prepare(self, obs: dict[str, Any]) -> PreparedSample:
        task = obs.get("task_type")
        if not isinstance(task, str) or not task.strip():
            raise KeyError("bpv5_stageb request missing non-empty task_type")
        task = task.strip()
        images = resolve_images(obs)
        state14 = resolve_state(obs)
        state80 = _pad_state14_to_80(state14)
        sample: dict[str, Any] = {OBS_STATE: torch.from_numpy(state80)}
        for key in CAMERA_ALIASES:
            history, _ = resolve_image_history(
                images,
                key,
                strict=False,
                frames=1,
                height=self.args.request_image_height,
                width=self.args.request_image_width,
            )
            sample[key] = hwc_tensor(history)
        sample = self.resize_fn(sample)
        out = self.image_processor([sample[key][0] for key in CAMERA_ALIASES], do_rescale=False)
        pixel_values = torch.as_tensor(out["pixel_values"])
        grid = torch.as_tensor(out["image_grid_thw"], dtype=torch.long)
        state_norm = self.normalize_state_fn({OBS_STATE: sample[OBS_STATE]})[OBS_STATE]
        demo, demo_info = self._select(obs, task)
        current_text = str(obs.get("task") or obs.get("prompt") or task)
        demo_ids, demo_attn = self._tokens(current_text)
        cur_ids, cur_attn = self._tokens(current_text)
        # Batch layout with S=1 (collate_nested cats on dim=0 across requests)
        inputs = {
            OBS_STATE: state_norm[None, None].to(self.device, self.runtime_dtype),  # [1,1,80]
            "obs.pixel_values": pixel_values[None].to(self.device, self.runtime_dtype),
            "obs.image_grid_thw": grid[None].to(self.device),
            "obs.camera_valid": torch.ones(1, 1, 3, dtype=torch.bool, device=self.device),
            "demo.feats": demo["demo.feats"][None].to(self.device, self.runtime_dtype),
            "demo.time": demo["demo.time"][None].to(self.device, self.runtime_dtype),
            "demo.camera_valid": demo["demo.camera_valid"][None].to(self.device),
            "demo_text.input_ids": demo_ids[None, None].to(self.device),
            "demo_text.attention_mask": demo_attn[None, None].to(self.device),
            "current_text.input_ids": cur_ids[None, None].to(self.device),
            "current_text.attention_mask": cur_attn[None, None].to(self.device),
        }
        prepared = PreparedSample(inputs, state80)
        prepared.demo_info = demo_info
        return prepared

    def postprocess(self, action_pred: torch.Tensor, states: list[np.ndarray]) -> list[dict[str, Any]]:
        """Unnormalize 80-D, apply delta, then slice native Aloha 14-D for Damntwin."""
        results = []
        for batch_idx, state80 in enumerate(states):
            model_action = action_pred[batch_idx, : self.infer_horizon, :80]
            action = self.unnormalize_action_fn({ACTION: model_action})[ACTION]
            model_action_np = model_action.detach().cpu().numpy().astype(np.float32)
            action_np = action.detach().cpu().numpy().astype(np.float32)
            if self.action_mode == "delta" and self.delta_mask is not None:
                state_pad = np.zeros_like(self.delta_mask, dtype=np.float32)
                usable = min(len(state_pad), len(state80))
                state_pad[:usable] = state80[:usable]
                dims = min(action_np.shape[-1], len(self.delta_mask))
                action_np[:, :dims] += state_pad[None, :dims] * self.delta_mask[None, :dims]
            action14 = _slice_80_to_14(action_np)
            model14 = _slice_80_to_14(model_action_np)
            results.append(
                {
                    "actions": action14,
                    "action": action14[0],
                    "model_actions": model14,
                    "model_action": model14[0],
                }
            )
        return results

    def _infer_prepared(self, prepared: list[PreparedSample]) -> list[dict[str, Any]]:
        results = super()._infer_prepared(prepared)
        for result, item in zip(results, prepared):
            result.update(getattr(item, "demo_info", {}))
        return results


ADAPTER = BPV5StageBAdapter
