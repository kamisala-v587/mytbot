"""BPv4CDX Stage B（bpv4cdx_b，含 B-3 LoRA / B-4 可训练示教编码器）：当前三路图像 + state + 示教 Z_demo。

与训练主路径数据集（lerobot.policies.BPV4CDXB.dataset_main）逐项一致：
  当前观测  三路相机 resize_with_pad(224) -> Qwen3.5 processor；token = [示教占位 num_demo_tokens + 2]
            + [obs_start] + 三路图像 + [obs_end]（占位 id 0，模型替换其 embedding）
  state     checkpoint 的 stats.json 做 mean_std 归一化（基类）；动作反归一化后按 delta mask 加回 state（基类）
  示教      B-4（checkpoint 带 demo_encoder）：示教轨迹的 AE-3 ViT 缓存（训练配置 dataset.demo_cache_dir，P5 或
            P64）在线编码；P64 按示教编码器的 spatial_tokens 池化。无 demo_encoder 的 checkpoint 读 Z_demo 缓存
            （dataset.zdemo_dir）
示教来源：BP 映射里 task_type 对应的 LeRobot repo（bprobotwin 服务端把 task_type 改写为候选集编号，即该候选集的
repo）。环境变量 BPV4CDXB_DEMO_SELECT：
  episode（默认）  每个回合选一条：按 (seed, task_type, episode_uid) 固定随机；请求没有 episode_uid 时，每个连接
                   在回合开头（reset / 首次请求）按该任务的第几个回合选
  set              每个候选集固定一条：按 (seed, task_type) 固定随机，该候选集的所有回合用同一条示教
种子默认 0，环境变量 BPV4CDXB_DEMO_SEED 可改。回复带 bp_demo_episode / bp_demo_select（客户端记进 episodes.jsonl
的 server 字段）；选择记录也写到 <debug_dir>/bp_video_cache/bpv4cdx_b/selection_log.jsonl。
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

import torch

from .base import PolicyAdapter, PreparedSample
from .bpvav2 import fixed_bp_video_cache_dir, load_bp_task_sources
from .common import (CAMERA_ALIASES, OBS_STATE, hwc_tensor, load_train_config_or_none, resolve_image_history,
                     resolve_images, resolve_state)

import lerobot.policies.BPV4CDXB.configuration_bpv4cdxb  # noqa: F401,E402  registers the bpv4cdx_b config
from lerobot.policies.BPV4CDX.dataset import pool_spatial  # noqa: E402
from lerobot.policies.BPV4CDX.paths import rel_id  # noqa: E402
from lerobot.policies.BPV4CDXB.dataset import PLACEHOLDER_ID  # noqa: E402
from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision  # noqa: E402

CHOICE_CACHE = 512
LOCAL_FROZEN = "/vla/workspace/models/BPVA/BPv4CDX/StageA_AE_frozen/0750000"
LOCAL_DEMO_CACHE = "/vla/workspace/data/robotwin2.5_vit_cache"  # BP RoboTwin AE-3 ViT records (P64)


class BPV4CDXBAdapter(PolicyAdapter):
    name = "bpv4cdx_b"
    family = "bp"
    aliases = ("bpv4cdx_b", "bpv4cdxb")

    def _check_type(self) -> None:
        if self.config.type != "bpv4cdx_b":
            raise ValueError(f"POLICY=bpv4cdx_b 但 checkpoint type={self.config.type!r}。")

    def before_policy_init(self) -> None:
        # exported checkpoints (e.g. LoRA merged) carry no train_config.json; merge_info.json names the source checkpoint
        if self.train_cfg is None and (self.ckpt_dir / "merge_info.json").is_file():
            source = json.loads((self.ckpt_dir / "merge_info.json").read_text()).get("source_checkpoint")
            if source:
                logging.info("bpv4cdx_b: train_config.json from the merge source %s", source)
                self.train_cfg = load_train_config_or_none(Path(source))
        # checkpoints trained on another machine (B200 /home/jovyan paths): use this machine's frozen delivery
        frozen = getattr(self.config, "frozen_encoder", "")
        if frozen and not Path(frozen).is_dir():
            local = os.environ.get("BPV4CDXB_FROZEN", LOCAL_FROZEN)
            logging.info("bpv4cdx_b: frozen_encoder %s not found, using %s", frozen, local)
            self.config.frozen_encoder = local
        # init_from is the training start point (B-2 / B-3 / B-4 checkpoint); every weight is in this checkpoint
        if getattr(self.config, "init_from", ""):
            logging.info("bpv4cdx_b: ignore init_from=%s (weights come from %s)", self.config.init_from, self.ckpt_dir)
            self.config.init_from = ""

    def setup_policy(self) -> None:
        dataset_cfg = getattr(self.train_cfg, "dataset", None)
        if dataset_cfg is None:
            raise RuntimeError(f"bpv4cdx_b 需要 checkpoint 的 train_config.json: {self.ckpt_dir}")
        processor_path = getattr(dataset_cfg, "qwen3_5_processor_path", None) or self.config.qwen3_5_pretrained_path
        if not Path(str(processor_path)).exists():
            processor_path = self.config.qwen3_5_pretrained_path  # remapped to this machine by the base adapter
        self.image_processor, self.token_ids = load_qwen35_vision(str(processor_path))
        self.online_demo = getattr(self.policy.model, "demo_encoder", None) is not None
        if self.online_demo:
            # BPV4CDXB_DEMO_CACHE_DIR overrides; a training-side path that does not exist here falls back to the
            # local BP RoboTwin cache (checkpoints trained elsewhere, e.g. B-4 on B200)
            demo_dir = os.environ.get("BPV4CDXB_DEMO_CACHE_DIR") or str(dataset_cfg.demo_cache_dir)
            if not Path(demo_dir).is_dir():
                logging.info("bpv4cdx_b: demo cache %s not found, using %s", demo_dir, LOCAL_DEMO_CACHE)
                demo_dir = LOCAL_DEMO_CACHE
            self.demo_root = Path(demo_dir)
            enc_cfg = json.loads((Path(self.config.frozen_encoder) / "pretrained_model" / "config.json").read_text())
            self.demo_spatial = int(enc_cfg["spatial_tokens"])
        else:
            self.demo_root = Path(str(dataset_cfg.zdemo_dir))
            self.demo_spatial = None
        if not self.demo_root.is_dir():
            raise FileNotFoundError(f"bpv4cdx_b 示教缓存目录不存在: {self.demo_root}")
        if not self.args.bp_mapping_path:
            raise ValueError("bpv4cdx_b 需要 bp_mapping_path（每个任务 / 候选集的示教数据集）。")
        self.mapping_path, self.sources = load_bp_task_sources(self.args.bp_mapping_path)
        self.demo_tokens = int(self.config.num_demo_tokens) + 2 if self.config.demo_input == "video" else 0
        self.markers = bool(self.config.obs_marker_tokens)
        self.select_seed = int(os.environ.get("BPV4CDXB_DEMO_SEED", "0"))
        self.select_mode = os.environ.get("BPV4CDXB_DEMO_SELECT", "episode")
        if self.select_mode not in {"episode", "set"}:
            raise ValueError(f"BPV4CDXB_DEMO_SELECT must be episode or set, got {self.select_mode!r}")
        self.log_dir = fixed_bp_video_cache_dir(self.args.debug_dir, self.name)
        self.select_log = self.log_dir / "selection_log.jsonl"
        self._pools: dict[str, tuple[str, Path, list[int]]] = {}
        self._choice: OrderedDict[tuple[str, str], dict[str, torch.Tensor]] = OrderedDict()
        self._episode_count: dict[str, int] = {}
        self._lock = threading.Lock()
        self._ids = None
        self._metadata_extra = {
            "bp_mapping_path": str(self.mapping_path),
            "bp_selection": ("random per episode (seed, task_type, episode_uid)" if self.select_mode == "episode"
                             else "one per candidate set (seed, task_type)"),
            "bp_selection_seed": self.select_seed,
            "bp_demo_input": "online P5 / P64 (demo_encoder)" if self.online_demo else "cached Z_demo",
            "bp_demo_cache_dir": str(self.demo_root),
            "bp_demo_spatial_tokens": self.demo_spatial,
            "bp_configured_tasks": len(self.sources),
            "required_image_history": 1,
            "use_language": False,
        }
        logging.info("bpv4cdx_b | tasks=%d | demo=%s %s | mapping=%s", len(self.sources),
                     self._metadata_extra["bp_demo_input"], self.demo_root, self.mapping_path)

    def build_metadata(self, chunk_size: int) -> dict[str, Any]:
        metadata = super().build_metadata(chunk_size)
        metadata.update(getattr(self, "_metadata_extra", {}))
        return metadata

    def on_request(self, connection_id: str, obs: dict[str, Any]) -> None:
        obs["__bp_connection"] = connection_id
        super().on_request(connection_id, obs)

    # ------------------------------------------------------------------ demonstrations
    def _demo_dir(self, repo: str) -> Path:
        """The repo's cache directory: <root>/<rel id>, or <root>/<any prefix>/<task>/<variant> (a copy of the same
        LeRobot export under another root, e.g. robotwin2.5_bp/ for data_bp/bp_robotwin_lerobot/)."""
        direct = self.demo_root / rel_id(repo)
        if direct.is_dir():
            return direct
        parts = Path(repo).parts[-2:]
        found = sorted(p for p in self.demo_root.glob(f"*/{parts[0]}/{parts[1]}") if p.is_dir())
        if len(found) != 1:
            raise FileNotFoundError(f"bpv4cdx_b: no unique demo cache dir for {repo} under {self.demo_root}: {found}")
        return found[0]

    def _pool(self, task: str) -> tuple[str, Path, list[int]]:
        if task in self._pools:
            return self._pools[task]
        if task not in self.sources:
            raise KeyError(f"bpv4cdx_b: BP 映射中没有任务 {task!r}")
        repo = str(self.sources[task].dataset_paths[0])
        cache_dir = self._demo_dir(repo)
        episodes = sorted(int(n[2:7]) for n in os.listdir(cache_dir)
                          if n.startswith("ep") and n.endswith(".pt") and ".tmp" not in n)
        if not episodes:
            raise FileNotFoundError(f"bpv4cdx_b: empty demo cache {cache_dir}")
        logging.info("bpv4cdx_b 示教池 | task=%s | episodes=%d | cache=%s", task, len(episodes), cache_dir)
        self._pools[task] = (repo, cache_dir, episodes)
        return self._pools[task]

    def _load_demo(self, cache_dir: Path, episode: int) -> dict[str, torch.Tensor]:
        rec = torch.load(cache_dir / f"ep{episode:05d}.pt", map_location="cpu", weights_only=False)
        if not self.online_demo:
            return {"demo.z": rec["z_demo"][None].to(self.device)}
        feats = rec["feats"]
        if feats.shape[2] != self.demo_spatial:
            if feats.shape[2] != 64:
                raise ValueError(f"{cache_dir} ep {episode}: {feats.shape[2]} spatial tokens, encoder uses {self.demo_spatial}")
            feats = pool_spatial(feats, self.demo_spatial)
        return {"demo.feats": feats[None].to(self.device), "demo.time": rec["time_ratio"].float()[None].to(self.device),
                "demo.camera_valid": rec["camera_valid"].bool()[None].to(self.device)}

    def _select(self, obs: dict[str, Any], task: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """-> (model inputs of the demo, {bp_demo_episode, bp_demo_select} for the reply)."""
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
            repo, cache_dir, episodes = self._pool(task)
            if self.select_mode == "set":
                index = random.Random(f"{self.select_seed}:{task}").randrange(len(episodes))
                record = {"task": task, "mode": "set"}
            elif uid is not None:
                index = random.Random(f"{self.select_seed}:{task}:{uid}").randrange(len(episodes))
                record = {"task": task, "episode_uid": uid}
            else:
                count = self._episode_count.get(task, 0)
                self._episode_count[task] = count + 1
                index = random.Random(f"{self.select_seed}:{task}:{count}").randrange(len(episodes))
                record = {"task": task, "connection": key[1], "task_episode": count}
            demo = self._load_demo(cache_dir, episodes[index])
            record.update(seed=self.select_seed, demo_repo=repo, demo_episode=episodes[index])
            with self.select_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            info = {"bp_demo_episode": int(episodes[index]), "bp_demo_select": self.select_mode}
            self._choice[key] = (demo, info)
            while len(self._choice) > CHOICE_CACHE:
                self._choice.popitem(last=False)
            return self._choice[key]

    # ------------------------------------------------------------------ current observation
    def _prompt_ids(self, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._ids is None:
            tid = self.token_ids
            ids = [PLACEHOLDER_ID] * self.demo_tokens + [PLACEHOLDER_ID] * self.markers
            types = [0] * len(ids)
            for g in grid:
                n_tok = int(g.prod()) // tid.spatial_merge_size**2
                ids += [tid.vision_start_token_id] + [tid.image_token_id] * n_tok + [tid.vision_end_token_id]
                types += [0] + [1] * n_tok + [0]
            ids += [PLACEHOLDER_ID] * self.markers
            types += [0] * self.markers
            self._ids = (torch.tensor(ids), torch.tensor(types))
        return self._ids

    def prepare(self, obs: dict[str, Any]) -> PreparedSample:
        task = obs.get("task_type")
        if not isinstance(task, str) or not task.strip():
            raise KeyError("bpv4cdx_b 请求缺少非空字符串 task_type。")
        images = resolve_images(obs)
        state_np = resolve_state(obs)
        sample: dict[str, Any] = {OBS_STATE: torch.from_numpy(state_np)}
        for key in CAMERA_ALIASES:
            history, _ = resolve_image_history(images, key, strict=False, frames=1,
                                               height=self.args.request_image_height,
                                               width=self.args.request_image_width)
            sample[key] = hwc_tensor(history)
        sample = self.resize_fn(sample)
        out = self.image_processor([sample[key][0] for key in CAMERA_ALIASES], do_rescale=False)
        pixel_values = torch.as_tensor(out["pixel_values"])
        grid = torch.as_tensor(out["image_grid_thw"], dtype=torch.long)
        ids, types = self._prompt_ids(grid)
        state = self.normalize_state_fn(sample)[OBS_STATE]
        demo, demo_info = self._select(obs, task.strip())
        inputs = {
            OBS_STATE: state[None, None].to(self.device, self.runtime_dtype),
            "obs.pixel_values": pixel_values[None].to(self.device, self.runtime_dtype),
            "obs.image_grid_thw": grid[None].to(self.device),
            "obs.input_ids": ids[None, None].to(self.device),
            "obs.mm_token_type_ids": types[None, None].to(self.device),
            "obs.attention_mask": torch.ones(1, 1, len(ids), dtype=torch.bool, device=self.device),
            **demo,
        }
        prepared = PreparedSample(inputs, state_np)
        prepared.demo_info = demo_info
        return prepared

    def _infer_prepared(self, prepared: list[PreparedSample]) -> list[dict[str, Any]]:
        results = super()._infer_prepared(prepared)
        for result, item in zip(results, prepared):
            result.update(getattr(item, "demo_info", {}))
        return results


ADAPTER = BPV4CDXBAdapter
