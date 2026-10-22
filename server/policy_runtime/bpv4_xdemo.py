"""BPv4 第二阶段（bpv4_xdemo）：当前三路图像 + state + 示教 BP（无语言、无进度）。

示教直接读第一阶段的关键帧 ViT 特征缓存（训练配置里的 bp_feature_cache_dir），不解码视频。
候选示教：BP 映射 YAML 中每个任务 dataset_path 的全部 episode。每个连接在 episode 开头（reset / 首次请求）选一条，
之后保持不变。selection.mode：
  best    当前头部首帧与候选示教首帧的 ViT token 均值余弦相似度最高的一条（与 v3 的 best50 可比）
  random  按 (seed, 任务, 该任务第几个 episode) 随机选一条
选择记录写到 <debug_dir>/bp_video_cache/bpv4_xdemo/selection_log.jsonl。
"""

from __future__ import annotations

import json
import logging
import random
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from .base import PolicyAdapter, PreparedSample
from .bpvav2 import fixed_bp_video_cache_dir, load_bp_task_sources
from .common import CAMERA_ALIASES, OBS_STATE, hwc_tensor, resolve_image_history, resolve_images, resolve_state

from lerobot.policies.BPV4BPEn.build_feature_cache import cache_path
from lerobot.policies.BPV4XDemo.dataset import first_grasp_arm
from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision
from lerobot.policies.names import is_bpv4_xdemo


class BPV4XDemoAdapter(PolicyAdapter):
    name = "bpv4_xdemo"
    family = "bp"
    aliases = ("bpv4_xdemo",)

    def _check_type(self) -> None:
        if not is_bpv4_xdemo(self.config.type):
            raise ValueError(f"POLICY=bpv4_xdemo 但 checkpoint type={self.config.type!r}。")

    def setup_policy(self) -> None:
        dataset_cfg = getattr(self.train_cfg, "dataset", None)
        processor_path = getattr(dataset_cfg, "qwen3_5_processor_path", None) or self.config.qwen3_5_pretrained_path
        self.image_processor, self.token_ids = load_qwen35_vision(str(processor_path))
        self.feature_cache_dir = str(getattr(dataset_cfg, "bp_feature_cache_dir",
                                             "/vla/workspace/cache/bpv4_bpen_vit_base800m_k32_v2"))
        self.num_keyframes = int(self.config.bp_num_keyframes)
        if not self.args.bp_mapping_path:
            raise ValueError("bpv4_xdemo 需要 bp_mapping_path（每个任务的示教数据集）。")
        self.mapping_path, self.sources = load_bp_task_sources(self.args.bp_mapping_path)
        selection = (yaml.safe_load(Path(self.mapping_path).read_text(encoding="utf-8")) or {}).get("selection") or {}
        self.select_mode = str(selection.get("mode", "best"))
        if self.select_mode not in {"best", "random"}:
            raise ValueError(f"selection.mode 必须是 best / random，得到 {self.select_mode!r}")
        self.select_seed = int(selection.get("seed", 0))
        self.log_dir = fixed_bp_video_cache_dir(self.args.debug_dir, self.name)
        self.select_log = self.log_dir / "selection_log.jsonl"
        self._pools: dict[str, list[dict[str, Any]]] = {}
        self._choice: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
        self._episode_count: dict[str, int] = {}
        self._lock = threading.Lock()
        self._ids = None
        self._metadata_extra = {
            "bp_mapping_path": str(self.mapping_path),
            "bp_selection": self.select_mode,
            "bp_selection_seed": self.select_seed,
            "bp_feature_cache_dir": self.feature_cache_dir,
            "bp_configured_tasks": sorted(self.sources),
            "required_image_history": 1,
            "use_language": False,
        }
        logging.info("bpv4_xdemo | tasks=%d | selection=%s | mapping=%s", len(self.sources), self.select_mode,
                     self.mapping_path)

    def build_metadata(self, chunk_size: int) -> dict[str, Any]:
        metadata = super().build_metadata(chunk_size)
        metadata.update(getattr(self, "_metadata_extra", {}))
        return metadata

    def on_request(self, connection_id: str, obs: dict[str, Any]) -> None:
        obs["__bp_connection"] = connection_id
        super().on_request(connection_id, obs)

    # ------------------------------------------------------------------ demonstrations
    def _pool(self, task: str) -> list[dict[str, Any]]:
        if task in self._pools:
            return self._pools[task]
        if task not in self.sources:
            raise KeyError(f"bpv4_xdemo: BP 映射中没有任务 {task!r}")
        repo = str(self.sources[task].dataset_paths[0])
        files = sorted(Path(repo, "data").rglob("*.parquet"))
        df = pd.concat(pd.read_parquet(f, columns=["episode_index", "observation.state"]) for f in files)
        arms = {int(ep): first_grasp_arm(np.stack(g["observation.state"].to_numpy())) for ep, g in df.groupby("episode_index")}
        pool = []
        for ep in sorted(arms):
            rec = torch.load(cache_path(self.feature_cache_dir, repo, ep), map_location="cpu", weights_only=False)
            k_eff = int(rec["frames"].shape[0])
            feats = torch.zeros((self.num_keyframes, *rec["feats"].shape[1:]), dtype=rec["feats"].dtype)
            feats[:k_eff] = rec["feats"]
            time = torch.zeros(self.num_keyframes)
            time[:k_eff] = rec["time_ratio"]
            pool.append({
                "episode": ep, "arm": arms[ep],
                "bp.feats": feats, "bp.time": time, "bp.visible": torch.arange(self.num_keyframes) < k_eff,
                "head": F.normalize(rec["feats"][0, 0].float().mean(0), dim=0),
            })
        logging.info("bpv4_xdemo 示教池 | task=%s | episodes=%d | repo=%s", task, len(pool), repo)
        self._pools[task] = pool
        return pool

    @torch.no_grad()
    def _head_descriptor(self, pixel_values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """ViT token mean of the current head image (first image), same frozen ViT as the feature cache."""
        vlm = self.policy.model.core.backbone.vlm
        patches = int(grid[0].prod())
        dtype = next(self.policy.model.core.backbone.vlm_visual.parameters()).dtype
        out = vlm.model.get_image_features(pixel_values[:patches].to(self.device, dtype), grid[:1].to(self.device))
        tokens = out.pooler_output
        tokens = tokens[0] if isinstance(tokens, (tuple, list)) else tokens
        return F.normalize(tokens.float().mean(0).cpu(), dim=0)

    def _select(self, obs: dict[str, Any], task: str, pixel_values, grid) -> dict[str, torch.Tensor]:
        key = (str(obs.get("__bp_connection", "")), task)
        with self._lock:
            if not bool(obs.get("reset")) and key in self._choice:
                return self._choice[key]
            pool = self._pool(task)
            count = self._episode_count.get(task, 0)
            self._episode_count[task] = count + 1
            record = {"task": task, "connection": key[0], "mode": self.select_mode, "task_episode": count}
            if self.select_mode == "best":
                head = self._head_descriptor(pixel_values, grid)
                sims = torch.stack([head @ item["head"] for item in pool])
                index = int(sims.argmax())
                record["sim"] = round(float(sims[index]), 4)
            else:
                index = random.Random(f"{self.select_seed}:{task}:{count}").randrange(len(pool))
            demo = pool[index]
            record.update(episode=demo["episode"], demo_arm=demo["arm"])
            with self.select_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            chosen = {k: demo[k][None].to(self.device) for k in ("bp.feats", "bp.visible")}
            chosen["bp.time"] = demo["bp.time"][None].to(self.device)
            self._choice[key] = chosen
            return chosen

    # ------------------------------------------------------------------ current observation
    def _prompt_ids(self, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._ids is None:
            tid, ids, types = self.token_ids, [], []
            for g in grid:
                n_tok = int(g.prod()) // tid.spatial_merge_size**2
                ids += [tid.vision_start_token_id] + [tid.image_token_id] * n_tok + [tid.vision_end_token_id]
                types += [0] + [1] * n_tok + [0]
            self._ids = (torch.tensor(ids), torch.tensor(types))
        return self._ids

    def prepare(self, obs: dict[str, Any]) -> PreparedSample:
        task = obs.get("task_type")
        if not isinstance(task, str) or not task.strip():
            raise KeyError("bpv4_xdemo 请求缺少非空字符串 task_type。")
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
        demo = self._select(obs, task.strip(), pixel_values, grid)
        inputs = {
            OBS_STATE: state[None, None].to(self.device, self.runtime_dtype),
            "observation.pixel_values": pixel_values[None].to(self.device, self.runtime_dtype),
            "observation.image_grid_thw": grid[None].to(self.device),
            "observation.input_ids": ids[None, None].to(self.device),
            "observation.mm_token_type_ids": types[None, None].to(self.device),
            "observation.attention_mask": torch.ones(1, 1, len(ids), dtype=torch.bool, device=self.device),
            "query.progress": torch.zeros(1, 1, device=self.device),
            **demo,
        }
        return PreparedSample(inputs, state_np)


ADAPTER = BPV4XDemoAdapter
