"""bpv4_xdemo dataset: one item = (current trajectory Q, demonstration D of the same task, another episode).

Per item: S query frames on Q (stratified in time), each with the three current camera images (resized with pad and
packed by the Qwen3.5 image processor, exactly like the BP feature cache), the state and the 50-step action chunk
(delta + normalized like stage 1); plus the cached keyframe ViT features of D and its keyframe visibility.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from lerobot.policies.BPV4BPEn.build_feature_cache import cache_path
from lerobot.policies.BPV4BPEn.dataset import _holdout, _Repo
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_bpv4_xdemo import BPV4XDemoDatasetConfig

LEFT_GRIPPER, RIGHT_GRIPPER = 6, 13


def first_grasp_arm(state: np.ndarray) -> str | None:
    """'L' / 'R' for the gripper that closes first, None when no gripper closes."""
    best = None
    for arm, dim in (("L", LEFT_GRIPPER), ("R", RIGHT_GRIPPER)):
        if state.shape[1] <= dim:
            continue
        closed = state[:, dim] < 0.5
        idx = np.nonzero(closed[1:] & ~closed[:-1])[0]
        if len(idx) and (best is None or idx[0] < best[1]):
            best = (arm, int(idx[0]))
    return None if best is None else best[0]


class _XRepo(_Repo):
    """Stage-1 repo (states, actions, transforms, tasks) + frame timestamps and the video metadata for decoding."""

    def __init__(self, repo: str, cfg, train_cfg) -> None:
        super().__init__(repo, cfg, train_cfg)
        files = sorted(Path(repo, "data").rglob("*.parquet"))
        df = pd.concat(pd.read_parquet(f, columns=["episode_index", "timestamp"]) for f in files)
        self.timestamps = {int(ep): g["timestamp"].to_numpy(np.float64) for ep, g in df.groupby("episode_index")}
        self.arms = {ep: first_grasp_arm(s) for ep, s in self.states.items()}
        self.task_name = Path(repo).parent.name
        self._meta = None

    def video_frames(self, episode: int, frames: list[int], camera: str, backend: str) -> torch.Tensor:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.datasets.video_utils import decode_video_frames

        if self._meta is None:
            self._meta = LeRobotDatasetMetadata(self.repo)
        path = Path(self.repo) / self._meta.get_video_file_path(episode, camera)
        info = self._meta.episodes[episode]
        ts = [float(info[f"videos/{camera}/from_timestamp"]) + float(self.timestamps[episode][f]) for f in frames]
        return decode_video_frames(path, ts, 1e-4, backend)  # [S, C, H, W] float in [0, 1]


class BPV4XDemoDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: BPV4XDemoDatasetConfig, train_cfg, repo_ids: list[str], chunk_size: int) -> None:
        self.cfg, self.chunk_size = cfg, int(chunk_size)
        self.video_backend = getattr(cfg, "video_backend", None) or "torchcodec"
        self.repos = [_XRepo(r, cfg, train_cfg) for r in repo_ids]
        self.items: list[tuple[int, int]] = []
        for ri, repo in enumerate(self.repos):
            n = repo.total_episodes
            hold = _holdout(repo.repo, dict(cfg.val_holdout_last))
            eps = range(n) if cfg.split == "all" else range(n - hold, n) if cfg.split == "val" else range(n - hold)
            for ep in eps:
                if not cache_path(cfg.bp_feature_cache_dir, repo.repo, ep).is_file():
                    raise FileNotFoundError(f"missing feature cache for {repo.repo} episode {ep}; run build_feature_cache")
                self.items.append((ri, ep))
        # demonstration pools: same task (+ same category for category-paired tasks), same split
        self.group_of: dict[tuple[int, int], tuple[str, str]] = {}
        self.pools: dict[tuple[str, str], list[tuple[int, int]]] = {}
        dropped = 0
        for ri, ep in self.items:
            repo = self.repos[ri]
            group = (repo.task_name, self._category(repo.task_name, repo.tasks.get(ep, "")))
            if group[1] == "?":
                dropped += 1
                continue
            self.group_of[(ri, ep)] = group
            self.pools.setdefault(group, []).append((ri, ep))
        self.items = [it for it in self.items if it in self.group_of and len(self.pools[self.group_of[it]]) > 1]
        self.num_episodes = len(self.items)
        self.num_frames = sum(len(self.repos[ri].states[ep]) for ri, ep in self.items)
        self._image_processor = None
        self._token_ids = None
        logging.info("bpv4_xdemo dataset split=%s: %d trajectories, %d frames, %d repos, %d demo groups, "
                     "%d episodes without a category", cfg.split, self.num_episodes, self.num_frames,
                     len(self.repos), len(self.pools), dropped)

    def _category(self, task: str, text: str) -> str:
        keywords = dict(self.cfg.category_pairing).get(task)
        if not keywords:
            return ""
        hits = [k for k in keywords if re.search(rf"\b{re.escape(k)}", text.lower())]
        return hits[0] if len(hits) == 1 else "?"

    def __len__(self) -> int:
        return len(self.items)

    @property
    def stats(self):
        return {self.repos[0].robot_type: self.repos[0].stats} if self.repos else {}

    # ------------------------------------------------------------------ helpers (also used by offline evaluation)
    def pick_demo(self, i: int) -> tuple[int, int]:
        key = self.items[i]
        if float(torch.rand(())) < self.cfg.same_episode_prob:
            return key
        pool = [d for d in self.pools[self.group_of[key]] if d != key]
        return pool[int(torch.randint(len(pool), ()))]

    def query_frames(self, ri: int, ep: int) -> list[int]:
        length = len(self.repos[ri].states[ep])
        s = int(self.cfg.queries_per_trajectory)
        edges = np.linspace(0, length, s + 1).astype(int)
        return [int(torch.randint(edges[j], max(edges[j] + 1, edges[j + 1]), ()).item()) for j in range(s)]

    def demo_features(self, ri: int, ep: int, *, drop: bool) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        rec = torch.load(cache_path(cfg.bp_feature_cache_dir, self.repos[ri].repo, ep), map_location="cpu",
                         weights_only=False)
        k_max, k_eff = int(cfg.bp_num_keyframes), int(rec["frames"].shape[0])
        feats = torch.zeros((k_max, *rec["feats"].shape[1:]), dtype=rec["feats"].dtype)
        feats[:k_eff] = rec["feats"]
        time = torch.zeros(k_max)
        time[:k_eff] = rec["time_ratio"]
        valid = torch.arange(k_max) < k_eff
        visible = valid.clone()
        if drop and cfg.bp_keyframe_drop_prob > 0:
            dropped = (torch.rand(k_max) < cfg.bp_keyframe_drop_prob) & valid
            if int((valid & ~dropped).sum()) >= min(int(cfg.bp_min_visible_keyframes), k_eff):
                visible = valid & ~dropped
        return {"bp.feats": feats, "bp.time": time, "bp.kf_valid": valid, "bp.visible": visible}

    def current_inputs(self, ri: int, ep: int, frames: list[int]) -> dict[str, torch.Tensor]:
        from lerobot.transforms.utils import resize_with_pad

        if self._image_processor is None:
            from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision

            self._image_processor, self._token_ids = load_qwen35_vision(self.cfg.qwen3_5_processor_path)
        repo, size = self.repos[ri], int(self.cfg.image_size)
        per_cam = [repo.video_frames(ep, frames, cam, self.video_backend) for cam in self.cfg.bp_camera_keys]
        images = [resize_with_pad(per_cam[c][j][None], size, size)[0]
                  for j in range(len(frames)) for c in range(len(per_cam))]
        out = self._image_processor(images, do_rescale=False)
        grid = torch.as_tensor(out["image_grid_thw"], dtype=torch.long)
        tid = self._token_ids
        ids, types = [], []
        for g in grid[: len(per_cam)]:
            n_tok = int(g.prod()) // tid.spatial_merge_size**2
            ids += [tid.vision_start_token_id] + [tid.image_token_id] * n_tok + [tid.vision_end_token_id]
            types += [0] + [1] * n_tok + [0]
        s = len(frames)
        input_ids = torch.tensor(ids)[None].expand(s, -1).clone()
        return {
            "observation.pixel_values": torch.as_tensor(out["pixel_values"]),
            "observation.image_grid_thw": grid,
            "observation.input_ids": input_ids,
            "observation.mm_token_type_ids": torch.tensor(types)[None].expand(s, -1).clone(),
            "observation.attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
        }

    def build(self, query: tuple[int, int], demo: tuple[int, int], frames: list[int], *, drop: bool) -> dict:
        ri, ep = query
        repo = self.repos[ri]
        out = {k: [] for k in (OBS_STATE, ACTION, ACTION_VALID_MASK)}
        for f in frames:
            q = repo.query(ep, f, self.chunk_size)
            for key in out:
                out[key].append(q[key])
        batch = {key: torch.stack(v) for key, v in out.items()}
        batch.update(self.current_inputs(ri, ep, frames))
        batch.update(self.demo_features(*demo, drop=drop))
        length = len(repo.states[ep])
        batch.update({
            "query.frame": torch.tensor(frames, dtype=torch.long),
            "query.progress": torch.tensor([f / max(1, length - 1) for f in frames], dtype=torch.float32),
            "pair.same_episode": torch.tensor(demo == query),
            "pair.same_arm": torch.tensor(repo.arms.get(ep) == self.repos[demo[0]].arms.get(demo[1])),
        })
        return batch

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        query = self.items[i]
        return self.build(query, self.pick_demo(i), self.query_frames(*query), drop=self.cfg.split == "train")


def make_bpv4_xdemo_dataset(train_cfg) -> tuple[BPV4XDemoDataset, dict]:
    from lerobot.datasets.factory import resolve_repo_ids

    repo_ids = resolve_repo_ids(train_cfg)
    ds = BPV4XDemoDataset(train_cfg.dataset, train_cfg, repo_ids, chunk_size=int(train_cfg.policy.chunk_size))
    return ds, ds.stats


def eval_dataset(train_config_json: dict, repo_ids: list[str], split: str = "val") -> BPV4XDemoDataset:
    """Rebuild the dataset from a checkpoint's train_config.json (offline evaluation tools)."""
    d = dict(train_config_json["dataset"])
    known = set(BPV4XDemoDatasetConfig.__dataclass_fields__)
    kwargs = {k: v for k, v in d.items() if k in known and k not in ("type", "image_transforms")}
    kwargs["split"] = split
    cfg = BPV4XDemoDatasetConfig(**kwargs)
    return BPV4XDemoDataset(cfg, SimpleNamespace(dataset=cfg), repo_ids,
                            chunk_size=int(train_config_json["policy"]["chunk_size"]))
