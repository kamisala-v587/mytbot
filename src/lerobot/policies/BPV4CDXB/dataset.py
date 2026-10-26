"""bpv4cdx Stage B dataset (`bpv4cdx Stage B 开发文档 v1.md`, section 3).

One item = (query episode j, demo episode i of the same candidate set) x S time steps of j:
  demo.z               [M, D]  cached Z_demo of i (BPV4CDXB.zdemo_cache)
  obs.*                        the three current images of j at each step (q35 image processing, 224) and the
                               prefix input ids: placeholders at the demo / marker positions (see modeling)
  observation.state / action / action_is_valid   j's state and 50-step action chunk at each step (q35 transforms)
The train sampler mixes single pairs with contrast groups (same anchor, every goal, consecutive in the batch).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.policies.BPV4CDX.repo_data import RepoData
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_bpv4cdxb import BPV4CDXBDatasetConfig

PLACEHOLDER_ID = 0  # a plain text token: sequential rope positions; its embedding is replaced by the model


class _Repo(RepoData):
    """RepoData + frame timestamps for video decoding."""

    def __init__(self, repo: str, cfg, train_cfg) -> None:
        super().__init__(repo, cfg, train_cfg)
        files = sorted(Path(repo, "data").rglob("*.parquet"))
        df = pd.concat(pd.read_parquet(f, columns=["episode_index", "timestamp"]) for f in files)
        self.timestamps = {int(ep): g["timestamp"].to_numpy(np.float64) for ep, g in df.groupby("episode_index")}
        self._meta = None

    def video_frames(self, episode: int, frames: list[int], camera: str, backend: str) -> torch.Tensor:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.datasets.video_utils import decode_video_frames

        if self._meta is None:
            self._meta = LeRobotDatasetMetadata(self.repo)
        path = Path(self.repo) / self._meta.get_video_file_path(episode, camera)
        info = self._meta.episodes[episode]
        ts = [float(info[f"videos/{camera}/from_timestamp"]) + float(self.timestamps[episode][f]) for f in frames]
        return decode_video_frames(path, ts, 1e-4, backend)  # [S, C, H, W] in [0, 1]


class BPV4CDXBDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: BPV4CDXBDatasetConfig, train_cfg, policy_cfg) -> None:
        self.cfg, self.policy_cfg = cfg, policy_cfg
        self.chunk = int(policy_cfg.chunk_size)
        self.demo_tokens = int(policy_cfg.num_demo_tokens) + 2 if policy_cfg.demo_input == "video" else 0
        self.markers = bool(policy_cfg.obs_marker_tokens)
        self.backend = getattr(cfg, "video_backend", None) or "torchcodec"
        self.index = json.loads(Path(cfg.index).read_text())
        root = Path(cfg.data_root or self.index["root"])
        eps, split = self.index["episodes"], self.index["split"]

        def keep(eid: str) -> bool:
            return cfg.split == "all" or split[eps[eid]["anchor"]] == cfg.split

        if cfg.fixed_pairs:
            self.items = [(q, d) for q, d in cfg.fixed_pairs]
        else:
            queries = [e for e in sorted(eps) if keep(e) and (not cfg.query_whitelist or e in cfg.query_whitelist)]
            self.items = [(q, None) for q in queries]
        repos = sorted({eps[e]["repo"] for pair in self.items for e in pair if e} |
                       {eps[d["demo"]]["repo"] for q, _ in self.items for d in self.index["matched"][q]})
        self.repos = {r: _Repo(str(root / r), cfg, train_cfg) for r in repos}
        self.zdemo_dir = Path(cfg.zdemo_dir)
        manifest = self.zdemo_dir / "manifest.json"
        self.zdemo_manifest = json.loads(manifest.read_text()) if manifest.is_file() else {}
        # contrast groups among the selected queries: anchor -> {candidate set: [query item indices]}
        self.groups: dict[str, dict[str, list[int]]] = {}
        for i, (q, _) in enumerate(self.items):
            self.groups.setdefault(eps[q]["anchor"], {}).setdefault(eps[q]["set"], []).append(i)
        self.groups = {a: g for a, g in self.groups.items() if len(g) > 1}
        self._processor = None
        self._token_ids = None
        self.num_episodes = len(self.items)
        self.num_frames = sum(int(eps[q]["length"]) for q, _ in self.items)
        logging.info("bpv4cdx_b dataset split=%s: %d query episodes, %d repos, %d contrast groups, Z_demo %s",
                     cfg.split, len(self.items), len(self.repos), len(self.groups), self.zdemo_dir)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def stats(self):
        r = next(iter(self.repos.values()))
        return {r.robot_type: r.stats}

    # ------------------------------------------------------------------ pieces (also used by the G9 evaluation)
    def pick_demo(self, query: str) -> str:
        """Draw a demo from the query's matched pool. Matched entries must be same-task (dev doc §3.2)."""
        cands = self.index["matched"][query]
        demo = cands[int(torch.randint(len(cands), ()))]["demo"]
        eps = self.index["episodes"]
        # same-task hard check: prefer explicit task text / task id when present
        q_task = eps[query].get("task_text") or eps[query].get("task")
        d_task = eps[demo].get("task_text") or eps[demo].get("task")
        if q_task is not None and d_task is not None and q_task != d_task:
            raise RuntimeError(f"matched demo crosses task: query={query!r} ({q_task!r}) demo={demo!r} ({d_task!r})")
        return demo

    def query_frames(self, query: str, *, fixed: bool = False) -> list[int]:
        length = int(self.index["episodes"][query]["length"])
        s = int(self.cfg.queries_per_item)
        if fixed or self.cfg.all_frames:
            return [int(x) for x in np.linspace(0, max(0, length - 2), s).round()]
        edges = np.linspace(0, length, s + 1).astype(int)
        return [int(torch.randint(edges[j], max(edges[j] + 1, edges[j + 1]), ()).item()) for j in range(s)]

    def zdemo(self, demo: str) -> torch.Tensor:
        e = self.index["episodes"][demo]
        rec = torch.load(self.zdemo_dir / e["repo"] / f"ep{int(e['episode']):05d}.pt", map_location="cpu",
                         weights_only=False)
        return rec["z_demo"]

    def current_inputs(self, query: str, frames: list[int]) -> dict[str, torch.Tensor]:
        from lerobot.transforms.utils import resize_with_pad

        if self._processor is None:
            from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision

            self._processor, self._token_ids = load_qwen35_vision(self.cfg.qwen3_5_processor_path)
        e = self.index["episodes"][query]
        repo, size = self.repos[e["repo"]], int(self.cfg.image_size)
        per_cam = [repo.video_frames(int(e["episode"]), frames, cam, self.backend) for cam in self.cfg.camera_keys]
        images = [resize_with_pad(per_cam[c][j][None], size, size)[0] for j in range(len(frames)) for c in range(len(per_cam))]
        out = self._processor(images, do_rescale=False)
        grid = torch.as_tensor(out["image_grid_thw"], dtype=torch.long)
        tid = self._token_ids
        ids = [PLACEHOLDER_ID] * self.demo_tokens + [PLACEHOLDER_ID] * self.markers
        types = [0] * len(ids)
        for g in grid[: len(per_cam)]:
            n_tok = int(g.prod()) // tid.spatial_merge_size**2
            ids += [tid.vision_start_token_id] + [tid.image_token_id] * n_tok + [tid.vision_end_token_id]
            types += [0] + [1] * n_tok + [0]
        ids += [PLACEHOLDER_ID] * self.markers
        types += [0] * self.markers
        s = len(frames)
        input_ids = torch.tensor(ids)[None].expand(s, -1).clone()
        return {"obs.pixel_values": torch.as_tensor(out["pixel_values"]), "obs.image_grid_thw": grid,
                "obs.input_ids": input_ids, "obs.mm_token_type_ids": torch.tensor(types)[None].expand(s, -1).clone(),
                "obs.attention_mask": torch.ones_like(input_ids, dtype=torch.bool)}

    def build(self, query: str, demo: str, frames: list[int], *, contrast: bool = False) -> dict:
        e = self.index["episodes"][query]
        repo = self.repos[e["repo"]]
        out = {k: [] for k in (OBS_STATE, ACTION, ACTION_VALID_MASK)}
        for f in frames:
            q = repo.query(int(e["episode"]), f, self.chunk)
            for k in out:
                out[k].append(q[k])
        batch = {k: torch.stack(v) for k, v in out.items()}
        batch.update(self.current_inputs(query, frames))
        if self.demo_tokens:
            batch["demo.z"] = self.zdemo(demo)
        batch.update({"query.frame": torch.tensor(frames, dtype=torch.long),
                      "query.progress": torch.tensor([f / max(1, int(e["length"]) - 1) for f in frames]),
                      "pair.contrast": torch.tensor(contrast)})
        return batch

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        contrast = isinstance(i, tuple)
        i = i[0] if contrast else i
        query, demo = self.items[i]
        return self.build(query, demo or self.pick_demo(query), self.query_frames(query), contrast=contrast)

    def make_train_sampler(self, seed: int):
        if self.cfg.split != "train" or self.cfg.fixed_pairs or not self.groups:
            return None
        return ContrastSampler(self, seed, float(self.cfg.contrast_prob))


class ContrastSampler(torch.utils.data.Sampler):
    """Single queries mixed with contrast groups: one query of every goal of an anchor, yielded consecutively (tagged
    as (index,) tuples so the item records pair.contrast). Same order on every rank."""

    def __init__(self, ds: BPV4CDXBDataset, seed: int, prob: float) -> None:
        self.ds, self.seed, self.prob, self.epoch = ds, seed, prob, 0
        self.anchors = sorted(ds.groups)

    def __len__(self) -> int:
        return len(self.ds)

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        self.epoch += 1
        n, out = len(self.ds), []
        while len(out) < n:
            if rng.random() < self.prob:
                group = self.ds.groups[self.anchors[int(rng.integers(len(self.anchors)))]]
                out += [(int(rng.choice(members)),) for members in group.values()]
            else:
                out.append(int(rng.integers(n)))
        return iter(out[:n])


def make_bpv4cdxb_dataset(train_cfg) -> tuple[BPV4CDXBDataset, dict]:
    ds = BPV4CDXBDataset(train_cfg.dataset, train_cfg, train_cfg.policy)
    return ds, ds.stats
