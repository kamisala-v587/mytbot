"""bpv4cdx Stage B main-path dataset (`bpv4cdx Stage B 开发文档 v1.md` 3.2 / 3.3): same-task pairing over the
pre-training repos, any embodiment.

Index: BPV4CDXB.pairs_pretrain (buckets = repo x identical description text, > 3 trajectories, AE-3 split).
One item = query episode j of the split x `queries_per_item` time steps; the demo i is drawn from j's bucket and
split (i != j, Z_demo cached), from the other arm with `cross_arm_prob` when the bucket has one. Episodes whose
Z_demo is not cached yet are left out of the demo pools (training on a partial cache), queries without any demo
are left out. Output keys are the ones of BPV4CDXB.dataset (the model is shared), plus pair.cross_arm.
Cameras: the embodiment's image0..image2 sources (BPV4CDX.embodiment); a missing camera is a black image.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.BPV4CDX.paths import resolve_rel
from lerobot.policies.BPV4CDX.repo_data import RepoData
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_bpv4cdxb import BPV4CDXBDatasetConfig
from .dataset import PLACEHOLDER_ID

ARMS = ("L", "R")


class BPV4CDXBPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: BPV4CDXBDatasetConfig, train_cfg, policy_cfg) -> None:
        self.cfg, self.policy_cfg = cfg, policy_cfg
        self.chunk = int(policy_cfg.chunk_size)
        self.demo_tokens = int(policy_cfg.num_demo_tokens) + 2 if policy_cfg.demo_input == "video" else 0
        self.markers = bool(policy_cfg.obs_marker_tokens)
        self.backend = getattr(cfg, "video_backend", None) or "torchcodec"
        self.index = json.loads(Path(cfg.index).read_text())
        buckets = self.index["buckets"][: cfg.max_buckets] if cfg.max_buckets else self.index["buckets"]
        roots = [r for r in (cfg.data_root.split(":") if cfg.data_root else self.index["roots"]) if r]
        used = sorted({b["repo"] for b in buckets})
        infos = [self.index["repos"][i] for i in used]
        paths = [resolve_rel(r["rel"], roots) for r in infos]
        missing = [r["rel"] for r, p in zip(infos, paths) if p is None]
        if missing:
            raise FileNotFoundError(f"{len(missing)} index repos not found under {roots}: {missing[:3]}")
        import datasets
        from tqdm import tqdm

        datasets.disable_progress_bars()
        tqdm.get_lock()
        store = cfg.array_store or None
        with ThreadPoolExecutor(max(1, int(cfg.load_threads))) as pool:
            repos = list(pool.map(lambda p: RepoData(p, cfg, train_cfg, store_root=store), paths))
        remap = {old: new for new, old in enumerate(used)}
        self.repos = repos
        self.zdemo_dir = Path(cfg.zdemo_dir)
        manifest = self.zdemo_dir / "manifest.json"
        self.zdemo_manifest = json.loads(manifest.read_text()) if manifest.is_file() else {}
        have = {}
        for ri, repo in enumerate(repos):
            d = self.zdemo_dir / repo.rel
            have[ri] = {int(n[2:7]) for n in os.listdir(d) if n.endswith(".pt")} if d.is_dir() else set()
        # buckets: repo, text, per split the demo pool (Z_demo cached) and the arm of every episode
        self.buckets = []
        for b in buckets:
            ri = remap[b["repo"]]
            self.buckets.append({
                "repo": ri, "text": b["text"], "length": {int(k): int(v) for k, v in b["length"].items()},
                "arm": {int(k): v for k, v in b["arm"].items()},
                "pool": {"train": [e for e in b["train"] if e in have[ri]], "val": [e for e in b["heldout"] if e in have[ri]]},
                "eps": {"train": list(b["train"]), "val": list(b["heldout"])},
            })
        splits = ["train", "val"] if cfg.split == "all" else [cfg.split]
        self.items: list[tuple[int, int, str]] = []
        no_demo = 0
        for bi, b in enumerate(self.buckets):
            for sp in splits:
                pool = set(b["pool"][sp])
                for e in b["eps"][sp]:
                    if pool - {e}:
                        self.items.append((bi, e, sp))
                    else:
                        no_demo += 1
        self._processor = None
        self._token_ids = None
        self._meta: dict[int, object] = {}
        self.num_episodes = len(self.items)
        self.num_frames = sum(self.buckets[bi]["length"][e] for bi, e, _ in self.items)
        logging.info("bpv4cdx_b same_task split=%s: %d query episodes (%d without a cached demo), %d buckets, %d repos, "
                     "Z_demo %s", cfg.split, len(self.items), no_demo, len(self.buckets), len(repos), self.zdemo_dir)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def stats(self):
        out = {}
        for repo in self.repos:
            out.setdefault(repo.robot_type, repo.stats)
        return out

    # ------------------------------------------------------------------ pieces (also used by the G9 evaluation)
    def key(self, bi: int, ep: int) -> str:
        return f"{self.repos[self.buckets[bi]['repo']].rel}|{int(ep)}"

    def demo_pool(self, bi: int, ep: int, split: str, arm: str | None = None, *, other_arm: bool = False) -> list[int]:
        b = self.buckets[bi]
        pool = [d for d in b["pool"][split] if d != ep]
        if other_arm:
            qa = b["arm"].get(ep)
            pool = [d for d in pool if qa in ARMS and b["arm"].get(d) in ARMS and b["arm"][d] != qa]
        elif arm is not None:
            pool = [d for d in pool if b["arm"].get(d) == arm]
        return pool

    def pick_demo(self, bi: int, ep: int, split: str) -> tuple[int, bool]:
        """Demo of the same bucket (identical description text) and split; (episode, cross-arm)."""
        b = self.buckets[bi]
        cross = self.demo_pool(bi, ep, split, other_arm=True)
        if cross and float(torch.rand(())) < float(self.cfg.cross_arm_prob):
            return cross[int(torch.randint(len(cross), ()))], True
        pool = self.demo_pool(bi, ep, split)
        d = pool[int(torch.randint(len(pool), ()))]
        qa, da = b["arm"].get(ep), b["arm"].get(d)
        return d, bool(qa in ARMS and da in ARMS and qa != da)

    def query_frames(self, bi: int, ep: int, *, fixed: bool = False) -> list[int]:
        length = self.buckets[bi]["length"][ep]
        s = int(self.cfg.queries_per_item)
        if fixed or self.cfg.all_frames:
            return [int(x) for x in np.linspace(0, max(0, length - 2), s).round()]
        edges = np.linspace(0, length, s + 1).astype(int)
        return [int(torch.randint(edges[j], max(edges[j] + 1, edges[j + 1]), ()).item()) for j in range(s)]

    def zdemo(self, bi: int, ep: int) -> torch.Tensor:
        rel = self.repos[self.buckets[bi]["repo"]].rel
        rec = torch.load(self.zdemo_dir / rel / f"ep{int(ep):05d}.pt", map_location="cpu", weights_only=False)
        return rec["z_demo"]

    def _video_frames(self, ri: int, episode: int, frames: list[int], camera: str) -> torch.Tensor:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.datasets.video_utils import decode_video_frames

        repo = self.repos[ri]
        if ri not in self._meta:
            self._meta[ri] = LeRobotDatasetMetadata(repo.repo)
        meta = self._meta[ri]
        path = Path(repo.repo) / meta.get_video_file_path(episode, camera)
        start = float(meta.episodes[episode][f"videos/{camera}/from_timestamp"])
        ts = [start + f / repo.fps for f in frames]
        return decode_video_frames(path, ts, 1e-4, self.backend)  # [S, C, H, W] in [0, 1]

    def current_inputs(self, bi: int, ep: int, frames: list[int]) -> dict[str, torch.Tensor]:
        from lerobot.transforms.utils import resize_with_pad

        if self._processor is None:
            from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision

            self._processor, self._token_ids = load_qwen35_vision(self.cfg.qwen3_5_processor_path)
        ri = self.buckets[bi]["repo"]
        size = int(self.cfg.image_size)
        per_cam = []
        for cam in self.repos[ri].emb.cameras:
            if cam is None:
                per_cam.append(None)
                continue
            per_cam.append(resize_with_pad(self._video_frames(ri, ep, frames, cam), size, size))
        black = torch.zeros(3, size, size)
        images = [black if c is None else c[j] for j in range(len(frames)) for c in per_cam]
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

    def build(self, bi: int, ep: int, demo: tuple[int, int], frames: list[int], *, cross: bool = False) -> dict:
        """demo = (bucket, episode) of the demo (blank: the model's blank_demo flag zeroes Z_demo)."""
        repo = self.repos[self.buckets[bi]["repo"]]
        out = {k: [] for k in (OBS_STATE, ACTION, ACTION_VALID_MASK)}
        for f in frames:
            q = repo.query(int(ep), f, self.chunk)
            for k in out:
                out[k].append(q[k])
        batch = {k: torch.stack(v) for k, v in out.items()}
        batch.update(self.current_inputs(bi, ep, frames))
        if self.demo_tokens:
            batch["demo.z"] = self.zdemo(*demo)
        length = self.buckets[bi]["length"][ep]
        batch.update({"query.frame": torch.tensor(frames, dtype=torch.long),
                      "query.progress": torch.tensor([f / max(1, length - 1) for f in frames]),
                      "pair.contrast": torch.tensor(False), "pair.cross_arm": torch.tensor(cross)})
        return batch

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        bi, ep, split = self.items[i]
        demo, cross = self.pick_demo(bi, ep, split)
        return self.build(bi, ep, (bi, demo), self.query_frames(bi, ep), cross=cross)

    def make_train_sampler(self, seed: int):
        return None

    def make_heldout_monitor(self):
        if not self.cfg.heldout_monitor or self.cfg.split != "train":
            return None
        from .gate_g9 import G9Monitor

        return G9Monitor(self)
