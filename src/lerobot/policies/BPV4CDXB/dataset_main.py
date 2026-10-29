"""Stage B main-path dataset (StageB实施.md S3) on the bucket index of `pairs_main.py`.

One item = query episode j (drawn length-weighted by the train sampler) x `queries_per_item` time steps of j, with a
demo i != j drawn from j's bucket (same robot type + identical task text, same split). Same item format as the P1
dataset (`dataset.py`), so the model is unchanged:
  demo.z            Z_demo of i (`<index zdemo_dir>/<rel>/epXXXXX.pt`)
  obs.*             three current images of j per time step; the robot's cameras are mapped to the 3 slots
                    (head, left / single wrist, right wrist) as in Stage A, a missing camera is a black image
  observation.state / action / action_is_valid   j's state and 50-step chunk (q35 transforms of the robot type)
Repo metadata is loaded by worker processes (it is pure Python); state / action come from the AE-3 array store.
"""

from __future__ import annotations

import copy
import json
import logging
import multiprocessing
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from lerobot.policies.BPV4CDX.repo_data import RepoData
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION, OBS_STATE

from .dataset import PLACEHOLDER_ID

logger = logging.getLogger(__name__)
FRAME_RETRIES = 8


class FrameDecodeError(RuntimeError):
    """A requested video frame is missing from the file or farther than `video_tolerance_s` from its time."""


def _load_repo(job) -> RepoData:
    path, cfg, dataset_cfg, store = job
    repo = RepoData(path, cfg, SimpleNamespace(dataset=dataset_cfg), store_root=store, load_arrays=False)
    info = json.loads(Path(path, "meta/info.json").read_text())
    cams = [c for c in repo.emb.cameras if c]
    cols = ["episode_index"] + [f"videos/{c}/{k}" for c in cams for k in ("chunk_index", "file_index", "from_timestamp")]
    ep = pd.concat(pd.read_parquet(f, columns=cols) for f in sorted(Path(path, "meta/episodes").rglob("*.parquet")))
    repo.video = {"template": info["video_path"], "fps": float(info["fps"]),
                  "row": {int(e): i for i, e in enumerate(ep["episode_index"])},
                  "cams": {c: (ep[f"videos/{c}/chunk_index"].to_numpy(np.int64), ep[f"videos/{c}/file_index"].to_numpy(np.int64),
                               ep[f"videos/{c}/from_timestamp"].to_numpy(np.float64)) for c in cams}}
    return repo


class BPV4CDXBMainDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, train_cfg, policy_cfg, index: dict) -> None:
        from lerobot.policies.BPV4CDX.paths import resolve_rel

        self.cfg, self.policy_cfg = cfg, policy_cfg
        self.chunk = int(policy_cfg.chunk_size)
        self.demo_tokens = int(policy_cfg.num_demo_tokens) + 2 if policy_cfg.demo_input == "video" else 0
        self.markers = bool(policy_cfg.obs_marker_tokens)
        self.backend = getattr(cfg, "video_backend", None) or "torchcodec"
        self.split = "val" if cfg.split == "val" else "train"
        self.buckets = index["buckets"]
        self.zdemo_dir = Path(cfg.zdemo_dir or index["zdemo_dir"])
        eps = index["episodes"]
        self.items = [eid for b in self.buckets for eid in b[self.split]]
        if cfg.query_whitelist:
            keep = set(cfg.query_whitelist)
            self.items = [e for e in self.items if e in keep]
        self.ep = {eid: eps[eid] for b in self.buckets for eid in b[self.split]}
        rels = sorted({self.ep[e][0] for e in self.ep})
        roots = [r for r in (index.get("data_roots") or [cfg.data_root]) if r]
        paths = {rel: resolve_rel(rel, roots) for rel in rels}
        missing = [r for r, p in paths.items() if p is None]
        if missing:
            raise FileNotFoundError(f"{len(missing)} repos not found under {roots}: {missing[:3]}")
        store = index.get("array_store") or None
        self.store = store
        jobs = [(paths[rel], cfg, cfg, store) for rel in rels]
        procs = min(int(index.get("load_procs", 32)), len(jobs))
        with ProcessPoolExecutor(procs, mp_context=multiprocessing.get_context("spawn")) as pool:
            loaded = list(pool.map(_load_repo, jobs, chunksize=max(1, len(jobs) // (procs * 4))))
        self.repos = dict(zip(rels, loaded))
        for repo in self.repos.values():
            if repo.states is None:
                repo.open_arrays(store)
        self._processor = None
        self._token_ids = None
        self.num_episodes = len(self.items)
        self.num_frames = sum(int(self.ep[e][2]) for e in self.items)
        logging.info("bpv4cdx_b main dataset split=%s: %d query episodes, %d buckets, %d repos, Z_demo %s",
                     self.split, len(self.items), sum(bool(b[self.split]) for b in self.buckets), len(self.repos),
                     self.zdemo_dir)

    def __len__(self) -> int:
        return len(self.items)

    # DataLoader workers are spawned (see make_main_dataset): the pickled copy carries no memory-mapped arrays
    # (pickling a memmap copies the whole array); each worker reopens them
    def __getstate__(self):
        state = self.__dict__.copy()
        repos = {}
        for rel, r in self.repos.items():
            c = copy.copy(r)
            c.states = c.actions = None
            repos[rel] = c
        state["repos"], state["_processor"], state["_token_ids"] = repos, None, None
        state.pop("_decoders", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for r in self.repos.values():
            if r.states is None:
                r.open_arrays(self.store)

    @property
    def stats(self):
        out = {}
        for r in self.repos.values():
            out.setdefault(r.robot_type, r.stats)
        return out

    # ------------------------------------------------------------------ pieces (also used by the G9 evaluation)
    def pick_demo(self, query: str, rng: np.random.Generator | None = None) -> str:
        """Another episode of the query's bucket and split (same robot type + identical task text)."""
        bi = self.ep[query][3]
        pool = self.buckets[bi][self.split]
        while True:
            k = int(rng.integers(len(pool))) if rng is not None else int(torch.randint(len(pool), ()))
            demo = pool[k]
            if demo != query:
                break
        if self.ep[demo][3] != bi:
            raise RuntimeError(f"demo crosses bucket: {query} -> {demo}")
        return demo

    def query_frames(self, query: str, *, fixed: bool = False) -> list[int]:
        length = int(self.ep[query][2])
        s = int(self.cfg.queries_per_item)
        if fixed or self.cfg.all_frames:
            return [int(x) for x in np.linspace(0, max(0, length - 2), s).round()]
        edges = np.linspace(0, length, s + 1).astype(int)
        return [int(torch.randint(edges[j], max(edges[j] + 1, edges[j + 1]), ()).item()) for j in range(s)]

    def zdemo(self, demo: str) -> torch.Tensor:
        rel, ep = self.ep[demo][0], int(self.ep[demo][1])
        return torch.load(self.zdemo_dir / rel / f"ep{ep:05d}.pt", map_location="cpu", weights_only=False)["z_demo"]

    def demo_p5(self, demo: str) -> dict[str, torch.Tensor]:
        """B-4: the demo's AE-3 P5 record for online encoding (`demo_features="p5"`)."""
        rel, ep = self.ep[demo][0], int(self.ep[demo][1])
        rec = torch.load(Path(self.cfg.demo_cache_dir) / rel / f"ep{ep:05d}.pt", map_location="cpu", weights_only=False)
        return {"demo.feats": rec["feats"], "demo.time": rec["time_ratio"].float(),
                "demo.camera_valid": rec["camera_valid"].bool()}

    def _frames(self, repo: RepoData, episode: int, frames: list[int], cam: str) -> torch.Tensor:
        # seek_mode="approximate": the "exact" mode scans the whole concatenated v3 file on open (about 2 s for
        # 160k frames); on these constant-fps videos both modes return identical frames (checked on 5 sources)
        from torchcodec.decoders import VideoDecoder

        v = repo.video
        chunk, file, start = (a[v["row"][episode]] for a in v["cams"][cam])
        path = str(Path(repo.repo) / v["template"].format(video_key=cam, chunk_index=int(chunk), file_index=int(file)))
        cache = self.__dict__.setdefault("_decoders", OrderedDict())
        if path in cache:
            cache.move_to_end(path)
        else:
            cache[path] = VideoDecoder(path, seek_mode="approximate")
            if len(cache) > 12:
                cache.popitem(last=False)
        dec = cache[path]
        fps = dec.metadata.average_fps
        want = [float(start) + f / v["fps"] for f in frames]
        idx = [round(t * fps) for t in want]
        last = int(dec.metadata.num_frames or 0) - 1  # header estimate in approximate mode, may still be too high
        if last >= 0:
            idx = [min(i, last) for i in idx]
        where = f"repo={repo.rel} episode={episode} camera={cam} frames={frames} video_idx={idx} last={last} file={path}"
        try:
            out = dec.get_frames_at(indices=idx)
        except RuntimeError as e:
            raise FrameDecodeError(f"{where}: {e}") from e
        gap = float((out.pts_seconds.double() - torch.tensor(want, dtype=torch.float64)).abs().max())
        tol = float(getattr(self.cfg, "video_tolerance_s", 0.5))
        if gap > tol:
            raise FrameDecodeError(f"{where}: decoded pts off by {gap:.3f} s > tolerance {tol} s")
        return out.data.float() / 255.0  # [S, C, H, W] in [0, 1]

    def current_inputs(self, query: str, frames: list[int]) -> dict[str, torch.Tensor]:
        from lerobot.transforms.utils import resize_with_pad

        if self._processor is None:
            from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision

            torch.set_num_threads(2)
            self._processor, self._token_ids = load_qwen35_vision(self.cfg.qwen3_5_processor_path)
        rel, ep = self.ep[query][0], int(self.ep[query][1])
        repo, size = self.repos[rel], int(self.cfg.image_size)
        per_slot = [self._frames(repo, ep, frames, cam) if cam else None for cam in repo.emb.cameras]
        black = torch.zeros(3, size, size)
        images = [resize_with_pad(per_slot[c][j][None], size, size)[0] if per_slot[c] is not None else black
                  for j in range(len(frames)) for c in range(len(per_slot))]
        out = self._processor(images, do_rescale=False)
        grid = torch.as_tensor(out["image_grid_thw"], dtype=torch.long)
        tid = self._token_ids
        ids = [PLACEHOLDER_ID] * self.demo_tokens + [PLACEHOLDER_ID] * self.markers
        types = [0] * len(ids)
        for g in grid[: len(per_slot)]:
            n_tok = int(g.prod()) // tid.spatial_merge_size**2
            ids += [tid.vision_start_token_id] + [tid.image_token_id] * n_tok + [tid.vision_end_token_id]
            types += [0] + [1] * n_tok + [0]
        ids += [PLACEHOLDER_ID] * self.markers
        types += [0] * self.markers
        s = len(frames)
        input_ids = torch.tensor(ids)[None].expand(s, -1).clone()
        pixels = torch.as_tensor(out["pixel_values"])
        if getattr(self.cfg, "pixel_transfer_dtype", "float32") == "bfloat16":
            pixels = pixels.to(torch.bfloat16)  # the model casts to the ViT dtype (bf16) first: identical ViT input
        return {"obs.pixel_values": pixels, "obs.image_grid_thw": grid,
                "obs.input_ids": input_ids, "obs.mm_token_type_ids": torch.tensor(types)[None].expand(s, -1).clone(),
                "obs.attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
                "obs.camera_valid": torch.tensor([c is not None for c in repo.emb.cameras])}

    def build(self, query: str, demo: str, frames: list[int]) -> dict:
        rel, ep = self.ep[query][0], int(self.ep[query][1])
        repo = self.repos[rel]
        out = {k: [] for k in (OBS_STATE, ACTION, ACTION_VALID_MASK)}
        for f in frames:
            q = repo.query(ep, f, self.chunk)
            for k in out:
                out[k].append(q[k])
        batch = {k: torch.stack(v) for k, v in out.items()}
        batch.update(self.current_inputs(query, frames))
        if self.demo_tokens:
            if getattr(self.cfg, "demo_features", "zdemo") == "p5":
                batch.update(self.demo_p5(demo))
            else:
                batch["demo.z"] = self.zdemo(demo)
        length = int(self.ep[query][2])
        batch.update({"query.frame": torch.tensor(frames, dtype=torch.long),
                      "query.progress": torch.tensor([f / max(1, length - 1) for f in frames]),
                      "pair.contrast": torch.tensor(False)})
        return batch

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        query = self.items[i]
        for attempt in range(1, FRAME_RETRIES + 1):
            try:
                return self.build(query, self.pick_demo(query), self.query_frames(query))
            except FrameDecodeError as e:
                # spawned workers have no logging handlers: the last-resort handler writes WARNING to stderr
                logger.warning("[BPV4CDXB frame-skip] item %d query %s attempt %d/%d, redrawing: %s",
                               i, query, attempt, FRAME_RETRIES, e)
                query = self.items[int(torch.randint(len(self.items), ()))]
        raise RuntimeError(f"[BPV4CDXB frame-skip] item {i}: {FRAME_RETRIES} draws failed to decode")

    def make_train_sampler(self, seed: int):
        """Query episodes drawn proportionally to their length (= uniform over frames), as in Stage A."""
        if self.split != "train":
            return None
        from lerobot.policies.BPV4CDX.dataset import LengthTaskSampler

        lengths = [int(self.ep[e][2]) for e in self.items]
        buckets = [str(self.ep[e][3]) for e in self.items]
        # 1000 passes: one DataLoader epoch outlasts any planned run (~4.5M steps at 128 queries per step), so the
        # spawned workers never restart; a finite length is still needed for the accelerate batch sharding
        return MultiPassSampler(LengthTaskSampler(lengths, buckets, seed), int(getattr(self, "sampler_passes", 1000)))


class MultiPassSampler(torch.utils.data.Sampler[int]):
    """`passes` epochs of the base sampler (each a fresh draw) as one DataLoader epoch: the spawned workers, which
    take minutes to start (they reload the dataset), restart once per `passes` passes instead of every pass."""

    def __init__(self, base, passes: int) -> None:
        self.base, self.passes = base, max(1, int(passes))

    def __len__(self) -> int:
        return len(self.base) * self.passes

    def __iter__(self):
        for _ in range(self.passes):
            yield from iter(self.base)


def make_main_dataset(train_cfg, index: dict):
    # forked DataLoader workers deadlock once the trainer has initialised CUDA (image processor / video decoding);
    # the DataLoader uses the default start method, so make it spawn
    torch.multiprocessing.set_start_method("spawn", force=True)
    ds = BPV4CDXBMainDataset(train_cfg.dataset, train_cfg, train_cfg.policy, index)
    return ds, ds.stats
