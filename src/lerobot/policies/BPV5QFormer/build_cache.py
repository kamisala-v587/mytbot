"""BPv5 demonstration cache: wall ViT P5 features of 32 S-U frames x 3 camera slots for every 80-D episode.

Per episode: states (state_80D, state_mask) -> S-U on the 80-D state (sampling80d) -> the 32 frames of the head /
left / right slot cameras (data80d.CAMERAS) decoded directly with torchcodec -> resize_with_pad to 224 -> Qwen3.5
processor -> wall ViT (only model.visual of the exported VLM) -> 64 tokens x 2560 per image -> P5 pooling (2x2 block
means + global mean) -> <cache>/<relative repo id>/epXXXXXX.pt, feats [32, 3, 5, 2560] bf16 (a missing camera slot is
zero with camera_valid False). Image path identical to AE-3 (BPV4CDX.build_cache) apart from the ViT.

Shard r of w gets a contiguous block of the (repo, episode) list; existing files are skipped; failing episodes go to
<cache>/_failures/rank<r>.jsonl. Launcher: tools/bpv5/build_cache.sh.

usage: python -m lerobot.policies.BPV5QFormer.build_cache --cache_dir <dir> [--sources robotwin,dom] [--rank 0 --world 1]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.BPV4CDX.dataset import pool_spatial
from lerobot.transforms.utils import resize_with_pad

from . import data80d as D
from .sampling80d import SAMPLER, SAMPLER_VERSION, sample_su80

VLM = "/vla/workspace/models/wall-oss/wall-oss-1.0-lite-vlm"


def cache_path(cache_dir: str | Path, rel: str, ep: int) -> Path:
    return Path(cache_dir) / rel / f"ep{int(ep):06d}.pt"


def _state_std_for_su(rel: str) -> np.ndarray:
    """S-U speed normalization: prefer group norm stats, else leaf meta/stats.json, else ones."""
    try:
        key = "__single__" if D.STATS_FILE is not None else rel
        idx = D.stats_index()
        if key in idx:
            return np.asarray(D.group_stats(idx[key])["state_80D"]["std"], dtype=np.float32)
    except Exception:  # noqa: BLE001
        pass
    leaf_stats = D.ROOT / rel / "meta" / "stats.json"
    if leaf_stats.is_file():
        try:
            st = json.loads(leaf_stats.read_text())
            if "state_80D" in st and "std" in st["state_80D"]:
                return np.asarray(st["state_80D"]["std"], dtype=np.float32)
        except Exception:  # noqa: BLE001
            pass
    # robotwin2.5 native-14 (no 80D group stats yet): uniform scale is enough for activity detection
    return np.ones(D.ACTION_DIM, dtype=np.float32)


def _batched_attention(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
    """Qwen3_5VisionAttention.forward for images of one length: one SDPA call over [images, heads, L, d] instead of
    transformers' per-image loop (its SDPA path without flash-attn); same math. Other lengths: the original."""
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if int(lengths.min()) != int(lengths.max()):
        return type(self).forward(self, hidden_states, cu_seqlens, position_embeddings=position_embeddings, **kwargs)
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision

    n, length = len(lengths), int(lengths[0])
    q, k, v = self.qkv(hidden_states).reshape(hidden_states.shape[0], 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    q, k, v = (t.reshape(n, length, self.num_heads, -1).transpose(1, 2) for t in (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scaling)
    return self.proj(out.transpose(1, 2).reshape(hidden_states.shape[0], -1))


def load_vision(vlm_path: str, device, batched_attention: bool = True):
    """The ViT of the exported Qwen3.5 checkpoint only (same outputs as the full model's get_image_features)."""
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    cfg = AutoConfig.from_pretrained(vlm_path).vision_config
    cfg._attn_implementation = "sdpa"
    vision = Qwen3_5VisionModel(cfg)
    with safe_open(str(Path(vlm_path) / "model.safetensors"), "pt") as f:
        state = {k[len("model.visual."):]: f.get_tensor(k) for k in f.keys() if k.startswith("model.visual.")}
    vision.load_state_dict(state, strict=True)
    if batched_attention:
        import types

        for block in vision.blocks:
            block.attn.forward = types.MethodType(_batched_attention, block.attn)
    return vision.to(device, torch.bfloat16).eval()


def vision_tokens(vision, pixels: torch.Tensor, grids: torch.Tensor) -> torch.Tensor:
    """[N images] -> [N, 64, 2560] (merger output, as Qwen3_5Model.get_image_features)."""
    out = vision(pixels, grid_thw=grids).pooler_output
    merge = int(vision.spatial_merge_size)
    return torch.stack(torch.split(out, [int(v) for v in (grids.prod(-1) // merge**2).tolist()]))


class EpisodeFrames(torch.utils.data.Dataset):
    """One item per (repo, episode): sampling plan + processor outputs for the 32 slots x the present cameras."""

    def __init__(self, items: list[tuple[str, int]], num_slots: int, size: int, vlm_path: str,
                 gpu_decode_pixels: int = 1280 * 720) -> None:
        self.items, self.k, self.size, self.vlm_path = items, num_slots, size, vlm_path
        self.gpu_decode_pixels = gpu_decode_pixels
        self._processor = None
        self._states = D.StateReader()
        self._decoders: dict[tuple[str, str], object] = {}
        self._cpu_only: set[str] = set()  # videos NVDEC could not decode (e.g. some galaxea streams)
        self._threads_set = False

    def __len__(self) -> int:
        return len(self.items)

    def _decoder(self, path: Path, device: str):
        """Cached decoders: a few CPU ones; one GPU one (NVDEC sessions and their frame pools hold GPU memory)."""
        from torchcodec.decoders import VideoDecoder

        key = (str(path), device)
        if key not in self._decoders:
            same = [k for k in self._decoders if k[1] == device]
            if len(same) >= (1 if device == "cuda" else 4):
                self._decoders.pop(same[0])
            self._decoders[key] = VideoDecoder(key[0], seek_mode="exact", device=device)
        return self._decoders[key]

    def _device(self, rel: str, cam: str, path: Path) -> str:
        """NVDEC for large videos (abc 1920x1200 HEVC: 36 s -> 3 s per episode on CPU vs GPU), CPU otherwise."""
        shape = D.info(rel)["features"][cam].get("shape") or [0, 0, 0]
        h, w = sorted(int(x) for x in shape)[-2:]
        big = self.gpu_decode_pixels and h * w >= self.gpu_decode_pixels
        return "cuda" if big and str(path) not in self._cpu_only else "cpu"

    def _frames(self, path: Path, device: str, t0: float, fps: float, plan_frames) -> torch.Tensor:
        """The plan's frames of one camera, resized with pad to [K, 3, size, size] float on the CPU."""
        decoder = self._decoder(path, device)
        # as LeRobotDataset (video_utils.decode_video_frames_torchcodec): index = round(time * average fps);
        # looking up the frame shown at time t0 + f / fps lands on frame boundaries and can return f - 1.
        # A few repos have videos a little shorter than their states: clamp to the last frame.
        vfps, last = decoder.metadata.average_fps, decoder.metadata.num_frames - 1
        idx = [min(round((t0 + float(f) / fps) * vfps), last) for f in plan_frames]
        batch = decoder.get_frames_at(indices=idx).data
        if batch.device.type != device:
            raise RuntimeError(f"decoder returned frames on {batch.device}")
        # 8 frames at a time: a float copy of 32 x 1920 x 1200 frames is ~0.9 GB
        return torch.cat([resize_with_pad(batch[i:i + 8].float() / 255.0, self.size, self.size).cpu()
                          for i in range(0, len(batch), 8)])

    def __getitem__(self, i: int):
        rel, ep_idx = self.items[i]
        if not self._threads_set:  # one decode worker per process slot: keep torch from using every core
            torch.set_num_threads(2)
            self._threads_set = True
        try:
            ep = D.episodes(rel)[ep_idx]
            state, valid = self._states.episode(rel, ep)
            std = _state_std_for_su(rel)
            plan = sample_su80(state, valid, std, self.k)
            fps = float(D.info(rel)["fps"])
            slots = D.camera_slots(rel)
            if not any(slots):
                raise ValueError(f"no mapped camera in {rel}")
            frames = {}
            for cam in (c for c in slots if c):
                chunk, file, t0 = ep["videos"][cam]
                path = D.video_path(rel, cam, chunk, file)
                device = self._device(rel, cam, path)
                try:
                    frames[cam] = self._frames(path, device, t0, fps, plan.frames)
                except Exception:  # noqa: BLE001 - NVDEC cannot decode every stream: this video on the CPU
                    if device != "cuda":
                        raise
                    self._decoders.pop((str(path), device), None)
                    self._cpu_only.add(str(path))
                    frames[cam] = self._frames(path, "cpu", t0, fps, plan.frames)
            images = [frames[c][j] for j in range(self.k) for c in slots if c]  # slot-major
            if self._processor is None:
                from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision

                torch.set_num_threads(2)
                self._processor, _ = load_qwen35_vision(self.vlm_path)
            out = self._processor(images, do_rescale=False)
            return {"rel": rel, "episode": ep_idx, "plan": plan, "pixels": torch.as_tensor(out.pixel_values),
                    "grids": torch.as_tensor(out.image_grid_thw, dtype=torch.long), "slots": slots,
                    "fps": fps}
        except Exception as e:  # noqa: BLE001 - one broken episode must not stop the build
            return {"rel": rel, "episode": ep_idx, "error": f"{type(e).__name__}: {e}"}


class EpisodeStream(torch.utils.data.IterableDataset):
    """Worker w iterates its own block of episodes in order, so a video file opened (exact index scan, up to ~1 min
    for a 480k-frame hifi_umi file) serves all following episodes of that worker."""

    def __init__(self, blocks: list[list[tuple[str, int]]], **kwargs) -> None:
        self.blocks, self.kwargs = blocks, kwargs

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        block = self.blocks[info.id if info else 0]
        frames = EpisodeFrames(block, **self.kwargs)
        for i in range(len(block)):
            yield frames[i]


def record(item: dict, feats: torch.Tensor, vlm_path: str) -> dict:
    """feats [32 * n_present, 64, D] slot-major -> P5 [32, 3, 5, D], zeros for missing camera slots."""
    plan, slots = item["plan"], item["slots"]
    k, present = len(plan.frames), [c is not None for c in slots]
    f = feats.reshape(k, sum(present), feats.shape[-2], feats.shape[-1])
    full = torch.zeros(k, 3, feats.shape[-2], feats.shape[-1], dtype=feats.dtype, device=feats.device)
    full[:, [j for j in range(3) if present[j]]] = f
    return {
        "feats": pool_spatial(full.float(), 5).to(torch.bfloat16).cpu(),
        "frames": torch.as_tensor(plan.frames, dtype=torch.long),
        "time_ratio": torch.as_tensor(plan.frames / max(1, plan.length - 1), dtype=torch.float32),
        "camera_valid": torch.tensor(present, dtype=torch.bool)[None].expand(k, 3).clone(),
        "cameras": slots,
        "active": tuple(int(x) for x in plan.active),
        "length": int(plan.length),
        "repeats": int(plan.repeats),
        "fps": item["fps"],
        "sampler": SAMPLER,
        "sampler_version": SAMPLER_VERSION,
        "vit_source": vlm_path,
        "spatial": 5,
        "rel": item["rel"],
    }


def save(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(rec, tmp)
    os.replace(tmp, path)


def all_items(args) -> list[tuple[str, int]]:
    """Every (repo, episode) of the selection; written once to --items_file and reused by all shards."""
    if args.items_file and Path(args.items_file).is_file():
        return [tuple(x) for x in json.loads(Path(args.items_file).read_text())]
    items = []
    for rel in D.repos(args.sources.split(",") if args.sources else None):
        eps = sorted(D.episodes(rel))
        items += [(rel, e) for e in (eps[: args.limit_per_repo] if args.limit_per_repo else eps)]
    if args.items_file:
        tmp = Path(f"{args.items_file}.{os.getpid()}.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(items))
        os.replace(tmp, args.items_file)
    return items


def shard(items: list[tuple[str, int]], rank: int, world: int, chunk: int = 5000) -> list[tuple[str, int]]:
    """Runs of at most `chunk` consecutive episodes of one repo, dealt round-robin to the shards: every shard mixes
    sources (abc's slow HEVC, dom's 207k-episode repo) and keeps repo-contiguous runs for its decoders."""
    runs, cur = [], []
    for it in items:
        if cur and (it[0] != cur[-1][0] or len(cur) >= chunk):
            runs.append(cur)
            cur = []
        cur.append(it)
    if cur:
        runs.append(cur)
    return [it for i, run in enumerate(runs) if i % world == rank for it in run]


def pending(args) -> list[tuple[str, int]]:
    """Missing cache eps first, then shard.

    Filter-before-shard so resume redistributes leftovers across the *current*
    world (e.g. old ranks 8–15 already done under world=16 no longer leave
    GPUs idle when relaunched on 8 cards).
    """
    listed: dict[Path, set[str]] = {}
    missing: list[tuple[str, int]] = []
    for rel, e in all_items(args):  # one directory listing per repo
        p = cache_path(args.cache_dir, rel, e)
        if p.parent not in listed:
            listed[p.parent] = set(os.listdir(p.parent)) if p.parent.is_dir() else set()
        if p.name not in listed[p.parent]:
            missing.append((rel, e))
    return shard(missing, args.rank, args.world)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--vlm_path", default=VLM)
    ap.add_argument("--sources", default="", help="','-separated sources (default all)")
    ap.add_argument("--num_slots", type=int, default=32)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--episodes_per_batch", type=int, default=4)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--limit_per_repo", type=int, default=0, help="first N episodes per repo (smoke tests)")
    ap.add_argument("--limit", type=int, default=0, help="first N pending episodes of this shard (smoke tests)")
    ap.add_argument("--items_file", default="", help="(repo, episode) list shared by all shards; built if missing")
    ap.add_argument("--gpu_decode_pixels", type=int, default=1280 * 720,
                    help="videos with at least this many pixels are decoded on the GPU (NVDEC); 0 = CPU only")
    ap.add_argument("--only_items", action="store_true", help="write --items_file and exit")
    args = ap.parse_args()
    if args.only_items:
        logging.basicConfig(level=logging.INFO)
        logging.info("items: %d -> %s", len(all_items(args)), args.items_file)
        return
    logging.basicConfig(level=logging.INFO, format=f"[rank {args.rank}] %(asctime)s %(message)s")
    items = pending(args)
    if args.limit:
        items = items[: args.limit]
    logging.info(
        "pending episodes: %d (rank=%d world=%d; filter-then-shard resume)",
        len(items),
        args.rank,
        args.world,
    )
    # compact progress line for the launcher aggregator
    prog = Path(args.cache_dir) / "_logs" / f"progress_rank{args.rank}.txt"
    prog.parent.mkdir(parents=True, exist_ok=True)
    prog.write_text(f"rank={args.rank} status=start pending={len(items)} done=0 failed=0 rate=0 eta_h=-1\n")
    if not items:
        prog.write_text(f"rank={args.rank} status=idle pending=0 done=0 failed=0 rate=0 eta_h=0\n")
        return
    device = torch.device("cuda")
    vision = load_vision(args.vlm_path, device)
    workers = max(1, args.num_workers)
    # each worker owns a contiguous block (its state file and video decoders stay warm)
    blocks = [items[i * len(items) // workers:(i + 1) * len(items) // workers] for i in range(workers)]
    order = items
    loader = torch.utils.data.DataLoader(
        EpisodeStream(blocks, num_slots=args.num_slots, size=args.image_size, vlm_path=args.vlm_path,
                      gpu_decode_pixels=args.gpu_decode_pixels),
        batch_size=None, num_workers=args.num_workers, prefetch_factor=4 if args.num_workers else None,
        # episodes cost 0.2 s (480p AV1) to 3 s (abc 1920x1200 HEVC): take whatever is ready, do not wait in order
        in_order=False,
        multiprocessing_context="spawn" if args.num_workers else None)
    fail_dir = Path(args.cache_dir) / "_failures"
    fail_dir.mkdir(parents=True, exist_ok=True)
    fail_file = fail_dir / f"rank{args.rank}.jsonl"
    t0, done, failed, batch = time.time(), 0, 0, []
    log_every = max(1, int(os.environ.get("CACHE_LOG_EVERY", "50")))

    def write_prog(status: str) -> None:
        rate = done / max(1e-6, time.time() - t0)
        left = max(0, len(order) - done - failed)
        eta = left / max(1e-6, rate) / 3600 if done else -1
        prog.write_text(
            f"rank={args.rank} status={status} pending={len(order)} done={done} failed={failed} "
            f"rate={rate:.3f} eta_h={eta:.2f}\n"
        )

    def flush(batch):
        pixels = torch.cat([b["pixels"] for b in batch]).to(device, torch.bfloat16)
        grids = torch.cat([b["grids"] for b in batch]).to(device)
        with torch.no_grad():
            feats = vision_tokens(vision, pixels, grids)
        for b, f in zip(batch, torch.split(feats, [len(b["grids"]) for b in batch])):
            save(cache_path(args.cache_dir, b["rel"], b["episode"]), record(b, f, args.vlm_path))

    for item in loader:
        if "error" in item:
            failed += 1
            with fail_file.open("a") as fh:
                fh.write(json.dumps({"rel": item["rel"], "episode": int(item["episode"]), "error": item["error"]}) + "\n")
            continue
        batch.append(item)
        if len(batch) >= args.episodes_per_batch:
            flush(batch)
            done += len(batch)
            batch = []
            if done % log_every < args.episodes_per_batch:
                rate = done / max(1e-6, time.time() - t0)
                logging.info("%d / %d done, %d failed, %.2f ep/s, eta %.1f h", done, len(order), failed, rate,
                             (len(order) - done - failed) / max(1e-6, rate) / 3600)
                write_prog("run")
    if batch:
        flush(batch)
        done += len(batch)
    write_prog("done")
    logging.info("done: %d built, %d failed (%s), %.1f min", done, failed, fail_file, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
