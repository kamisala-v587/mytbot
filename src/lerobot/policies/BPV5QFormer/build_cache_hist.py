"""BPv5 Stage B history cache: wall ViT P5 of the head camera on a control-step grid.

Per episode: frames 0, 25, 50, … < length of the head slot only -> decode (chunked) ->
resize_with_pad 224 -> Qwen3.5 processor -> wall ViT -> 64 x 2560 -> P5 [5, 2560] ->
<cache>/<rel>/epXXXXXX.pt with feats [N, 5, 2560] bf16 and frames [N].

Training later gathers frame_id < t (at most 30), left-pads with hist_valid; current three-view
obs is separate. Chunked decode/encode so long episodes (e.g. hifi_umi) stay in memory bounds.

No head camera -> empty feats/frames (still written so shards do not retry). Failures ->
_failures/rank*.jsonl. Launcher: tools/bpv5/build_cache_hist_dist.sh

usage: python -m lerobot.policies.BPV5QFormer.build_cache_hist --cache_dir <dir> [--stride 25] [--rank 0 --world 1]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch

from lerobot.policies.BPV4CDX.dataset import pool_spatial
from lerobot.transforms.utils import resize_with_pad

from . import data80d as D
from .build_cache import (
    VLM,
    all_items,
    cache_path,
    load_vision,
    save,
    shard,
    vision_tokens,
)

STRIDE_DEFAULT = 25
CACHE_KIND = "hist_head_s25"
# Decode / processor / ViT batch size in number of history frames (one camera).
FRAME_CHUNK = 32


def hist_frames(length: int, stride: int) -> list[int]:
    """Control-frame indices from 0 with fixed stride (exclusive of length)."""
    if length <= 0:
        return []
    return list(range(0, int(length), int(stride)))


class EpisodeHistChunks(torch.utils.data.IterableDataset):
    """Yields decode chunks for each (repo, episode). Long tracks stream as many chunks; empty /
    no-head episodes yield a single empty marker so the writer still saves a stub .pt."""

    def __init__(self, blocks: list[list[tuple[str, int]]], stride: int, size: int, vlm_path: str,
                 frame_chunk: int = FRAME_CHUNK, gpu_decode_pixels: int = 1280 * 720) -> None:
        self.blocks = blocks
        self.stride = int(stride)
        self.size = size
        self.vlm_path = vlm_path
        self.frame_chunk = int(frame_chunk)
        self.gpu_decode_pixels = gpu_decode_pixels

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        block = self.blocks[info.id if info else 0]
        worker = _HistWorker(self.stride, self.size, self.vlm_path, self.frame_chunk, self.gpu_decode_pixels)
        for rel, ep_idx in block:
            yield from worker.iter_episode(rel, ep_idx)


class _HistWorker:
    def __init__(self, stride: int, size: int, vlm_path: str, frame_chunk: int, gpu_decode_pixels: int) -> None:
        self.stride, self.size, self.vlm_path = stride, size, vlm_path
        self.frame_chunk = frame_chunk
        self.gpu_decode_pixels = gpu_decode_pixels
        self._processor = None
        self._decoders: dict[tuple[str, str], object] = {}
        self._cpu_only: set[str] = set()
        self._threads_set = False

    def _decoder(self, path: Path, device: str):
        from torchcodec.decoders import VideoDecoder

        key = (str(path), device)
        if key not in self._decoders:
            same = [k for k in self._decoders if k[1] == device]
            if len(same) >= (1 if device == "cuda" else 4):
                self._decoders.pop(same[0])
            self._decoders[key] = VideoDecoder(key[0], seek_mode="exact", device=device)
        return self._decoders[key]

    def _device(self, rel: str, cam: str, path: Path) -> str:
        shape = D.info(rel)["features"][cam].get("shape") or [0, 0, 0]
        h, w = sorted(int(x) for x in shape)[-2:]
        big = self.gpu_decode_pixels and h * w >= self.gpu_decode_pixels
        return "cuda" if big and str(path) not in self._cpu_only else "cpu"

    def _frames(self, path: Path, device: str, t0: float, fps: float, plan_frames: list[int]) -> torch.Tensor:
        decoder = self._decoder(path, device)
        vfps, last = decoder.metadata.average_fps, decoder.metadata.num_frames - 1
        idx = [min(round((t0 + float(f) / fps) * vfps), last) for f in plan_frames]
        batch = decoder.get_frames_at(indices=idx).data
        if batch.device.type != device:
            raise RuntimeError(f"decoder returned frames on {batch.device}")
        return torch.cat([
            resize_with_pad(batch[i:i + 8].float() / 255.0, self.size, self.size).cpu()
            for i in range(0, len(batch), 8)
        ])

    def _ensure_processor(self):
        if self._processor is None:
            from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision

            torch.set_num_threads(2)
            self._processor, _ = load_qwen35_vision(self.vlm_path)

    def iter_episode(self, rel: str, ep_idx: int):
        if not self._threads_set:
            torch.set_num_threads(2)
            self._threads_set = True
        try:
            ep = D.episodes(rel)[ep_idx]
            length = int(ep["length"])
            fps = float(D.info(rel)["fps"])
            head = D.camera_slots(rel)[0]
            frames_idx = hist_frames(length, self.stride)
            meta = {"rel": rel, "episode": ep_idx, "length": length, "fps": fps, "head": head,
                    "frames_all": frames_idx}
            if head is None or not frames_idx:
                yield {**meta, "empty": True, "chunk_i": 0, "n_chunks": 1}
                return
            chunk, file, t0 = ep["videos"][head]
            path = D.video_path(rel, head, chunk, file)
            device = self._device(rel, head, path)
            n_chunks = (len(frames_idx) + self.frame_chunk - 1) // self.frame_chunk
            self._ensure_processor()
            for chunk_i in range(n_chunks):
                sl = frames_idx[chunk_i * self.frame_chunk:(chunk_i + 1) * self.frame_chunk]
                try:
                    images = self._frames(path, device, t0, fps, sl)
                except Exception:  # noqa: BLE001 - NVDEC cannot decode every stream
                    if device != "cuda":
                        raise
                    self._decoders.pop((str(path), device), None)
                    self._cpu_only.add(str(path))
                    device = "cpu"
                    images = self._frames(path, device, t0, fps, sl)
                out = self._processor([images[j] for j in range(len(sl))], do_rescale=False)
                yield {
                    **meta, "empty": False, "chunk_i": chunk_i, "n_chunks": n_chunks,
                    "frames_idx": sl,
                    "pixels": torch.as_tensor(out.pixel_values),
                    "grids": torch.as_tensor(out.image_grid_thw, dtype=torch.long),
                }
        except Exception as e:  # noqa: BLE001
            yield {"rel": rel, "episode": ep_idx, "error": f"{type(e).__name__}: {e}"}


def record(meta: dict, feats: torch.Tensor | None, frames_idx: list[int], vlm_path: str, stride: int) -> dict:
    if feats is None or len(frames_idx) == 0:
        p5 = torch.zeros(0, 5, 2560, dtype=torch.bfloat16)
        frames = torch.zeros(0, dtype=torch.long)
    else:
        # feats [N, 64, 2560] -> [N, 1, 64, D] -> P5 -> squeeze camera
        p5 = pool_spatial(feats[:, None].float(), 5).to(torch.bfloat16).cpu()[:, 0]
        frames = torch.as_tensor(frames_idx, dtype=torch.long)
    return {
        "feats": p5,
        "frames": frames,
        "length": int(meta["length"]),
        "fps": float(meta["fps"]),
        "stride": int(stride),
        "camera": meta.get("head"),
        "vit_source": vlm_path,
        "spatial": 5,
        "kind": CACHE_KIND,
        "rel": meta["rel"],
    }


def pending(args) -> list[tuple[str, int]]:
    items = shard(all_items(args), args.rank, args.world)
    listed: dict[Path, set[str]] = {}
    out = []
    for rel, e in items:
        p = cache_path(args.cache_dir, rel, e)
        if p.parent not in listed:
            listed[p.parent] = set(os.listdir(p.parent)) if p.parent.is_dir() else set()
        if p.name not in listed[p.parent]:
            out.append((rel, e))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--vlm_path", default=VLM)
    ap.add_argument("--sources", default="", help="','-separated sources (default all)")
    ap.add_argument("--stride", type=int, default=STRIDE_DEFAULT)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--frame_chunk", type=int, default=FRAME_CHUNK,
                    help="history frames per decode/ViT chunk (memory bound for long episodes)")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--limit_per_repo", type=int, default=0, help="first N episodes per repo (smoke)")
    ap.add_argument("--limit", type=int, default=0, help="first N pending episodes of this shard (smoke)")
    ap.add_argument("--items_file", default="", help="(repo, episode) list shared by all shards")
    ap.add_argument("--gpu_decode_pixels", type=int, default=1280 * 720,
                    help="videos with at least this many pixels decode on GPU (NVDEC); 0 = CPU only")
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
    logging.info("pending episodes: %d (stride=%d, frame_chunk=%d)", len(items), args.stride, args.frame_chunk)
    if not items:
        return
    device = torch.device("cuda")
    vision = load_vision(args.vlm_path, device)
    workers = max(1, args.num_workers)
    blocks = [items[i * len(items) // workers:(i + 1) * len(items) // workers] for i in range(workers)]
    order = items
    loader = torch.utils.data.DataLoader(
        EpisodeHistChunks(blocks, stride=args.stride, size=args.image_size, vlm_path=args.vlm_path,
                          frame_chunk=args.frame_chunk, gpu_decode_pixels=args.gpu_decode_pixels),
        batch_size=None, num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers else None, in_order=False,
        multiprocessing_context="spawn" if args.num_workers else None)
    fail_dir = Path(args.cache_dir) / "_failures"
    fail_dir.mkdir(parents=True, exist_ok=True)
    fail_file = fail_dir / f"rank{args.rank}.jsonl"
    t0, done, failed = time.time(), 0, 0
    # Accumulate chunked features per episode (out-of-order workers).
    pending_eps: dict[tuple[str, int], dict] = {}

    def encode_chunk(pixels: torch.Tensor, grids: torch.Tensor) -> torch.Tensor:
        pixels = pixels.to(device, torch.bfloat16)
        grids = grids.to(device)
        with torch.no_grad():
            return vision_tokens(vision, pixels, grids).float().cpu()

    def finish_episode(key: tuple[str, int]) -> None:
        nonlocal done
        st = pending_eps.pop(key)
        meta = st["meta"]
        if st.get("empty"):
            save(cache_path(args.cache_dir, meta["rel"], meta["episode"]),
                 record(meta, None, [], args.vlm_path, args.stride))
        else:
            chunks = st["chunks"]
            order_i = sorted(chunks)
            feats = torch.cat([chunks[i][0] for i in order_i], dim=0)
            frames = [f for i in order_i for f in chunks[i][1]]
            save(cache_path(args.cache_dir, meta["rel"], meta["episode"]),
                 record(meta, feats, frames, args.vlm_path, args.stride))
        done += 1
        if done % 50 == 0 or done == len(order):
            rate = done / max(1e-6, time.time() - t0)
            logging.info("%d / %d done, %d failed, %.2f ep/s, eta %.1f h", done, len(order), failed, rate,
                         (len(order) - done - failed) / max(1e-6, rate) / 3600)

    for item in loader:
        if "error" in item:
            failed += 1
            key = (item["rel"], int(item["episode"]))
            pending_eps.pop(key, None)
            with fail_file.open("a") as fh:
                fh.write(json.dumps({"rel": item["rel"], "episode": int(item["episode"]), "error": item["error"]}) + "\n")
            continue
        key = (item["rel"], int(item["episode"]))
        if item.get("empty"):
            pending_eps[key] = {"meta": item, "empty": True, "got": 1, "need": 1}
            finish_episode(key)
            continue
        st = pending_eps.get(key)
        if st is None:
            st = pending_eps[key] = {
                "meta": item, "empty": False, "chunks": {}, "got": 0, "need": int(item["n_chunks"]),
            }
        feats = encode_chunk(item["pixels"], item["grids"])
        st["chunks"][int(item["chunk_i"])] = (feats, list(item["frames_idx"]))
        st["got"] += 1
        if st["got"] >= st["need"]:
            finish_episode(key)

    if pending_eps:
        logging.warning("incomplete episodes left: %d (not saved)", len(pending_eps))
        for key, st in list(pending_eps.items()):
            logging.warning("  %s ep%s got %d/%d", key[0], key[1], st["got"], st["need"])
    logging.info("done: %d built, %d failed (%s), %.1f min", done, failed, fail_file, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
