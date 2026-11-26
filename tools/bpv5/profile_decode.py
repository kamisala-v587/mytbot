"""Where the cache worker's time goes for a few episodes: state read, decoder open, frame decode, resize, processor.
Second episode of a repo shows the warm cost (decoders / state file cached).

usage: python tools/bpv5/profile_decode.py
"""

from __future__ import annotations

import time

import torch

from lerobot.policies.BPV5QFormer import data80d as D
from lerobot.policies.BPV5QFormer.build_cache import VLM
from lerobot.policies.BPV5QFormer.sampling80d import sample_su80
from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision
from lerobot.transforms.utils import resize_with_pad
from torchcodec.decoders import VideoDecoder
import numpy as np

processor, _ = load_qwen35_vision(VLM)
torch.set_num_threads(2)
reader = D.StateReader()
for rel, eps in [("robotwin/adjust_bottle/aloha-agilex_clean_50", [3, 4]), ("abc/train__arrange_the_flowers_into_the_vase__zedx", [1, 2]),
                 ("interndata/articulation_tasks/franka/open_the_microwave_part2", [10, 11]), ("dom", [100, 101]),
                 ("hifi_umi/chunk-0001/part-0000", [5, 6])]:
    decoders = {}
    for e in eps:
        t = {}
        t0 = time.time()
        ep = D.episodes(rel)[e]
        t["meta"] = time.time() - t0
        t0 = time.time()
        state, valid = reader.episode(rel, ep)
        std = np.asarray(D.group_stats(D.stats_index()[rel])["state_80D"]["std"], dtype=np.float32)
        plan = sample_su80(state, valid, std)
        t["state+plan"] = time.time() - t0
        fps = float(D.info(rel)["fps"])
        frames = []
        t["open"] = t["decode"] = t["resize"] = 0.0
        for cam in (c for c in D.camera_slots(rel) if c):
            chunk, file, t0v = ep["videos"][cam]
            path = str(D.video_path(rel, cam, chunk, file))
            s = time.time()
            if path not in decoders:
                decoders[path] = VideoDecoder(path, seek_mode="exact")
            dec = decoders[path]
            t["open"] += time.time() - s
            s = time.time()
            idx = [min(round((t0v + float(f) / fps) * dec.metadata.average_fps), dec.metadata.num_frames - 1) for f in plan.frames]
            b = dec.get_frames_at(indices=idx).data
            t["decode"] += time.time() - s
            s = time.time()
            frames.append(resize_with_pad(b.float() / 255.0, 224, 224))
            t["resize"] += time.time() - s
        s = time.time()
        images = [f[j] for j in range(32) for f in frames]
        processor(images, do_rescale=False)
        t["processor"] = time.time() - s
        m = dec.metadata
        print(f"{rel[:45]:45s} ep{e:<5d} len {ep['length']:5d} video {m.width}x{m.height} {m.codec} "
              f"frames_in_file {m.num_frames} | " + " ".join(f"{k} {v:.2f}s" for k, v in t.items()))
