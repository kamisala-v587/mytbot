"""Frame-convention probe (F1 preflight §3.5) — one episode, one frame, full chain.

Validates BEFORE the 3500-episode build:
  torchcodec frame is 3-channel RGB uint8 [3,H,W]  (torchcodec returns RGB by
  construction; the RGB-ness itself is only truly proven by parity A against an
  upstream cached record) -> float32/255 in [0,1] -> resize_with_pad(224, 224)
  -> Qwen3.5 processor(images, do_rescale=False) -> image_grid_thw row whose
  merged token count is 64 (the frozen front-end fingerprint: 224 px, 64
  tokens/image). Also times batched get_frames_at vs per-index reads and checks
  duplicate-index mapping, the two reader behaviors the producer relies on.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from episode_source import CAMERAS, load_episode
from build_feats import read_frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", required=True, help="episode json to probe")
    ap.add_argument("--qwen_path", default="/vla/workspace/models/q35/base_800m")
    args = ap.parse_args()

    ep = load_episode(args.episode)
    print(f"episode {ep['key']}: num_frames={ep['num_frames']} state={ep['state'].shape}")

    from torchcodec.decoders import VideoDecoder

    dec = VideoDecoder(ep["videos"]["ego"])
    frame = dec[0].data
    assert frame.ndim == 3 and frame.shape[0] == 3, f"expected [3,H,W], got {tuple(frame.shape)}"
    assert frame.dtype == torch.uint8
    print(f"reader frame: {tuple(frame.shape)} {frame.dtype} rgb (3-channel; channel order "
          f"proven by parity A vs upstream cache)")

    f = frame.to(torch.float32).div_(255.0)
    assert float(f.min()) >= 0.0 and float(f.max()) <= 1.0

    from lerobot.transforms.utils import resize_with_pad

    img = resize_with_pad(f[None], 224, 224)[0]  # [3,224,224] — the upstream build_cache call
    assert tuple(img.shape) == (3, 224, 224) and float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    print(f"resized: {tuple(img.shape)} range [{float(img.min()):.4f}, {float(img.max()):.4f}]")
    from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision

    processor, token_ids = load_qwen35_vision(args.qwen_path)
    out = processor([img], do_rescale=False)
    pixels = torch.as_tensor(out.pixel_values)
    grids = torch.as_tensor(out.image_grid_thw, dtype=torch.long)
    merge = int(token_ids.spatial_merge_size)
    toks = int(grids[0, 1]) // merge * (int(grids[0, 2]) // merge)
    print(f"processor: pixel_values {tuple(pixels.shape)} {pixels.dtype}, image_grid_thw {grids.tolist()} "
          f"(merge {merge} -> {toks} tokens/image)")
    assert toks == 64, f"expected 64 merged tokens/image, got {toks}"

    # reader behaviors the producer relies on: batched indexed read + repeats mapping
    idx = [0, 0, 5, ep["num_frames"] - 1]
    sel = read_frames(ep["videos"]["ego"], np.asarray(idx), ep["num_frames"], "ego", ep["key"])
    assert tuple(sel.shape) == (4, 3, int(dec.metadata.height), int(dec.metadata.width))
    assert sel.dtype == torch.float32 and float(sel.max()) <= 1.0
    assert torch.equal(sel[0], sel[1]), "duplicate index must map to the same frame"

    t0 = time.time()
    read_frames(ep["videos"]["ego"], np.linspace(0, ep["num_frames"] - 1, 32).astype(int),
                ep["num_frames"], "ego", ep["key"])
    t_batch = time.time() - t0
    t0 = time.time()
    dec2 = VideoDecoder(ep["videos"]["ego"])
    for i in np.linspace(0, ep["num_frames"] - 1, 32).astype(int):
        dec2[int(i)]
    t_single = time.time() - t0
    print(f"32-frame read: batched get_frames_at {t_batch:.2f}s vs per-index {t_single:.2f}s "
          f"({ep['num_frames']} frames, {len(CAMERAS)} cameras/episode)")
    print("PROBE PASS — reader/resize/processor chain OK, 64 tokens/image")


if __name__ == "__main__":
    main()
