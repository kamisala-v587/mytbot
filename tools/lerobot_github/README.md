# Minimal LeRobot episode splitter

This package is a narrow local adaptation of `lerobot.datasets.dataset_tools` from the
Hugging Face LeRobot project (Apache-2.0). It only creates one v3.0 subset from explicit,
complete episode indices. It intentionally does not import or require the reference checkout
in `/home/jovyan/workspace/.cache/lerobot-github`.

`split_episodes` keeps all non-visual fields and only requested cameras, remaps episode/frame/task
metadata, re-encodes each selected video episode, greedily packs measured files, validates final
multi-episode file sizes, and returns mappings/statistics/warnings for a manifest. The callback
receives immutable `ProgressEvent` objects with stage/repo/task/episode/camera/file context.

The implementation preserves the source codec and pixel format when PyAV can encode them. If
that is unavailable it falls back to H.264/yuv420p-compatible output and records the selected
`(codec, pix_fmt)` pair in `SplitResult.encoder_fallbacks`. A camera uses one encoding pair for all
of its episode segments, and segment compatibility is checked before concat. Output directories
must not exist or be empty; atomic rename and cleanup policy belong to the caller.

Video statistics are never copied from the source after transcoding. They are recomputed per
output episode by decoding the final packed files using the measured PTS boundaries; non-visual
statistics continue to come from complete source episode metadata.

## Attribution

Copyright 2024-2025 The Hugging Face Inc. team. Adapted under Apache License 2.0.
See https://www.apache.org/licenses/LICENSE-2.0.
