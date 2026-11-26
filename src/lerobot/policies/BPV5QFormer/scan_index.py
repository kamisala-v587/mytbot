"""Build the BPv5 membership index so training does not rescan leaves.

Usage:
  python -m lerobot.policies.BPV5QFormer.scan_index --config configs/bpv5/stage_a/stage_a_main.jsonc
"""

from __future__ import annotations

import argparse
import dataclasses
import logging

from lerobot.policies.BPV5QFormer.configuration_bpv5qformer import BPV5QFormerDatasetConfig
from lerobot.policies.BPV5QFormer.dataset import (
    build_scan_index,
    save_scan_index,
    scan_index_path,
    try_load_scan_index,
)
from lerobot.utils.jsonc import load_jsonc


def _cfg_from_jsonc(path: str) -> BPV5QFormerDatasetConfig:
    raw = load_jsonc(path)["dataset"]
    fields = {f.name for f in dataclasses.fields(BPV5QFormerDatasetConfig)}
    return BPV5QFormerDatasetConfig(**{k: v for k, v in raw.items() if k in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = _cfg_from_jsonc(args.config)
    path = scan_index_path(cfg)
    if path.is_file() and not args.force and try_load_scan_index(cfg) is not None:
        logging.info("scan index already matches this config: %s", path)
        return
    by_repo, missing = build_scan_index(cfg, args.workers)
    if not by_repo:
        raise SystemExit(f"scan kept no trajectories under {cfg.cache_dir}")
    written = save_scan_index(cfg, by_repo, missing)
    episodes = sum(len(v) for v in by_repo.values())
    frames = sum(length for rows in by_repo.values() for _rel, _ep, length in rows)
    logging.info(
        "ready: %s repos=%d episodes=%d frames=%d miss_cache=%d",
        written,
        len(by_repo),
        episodes,
        frames,
        missing,
    )


if __name__ == "__main__":
    main()
