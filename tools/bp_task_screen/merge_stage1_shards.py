"""Merge stage-1 seed-search shards (data/<task>/<config>_shard<k>) into data/<task>/<config>.

Appends the shard seeds to seed.txt and renumbers _traj_data/episode<j>.pkl after the main directory's episodes,
up to --max_episodes. The main stage-1 process of the task must be stopped first.

Usage: python merge_stage1_shards.py <robotwin_data_dir>/<task>/<config> [--max_episodes 100]
"""

import argparse
import shutil
from pathlib import Path


def seeds(d: Path) -> list[int]:
    f = d / "seed.txt"
    return [int(s) for s in f.read_text().split()] if f.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("main_dir")
    ap.add_argument("--max_episodes", type=int, default=100)
    args = ap.parse_args()
    main_dir = Path(args.main_dir)
    merged = seeds(main_dir)
    (main_dir / "_traj_data").mkdir(exist_ok=True)
    for shard in sorted(main_dir.parent.glob(main_dir.name + "_shard*")):
        for j, seed in enumerate(seeds(shard)):
            if len(merged) >= args.max_episodes:
                break
            src = shard / "_traj_data" / f"episode{j}.pkl"
            if not src.exists():
                continue
            shutil.copy(src, main_dir / "_traj_data" / f"episode{len(merged)}.pkl")
            merged.append(seed)
    (main_dir / "seed.txt").write_text("".join(f"{s} " for s in merged))
    print(main_dir.parent.name, len(merged))


if __name__ == "__main__":
    main()
