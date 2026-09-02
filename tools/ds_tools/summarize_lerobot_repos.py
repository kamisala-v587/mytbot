#!/usr/bin/env python3
"""Summarize basic metadata for LeRobot repos listed in a txt file.

Usage:
  python tools/summarize_lerobot_repos.py
  python tools/summarize_lerobot_repos.py configs/ds_ids/Baidunyun/pretrain_data_ids_baiduyun.txt
  python tools/summarize_lerobot_repos.py /path/to/repo_ids.txt --json
  python tools/summarize_lerobot_repos.py repo_ids.txt --root /public-data/dataset

The input txt should contain one repo path/id per line. Blank lines and lines
starting with '#' are ignored. Absolute paths are read directly; relative repo
ids are resolved under --root when provided.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_TXT = "/vla/workspace/my_tbot/configs/ds_ids/Baidunyun/pretrain_data_ids_baiduyun.txt"


def read_repo_list(path: Path) -> list[str]:
    repos: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        repos.append(line)
    return repos


def repo_meta_path(repo: str, root: Path | None = None) -> Path:
    repo_path = Path(repo).expanduser()
    if not repo_path.is_absolute() and root is not None:
        repo_path = root.expanduser() / repo_path
    return repo_path / "meta" / "info.json"


def load_info(repo: str, root: Path | None = None) -> dict[str, Any]:
    info_path = repo_meta_path(repo, root)
    if not info_path.is_file():
        raise FileNotFoundError(f"missing meta/info.json: {info_path}")
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_hours(hours: float) -> str:
    minutes = hours * 60.0
    return f"{hours:.3f} h ({minutes:.1f} min)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LeRobot repo ids listed in a txt file.")
    parser.add_argument(
        "txt",
        nargs="?",
        default=DEFAULT_TXT,
        help="Text file containing one repo path/id per line.",
    )
    parser.add_argument("--root", type=Path, default=None, help="Common root for relative repo ids.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text.")
    args = parser.parse_args()

    txt_path = Path(args.txt).expanduser()
    repos = read_repo_list(txt_path)
    rows: list[dict[str, Any]] = []
    fps_counter: Counter[float] = Counter()
    total_frames = 0
    total_episodes = 0
    total_seconds = 0.0
    errors: list[dict[str, str]] = []

    for repo in repos:
        try:
            info = load_info(repo, args.root)
            frames = int(info.get("total_frames", 0) or 0)
            episodes = int(info.get("total_episodes", 0) or 0)
            fps = float(info.get("fps", 0) or 0)
            seconds = frames / fps if fps > 0 else 0.0
            total_frames += frames
            total_episodes += episodes
            total_seconds += seconds
            if fps > 0:
                fps_counter[fps] += 1
            rows.append(
                {
                    "repo": repo,
                    "total_frames": frames,
                    "total_episodes": episodes,
                    "fps": fps,
                    "hours": seconds / 3600.0,
                }
            )
        except Exception as exc:  # Keep summarizing other repos.
            errors.append({"repo": repo, "error": str(exc)})

    fps_values = sorted(fps_counter)
    summary: dict[str, Any] = {
        "source_txt": str(txt_path),
        "root": str(args.root.expanduser()) if args.root else None,
        "repo_count": len(repos),
        "loaded_repo_count": len(rows),
        "failed_repo_count": len(errors),
        "total_frames": total_frames,
        "total_episodes": total_episodes,
        "fps_values": fps_values,
        "fps_counts": {str(k): v for k, v in sorted(fps_counter.items())},
        "estimated_hours": total_seconds / 3600.0,
        "repos": rows,
        "errors": errors,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print(f"Source txt        : {summary['source_txt']}")
    if args.root:
        print(f"Repo root         : {summary['root']}")
    print(
        f"Repo count        : {summary['repo_count']} "
        f"(loaded={summary['loaded_repo_count']}, failed={summary['failed_repo_count']})"
    )
    print(f"Total frames      : {summary['total_frames']:,}")
    print(f"Total trajectories: {summary['total_episodes']:,}")
    if len(fps_values) == 1:
        print(f"FPS               : {fps_values[0]:g}")
    else:
        print(f"FPS values        : {summary['fps_counts']}")
    print(f"Estimated duration: {fmt_hours(summary['estimated_hours'])}")

    print("\nPer repo:")
    for row in rows:
        print(
            f"- {row['repo']} | frames={row['total_frames']:,} | "
            f"episodes={row['total_episodes']:,} | fps={row['fps']:g} | duration={fmt_hours(row['hours'])}"
        )

    if errors:
        print("\nErrors:")
        for item in errors:
            print(f"- {item['repo']}: {item['error']}")

    print("\nSummary:")
    print(f"Source txt        : {summary['source_txt']}")
    print(
        f"Repo count        : {summary['repo_count']} "
        f"(loaded={summary['loaded_repo_count']}, failed={summary['failed_repo_count']})"
    )
    print(f"Total frames      : {summary['total_frames']:,}")
    print(f"Total trajectories: {summary['total_episodes']:,}")
    print(f"Estimated duration: {fmt_hours(summary['estimated_hours'])}")


if __name__ == "__main__":
    main()
