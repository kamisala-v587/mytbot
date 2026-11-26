#!/usr/bin/env python3
"""Build stats_index.json for Stage A pretrain sources (agibot/interndata/droid/molmoact).

Maps each cleaned_root relative leaf → absolute mixed stats.json (delta joints / abs gripper·quat·vel).
Reuses already-computed AgiBot + InternData mixed stats; expects DROID / MolmoAct mixed outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/vla/workspace/cache/bpv5_cleaned_root")
STATS_PT = Path("/vla/workspace/my_tbot/norm_stats_bpv5_pt")

# source → (stats_dir relative to STATS_PT, robot_type key normalizer)
SOURCE_STATS = {
    "agibot": STATS_PT / "agibotworld_full_20261106_mixed_80d",
    "interndata": STATS_PT / "interndata_a1_v3_full_20261114_mixed_80d",
    "droid": STATS_PT / "droid_v3_clean20261114_mixed_80d",
    "molmoact": STATS_PT / "molmoact_v3_mixed_80d",
}


def discover_leaves(source: str) -> list[str]:
    """All LeRobot leaves under ROOT/<source> (any depth; follow symlinks)."""
    import os

    base = ROOT / source
    out: list[str] = []
    if not base.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(base, followlinks=True):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if "info.json" in filenames and Path(dirpath).name == "meta":
            leaf = Path(dirpath).parent
            try:
                rel = leaf.relative_to(ROOT)
            except ValueError:
                continue
            if any(part.startswith(".") for part in rel.parts):
                continue
            out.append(str(rel))
            dirnames.clear()
    return sorted(set(out))


def robot_type(rel: str) -> str:
    info = json.loads((ROOT / rel / "meta" / "info.json").read_text())
    return str(info.get("robot_type") or "unknown")


def resolve_stats(stats_dir: Path, rt: str) -> Path | None:
    """Match robot_type to a group folder (case-insensitive, slug-tolerant)."""
    if not stats_dir.is_dir():
        return None
    candidates = {p.name: p / "stats.json" for p in stats_dir.iterdir() if (p / "stats.json").is_file()}
    if rt in candidates:
        return candidates[rt]
    lower = {k.lower(): v for k, v in candidates.items()}
    if rt.lower() in lower:
        return lower[rt.lower()]
    # slug: spaces/underscores
    slug = rt.replace(" ", "_")
    if slug in candidates:
        return candidates[slug]
    if slug.lower() in lower:
        return lower[slug.lower()]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=STATS_PT / "stage_a_pretrain_mixed_80d",
    )
    ap.add_argument(
        "--sources",
        nargs="+",
        default=["agibot", "interndata", "droid", "molmoact"],
    )
    args = ap.parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    index: dict[str, str] = {}
    summary: dict = {"sources": {}, "missing": []}
    for src in args.sources:
        stats_dir = SOURCE_STATS[src]
        leaves = discover_leaves(src)
        ok = 0
        by_rt: dict[str, int] = {}
        for rel in leaves:
            rt = robot_type(rel)
            path = resolve_stats(stats_dir, rt)
            if path is None or not path.is_file():
                summary["missing"].append({"repo": rel, "robot_type": rt, "stats_dir": str(stats_dir)})
                continue
            index[rel] = str(path.resolve())
            ok += 1
            by_rt[rt] = by_rt.get(rt, 0) + 1
        summary["sources"][src] = {
            "stats_dir": str(stats_dir),
            "leaves": len(leaves),
            "indexed": ok,
            "by_robot_type": by_rt,
        }
        print(f"{src}: leaves={len(leaves)} indexed={ok} robots={by_rt}", flush=True)

    index_path = out_dir / "stats_index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    summary_path = out_dir / "summary.json"
    summary["n_index"] = len(index)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {index_path} n={len(index)} missing={len(summary['missing'])}", flush=True)
    if summary["missing"]:
        print("missing sample:", summary["missing"][:5], flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
