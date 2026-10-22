"""Repo lists, machine-independent repo ids, task names / canonical texts and the held-out split rules."""

from __future__ import annotations

import json
import re
from pathlib import Path

GENERIC_TEXT = "reconstruct the demonstrated action segment"

# machine-specific dataset roots, stripped to get a relative repo id shared by all machines (AE-3 plan 3.1)
ROOT_PREFIXES = ("/public-data/dataset/", "/share/", "/vla/workspace/data/")
# the same dataset stored under different names on different machines
ALIASES = {"lerobot_out/agibotworld": "agibotworld", "robochallenge": "RoboChallenge-lerobotv3.0",
           "robotwin2.0": "RoboTwin-LeRobot-v3.0"}


def read_repo_list(repo_file: str | Path, data_root: str | Path | None = None) -> list[str]:
    """One repo per line; relative lines are joined with `data_root`, absolute lines are kept."""
    repos = []
    for line in Path(repo_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        if not path.is_absolute():
            if data_root is None:
                raise ValueError(f"relative repo {line!r} in {repo_file} needs data_root")
            path = Path(data_root) / path
        repos.append(str(path))
    return repos


def rel_id(repo: str | Path) -> str:
    """Absolute repo path -> id shared by all machines, e.g.
    /share/RoboTwin-LeRobot-v3.0/place_shoe/aloha-agilex_clean_50 -> RoboTwin-LeRobot-v3.0/place_shoe/aloha-agilex_clean_50."""
    p = str(repo).rstrip("/")
    for pre in ROOT_PREFIXES:
        if p.startswith(pre):
            p = p[len(pre):]
            break
    else:
        p = p.lstrip("/")
    for old, new in ALIASES.items():
        if p == old or p.startswith(old + "/"):
            p = new + p[len(old):]
            break
    return p


def resolve_rel(rel: str, roots: list[str | Path]) -> str | None:
    """Relative repo id -> an existing local path under one of `roots` (aliases tried both ways)."""
    names = {rel}
    for old, new in ALIASES.items():
        if rel.startswith(new + "/"):
            names.add(old + rel[len(new):])
    for root in roots:
        for name in sorted(names):
            path = Path(root) / name
            if (path / "meta" / "info.json").is_file():
                return str(path)
    return None


def is_robotwin_like(repo: str | Path) -> bool:
    """RoboTwin layout `<task>/aloha-agilex_<variant>` (RoboTwin data and bp_probe)."""
    return Path(repo).name.startswith("aloha-agilex")


_SUFFIXES = (re.compile(r"_part\d+$"), re.compile(r"(_(left|right))+(_arm)?$"),
             re.compile(r"_(left|right)_arms?(?=_|$)"))  # the last one anywhere: pull_x_left_arm_top -> pull_x_top


def task_name(repo: str | Path) -> str:
    """Task of a repo. RoboTwin layout `<task>/<variant>` -> `<task>` (unchanged since AE-1). Other datasets keep one
    task per folder; trailing arm / part markers (`_left_arm`, `_part1`, `_left_right`, ...) are removed so that the
    task name, and the canonical text built from it, carries no arm information (G1)."""
    if is_robotwin_like(repo):
        return Path(repo).parent.name
    path = Path(repo)
    if "pick_and_place_tasks" in path.parts:
        # InternData pick_and_place: <robot>/<category>/<object>, e.g. single_pick_left_arm/google_scan-book
        return f"{_clean(path.parent.name)}_{_clean(path.name)}"
    return _clean(path.name)


def _clean(name: str) -> str:
    changed = True
    while changed:
        changed = False
        for rx in _SUFFIXES:
            new = rx.sub("", name)
            if new != name and new:
                name, changed = new, True
    return name


def canonical_text(repo: str | Path) -> str:
    """One fixed text per task, from the task name only (no arm, object appearance or position)."""
    return task_name(repo).replace("_", " ")


def holdout_count(repo: str | Path, rules: dict[str, int]) -> int:
    """Number of held-out episodes at the end of a repo; rules match substrings of the repo path."""
    for key, n in rules.items():
        if key in str(repo):
            return int(n)
    return 0


def split_episodes(repo: str | Path, total: int, split: str, rules: dict[str, int],
                   heldout: list[int] | None = None) -> list[int]:
    """Episodes of `split`. With an explicit held-out list (split manifest) it is used as is; otherwise the last
    `holdout_count` episodes are held out (AE-1 / AE-2 rule)."""
    if heldout is None:
        hold = holdout_count(repo, rules)
        heldout = list(range(total - hold, total))
    held = set(int(e) for e in heldout)
    if split == "all":
        return list(range(total))
    if split == "val":
        return [e for e in range(total) if e in held]
    if split == "train":
        return [e for e in range(total) if e not in held]
    raise ValueError(f"unknown split {split!r}")


# --- AE-3 split manifest -----------------------------------------------------------------------------------------
def ae3_heldout(rel: str, total: int) -> list[int]:
    """AE-3 held-out rule (plan 3.2): RoboTwin clean_50 last 5; RoboTwin randomized_* episodes 90-99 (= the AE-2
    randomized_100 held-out, the first 100 of randomized_500 are the same episodes); every other repo the last
    k = min(10, max(2, round(1% of the episodes)))."""
    name = Path(rel).name
    if is_robotwin_like(rel):
        if "clean_50" in name:
            return list(range(max(0, total - 5), total))
        if "randomized" in name:
            return [e for e in range(90, 100) if e < total]
    k = min(10, max(2, round(0.01 * total)))
    k = min(k, max(0, total - 1))
    return list(range(total - k, total))


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest_repos(manifest: dict, roots: list[str | Path], sources: list[str] | None = None) -> list[tuple[str, dict]]:
    """[(local path, manifest entry)] for the manifest repos that exist locally (optionally only some sources)."""
    out = []
    for entry in manifest["repos"]:
        if sources and entry["rel"].split("/")[0] not in sources:
            continue
        local = resolve_rel(entry["rel"], roots)
        if local is not None:
            out.append((local, entry))
    return out
