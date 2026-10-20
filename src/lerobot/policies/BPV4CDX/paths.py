"""Repo lists (data root + relative `<task>/<variant>` names), canonical task text and the held-out split rule."""

from __future__ import annotations

from pathlib import Path

GENERIC_TEXT = "reconstruct the demonstrated action segment"


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


def task_name(repo: str | Path) -> str:
    """`<data_root>/<task>/<variant>` -> `<task>`."""
    return Path(repo).parent.name


def canonical_text(repo: str | Path) -> str:
    """One fixed text per task, from the task name only (no arm, object appearance or position)."""
    return task_name(repo).replace("_", " ")


def holdout_count(repo: str | Path, rules: dict[str, int]) -> int:
    """Number of held-out episodes at the end of a repo; rules match substrings of the repo path."""
    for key, n in rules.items():
        if key in str(repo):
            return int(n)
    return 0


def split_episodes(repo: str | Path, total: int, split: str, rules: dict[str, int]) -> range:
    hold = holdout_count(repo, rules)
    if split == "all":
        return range(total)
    if split == "val":
        return range(total - hold, total)
    if split == "train":
        return range(total - hold)
    raise ValueError(f"unknown split {split!r}")
