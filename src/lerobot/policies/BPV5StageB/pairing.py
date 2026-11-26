"""Stage B M1 demo pairing: L0 (same leaf) / L1 (same leaf + exact task text).

Source tiers for the current Stage B mix (see 数据集总览与示教配对v1.md §3):
  agibot / interndata / robotwin → L0
  droid / molmoact               → L1  (one leaf, many tasks)

Bucket rule (doc): size > 3 and i≠j. Placeholder task text "robot demonstration"
is not a valid L1 key (DROID empty instructions).

HOM wrong (eval only; not used as a training partner):
  different task, same source group, prefer same robot_type.
  RoboTwin clean vs random of the same task name are the SAME task (not wrong).
"""

from __future__ import annotations

import logging
import operator
import time
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import overload

from lerobot.policies.BPV5QFormer import data80d as D

TIERS = ("L0", "L1")
# Top-level source under cleaned_root. Unlisted sources default to L0.
SOURCE_TIER: dict[str, str] = {
    "agibot": "L0",
    "interndata": "L0",
    "robotwin": "L0",
    "robotwin25": "L0",
    "droid": "L1",
    "molmoact": "L1",
}
INVALID_L1_TEXT = frozenset({"", "robot demonstration"})
# Paths like robotwin/<task>/<variant>: variant dirs are not separate tasks for wrong.
_ROBOTWIN_SOURCES = frozenset({"robotwin", "robotwin25"})


def source_of(rel: str) -> str:
    return rel.split("/", 1)[0]


def task_identity(rel: str) -> str:
    """Task id for wrong pairing (merge RoboTwin clean/random variants)."""
    parts = rel.split("/")
    if parts and parts[0] in _ROBOTWIN_SOURCES and len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return rel


def robot_key(rel: str) -> str:
    """Embodiment key from leaf meta; empty/missing → 'unknown'."""
    try:
        raw = D.info(rel).get("robot_type")
    except Exception:
        raw = None
    text = str(raw or "").strip().lower()
    return text or "unknown"


def pairing_tier(rel: str, overrides: dict[str, str] | None = None) -> str:
    src = source_of(rel)
    if overrides and src in overrides:
        tier = overrides[src]
    else:
        tier = SOURCE_TIER.get(src, "L0")
    if tier not in TIERS:
        raise ValueError(f"pairing tier for {src} must be L0 or L1, got {tier!r}")
    return tier


@dataclass(frozen=True)
class SharedPartnerCandidates(Sequence[int]):
    """Pickleable read-only view excluding one episode from an immutable bucket.

    ``excluded_positions`` contains physical bucket positions minus their ordinal
    among the exclusions. It is shared per episode, not rebuilt per query.
    """

    bucket: tuple[int, ...]
    excluded: int
    excluded_positions: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.bucket) - len(self.excluded_positions)

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[int, ...]: ...

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("candidate index out of range")
        # One exclusion is O(1); repeated episodes require O(log occurrences).
        return self.bucket[index + bisect_right(self.excluded_positions, index)]

    def __iter__(self) -> Iterator[int]:
        return (ep for ep in self.bucket if ep != self.excluded)


def _shared_bucket(bucket: Sequence[int]) -> tuple[tuple[int, ...], dict[int, tuple[int, ...]]]:
    """Freeze and index a bucket once, retaining order and all duplicates."""
    frozen = tuple(int(ep) for ep in bucket)
    positions: dict[int, list[int]] = defaultdict(list)
    for position, ep in enumerate(frozen):
        positions[ep].append(position - len(positions[ep]))
    return frozen, {ep: tuple(indices) for ep, indices in positions.items()}


def choose_partner(candidates: Sequence[int], current: int, index: int) -> int:
    if isinstance(candidates, SharedPartnerCandidates) and candidates.excluded == int(current):
        if not candidates:
            raise ValueError(f"no partner for episode {current}")
        return candidates[int(index) % len(candidates)]
    others = [int(ep) for ep in candidates if int(ep) != int(current)]
    if not others:
        raise ValueError(f"no partner for episode {current}")
    return others[int(index) % len(others)]


def choose_wrong_partner(
    query_rel: str,
    query_ep: int,
    pool_eps: dict[str, list[int]],
    *,
    index: int,
    tier_overrides: dict[str, str] | None = None,
) -> tuple[str, int] | None:
    """Pick a HOM wrong demo: different task, same source; prefer same robot.

    Never crosses top-level source groups (agibot / interndata / …).
    RoboTwin ``…/clean_50`` vs ``…/randomized_500`` of the same task name are
    excluded (same ``task_identity``). For L1 leaves, also allows another
    episode in the same leaf with a different normalized task string.
    """
    src = source_of(query_rel)
    q_task = task_identity(query_rel)
    q_robot = robot_key(query_rel)
    q_ep = int(query_ep)

    same_robot: list[tuple[str, int]] = []
    same_source: list[tuple[str, int]] = []

    for rel, eps in pool_eps.items():
        if not eps or source_of(rel) != src:
            continue
        if task_identity(rel) == q_task:
            # Same physical task leaf (or RoboTwin variant). L1 may still use
            # a different task-text episode inside this leaf.
            if pairing_tier(rel, tier_overrides) != "L1" or rel != query_rel:
                continue
            q_text = task_key(query_rel, q_ep)
            if q_text is None:
                continue
            for ep in eps:
                ep = int(ep)
                if ep == q_ep:
                    continue
                other_text = task_key(rel, ep)
                if other_text is None or other_text == q_text:
                    continue
                item = (rel, ep)
                if robot_key(rel) == q_robot:
                    same_robot.append(item)
                same_source.append(item)
            continue
        # Different task leaf: take a deterministic episode from that leaf.
        ep = int(eps[int(index) % len(eps)])
        item = (rel, ep)
        if robot_key(rel) == q_robot:
            same_robot.append(item)
        same_source.append(item)

    pool = same_robot or same_source
    if not pool:
        return None
    return pool[int(index) % len(pool)]


def task_key(rel: str, ep: int, raw=None) -> str | None:
    """Normalized task string for an L1 key, or None if unusable."""
    if raw is None:
        text = D.task_text(rel, int(ep)).strip()
    else:
        text = D.normalize_task_text(raw).strip()
    if text in INVALID_L1_TEXT:
        return None
    return text


def _l1_text_by_ep(rel: str, pool: list[int]) -> dict[int, str | None]:
    """Bulk-load episode tasks once per leaf, then normalize only pool eps."""
    raw = D.episode_tasks(rel)
    return {int(ep): task_key(rel, int(ep), raw.get(int(ep))) for ep in pool}


def build_partner_map(
    items: list[tuple[str, int, int]],
    pool_eps: dict[str, list[int]],
    *,
    min_bucket: int = 4,
    tier_overrides: dict[str, str] | None = None,
    storage: str = "list",
    timing: bool = False,
) -> tuple[dict[tuple[str, int], Sequence[int]], dict[str, int]]:
    """Map (rel, query_ep) → partner episode ids under M1.

    Partners are drawn from ``pool_eps[rel]`` (same leaf). L1 further restricts
    to the same normalized task string. Queries whose M1 bucket has size
    ``< min_bucket`` are omitted from the map (caller should drop those items).
    ``storage="list"`` preserves legacy lists; ``"shared_bucket"`` returns
    read-only views sharing immutable buckets and per-episode exclusion indices.
    Bucket size is checked before excluding the query in either mode.
    """
    if min_bucket < 2:
        raise ValueError("min_bucket must be >= 2")

    if storage not in {"list", "shared_bucket"}:
        raise ValueError("storage must be 'list' or 'shared_bucket'")
    setup_started = time.perf_counter() if timing else 0.0

    # Precompute texts only for L1 leaves that appear in items.
    text_by_ep: dict[str, dict[int, str | None]] = {}
    l1_by_text: dict[str, dict[str, list[int]]] = {}
    for rel, _ep, _ in items:
        if pairing_tier(rel, tier_overrides) != "L1":
            continue
        if rel in text_by_ep:
            continue
        pool = [int(e) for e in pool_eps.get(rel, [])]
        text_by_ep[rel] = _l1_text_by_ep(rel, pool)
        buckets: dict[str, list[int]] = defaultdict(list)
        for ep, key in text_by_ep[rel].items():
            if key is not None:
                buckets[key].append(ep)
        l1_by_text[rel] = buckets

    if timing:
        logging.info("bpv5 stage B M1 L1 setup elapsed=%.6fs", time.perf_counter() - setup_started)
    mapping_started = time.perf_counter() if timing else 0.0

    # Only the opt-in path freezes/indexes pools; each leaf is processed once.
    shared_pools = {}
    shared_l1 = {}
    if storage == "shared_bucket":
        for rel in dict.fromkeys(rel for rel, _ep, _ in items):
            shared_pools[rel] = _shared_bucket(pool_eps.get(rel, []))
        shared_l1 = {
            rel: {key: _shared_bucket(bucket) for key, bucket in buckets.items()}
            for rel, buckets in l1_by_text.items()
        }

    partners: dict[tuple[str, int], Sequence[int]] = {}
    dropped = defaultdict(int)
    kept = defaultdict(int)
    for rel, ep, _ in items:
        ep = int(ep)
        tier = pairing_tier(rel, tier_overrides)
        pool = shared_pools[rel][0] if storage == "shared_bucket" else [int(e) for e in pool_eps.get(rel, [])]
        used_tier = tier
        if tier == "L0":
            bucket = pool
        else:
            key = text_by_ep.get(rel, {}).get(ep)
            bucket = l1_by_text[rel].get(key, []) if key is not None else []
            # Prefer L1; if text unusable or bucket too small, fall back to L0 (same leaf).
            if key is None or len(bucket) < int(min_bucket):
                dropped[f"{source_of(rel)}:L1_to_L0"] += 1
                bucket = pool
                used_tier = "L0"
        if len(bucket) < int(min_bucket):
            dropped[f"{source_of(rel)}:{used_tier}_small_bucket"] += 1
            continue
        if storage == "shared_bucket":
            frozen, positions = shared_pools[rel] if used_tier == "L0" else shared_l1[rel][key]
            others = SharedPartnerCandidates(frozen, ep, positions.get(ep, ()))
        else:
            others = [e for e in bucket if e != ep]
        if not others:
            dropped[f"{source_of(rel)}:{used_tier}_no_partner"] += 1
            continue
        partners[(rel, ep)] = others
        kept[f"{source_of(rel)}:{used_tier}"] += 1

    logging.info(
        "bpv5 stage B M1 partners: kept=%s dropped=%s min_bucket=%d",
        dict(kept),
        dict(dropped),
        min_bucket,
    )
    if timing:
        logging.info("bpv5 stage B M1 mapping elapsed=%.6fs", time.perf_counter() - mapping_started)
    return partners, {"kept": sum(kept.values()), "dropped": sum(dropped.values())}
