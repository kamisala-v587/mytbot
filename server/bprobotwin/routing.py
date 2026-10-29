"""Wrap any policy adapter for BP RoboTwin requests.

The client sends `task` (the candidate set's language) and `bp_candidate_set` with every frame.
- BP family (demonstration-conditioned, e.g. bpv4_xdemo / bpvav3x): `task_type` is replaced by the candidate set's key,
  so the adapter's usual "task_type -> demonstration dataset" lookup (BP mapping YAML generated from the index) draws
  the demonstration from that candidate set's LeRobot repo. Adapter-specific demo selection (per connection at reset,
  best / random, cached) is unchanged.
- VLA family: nothing is rewritten; the adapter reads `task` / `prompt` as language.
Responses get `bp_candidate_set` (and `bp_demo_repo` for BP) so the client can log what was used.
"""

from __future__ import annotations

import logging
from typing import Any

from .index import safe_key


class BPRoboTwinPolicy:
    def __init__(self, inner: Any, index: dict[str, dict], mapping_path: str) -> None:
        self.inner = inner
        self.index = index
        self.family = str(getattr(inner, "family", "vla"))
        self._metadata = {
            **dict(inner.metadata),
            "bprobotwin": True,
            "bp_candidate_sets": len(index),
            "bp_candidate_set_mapping": str(mapping_path),
            "bp_demo_source": "candidate set (bp_candidate_set)" if self.family == "bp" else "none (language only)",
        }
        self._warned_language: set[str] = set()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _route(self, obs: dict[str, Any]) -> str | None:
        cs = obs.get("bp_candidate_set")
        if not isinstance(cs, str) or not cs:
            if self.family == "bp":
                raise KeyError("BP RoboTwin request without bp_candidate_set")
            return None
        entry = self.index.get(cs)
        if entry is None:
            raise KeyError(f"unknown candidate set {cs!r} (not in the official BP RoboTwin index)")
        if self.family == "bp":
            obs["task_type"] = safe_key(cs)
        text = obs.get("prompt") or obs.get("task")
        if text != entry["instruction"] and cs not in self._warned_language:
            self._warned_language.add(cs)
            logging.warning("language of %s differs from its LeRobot instruction: %r", cs, text)
        return cs

    def on_request(self, connection_id: str, obs: dict[str, Any]) -> None:
        try:
            self._route(obs)
        except KeyError:
            pass  # reported by infer_batch
        self.inner.on_request(connection_id, obs)

    def infer_batch(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sets = [self._route(obs) for obs in observations]
        results = self.inner.infer_batch(observations)
        for result, cs in zip(results, sets):
            if cs is not None:
                result.setdefault("bp_candidate_set", cs)
                if self.family == "bp":
                    result.setdefault("bp_demo_repo", self.index[cs]["repo"])
        return results
