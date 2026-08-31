"""Asynchronous CPU/CUDA timing for BPVA model stages."""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import Any

from .metrics import StageRecord


@dataclass
class _Pending:
    stage: str
    rank: int
    step: int | None
    cpu_s: float
    start: Any = None
    end: Any = None


class DeviceStageTimer:
    """Measure CPU wall and optional CUDA-event time without synchronizing on exit."""

    def __init__(self, stage: str, rank: int, step: int | None):
        self.stage = stage
        self.rank = rank
        self.step = step
        self.began = 0.0
        self.start = None
        self.end = None
        self.pending: _Pending | None = None

    @staticmethod
    def _events():
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
        except (ImportError, RuntimeError):
            pass
        return None, None

    def __enter__(self):
        self.began = time.perf_counter()
        self.start, self.end = self._events()
        if self.start is not None:
            self.start.record()
        return self

    def __exit__(self, *_):
        if self.end is not None:
            self.end.record()
        self.pending = _Pending(
            self.stage,
            self.rank,
            self.step,
            time.perf_counter() - self.began,
            self.start,
            self.end,
        )


def resolve_pending(pending: list[_Pending]) -> list[StageRecord]:
    if any(item.end is not None for item in pending):
        try:
            import torch

            torch.cuda.synchronize()
        except (ImportError, RuntimeError):
            pass
    records = []
    for item in pending:
        device = None
        if item.start is not None:
            try:
                device = float(item.start.elapsed_time(item.end)) / 1000.0
            except RuntimeError:
                pass
        records.append(
            StageRecord(
                item.stage,
                item.cpu_s,
                item.rank,
                item.step,
                device_elapsed_s=device,
            )
        )
    return records


class ModelInstrumentation:
    """Install removable BPVA hooks; resolve synchronizes CUDA once per microstep."""

    DEFAULT_PATTERNS = {
        "qwen_visual": ("und_expert.visual", "qwen3_vl_with_expert.und_expert.visual"),
        "bp_encoder": ("bp_obs_encoder",),
        "mot": ("qwen3_vl_with_expert",),
        "da3": ("da3_teacher", "future_3d_output_decoder"),
        "cosmos": ("cosmos",),
    }
    METHOD_NAMES = (
        "embed_middle",
        "embed_suffix",
        "embed_prefix",
        "embed_prefix_with_behavior_prompt",
        "compute_3d_query_loss",
    )

    def __init__(self, model: Any, rank: int = 0):
        self.model = model
        self.rank = rank
        self.step: int | None = None
        self.records: list[StageRecord] = []
        self._handles: list[Any] = []
        self._methods: list[tuple[Any, str, Any]] = []
        self._pending: list[_Pending] = []
        self._active: dict[tuple[int, str], list[tuple[float, Any, Any]]] = {}

    def _register_module(self, module: Any, stage: str, seen: set[int]) -> None:
        if id(module) in seen:
            return
        seen.add(id(module))
        self._handles.extend(
            [
                module.register_forward_pre_hook(self._pre(stage)),
                module.register_forward_hook(self._post(stage)),
            ]
        )

    def install(self):
        modules = dict(self.model.named_modules())
        seen: set[int] = set()
        for stage, patterns in self.DEFAULT_PATTERNS.items():
            matches = [
                (name, module)
                for name, module in modules.items()
                if name
                and any(
                    name == pattern or name.startswith(pattern + ".")
                    for pattern in patterns
                )
            ]
            if matches:
                _, module = min(matches, key=lambda item: len(item[0]))
                self._register_module(module, stage, seen)

        key_map = modules.get("bp_obs_encoder.chunk_encoder.key_model_map")
        if key_map is not None:
            for module in key_map.values():
                self._register_module(module, "bp_vit", seen)

        for name in self.METHOD_NAMES:
            if hasattr(self.model, name):
                self._wrap_method(self.model, name, f"method.{name}")
        return self

    def _pre(self, stage: str):
        def hook(module, _inputs):
            timer = DeviceStageTimer(stage, self.rank, self.step)
            timer.__enter__()
            self._active.setdefault((id(module), stage), []).append(
                (timer.began, timer.start, timer.end)
            )

        return hook

    def _post(self, stage: str):
        def hook(module, _inputs, _output):
            stack = self._active.get((id(module), stage), [])
            if not stack:
                return
            began, start, end = stack.pop()
            if end is not None:
                end.record()
            self._pending.append(
                _Pending(
                    stage, self.rank, self.step, time.perf_counter() - began, start, end
                )
            )

        return hook

    def _wrap_method(self, owner: Any, name: str, stage: str):
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with DeviceStageTimer(stage, self.rank, self.step) as timer:
                result = original(*args, **kwargs)
            if timer.pending is not None:
                self._pending.append(timer.pending)
            return result

        self._methods.append((owner, name, original))
        setattr(owner, name, wrapped)

    def pop_pending(self, step: int | None = None) -> list[_Pending]:
        pending = self._pending
        self._pending = []
        for item in pending:
            if item.step is None:
                item.step = step
        return pending

    def resolve(self, step: int | None = None) -> list[StageRecord]:
        resolved = resolve_pending(self.pop_pending(step))
        self.records.extend(resolved)
        return resolved

    def uninstall(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for owner, name, original in reversed(self._methods):
            setattr(owner, name, original)
        self._methods.clear()

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.resolve()
        self.uninstall()
