"""Runtime-only dataset/video instrumentation safe for DataLoader workers."""

from __future__ import annotations

import functools
import heapq
import multiprocessing as mp
import os
import queue
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable


def _worker_id() -> int | None:
    try:
        from torch.utils.data import get_worker_info

        info = get_worker_info()
        return None if info is None else info.id
    except ImportError:
        return None


def _rank() -> int:
    for name in ("RANK", "LOCAL_RANK"):
        try:
            return int(os.environ.get(name, "0"))
        except ValueError:
            pass
    return 0


def _safe_put(event_queue: Any, event: dict[str, Any]) -> None:
    try:
        event_queue.put_nowait(event)
    except (queue.Full, BrokenPipeError, EOFError, OSError, AttributeError):
        pass


def _sampled(sample_rate: float) -> bool:
    return sample_rate >= 1 or (
        sample_rate > 0
        and hash((os.getpid(), time.time_ns())) % 1_000_000 < sample_rate * 1_000_000
    )


def _extract_dataset_meta(obj: Any) -> dict[str, Any]:
    current = getattr(obj, "current_ds", None)
    return {"repo_id": str(getattr(obj, "repo_id", getattr(current, "repo_id", "")))}


def _safe_index(value: Any) -> Any:
    """Only capture cheap, immutable index values (e.g. `int` for `__getitem__`).

    Some wrapped methods (e.g. `_build_prompt(self, current_sample)`) receive a
    mutable object (a dict) as their first positional argument that the caller
    continues to mutate in place immediately after the wrapped call returns
    (`current[BP_PREFIX] = self._build_prompt(current)`). Storing a reference to
    that object in the event dict and handing it to a background
    multiprocessing queue feeder thread races with that in-place mutation and
    can raise `RuntimeError: dictionary changed size during iteration` during
    pickling. Anything that is not already an immutable scalar is summarized
    instead of referenced.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return f"<{type(value).__name__}>"


def _dataset_wrapper(
    original: Callable[..., Any],
    event_queue: Any,
    stage: str,
    sample_rate: float,
    threshold_s: float,
) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            if _sampled(sample_rate) or elapsed >= threshold_s:
                _safe_put(
                    event_queue,
                    {
                        "kind": "sample",
                        "stage": stage,
                        "elapsed_s": elapsed,
                        "rank": _rank(),
                        "worker_id": _worker_id(),
                        "pid": os.getpid(),
                        "index": _safe_index(args[0]) if args else None,
                        "slow": elapsed >= threshold_s,
                        **_extract_dataset_meta(self),
                    },
                )

    wrapped.__bpva_instrumented__ = True
    return wrapped


def _video_wrapper(
    original: Callable[..., Any],
    event_queue: Any,
    sample_rate: float,
    threshold_s: float,
    stage: str,
) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(
        video_path: Any,
        timestamps: Any,
        tolerance_s: Any,
        backend: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        start = time.perf_counter()
        try:
            return original(
                video_path, timestamps, tolerance_s, backend, *args, **kwargs
            )
        finally:
            elapsed = time.perf_counter() - start
            if _sampled(sample_rate) or elapsed >= threshold_s:
                try:
                    requested = [float(value) for value in timestamps]
                except (TypeError, ValueError):
                    requested = str(timestamps)
                _safe_put(
                    event_queue,
                    {
                        "kind": "video",
                        "stage": stage,
                        "elapsed_s": elapsed,
                        "rank": _rank(),
                        "worker_id": _worker_id(),
                        "pid": os.getpid(),
                        "video_path": str(video_path),
                        "backend": str(backend),
                        "requested_timestamps": requested,
                        "slow": elapsed >= threshold_s,
                        "capability": "whole_decode_call_only",
                    },
                )

    wrapped.__bpva_instrumented__ = True
    return wrapped


class EventCollector:
    """Drain a bounded queue and retain bounded per-kind top-k events."""

    def __init__(
        self, max_queue: int = 4096, top_k: int = 100, context: str | None = None
    ):
        ctx = mp.get_context(context) if context else mp.get_context()
        self.queue = ctx.Queue(maxsize=max(1, max_queue))
        self.top_k = max(0, top_k)
        self._heaps: dict[str, list[tuple[float, int, dict[str, Any]]]] = {}
        self._seen: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._serial = 0
        self._lock = threading.RLock()
        self._stopped = False

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._drain, daemon=True)
            self._thread.start()
        return self

    def _drain(self):
        while not self._stop.is_set():
            try:
                self._accept(self.queue.get(timeout=0.1))
            except queue.Empty:
                continue
            except (EOFError, OSError):
                return

    def _accept(self, event: dict[str, Any]):
        with self._lock:
            self._accept_locked(event)

    def _accept_locked(self, event: dict[str, Any]):
        kind = str(event.get("kind", "unknown"))
        self._seen[kind] = self._seen.get(kind, 0) + 1
        self._serial += 1
        event = dict(event)
        event.setdefault(
            "event_id", f"r{event.get('rank', 0)}-p{event.get('pid', 0)}-{self._serial}"
        )
        if not self.top_k:
            return
        heap = self._heaps.setdefault(kind, [])
        item = (float(event.get("elapsed_s", 0)), self._serial, event)
        if len(heap) < self.top_k:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)

    def snapshot(self) -> dict[str, Any]:
        """Copy retained events and counters without stopping collection."""
        with self._lock:
            events = [dict(item[2]) for kind in sorted(self._heaps) for item in sorted(self._heaps[kind], reverse=True)]
            retained = {kind: len(heap) for kind, heap in self._heaps.items()}
            seen = dict(self._seen)
        return {"top_events": events, "stats": {"seen": seen, "retained": retained, "dropped": {kind: max(0, count - retained.get(kind, 0)) for kind, count in seen.items()}}}

    @property
    def stopped(self) -> bool:
        return self._stopped

    def stop(self) -> list[dict[str, Any]]:
        if self._stopped:
            return self.snapshot()["top_events"]
        self._stopped = True
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        while True:
            try:
                self._accept(self.queue.get_nowait())
            except (queue.Empty, EOFError, OSError):
                break
        try:
            self.queue.close()
            self.queue.cancel_join_thread()
        except (AttributeError, OSError, ValueError):
            pass
        return self.top_events

    @property
    def top_events(self) -> list[dict[str, Any]]:
        return self.snapshot()["top_events"]

    @property
    def stats(self) -> dict[str, Any]:
        return self.snapshot()["stats"]


class DataInstrumentation(AbstractContextManager):
    def __init__(
        self,
        *,
        event_queue: Any,
        sample_rate: float = 1.0,
        slow_sample_s: float = 1.0,
        slow_video_s: float = 0.5,
        video: bool = True,
    ):
        self.event_queue = event_queue
        self.sample_rate = sample_rate
        self.slow_sample_s = slow_sample_s
        self.slow_video_s = slow_video_s
        self.video = video
        self._patches: list[tuple[Any, str, Any]] = []

    def _patch(self, owner: Any, name: str, replacement: Any):
        if owner is not None and hasattr(owner, name):
            old = getattr(owner, name)
            if getattr(old, "__bpva_instrumented__", False):
                return
            self._patches.append((owner, name, old))
            setattr(owner, name, replacement(old))

    def install(self):
        try:
            return self._install()
        except BaseException:
            self.uninstall()
            raise

    def _install(self):
        from lerobot.datasets.behavior_prompt_dataset import (
            BehaviorPromptLeRobotDataset,
        )
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._patch(
            BehaviorPromptLeRobotDataset,
            "__getitem__",
            lambda old: _dataset_wrapper(
                old,
                self.event_queue,
                "bp_dataset_getitem",
                self.sample_rate,
                self.slow_sample_s,
            ),
        )
        self._patch(
            BehaviorPromptLeRobotDataset,
            "_build_prompt",
            lambda old: _dataset_wrapper(
                old,
                self.event_queue,
                "bp_build_prompt",
                self.sample_rate,
                self.slow_sample_s,
            ),
        )
        self._patch(
            LeRobotDataset,
            "__getitem__",
            lambda old: _dataset_wrapper(
                old,
                self.event_queue,
                "lerobot_getitem",
                self.sample_rate,
                self.slow_sample_s,
            ),
        )
        if self.video:
            from lerobot.datasets import lerobot_dataset, video_utils

            whole = lambda old: _video_wrapper(
                old,
                self.event_queue,
                self.sample_rate,
                self.slow_video_s,
                "decode_video_frames",
            )
            torchvision = lambda old: _video_wrapper(
                old,
                self.event_queue,
                self.sample_rate,
                self.slow_video_s,
                "decode_video_frames_torchvision",
            )
            self._patch(video_utils, "decode_video_frames", whole)
            self._patch(lerobot_dataset, "decode_video_frames", whole)
            self._patch(video_utils, "decode_video_frames_torchvision", torchvision)
        return self

    def uninstall(self):
        for owner, name, old in reversed(self._patches):
            setattr(owner, name, old)
        self._patches.clear()

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.uninstall()


_WORKER_INSTRUMENTATION: DataInstrumentation | None = None


@dataclass
class WorkerInstrumentation:
    """Picklable DataLoader worker initializer for spawn and fork contexts."""

    event_queue: Any
    sample_rate: float = 1.0
    slow_sample_s: float = 1.0
    slow_video_s: float = 0.5
    video: bool = True

    def __call__(self, worker_id: int) -> None:
        del worker_id
        global _WORKER_INSTRUMENTATION
        _WORKER_INSTRUMENTATION = DataInstrumentation(
            event_queue=self.event_queue,
            sample_rate=self.sample_rate,
            slow_sample_s=self.slow_sample_s,
            slow_video_s=self.slow_video_s,
            video=self.video,
        ).install()
