"""Runtime-only dataset/video instrumentation safe for DataLoader workers."""

from __future__ import annotations

import contextvars
import functools
import heapq
import itertools
import json
import multiprocessing as mp
import os
import queue
import threading
import time

from torch.utils.data import IterableDataset
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Iterator


BENCHMARK_PREFIX = "__benchmark_"
BENCHMARK_METADATA_KEY = "__benchmark_metadata_json"
_CONTEXT = contextvars.ContextVar("bpva_benchmark_context", default=())
_SAMPLE_EVENTS = contextvars.ContextVar("bpva_benchmark_sample_events", default=None)
_DECODE_CALL = contextvars.ContextVar("bpva_benchmark_decode_call", default=None)
_LOAD_COUNTER = itertools.count()


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



def _context_name() -> str:
    stack = _CONTEXT.get()
    return stack[-1] if stack else "unknown"


def _context_wrapper(original: Callable[..., Any], context: str, *, only_if_empty: bool = False):
    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        stack = _CONTEXT.get()
        if only_if_empty and stack:
            return original(*args, **kwargs)
        token = _CONTEXT.set((*stack, context))
        try:
            return original(*args, **kwargs)
        finally:
            _CONTEXT.reset(token)

    wrapped.__bpva_instrumented__ = True
    return wrapped


def _append_sample_event(event: dict[str, Any]) -> None:
    events = _SAMPLE_EVENTS.get()
    if events is not None:
        events.append(dict(event))


def _backend_wrapper(original: Callable[..., Any], backend_name: str):
    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        call = _DECODE_CALL.get()
        began = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            if call is not None:
                call.setdefault("backend_spans", []).append({
                    "backend": backend_name, "elapsed_s": time.perf_counter() - began
                })

    wrapped.__bpva_instrumented__ = True
    return wrapped


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
                event = {
                    "kind": "sample", "stage": stage, "elapsed_s": elapsed,
                    "rank": _rank(), "worker_id": _worker_id(), "pid": os.getpid(),
                    "index": _safe_index(args[0]) if args else None,
                    "slow": elapsed >= threshold_s, "context": _context_name(),
                    **_extract_dataset_meta(self),
                }
                _append_sample_event(event)
                _safe_put(event_queue, event)

    wrapped.__bpva_instrumented__ = True
    return wrapped


def _video_wrapper(
    original: Callable[..., Any], event_queue: Any, sample_rate: float,
    threshold_s: float, stage: str,
) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(video_path: Any, timestamps: Any, tolerance_s: Any, backend: Any = None,
                *args: Any, **kwargs: Any) -> Any:
        began = time.perf_counter()
        call: dict[str, Any] = {"backend_spans": []}
        token = _DECODE_CALL.set(call)
        error = None
        try:
            return original(video_path, timestamps, tolerance_s, backend, *args, **kwargs)
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            _DECODE_CALL.reset(token)
            elapsed = time.perf_counter() - began
            spans = call["backend_spans"]
            requested = str(backend)
            event = {
                "kind": "video", "stage": stage, "elapsed_s": elapsed,
                "rank": _rank(), "worker_id": _worker_id(), "pid": os.getpid(),
                "video_path": str(video_path), "requested_backend": requested,
                "effective_backend": spans[-1]["backend"] if spans else requested,
                "fallback": requested in {"pyav", "video_reader"} and any(
                    span["backend"] == "torchcodec" for span in spans
                ),
                "backend_spans": spans, "timestamp_count": len(timestamps),
                "slow": elapsed >= threshold_s, "error": error,
                "context": _context_name(),
            }
            _append_sample_event(event)
            if _sampled(sample_rate) or elapsed >= threshold_s:
                _safe_put(event_queue, event)

    wrapped.__bpva_instrumented__ = True
    return wrapped


def _instrument_sample(dataset: Any, sample_factory: Callable[[], Any], index: Any) -> Any:
    start_ns = time.perf_counter_ns()
    events: list[dict[str, Any]] = []
    token = _SAMPLE_EVENTS.set(events)
    try:
        sample = sample_factory()
    finally:
        _SAMPLE_EVENTS.reset(token)
    end_ns = time.perf_counter_ns()
    sample = dict(sample) if isinstance(sample, dict) else {"sample": sample}
    metadata = {
        "load_id": f"r{_rank()}-p{os.getpid()}-{next(_LOAD_COUNTER)}",
        "rank": _rank(), "worker_id": _worker_id(), "pid": os.getpid(),
        "index": _safe_index(index),
        "repo_id": _extract_dataset_meta(dataset)["repo_id"],
        "start_ns": start_ns, "end_ns": end_ns,
        "elapsed_s": (end_ns - start_ns) / 1e9, "events": events,
    }
    sample[BENCHMARK_METADATA_KEY] = json.dumps(metadata, separators=(",", ":"))
    return sample


class InstrumentedMapDataset:
    """Return collatable metadata that can be assigned a step after next()."""

    def __init__(self, dataset: Any):
        self.dataset = dataset
        for name in ("dataset_weights", "num_frames", "num_episodes", "meta", "datasets", "_lengths", "_cum_lengths"):
            if hasattr(dataset, name):
                setattr(self, name, getattr(dataset, name))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def __getitem__(self, index: Any) -> Any:
        return _instrument_sample(self.dataset, lambda: self.dataset[index], index)


class InstrumentedIterableDataset(IterableDataset):
    """Preserve IterableDataset recognition and instrument each yielded sample."""

    def __init__(self, dataset: IterableDataset):
        super().__init__()
        self.dataset = dataset

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def __iter__(self) -> Iterator[Any]:
        iterator = iter(self.dataset)
        sequence = 0
        while True:
            try:
                yield _instrument_iterable_sample(self.dataset, iterator, sequence)
            except StopIteration:
                return
            sequence += 1

    def __len__(self) -> int:
        length = getattr(self.dataset, "__len__", None)
        if not callable(length):
            raise TypeError(f"{type(self.dataset).__name__} has no length")
        return len(self.dataset)


def _iter_next(iterator: Iterator[Any]) -> Any:
    return next(iterator)


def _instrument_iterable_sample(dataset: Any, iterator: Iterator[Any], sequence: int) -> Any:
    return _instrument_sample(dataset, lambda: _iter_next(iterator), sequence)


def wrap_dataset_for_instrumentation(dataset: Any) -> Any:
    if isinstance(dataset, IterableDataset):
        return InstrumentedIterableDataset(dataset)
    return InstrumentedMapDataset(dataset)


def strip_benchmark_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_benchmark_metadata(item) for key, item in value.items()
                if not str(key).startswith(BENCHMARK_PREFIX)}
    if isinstance(value, list):
        return [strip_benchmark_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(strip_benchmark_metadata(item) for item in value)
    return value


def sample_records_from_batch(batch: Any, *, rank: int, step: int,
                              optimizer_step: int, microstep: int):
    if not isinstance(batch, dict) or BENCHMARK_METADATA_KEY not in batch:
        return [], []
    raw = batch[BENCHMARK_METADATA_KEY]
    values = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    rows = []
    for position, value in enumerate(values):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        metadata = json.loads(value)
        metadata.update({"report_step": step, "optimizer_step": optimizer_step,
                         "microstep": microstep, "sample_in_batch": position,
                         "delivery_rank": rank})
        rows.append(metadata)
    worker_rows = []
    worker_ids = sorted({row.get("worker_id") for row in rows}, key=lambda x: -1 if x is None else x)
    for worker_id in worker_ids:
        group = [row for row in rows if row.get("worker_id") == worker_id]
        start_ns = min(row["start_ns"] for row in group)
        end_ns = max(row["end_ns"] for row in group)
        worker_rows.append({"worker_id": worker_id, "pid": group[0].get("pid"),
                            "elapsed_s": (end_ns - start_ns) / 1e9,
                            "sample_count": len(group), "start_ns": start_ns,
                            "end_ns": end_ns})
    return rows, worker_rows


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
            lambda old: _context_wrapper(_dataset_wrapper(
                old, self.event_queue, "bp_build_prompt", self.sample_rate,
                self.slow_sample_s), "bp_prompt"),
        )
        self._patch(
            LeRobotDataset,
            "__getitem__",
            lambda old: _context_wrapper(_dataset_wrapper(
                old, self.event_queue, "lerobot_getitem", self.sample_rate,
                self.slow_sample_s), "current_obs", only_if_empty=True),
        )
        self._patch(
            LeRobotDataset, "_query_videos",
            lambda old: _context_wrapper(old, "current_obs", only_if_empty=True),
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
            self._patch(video_utils, "decode_video_frames_torchvision", lambda old: _backend_wrapper(old, "pyav"))
            self._patch(video_utils, "decode_video_frames_torchcodec", lambda old: _backend_wrapper(old, "torchcodec"))
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


@dataclass
class ComposedWorkerInit:
    original: Callable[[int], None] | None
    instrumentation: WorkerInstrumentation

    def __call__(self, worker_id: int) -> None:
        if self.original is not None:
            self.original(worker_id)
        self.instrumentation(worker_id)


def compose_worker_init(original: Callable[[int], None] | None,
                        instrumentation: WorkerInstrumentation) -> ComposedWorkerInit:
    return ComposedWorkerInit(original, instrumentation)
