"""Local instrumentation of the real trainer, never an alternate training loop."""
import ast
from contextlib import ExitStack, contextmanager
import inspect
import json
import sys
import textwrap
import time
from unittest.mock import patch

import torch


def loop_boundaries(train):
    """Resolve source statements (currently 1250/1547), fail closed on drift."""
    function = inspect.unwrap(train)
    source, first = inspect.getsourcelines(function)
    tree = ast.parse(textwrap.dedent("".join(source)))
    definition = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    loops = [(i, n) for i, n in enumerate(definition.body)
             if isinstance(n, ast.While) and ast.unparse(n.test) == "step < cfg.steps"]
    if len(loops) != 1:
        raise RuntimeError("Cannot identify unique train step loop")
    index, loop = loops[0]
    after = definition.body[index + 1]
    return function.__code__, first + loop.body[0].lineno - 1, first + after.lineno - 1


@contextmanager
def measure_training_loop(trainer, controller):
    import accelerate.data_loader as accelerated_loader

    code, start_line, end_line = loop_boundaries(trainer.train)
    if sys.gettrace() is not None:
        raise RuntimeError("Refusing to replace an existing Python tracer")
    active = False
    h2d_depth = 0

    def complete(now):
        nonlocal active
        if not active:
            return
        record = controller.current
        if record.get("step") != controller.step or "update_wall_s" not in record:
            raise RuntimeError("Iteration boundary reached without a completed update")
        record["iteration_end_monotonic_s"] = now
        record["iteration_wall_s"] = now - record["iteration_start_monotonic_s"]
        record["post_update_wall_s"] = now - record.pop("_update_return_monotonic_s")
        with (controller.directory / "steps.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if record["mode"] == "steady":
            controller.steady.append(dict(record))
        active = False

    def trace(frame, event, arg):
        nonlocal active
        if frame.f_code is not code:
            return None
        if event == "line" and frame.f_lineno in (start_line, end_line):
            now = time.perf_counter()
            complete(now)
            if frame.f_lineno == start_line:
                if frame.f_locals["step"] != controller.step:
                    raise RuntimeError("Trainer and profile step counters diverged")
                controller.current = {
                    "iteration_start_monotonic_s": now,
                    "dataloader_next_wall_s": 0.0,
                    "h2d": [], "gpu_telemetry": [],
                }
                active = True
        # Final iteration ends at the first statement after while, not train() return
        # (which includes distributed cleanup). Exceptions leave partial_record intact.
        return trace

    original_cycle = trainer.cycle
    original_pop = trainer._pop_worker_timing
    original_stats = trainer._collect_gpu_stats
    original_send = accelerated_loader.send_to_device
    original_update = trainer.update_policy

    def cycle(loader):
        iterator = original_cycle(loader)
        while True:
            started = time.perf_counter()
            try:
                value = next(iterator)
            except StopIteration:
                return
            if active:
                controller.current["dataloader_next_wall_s"] += time.perf_counter() - started
            yield value

    def pop_timing(batch):
        # Count samples before the original pops _timing; do not retain tensors.
        timing = batch.get("_timing") if isinstance(batch, dict) else None
        samples = timing.get("getitem_s") if isinstance(timing, dict) else None
        count = samples.numel() if isinstance(samples, torch.Tensor) else None
        result = original_pop(batch)
        if active:
            controller.current["worker_timing"] = result
            controller.current["worker_timing_sample_count"] = count
        return result

    def send(*args, **kwargs):
        nonlocal h2d_depth
        if not active or h2d_depth:
            return original_send(*args, **kwargs)
        # Accelerate's helper recursively calls its module global; count only the
        # outer batch call. Steps 11-30 must retain the real asynchronous path.
        diagnostic = controller.step >= 30
        if diagnostic:
            controller.sync()
        started = time.perf_counter()
        h2d_depth += 1
        try:
            result = original_send(*args, **kwargs)
        finally:
            h2d_depth -= 1
        enqueued = time.perf_counter()
        record = {"enqueue_wall_s": enqueued - started, "diagnostic_sync": diagnostic}
        if diagnostic:
            controller.sync()
            record["synchronized_wall_s"] = time.perf_counter() - started
            record["completion_wait_s"] = record["synchronized_wall_s"] - record["enqueue_wall_s"]
        controller.current["h2d"].append(record)
        return result

    def stats(*args, **kwargs):
        started = time.perf_counter()
        result = original_stats(*args, **kwargs)
        if active:
            controller.current["gpu_telemetry"].append({
                "wall_s": time.perf_counter() - started, "stats": result})
        return result

    def update(*args, **kwargs):
        result = original_update(*args, **kwargs)
        if active:
            controller.current["_update_return_monotonic_s"] = time.perf_counter()
        return result

    with ExitStack() as patches:
        patches.enter_context(worker_instrumentation())
        patches.enter_context(patch.object(trainer, "cycle", cycle))
        patches.enter_context(patch.object(trainer, "_pop_worker_timing", pop_timing))
        patches.enter_context(patch.object(trainer, "_collect_gpu_stats", stats))
        patches.enter_context(patch.object(trainer, "update_policy", update))
        patches.enter_context(patch.object(accelerated_loader, "send_to_device", send))
        sys.settrace(trace)
        try:
            yield
            if active:
                raise RuntimeError("Train exited without final iteration boundary")
        finally:
            sys.settrace(None)


def aggregate_windows(output, rank):
    """Last completing rank exports whole-window max; no extra collective/wait."""
    paths = [output / f"rank_{r:02d}" / "summary.json" for r in range(4)]
    if not all(path.is_file() for path in paths):
        return
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(row.get("status") != "complete" or row.get("count") != 20 for row in summaries):
        raise RuntimeError("Incomplete rank window summary")
    elapsed = [row["steady_window_elapsed_s"] for row in summaries]
    maximum = max(elapsed)
    result = {
        "schema_version": 3, "status": "complete", "steady_steps": [11, 30],
        "count": 20, "rank_window_elapsed_s": dict(enumerate(elapsed)),
        "max_rank_window_elapsed_s": maximum,
        "global_updates_per_s": 20 / maximum,
        "global_trajectories_per_s": 20 * 8 / maximum,
        "global_segments_per_s": 20 * 32 / maximum,
        "semantics": "max of each rank continuous full 20-step window; NOT sum of per-step rank maxima; no added barrier",
    }
    temporary = output / f"e2e_summary.rank_{rank:02d}.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output / "e2e_summary.json")


@contextmanager
def worker_instrumentation():
    """Add missing Stage B worker timing, inherited by Linux fork workers."""
    from lerobot.policies.BPV5StageB.dataset import BPV5StageBDataset

    cls = BPV5StageBDataset
    original_getitem = cls.__getitem__
    with ExitStack() as patches:
        def wrap(method, label):
            original = getattr(cls, method)

            def measured(self, *args, **kwargs):
                started = time.perf_counter()
                try:
                    return original(self, *args, **kwargs)
                finally:
                    timing = getattr(self, "_profile_worker_timing", None)
                    if timing is not None:
                        timing[label] += time.perf_counter() - started
            patches.enter_context(patch.object(cls, method, measured))

        wrap("_demo_episode", "cache_s")
        wrap("current_images", "frames_s")
        wrap("target", "xform_s")
        wrap("tokens", "xform_s")

        def getitem(self, index):
            started = time.perf_counter()
            timing = dict.fromkeys(("cache_s", "frames_s", "xform_s"), 0.0)
            self._profile_worker_timing = timing
            try:
                result = original_getitem(self, index)
                if "_timing" not in result:
                    timing["getitem_s"] = time.perf_counter() - started
                    result["_timing"] = {key: torch.tensor(value, dtype=torch.float32)
                                         for key, value in timing.items()}
                return result
            finally:
                self._profile_worker_timing = None
        patches.enter_context(patch.object(cls, "__getitem__", getitem))
        yield
