"""Instrumentation only. No model/dataset/optimizer training implementation here."""
from collections import defaultdict
from contextlib import ExitStack, contextmanager
import fnmatch
import json
import math
import os
import statistics
import time
import weakref
from unittest.mock import patch

import torch


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().item()
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


def storage(tensor):
    """Storage identity, not logical numel: views and shared weights count once."""
    if tensor.layout != torch.strided or tensor.device.type == "meta":
        return None
    s = tensor.untyped_storage()
    return (str(tensor.device), s._cdata), s.nbytes()


def group_name(name):
    # Exact owning module, with deterministic first-owner assignment for aliases.
    return name.rsplit(".", 1)[0] if "." in name else "<root>"


def storage_inventory(model, optimizer, *, include_records=False):
    """Non-overlapping physical storage by category and exact owning module."""
    seen = set()
    groups = defaultdict(lambda: defaultdict(int))
    skipped = defaultdict(int)
    aliases = defaultdict(int)
    records = []

    def add(t, owner, kind):
        info = storage(t)
        if info is None:
            skipped[kind] += 1
            return
        key, size = info
        if key in seen:
            aliases[kind] += 1
            return
        seen.add(key)
        groups[owner][f"{kind}:{t.device}"] += size
        if include_records:
            records.append(storage_record(key, size, owner, kind))

    names = {}
    for name, p in model.named_parameters():
        names[id(p)] = group_name(name)
        add(p, group_name(name), "parameter")
    for name, p in model.named_parameters():
        if p.grad is not None:
            add(p.grad, group_name(name), "gradient")
    # AcceleratedOptimizer exposes the real optimizer state through .state.
    for p, state in optimizer.state.items():
        for t in tensors(state):
            add(t, names.get(id(p), "<unmapped_optimizer_parameter>"), "optimizer_state")
    result = {"groups_bytes": dict(groups), "unsupported_tensor_counts": dict(skipped),
            "deduplicated_alias_counts": dict(aliases),
            "assignment": "global storage dedup; parameter > gradient > optimizer; first owner",
            "excludes": "buffers, DDP buckets, allocator slack, workspaces, activations"}
    if include_records:
        result["storage_records"] = records
    return result


def trainable_grad_none_inventory(model):
    """Diagnostic snapshot after backward, before optimizer step/zero_grad.

    A None gradient on this batch does not prove the parameter is always unused.
    """
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    names = [name for name, p in trainable if p.grad is None]
    return {"trainable_parameter_count": len(trainable), "grad_none_count": len(names),
            "grad_none_names": names,
            "scope": "current diagnostic backward only; not proof of globally unused parameters"}


def storage_record(key, size, owner, kind):
    return {"key": f"{key[0]}:{key[1]}", "device": key[0], "bytes": size,
            "owner": owner, "kind": kind}


def forward_memory_decomposition(inventory, saved, allocated, device):
    """Same-instant allocated accounting; identities are local to this snapshot."""
    categories = dict.fromkeys(("parameter", "optimizer_state", "gradient", "saved_activation"), 0)
    seen = set()
    excluded = set(saved["excluded_storage_keys"])
    if any(record["key"] in excluded for record in saved["storage_records"]):
        raise RuntimeError("Saved storage overlaps parameter/buffer/batch exclusion")
    for record in inventory["storage_records"] + saved["storage_records"]:
        if record["device"] != str(device):
            continue
        key = record["key"]
        if key in seen:
            raise RuntimeError(f"Forward storage categories overlap: {key}")
        seen.add(key)
        categories[record["kind"]] += record["bytes"]
    residual = allocated - sum(categories.values())
    if residual < 0:
        raise RuntimeError("Forward storage accounting exceeds same-instant CUDA allocated")
    categories["other_residual"] = residual
    return {"allocated_bytes": allocated, "device": str(device), "categories_bytes": categories,
            "categories_fraction": {k: v / allocated if allocated else 0 for k, v in categories.items()},
            "disjoint_storage_check": True, "sum_equals_allocated_check": True,
            "scope": "same-instant forward end CUDA allocated; NOT reserved or peak",
            "residual_includes": "buffers, batch inputs, unsaved activations, DDP buckets, workspaces, allocator rounding"}


class SavedActivations:
    """Return the ORIGINAL tensor to autograd; instrumentation retains weakrefs only.

    Returning a weakref as the packed object is incorrect: autograd must own saved
    tensors until backward. We add no ownership beyond autograd's normal pack.
    """
    def __init__(self, model, batch, stack):
        self.excluded = set()
        for t in (*model.parameters(), *model.buffers()):
            info = storage(t)
            if info:
                self.excluded.add(info[0])
        for t in tensors(batch):
            info = storage(t)
            if info:
                self.excluded.add(info[0])
        self.stack = stack
        self.refs = []
        self.forward_snapshot = None

    def pack(self, t):
        info = storage(t)
        if info and info[0] not in self.excluded:
            self.refs.append((weakref.ref(t), self.stack[-1] if self.stack else "<functional>"))
        return t

    @staticmethod
    def unpack(t):
        return t

    def snapshot(self):
        seen = set()
        groups = defaultdict(int)
        alive = 0
        records = []
        for ref, owner in self.refs:
            t = ref()
            if t is None:
                continue
            alive += 1
            info = storage(t)
            if info and info[0] not in seen and info[0] not in self.excluded:
                seen.add(info[0])
                groups[owner] += info[1]
                records.append(storage_record(info[0], info[1], owner, "saved_activation"))
        return {"storage_records": records,
                "excluded_storage_keys": sorted(f"{device}:{identity}" for device, identity in self.excluded),
                "groups_storage_bytes": dict(groups), "total_storage_bytes": sum(groups.values()),
                "live_tensor_refs": alive, "saved_pack_calls": len(self.refs),
                "scope": "live saved tensors at forward end, NOT peak memory decomposition",
                "exclusions": "parameter, buffer and batch-input storages; global storage dedup",
                "attribution": "first live save owner (deepest function/module scope), shared storage once"}


class ProfileController:
    def __init__(self, directory, cfg):
        self.directory = directory
        self.step = 0
        self.device = None
        self.current = {}
        self.steady = []
        self.initialized = False
        self.step_highwater = {"allocated_bytes": 0, "reserved_bytes": 0}
        write_json(directory / "metadata.json", {
            "schema_version": 3, "rank": int(os.environ.get("RANK", 0)),
            "world_size": int(os.environ.get("WORLD_SIZE", 1)), "torch_version": torch.__version__,
            "config": "configs/bpv5/stage_b/v2/test.jsonc", "resolved_config": cfg.to_dict(),
            "environment": {name: os.environ.get(name) for name in (
                "LEROBOT_DDP_FIND_UNUSED_PARAMETERS", "LEROBOT_DDP_TIMEOUT_SEC", "DDP_TIMEOUT_SEC",
                "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM",
                "NCCL_DEBUG", "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "BPV5_PROFILE_MODULES",
            )},
            "schedule": {"warmup": [1, 10], "steady": [11, 30], "phases": [31, 34],
                         "modules_and_saved": [35, 38], "torch_profiler": [39, 40],
                         "functions_and_forward_storage": [35, 40], "saved_hooks": [35, 40]},
            "units": {"memory": "bytes", "wall": "seconds", "events": "milliseconds",
                      "operators": "microseconds"},
            "steady_boundary": "train loop start through next loop start; includes loader/implicit H2D, update, telemetry, logging/pbar",
            "update_boundary": "synchronized update_policy only (subset of iteration)",
            "batch": {"per_rank_trajectories": cfg.batch_size,
                      "global_trajectories": cfg.batch_size * 4,
                      "global_segments": cfg.batch_size * 4 * cfg.dataset.segments_per_trajectory},
            "preflight": {"enabled": cfg.policy.preflight_memory_check,
                          "extra_real_training_steps": 2 if cfg.policy.preflight_memory_check else 0,
                          "scope": "before measured 40 steps; same batch, LR=0, optimizer state cleared; not a no-op"},
            "telemetry": {"gpu_stats_freq": cfg.gpu_stats_freq,
                          "scope": "rank 0 every N optimizer updates; 0 disables; other ranks may wait in logging collective"},
            "h2d": "Accelerate send_to_device top-level enqueue; synchronized transfer diagnosis only in steps 31-40",
            "worker_timing": "mean/max sample workload from _pop_worker_timing; prefetch overlaps train loop, NEVER add to iteration wall",
            "warning": "diagnostic wall times are synchronized/instrumented, not steady throughput",
        })

    def memory(self):
        if self.device is None:
            return {}
        return {"allocated_bytes": torch.cuda.memory_allocated(self.device),
                "reserved_bytes": torch.cuda.memory_reserved(self.device)}

    def peak(self):
        latest = self.local_peak()
        for key, value in latest.items():
            self.step_highwater[key] = max(self.step_highwater[key], value)
        return dict(self.step_highwater)

    def local_peak(self):
        if self.device is None:
            return {}
        return {"allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "reserved_bytes": torch.cuda.max_memory_reserved(self.device)}

    def sync(self):
        torch.cuda.synchronize(self.device)

    @contextmanager
    def phase(self, name, model=None, optimizer=None, saved=None):
        # Phase wrappers are siblings; never wrap an inner optimizer as well as
        # AcceleratedOptimizer. Merge high water before resetting phase counters.
        self.sync()
        before = self.memory()
        self.peak()
        torch.cuda.reset_peak_memory_stats(self.device)
        phase_record = {"name": name, "before": before, "status": "running"}
        self.current.setdefault("phases", []).append(phase_record)
        with torch.profiler.record_function(f"bpv5/phase/{name}"):
            started = time.perf_counter()
            try:
                yield
                self.sync()
            except BaseException:
                phase_record["status"] = "failed"
                phase_record["memory_at_failure"] = self.memory()
                raise
            else:
                phase_record.update({"wall_s": time.perf_counter() - started,
                                     "after": self.memory(), "status": "ok",
                                     "phase_peak": self.local_peak(),
                                     "whole_step_peak_so_far": self.peak()})
                phase_record["allocated_delta_bytes"] = (
                    phase_record["after"]["allocated_bytes"] - before["allocated_bytes"])
                if name == "forward" and saved is not None:
                    # No CUDA work occurs between allocator observation and storage enumeration.
                    inventory = storage_inventory(model, optimizer, include_records=True)
                    saved.forward_snapshot = saved.snapshot()
                    self.current["storage_forward_end"] = inventory
                    self.current["memory_forward_end"] = forward_memory_decomposition(
                        inventory, saved.forward_snapshot,
                        phase_record["after"]["allocated_bytes"], self.device)
                if name == "backward_ddp":
                    self.current["storage_after_backward"] = storage_inventory(model, optimizer)
                    self.current["trainable_grad_none_after_backward"] = trainable_grad_none_inventory(model)

    @contextmanager
    def phase_wrappers(self, values, model, saved=None):
        policy, opt, accel = values["policy"], values["optimizer"], values["accelerator"]
        with ExitStack() as patches:
            def wrap(obj, method, label):
                original = getattr(obj, method)

                def measured(*a, **kw):
                    with self.phase(label, model, opt, saved):
                        return original(*a, **kw)
                patches.enter_context(patch.object(obj, method, measured))

            wrap(policy, "forward", "forward")
            wrap(accel, "backward", "backward_ddp")
            wrap(opt, "step", "optimizer")
            wrap(opt, "zero_grad", "zero_grad")
            if values["grad_clip_norm"] > 0:
                wrap(accel, "clip_grad_norm_", "clip")
            else:
                wrap(torch.nn.utils, "clip_grad_norm_", "clip")
            scheduler = values.get("lr_scheduler")
            if scheduler is not None:
                wrap(scheduler, "step", "scheduler")
            if callable(getattr(model, "update", None)):
                wrap(model, "update", "policy_update")
            yield

    @contextmanager
    def module_hooks(self, model):
        # All modules supply attribution stack, but only selected boundaries
        # receive events/ranges. Default boundaries include transformer blocks.
        patterns = os.environ.get("BPV5_PROFILE_MODULES", "").split(",")
        patterns = [p.strip() for p in patterns if p.strip()]
        stack, pending, events, handles = [], defaultdict(list), [], []
        selected = []

        def enter(name, selected_module):
            def hook(module, args):
                functional = next((frame for frame in reversed(stack)
                                   if frame.startswith("function/") and "::" not in frame), None)
                stack.append(f"{functional}::{name}" if functional else name)
                if selected_module:
                    scope = torch.profiler.record_function(f"bpv5/module/{name}")
                    scope.__enter__()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    pending[name].append((start, end, scope))
            return hook

        def leave(name, selected_module):
            def hook(module, args, output):
                if selected_module and pending[name]:
                    start, end, scope = pending[name].pop()
                    end.record()
                    scope.__exit__(None, None, None)
                    events.append((name, start, end))
                if stack:
                    stack.pop()
            return hook

        try:
            for name, module in model.named_modules():
                name = name or "<root>"
                chosen = (any(fnmatch.fnmatchcase(name, p) for p in patterns) if patterns else
                          (name.count(".") <= 3 or
                           any(s in type(module).__name__.lower() for s in
                               ("decoderlayer", "encoderlayer", "block", "attention"))))
                if chosen:
                    selected.append(name)
                handles.append(module.register_forward_pre_hook(enter(name, chosen)))
                handles.append(module.register_forward_hook(leave(name, chosen), always_call=True))
            self.current["selected_modules"] = selected
            yield stack, events
        finally:
            for handle in handles:
                handle.remove()
            # Close ranges after exceptional forwards even if a pre-hook failed.
            for items in pending.values():
                for _, _, scope in reversed(items):
                    scope.__exit__(None, None, None)

    @contextmanager
    def functional_wrappers(self, policy, stack, events):
        """Only diagnostics 35–40: preserve bound/static calling convention and outputs."""
        model = getattr(policy, "model", policy)
        core = getattr(model, "core", None)
        backbone = getattr(core, "backbone", None)
        vlm_ids = set()
        if backbone is not None:
            language = getattr(backbone, "vlm_language_model", None)
            if language is not None:
                vlm_ids = {id(m) for m in language.modules()}

        with ExitStack() as patches:
            def wrap(obj, method, label):
                if obj is None or not callable(getattr(obj, method, None)):
                    return
                original = getattr(obj, method)

                def measured(*args, **kwargs):
                    name = label(args, kwargs) if callable(label) else label
                    stack.append("function/" + name)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    try:
                        with torch.profiler.record_function("bpv5/function/" + name):
                            start.record()
                            try:
                                return original(*args, **kwargs)
                            finally:
                                end.record()
                                events.append(("function/" + name, start, end))
                    finally:
                        stack.pop()
                patches.enter_context(patch.object(obj, method, measured))

            # demo_encoder already goes through nn.Module.__call__: existing hooks suffice.
            wrap(model, "build_prefix", "prefix/build_prefix")
            wrap(model, "_vision", "prefix/vision")
            wrap(getattr(backbone, "vlm_visual", None), "forward", "vision/visual_only")
            wrap(model, "_proprio", "prefix/proprio")
            wrap(core, "_forward_joint", "joint/forward")
            wrap(backbone, "forward_joint", "joint/backbone")
            wrap(getattr(backbone, "expert", None), "forward_joint", "joint/expert")
            wrap(backbone, "forward_vlm", "vlm/forward")
            wrap(backbone, "_run_vlm_full_layer", "vlm/full_attention_layer")
            wrap(backbone, "_run_linear_stream", lambda a, kw:
                 "vlm/linear_attention" if id(kw.get("mixer")) in vlm_ids else "expert/linear_attention")
            wrap(backbone, "_project_full", lambda a, kw:
                 ("vlm" if id(a[0] if a else kw.get("attention")) in vlm_ids else "expert") + "/attention_projection")
            wrap(backbone, "_run_attention", lambda a, kw:
                 "joint/attention" if any(frame == "function/joint/forward" for frame in stack)
                 else "vlm/attention")
            wrap(backbone, "_rotary_embeddings", "vlm/rotary_embeddings")
            self.current["functional_attribution"] = "function scopes share saved stack; child modules retain functional prefix"
            yield

    def export_profiler(self, profiler):
        stem = f"step_{self.step:02d}"
        profiler.export_chrome_trace(str(self.directory / f"{stem}.trace.json"))
        rows = []
        for event in profiler.key_averages():
            # Device terminology is version-dependent; old CUDA names are the
            # same metric. Never substitute inclusive cuda/device_time_total.
            device_self = getattr(event, "self_device_time_total", None)
            metric = "self_device_time_total"
            if device_self is None:
                device_self = getattr(event, "self_cuda_time_total", None)
                metric = "self_cuda_time_total"
            if device_self is None:
                raise RuntimeError("PyTorch profiler has no self device-time metric")
            rows.append({"name": event.key, "count": event.count,
                         "self_device_time_us": device_self,
                         "self_cpu_time_us": event.self_cpu_time_total, "source_metric": metric,
                         "device_type": str(event.device_type).rsplit(".", 1)[-1],
                         "is_user_annotation": bool(getattr(event, "is_user_annotation", False))})
        kernels = sorted((r for r in rows if r["device_type"] == "CUDA"),
                         key=lambda r: r["self_device_time_us"], reverse=True)
        cpu = sorted((r for r in rows if r["device_type"] == "CPU"),
                     key=lambda r: r["self_cpu_time_us"], reverse=True)
        write_json(self.directory / f"{stem}.operators.json", {
            "schema_version": 2, "operators": rows,
            "cuda_kernels": kernels, "cpu_events": cpu,
            "high_cost_top_30": kernels[:30],
            "sort": "CUDA kernels: self_device_time_us; CPU events: self_cpu_time_us; NEVER mixed",
            "warning": "CPU user annotations can carry device attribution, not exclusive kernel cost; concurrent kernels may overlap",
        })

    def update(self, original, args, kwargs, values):
        self.step += 1
        accel = values["accelerator"]
        self.device = accel.device
        if accel.num_processes != 4 or accel.gradient_accumulation_steps != 1:
            raise RuntimeError("Runtime requires four ranks, accumulation=1")
        if self.device.type != "cuda":
            raise RuntimeError("Runtime device must be CUDA")
        self.current.update({"step": self.step, "rank": accel.process_index})
        mode = ("warmup" if self.step <= 10 else "steady" if self.step <= 30 else
                "phases" if self.step <= 34 else "modules_saved" if self.step <= 38 else "profiler")
        self.current["mode"] = mode
        model = accel.unwrap_model(values["policy"], keep_fp32_wrapper=True)
        if not self.initialized:
            enabled = [name for name, m in model.named_modules()
                       if getattr(m, "gradient_checkpointing", False) is True]
            if enabled:
                raise RuntimeError(f"Runtime gradient checkpointing enabled: {enabled}")
            write_json(self.directory / "device.json", {
                "device": str(self.device), "name": torch.cuda.get_device_name(self.device),
                "total_memory_bytes": torch.cuda.get_device_properties(self.device).total_memory,
                "storage_before_first_update": storage_inventory(model, values["optimizer"]),
            })
            self.initialized = True
        diagnostic = self.step > 30
        profiler = None
        saved = None
        events = []
        with ExitStack() as context:
            if mode in {"modules_saved", "profiler"}:
                module_stack, events = context.enter_context(self.module_hooks(model))
                context.enter_context(self.functional_wrappers(model, module_stack, events))
                saved = SavedActivations(model, values["batch"], module_stack)
                context.enter_context(torch.autograd.graph.saved_tensors_hooks(saved.pack, saved.unpack))
            if diagnostic:
                context.enter_context(self.phase_wrappers(values, model, saved))
            if mode == "profiler":
                profiler = context.enter_context(torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False, profile_memory=True, with_stack=False))
            self.sync()
            torch.cuda.reset_peak_memory_stats(self.device)
            self.step_highwater = {"allocated_bytes": 0, "reserved_bytes": 0}
            self.current["before"] = self.memory()
            started = time.perf_counter()
            result = original(*args, **kwargs)
            self.sync()
            self.current["update_wall_s"] = time.perf_counter() - started
            self.current["after"] = self.memory()
            self.current["whole_step_peak"] = self.peak()
            if not result[2]:
                raise RuntimeError("Unexpected non-sync update with accumulation=1")
        self.current["loss"] = scalar(result[0].loss.val)
        self.current["loss_components"] = {
            k: scalar(v) for k, v in result[1].items() if k.startswith("loss")
            and (isinstance(v, (int, float)) or isinstance(v, torch.Tensor) and v.numel() == 1)
        }
        if saved:
            self.current["saved_activations_forward_end"] = saved.forward_snapshot
        if events:
            self.current["module_calls"] = [
                {"name": name, "inclusive_cuda_ms": start.elapsed_time(end)}
                for name, start, end in events]
            self.current["module_timing_semantics"] = (
                "inclusive stream CUDA intervals; parent/child/calls may overlap; NEVER sum across modules")
        if diagnostic:
            self.current["storage_after_update"] = storage_inventory(model, values["optimizer"])
            self.current["phase_peak_semantics"] = (
                "phase_peak is absolute allocator high water, not incremental; "
                "whole_step_peak_so_far merges all intervals; never sum peaks")
        if profiler:
            self.export_profiler(profiler)
        return result

    def finish(self):
        if self.step != 40 or len(self.steady) != 20:
            raise RuntimeError(f"Incomplete profile: {self.step} updates, {len(self.steady)} steady")
        durations = [r["update_wall_s"] for r in self.steady]
        iteration_durations = [r["iteration_wall_s"] for r in self.steady]
        window = self.steady[-1]["iteration_end_monotonic_s"] - self.steady[0]["iteration_start_monotonic_s"]
        ordered = sorted(durations)
        write_json(self.directory / "summary.json", {
            "status": "complete", "steady_steps": [11, 30], "count": len(durations),
            "iteration_wall_s": {"mean": statistics.mean(iteration_durations),
                                 "median": statistics.median(iteration_durations),
                                 "min": min(iteration_durations), "max": max(iteration_durations)},
            "steady_window_elapsed_s": window,
            "steady_window_start_monotonic_s": self.steady[0]["iteration_start_monotonic_s"],
            "steady_window_end_monotonic_s": self.steady[-1]["iteration_end_monotonic_s"],
            "throughput_semantics": "global updates/s = 20 / max_rank(steady_window_elapsed_s); trajectories/s = updates/s * 8; segments/s = trajectories/s * 4; NEVER sum per-step rank maxima",
            "update_wall_s": {"mean": statistics.mean(durations), "median": statistics.median(durations),
                              "min": min(durations), "max": max(durations),
                              "p95_nearest_rank": ordered[math.ceil(.95 * len(ordered)) - 1]},
            "max_step_peak_allocated_bytes": max(r["whole_step_peak"]["allocated_bytes"] for r in self.steady),
            "max_step_peak_reserved_bytes": max(r["whole_step_peak"]["reserved_bytes"] for r in self.steady),
            "losses": [{"step": r["step"], "loss": r["loss"]} for r in self.steady],
            "scope": "this rank only; full E2E iteration and continuous 20-step window; synchronized update subset; no diagnostic H2D sync/profiler/module hooks in 11-30",
        })
