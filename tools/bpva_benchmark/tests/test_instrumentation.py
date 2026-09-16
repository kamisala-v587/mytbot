import time
import threading

import torch

from tools.bpva_benchmark.data_instrumentation import (
    EventCollector,
    _dataset_wrapper,
    _safe_index,
)
from tools.bpva_benchmark.model_instrumentation import ModelInstrumentation


class Dataset:
    repo_id = "fake/repo"

    def get(self, index):
        return index


def test_data_wrapper_and_queue():
    collector = EventCollector(max_queue=8, top_k=2).start()
    original = Dataset.get
    Dataset.get = _dataset_wrapper(original, collector.queue, "getitem", 1.0, 0.0)
    try:
        assert Dataset().get(3) == 3
    finally:
        Dataset.get = original
    time.sleep(0.15)
    events = collector.stop()
    assert events[0]["stage"] == "getitem"
    assert events[0]["repo_id"] == "fake/repo"


def test_safe_index_summarizes_mutable_args():
    """Regression test: wrapped methods like `_build_prompt(self, current_sample)`
    receive a dict that the caller mutates in place right after the call
    returns (`current[BP_PREFIX] = self._build_prompt(current)`). If the event
    dict captured a reference to that dict instead of a summary, the
    background multiprocessing queue feeder thread can race with the mutation
    and raise `RuntimeError: dictionary changed size during iteration` while
    pickling."""
    assert _safe_index(3) == 3
    assert _safe_index("idx") == "idx"
    assert _safe_index(None) is None
    mutable = {"a": 1}
    summarized = _safe_index(mutable)
    assert summarized == "<dict>"
    mutable["b"] = 2  # must not affect the already-captured summary
    assert summarized == "<dict>"


def test_dataset_wrapper_does_not_reference_mutable_first_arg():
    collector = EventCollector(max_queue=8, top_k=2).start()

    def build(self, current_sample):
        return {"ok": True}

    wrapped = _dataset_wrapper(build, collector.queue, "build_prompt", 1.0, 0.0)
    mutable_arg = {"x": 1}
    Dataset.build_prompt = wrapped
    try:
        Dataset().build_prompt(mutable_arg)
        mutable_arg["y"] = 2  # simulate caller mutating the dict right after return
    finally:
        del Dataset.build_prompt
    time.sleep(0.15)
    events = collector.stop()
    assert events[0]["index"] == "<dict>"


def test_collector_is_bounded_per_kind():
    collector = EventCollector(max_queue=32, top_k=3)
    for index in range(20):
        collector._accept(
            {"kind": "sample", "elapsed_s": float(index), "rank": 0, "pid": 1}
        )
        collector._accept(
            {"kind": "video", "elapsed_s": float(index), "rank": 0, "pid": 1}
        )
    events = collector.stop()
    assert len(events) == 6
    assert collector.stats["seen"] == {"sample": 20, "video": 20}
    assert collector.stats["dropped"] == {"sample": 17, "video": 17}
    assert len({event["event_id"] for event in events}) == 6


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cosmos = torch.nn.Linear(2, 2)

    def embed_middle(self, x):
        return self.cosmos(x)

    def forward(self, x):
        return self.embed_middle(x)


def test_model_wrapper_uninstall():
    model = Tiny()
    original = model.embed_middle
    inst = ModelInstrumentation(model).install()
    model(torch.ones(1, 2))
    records = inst.resolve(0)
    inst.uninstall()
    assert {record.stage for record in records} >= {"cosmos", "method.embed_middle"}
    assert model.embed_middle.__func__ is original.__func__


class FakeBPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bp_obs_encoder = torch.nn.Module()
        self.bp_obs_encoder.chunk_encoder = torch.nn.Module()
        shared = torch.nn.Linear(2, 2)
        self.bp_obs_encoder.chunk_encoder.key_model_map = torch.nn.ModuleDict(
            {"left": shared, "right": shared}
        )

    def forward(self, x):
        model = self.bp_obs_encoder.chunk_encoder.key_model_map["left"]
        return model(x)


def test_bp_vit_hooks_actual_shared_child_once():
    model = FakeBPModel()
    inst = ModelInstrumentation(model).install()
    model(torch.ones(1, 2))
    records = inst.resolve(0)
    inst.uninstall()
    assert [record.stage for record in records].count("bp_vit") == 1


class FakeSharedVisual(torch.nn.Module):
    def forward(self, x):
        return x + 1


class FakeChunkEncoder(torch.nn.Module):
    def __init__(self, visual: torch.nn.Module):
        super().__init__()
        self._visual = visual

    def _encode_images(self, x):
        return self._visual(x)

    def forward(self, x):
        encoded = self._encode_images(x)
        return encoded * 2


class FakeBPObsEncoder(torch.nn.Module):
    def __init__(self, chunk_encoder: FakeChunkEncoder):
        super().__init__()
        self.chunk_encoder = chunk_encoder

    def forward(self, x):
        return self.chunk_encoder(x)


class FakeBPVAv2Model(torch.nn.Module):
    """Mirrors BPVAv2: shared visual is an argument, not a key_model_map child."""

    def __init__(self):
        super().__init__()
        self.und_expert = torch.nn.Module()
        self.und_expert.visual = FakeSharedVisual()
        chunk = FakeChunkEncoder(self.und_expert.visual)
        self.bp_obs_encoder = FakeBPObsEncoder(chunk)

    def forward(self, x):
        current = self.und_expert.visual(x)
        bp = self.bp_obs_encoder(x)
        return current + bp


def test_bpvav2_bp_visual_encode_is_timed_and_split_from_current_visual():
    model = FakeBPVAv2Model()
    inst = ModelInstrumentation(model).install()
    inst.step = 3
    model(torch.ones(1, 2))
    records = inst.resolve(3)
    inst.uninstall()

    stages = [record.stage for record in records]
    assert stages.count("bp_encoder") == 1
    assert stages.count("bp_visual_encode") == 1
    assert stages.count("bp_qwen_visual") == 1
    assert stages.count("qwen_visual") == 1
    assert (
        model.bp_obs_encoder.chunk_encoder._encode_images.__func__
        is FakeChunkEncoder._encode_images
    )


def test_derive_bp_compressor_estimate_pairs_encoder_and_visual():
    from tools.bpva_benchmark.metrics import (
        StageRecord,
        derive_bp_compressor_estimates,
        summarize_records,
    )

    records = [
        StageRecord("data_wait", 0.004, rank=0, step=1),
        StageRecord(
            "bp_encoder", 1.0, rank=0, step=1, device_elapsed_s=0.90
        ),
        StageRecord(
            "bp_visual_encode", 0.8, rank=0, step=1, device_elapsed_s=0.75
        ),
        StageRecord("forward", 2.0, rank=0, step=1, device_elapsed_s=1.8),
    ]
    derived = derive_bp_compressor_estimates(records)
    assert len(derived) == 1
    assert derived[0].stage == "bp_compressor_est"
    assert abs(derived[0].elapsed_s - 0.2) < 1e-9
    assert abs(derived[0].device_elapsed_s - 0.15) < 1e-9

    summary = summarize_records(records)
    assert summary["bp_attribution"]["bp_visual_encode"]["mean_s"] == 0.8
    assert abs(summary["bp_attribution"]["bp_compressor_est"]["mean_s"] - 0.2) < 1e-9
    assert summary["bp_attribution"]["data_wait"]["mean_s"] == 0.004
    assert "Qwen visual encode" in summary["bp_attribution"]["interpretation"]


def test_format_terminal_summary_includes_bp_attribution_block():
    from tools.bpva_benchmark.metrics import StageRecord, summarize_records
    from tools.bpva_benchmark.reporting import format_terminal_summary

    summary = summarize_records(
        [
            StageRecord("data_wait", 0.01, rank=0, step=0),
            StageRecord("bp_encoder", 1.2, rank=0, step=0, device_elapsed_s=1.1),
            StageRecord(
                "bp_visual_encode", 1.0, rank=0, step=0, device_elapsed_s=0.95
            ),
        ]
    )
    summary["metadata"] = {"dataset_count": 12, "total_frames": 3456}
    text = format_terminal_summary(summary)
    assert "视觉/BP 归因" in text
    assert "bp_visual_encode" in text
    assert "bp_compressor_est" in text
    assert "数据集总个数: 12" in text
    assert "数据集总帧数: 3,456" in text


def test_collector_concurrent_snapshot():
    collector = EventCollector(max_queue=256, top_k=5).start()
    def produce():
        for index in range(100):
            collector._accept({"kind": "sample", "elapsed_s": index, "rank": 0, "pid": 1})
    thread = threading.Thread(target=produce); thread.start()
    snapshots = [collector.snapshot() for _ in range(20)]
    thread.join(); final = collector.snapshot(); collector.stop()
    assert all("top_events" in item and "stats" in item for item in snapshots)
    assert final["stats"]["seen"]["sample"] == 100


def test_monitor_concurrent_snapshot(monkeypatch):
    from tools.bpva_benchmark.system_monitor import SystemMonitor
    monitor = SystemMonitor(.1)
    def append():
        for index in range(100):
            with monitor._lock: monitor.samples.append({"index": index})
    thread = threading.Thread(target=append); thread.start()
    snapshots = [monitor.snapshot() for _ in range(20)]
    thread.join(); monitor.stop()
    assert all("samples" in item and "errors" in item for item in snapshots)
    assert len(monitor.snapshot()["samples"]) == 100


def test_recursive_strip_and_sample_batch_association():
    import json
    from tools.bpva_benchmark.data_instrumentation import (
        BENCHMARK_METADATA_KEY,
        sample_records_from_batch,
        strip_benchmark_metadata,
    )

    metadata = {
        "load_id": "load-1", "worker_id": 2, "pid": 42,
        "start_ns": 100, "end_ns": 350, "elapsed_s": 2.5e-7,
        "repo_id": "repo", "events": [],
    }
    batch = {
        "x": torch.ones(1),
        BENCHMARK_METADATA_KEY: [json.dumps(metadata)],
        "nested": {"__benchmark_secret": 1, "keep": 2},
    }
    samples, workers = sample_records_from_batch(
        batch, rank=3, step=7, optimizer_step=5, microstep=9
    )
    stripped = strip_benchmark_metadata(batch)

    assert BENCHMARK_METADATA_KEY not in stripped
    assert stripped["nested"] == {"keep": 2}
    assert samples[0]["report_step"] == 7
    assert samples[0]["delivery_rank"] == 3
    assert workers == [{"worker_id": 2, "pid": 42, "elapsed_s": 2.5e-07,
                        "sample_count": 1, "start_ns": 100, "end_ns": 350}]


def test_context_and_backend_fallback_are_attributed():
    import queue
    from tools.bpva_benchmark.data_instrumentation import (
        _backend_wrapper, _context_wrapper, _video_wrapper,
    )

    events = queue.Queue()
    def torchcodec(*_args):
        return "frames"
    backend = _backend_wrapper(torchcodec, "torchcodec")
    def decode(path, timestamps, tolerance, requested):
        return backend(path, timestamps, tolerance)
    wrapped = _context_wrapper(
        _video_wrapper(decode, events, 1.0, 0.0, "decode_video_frames"),
        "bp_prompt",
    )

    assert wrapped("v.mp4", [1.0], 0.01, "pyav") == "frames"
    event = events.get_nowait()
    assert event["context"] == "bp_prompt"
    assert event["requested_backend"] == "pyav"
    assert event["effective_backend"] == "torchcodec"
    assert event["fallback"] is True


def test_compose_worker_init_calls_original_first():
    from tools.bpva_benchmark.data_instrumentation import compose_worker_init
    calls = []
    composed = compose_worker_init(lambda worker: calls.append(("original", worker)),
                                   lambda worker: calls.append(("instrument", worker)))
    composed(4)
    assert calls == [("original", 4), ("instrument", 4)]


def test_iterable_instrumentation_preserves_type_and_metadata():
    import json
    from torch.utils.data import DataLoader, IterableDataset
    from tools.bpva_benchmark.data_instrumentation import (
        BENCHMARK_METADATA_KEY,
        InstrumentedIterableDataset,
        wrap_dataset_for_instrumentation,
    )

    class Stream(IterableDataset):
        repo_id = "stream/repo"
        def __iter__(self):
            yield {"value": torch.tensor(1)}
            yield {"value": torch.tensor(2)}

    wrapped = wrap_dataset_for_instrumentation(Stream())
    assert isinstance(wrapped, InstrumentedIterableDataset)
    assert isinstance(wrapped, IterableDataset)
    batch = next(iter(DataLoader(wrapped, batch_size=2, num_workers=0)))
    metadata = [json.loads(value) for value in batch[BENCHMARK_METADATA_KEY]]
    assert batch["value"].tolist() == [1, 2]
    assert [row["index"] for row in metadata] == [0, 1]
    assert {row["repo_id"] for row in metadata} == {"stream/repo"}
