import time

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
