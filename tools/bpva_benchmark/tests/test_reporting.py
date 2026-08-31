import csv
import json

import pytest
from tools.bpva_benchmark.metrics import StageRecord
from tools.bpva_benchmark.reporting import write_report


def test_output_schema(tmp_path):
    summary = write_report(
        tmp_path,
        [StageRecord("forward", 0.1)],
        gpu_samples=[{"index": "0", "memory_used_mib": "10"}],
        slow_samples=[{"x": 1}],
        slow_videos=[],
    )
    expected = {
        "summary.json",
        "stages.csv",
        "gpu_samples.csv",
        "slow_samples.jsonl",
        "slow_videos.jsonl",
    }
    assert {p.name for p in tmp_path.iterdir()} == expected
    data = json.loads((tmp_path / "summary.json").read_text())
    assert data["record_count"] == 1 and "stages" in data
    rows = list(csv.DictReader((tmp_path / "stages.csv").open()))
    assert rows[0]["stage"] == "forward"


def test_decile_boundaries():
    from tools.bpva_benchmark.reporting import snapshot_boundaries

    assert snapshot_boundaries(1) == (1,)
    assert snapshot_boundaries(7) == (1, 2, 3, 4, 5, 6, 7)
    assert snapshot_boundaries(10) == tuple(range(1, 11))
    assert snapshot_boundaries(11) == (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    assert snapshot_boundaries(100) == tuple(range(10, 101, 10))


def test_atomic_partial_overwrite_and_manifest(tmp_path):
    from tools.bpva_benchmark.reporting import RunSession, write_manifest, write_partial

    session = RunSession(tmp_path, "generation", 0, 1, 123.0)
    write_manifest(session, "running")
    path = write_partial(
        session,
        kind="data",
        phase="measure",
        completed=1,
        total=2,
        records=[StageRecord("x", 0.1)],
    )
    write_partial(
        session,
        kind="data",
        phase="measure",
        completed=2,
        total=2,
        records=[StageRecord("x", 0.2)],
    )
    partial = json.loads(path.read_text())
    assert partial["completed"] == 2 and partial["records"][0]["elapsed_s"] == 0.2
    assert not list(path.parent.glob(".*"))
    write_manifest(session, "completed")
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "completed"


def test_progress_log(capsys, tmp_path):
    import time
    from types import SimpleNamespace
    from tools.bpva_benchmark.reporting import log_progress

    accelerator = SimpleNamespace(is_main_process=True)
    log_progress(
        accelerator,
        phase="measure",
        completed=1,
        total=10,
        last_elapsed_s=0.2,
        started=time.perf_counter() - 0.5,
        path=tmp_path,
    )
    output = capsys.readouterr().out
    assert (
        "measure 1/10 (10%)" in output
        and "last=" in output
        and "ETA=" in output
        and "partial=" in output
    )


def test_phase_progress_has_independent_warmup_and_measure_boundaries():
    import time
    from tools.bpva_benchmark.reporting import PhaseProgress

    warmup = PhaseProgress("warmup", 11, time.perf_counter())
    warmup_hits = [step for step in range(1, 12) if warmup.advance(step)]
    measure = PhaseProgress("measure", 7, time.perf_counter())
    measure_hits = [step for step in range(1, 8) if measure.advance(step)]
    disabled = PhaseProgress("warmup", 0, time.perf_counter())

    assert warmup_hits == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert measure_hits == [1, 2, 3, 4, 5, 6, 7]
    assert not disabled.advance(1)
    assert warmup.phase == "warmup" and warmup.completed == 11
    assert measure.phase == "measure" and measure.completed == 7


def test_non_main_rank_does_not_write_manifest(tmp_path):
    from tools.bpva_benchmark.reporting import RunSession, write_manifest

    session = RunSession(tmp_path, "generation", 1, 2, 123.0)
    write_manifest(session, "failed", error={"type": "Failure"})
    assert not (tmp_path / "manifest.json").exists()


def test_record_failure_does_not_mask_snapshot_or_manifest_errors(
    tmp_path, monkeypatch, capsys
):
    from types import SimpleNamespace
    from tools.bpva_benchmark import reporting

    session = reporting.RunSession(tmp_path, "generation", 0, 1, 123.0)
    accelerator = SimpleNamespace(process_index=0)

    def broken_snapshot(_details):
        raise OSError("snapshot unavailable")

    def broken_manifest(*_args, **_kwargs):
        raise OSError("manifest unavailable")

    monkeypatch.setattr(reporting, "write_manifest", broken_manifest)
    original = RuntimeError("original failure")
    details = reporting.record_failure(session, accelerator, original, broken_snapshot)

    output = capsys.readouterr().out
    assert details["message"] == "original failure"
    assert "snapshot unavailable" in output
    assert "manifest unavailable" in output
    assert "original failure" in output


def test_completed_log_includes_output_path(capsys, tmp_path):
    from types import SimpleNamespace
    from tools.bpva_benchmark.reporting import log_phase

    log_phase(SimpleNamespace(is_main_process=True), "completed", f"output={tmp_path}")
    assert f"[bpva-benchmark] completed: output={tmp_path}" in capsys.readouterr().out


def test_exact_session_cleans_only_old_partials(tmp_path):
    from types import SimpleNamespace
    from tools.bpva_benchmark.reporting import create_run_session

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "rank-00000.json").write_text("old")
    (partial / "notes.txt").write_text("keep")
    (tmp_path / "summary.json").write_text("keep")
    accelerator = SimpleNamespace(process_index=0, num_processes=1)
    session = create_run_session(tmp_path, accelerator, exact=True)

    assert session.generation
    assert not (partial / "rank-00000.json").exists()
    assert (partial / "notes.txt").read_text() == "keep"
    assert (tmp_path / "summary.json").read_text() == "keep"


def _complete_partial(session, *, elapsed, event, stats, monitor):
    from tools.bpva_benchmark.reporting import write_partial

    write_partial(
        session,
        kind="data",
        phase="local_finalize",
        completed=1,
        total=1,
        records=[StageRecord("next_dataloader", elapsed, rank=session.rank)],
        collector={"top_events": [event], "stats": stats},
        monitor=monitor,
        status="local_complete",
    )


def test_load_and_merge_two_rank_local_complete_partials(tmp_path):
    from tools.bpva_benchmark.data_benchmark import merge_data_partials
    from tools.bpva_benchmark.reporting import (
        RunSession,
        load_local_complete_partials,
    )

    generation = "current"
    _complete_partial(
        RunSession(tmp_path, generation, 0, 2, 123.0),
        elapsed=0.1,
        event={"kind": "sample", "elapsed_s": 1.0},
        stats={"seen": {"sample": 1}},
        monitor={"samples": [{"index": "0"}], "errors": ["monitor warning"]},
    )
    _complete_partial(
        RunSession(tmp_path, generation, 1, 2, 123.0),
        elapsed=0.2,
        event={"kind": "video", "elapsed_s": 2.0},
        stats={"seen": {"video": 1}},
        monitor=None,
    )

    partials = load_local_complete_partials(
        tmp_path,
        generation=generation,
        world_size=2,
        timeout_s=0.2,
        poll_interval_s=0.01,
    )
    merged = merge_data_partials(partials)

    assert [record.rank for record in merged["records"]] == [0, 1]
    assert len(merged["slow_samples"]) == 1
    assert len(merged["slow_videos"]) == 1
    assert merged["event_collector_by_rank"] == [
        {"seen": {"sample": 1}},
        {"seen": {"video": 1}},
    ]
    assert merged["gpu_samples"] == [{"index": "0"}]
    assert merged["monitor_errors"] == ["monitor warning"]



def test_merge_train_partials_collects_records_gpu_and_monitor_errors():
    from tools.bpva_benchmark.train_benchmark import merge_train_partials

    partials = [
        {
            "records": [StageRecord("forward", 0.1, rank=0).to_dict()],
            "monitor": {"samples": [{"index": "0"}], "errors": ["gpu warning"]},
        },
        {
            "records": [StageRecord("backward", 0.2, rank=1).to_dict()],
            "monitor": None,
        },
    ]

    merged = merge_train_partials(partials)

    assert [record.rank for record in merged["records"]] == [0, 1]
    assert [record.stage for record in merged["records"]] == ["forward", "backward"]
    assert merged["gpu_samples"] == [{"index": "0"}]
    assert merged["monitor_errors"] == ["gpu warning"]

def test_local_complete_loader_ignores_stale_and_running_until_timeout(tmp_path):
    from tools.bpva_benchmark.reporting import (
        RunSession,
        load_local_complete_partials,
        write_partial,
    )

    _complete_partial(
        RunSession(tmp_path, "old", 0, 2, 123.0),
        elapsed=0.1,
        event={"kind": "sample"},
        stats={},
        monitor=None,
    )
    write_partial(
        RunSession(tmp_path, "current", 1, 2, 123.0),
        kind="data",
        phase="measure",
        completed=2,
        total=2,
        records=[],
        collector={"top_events": [], "stats": {}},
    )

    with pytest.raises(TimeoutError) as error:
        load_local_complete_partials(
            tmp_path,
            generation="current",
            world_size=3,
            timeout_s=0.03,
            poll_interval_s=0.005,
        )

    message = str(error.value)
    assert "missing_ranks=[2]" in message
    assert "stale_generation" in message
    assert "status('running')" in message
