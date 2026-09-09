import pytest
from tools.bpva_benchmark.metrics import StageRecord, percentile, summarize_records


def test_percentile_and_multirank_summary():
    rows = [StageRecord("data", x, rank=i % 2) for i, x in enumerate([1, 2, 3, 4])]
    summary = summarize_records(rows)
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert summary["stages"]["data"]["p95_s"] == 3.85
    assert set(summary["ranks"]) == {"0", "1"}
    assert summary["bottlenecks"][0]["stage"] == "data"


def test_json_safe_nonfinite():
    assert StageRecord("x", float("inf")).to_dict()["elapsed_s"] is None


def test_cross_rank_same_step_straggler():
    from tools.bpva_benchmark.metrics import aggregate_step_stragglers
    rows = [
        StageRecord("microstep_wall", 0.4, rank=0, step=3, metadata={"microstep": 8}),
        StageRecord("microstep_wall", 0.7, rank=1, step=3, metadata={"microstep": 8}),
        StageRecord("microstep_wall", 0.5, rank=2, step=3, metadata={"microstep": 8}),
    ]
    result = aggregate_step_stragglers(rows)
    assert result[0]["rank_max_s"] == 0.7
    assert result[0]["straggler_rank"] == 1
    assert result[0]["rank_median_s"] == 0.5
    assert result[0]["rank_spread_s"] == pytest.approx(0.3)


def test_optimizer_step_wall_is_default_straggler_stage_and_uses_own_step():
    from tools.bpva_benchmark.metrics import aggregate_step_stragglers

    rows = [
        # Microstep ids deliberately differ: optimizer-step aggregation must use
        # StageRecord.step=0 rather than ending_microstep or metadata optimizer_step.
        StageRecord("optimizer_step_wall", 1.2, rank=0, step=0,
                    metadata={"optimizer_step": 4, "ending_microstep": 9}),
        StageRecord("optimizer_step_wall", 1.8, rank=1, step=0,
                    metadata={"optimizer_step": 4, "ending_microstep": 11}),
        StageRecord("optimizer_step_wall", 1.0, rank=0, step=1,
                    metadata={"optimizer_step": 5, "ending_microstep": 13}),
        StageRecord("optimizer_step_wall", 1.1, rank=1, step=1,
                    metadata={"optimizer_step": 5, "ending_microstep": 15}),
    ]

    result = [row for row in aggregate_step_stragglers(rows)
              if row["stage"] == "optimizer_step_wall"]
    assert [row["step"] for row in result] == [0, 1]
    assert result[0]["straggler_rank"] == 1
    assert result[0]["rank_max_s"] == 1.8
    assert result[0]["optimizer_step"] == 4
