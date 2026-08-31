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
