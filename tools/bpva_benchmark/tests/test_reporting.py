import csv
import json
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
