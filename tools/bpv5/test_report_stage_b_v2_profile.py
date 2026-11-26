"""CPU-only synthetic fixtures; never reads outputs or imports training/torch."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import report_stage_b_v2_profile as report
import create_profile_canvas as canvas


def fixture(root, schema=3, batch=2):
    config = {"batch_size": batch, "gradient_accumulation_steps": 1,
              "dataset": {"segments_per_trajectory": 4}}
    for rank in range(4):
        directory = root / f"rank_{rank:02d}"
        directory.mkdir(parents=True)
        meta = {"rank": rank, "world_size": 4, "schema_version": schema, "resolved_config": config}
        (directory / "metadata.json").write_text(json.dumps(meta))
        (directory / "device.json").write_text(json.dumps({"name": "synthetic CPU fixture"}))
        rows = []
        for step in range(1, 41):
            # Alternating slow ranks proves max(window) != sum(max(step)).
            whole = 2 + (.2 if step % 4 == rank else 0)
            start = rows[-1]["iteration_end_monotonic_s"] if rows else 100
            row = {"rank": rank, "step": step, "mode": "warmup" if step <= 10 else "steady" if step <= 30 else "phases" if step <= 34 else "modules_saved" if step <= 38 else "profiler",
                   "update_wall_s": 1., "whole_step_peak": {"allocated_bytes": 1000, "reserved_bytes": 2000},
                   "loss": 1. / step,
                   "iteration_wall_s": whole, "iteration_start_monotonic_s": start,
                   "iteration_end_monotonic_s": start + whole,
                   "dataloader_next_wall_s": .2, "post_update_wall_s": .5,
                   "h2d": [{"enqueue_wall_s": .05, "diagnostic_sync": step > 30,
                            "synchronized_wall_s": .1}],
                   "gpu_telemetry": [{"wall_s": .3}] if rank == 0 else [],
                   "worker_timing": {"getitem_s": .5, "getitem_max_s": .7, "cache_s": .1, "frames_s": .2, "xform_s": .1}}
            if step >= 31:
                row["phases"] = [{"name": name, "status": "ok", "wall_s": .1,
                                  "after": {"allocated_bytes": 1000}} for name in report.PHASES]
            if step >= 35:
                inventory = {"groups_bytes": {"model": {"parameter:cuda:0": 100}},
                             "storage_records": [{"key": "p", "kind": "parameter", "device": "cuda:0", "bytes": 100}]}
                saved = {"total_storage_bytes": 200, "groups_storage_bytes": {"vit": 200},
                         "excluded_storage_keys": ["p"], "storage_records": [{"key": "s", "kind": "saved_activation", "device": "cuda:0", "bytes": 200}]}
                row.update(storage_after_backward=inventory, storage_after_update=inventory,
                           storage_forward_end=inventory, saved_activations_forward_end=saved,
                           memory_forward_end={"allocated_bytes": 1000, "device": "cuda:0", "categories_bytes": {"parameter": 100, "optimizer_state": 0, "gradient": 0, "saved_activation": 200, "other_residual": 700}},
                           module_calls=[{"name": "function/" + name, "inclusive_cuda_ms": 10} for name in ("vision/visual_only", "prefix/vision", "prefix/build_prefix")])
            if step >= 39:
                (directory / f"step_{step:02d}.operators.json").write_text(json.dumps({"operators": [{"name": "CPU", "source_metric": "self_device_time_total", "device_type": "CPU", "self_device_time_us": 0, "self_cpu_time_us": 10, "count": 1}]}))
                (directory / f"step_{step:02d}.trace.json").write_text(json.dumps({"traceEvents": [{"cat": "kernel", "ph": "X", "name": "synthetic_kernel", "dur": 100}]}))
            rows.append(row)
        (directory / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        (directory / "summary.json").write_text(json.dumps({"status": "complete", "count": 20, "steady_steps": [11, 30], "steady_window_elapsed_s": rows[29]["iteration_end_monotonic_s"] - rows[10]["iteration_start_monotonic_s"]}))


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        fixture(self.root / "input")
        self.ranks = report.load_profile(self.root / "input")

    def summary(self):
        return report.summarize(self.ranks, 0, 35)

    def test_dynamic_batch_and_window(self):
        s = self.summary()
        self.assertEqual(s["batch"]["global_trajectories"], 8)
        self.assertEqual(s["batch"]["global_segments"], 32)
        self.assertEqual(s["synchronized_throughput"]["update_only_samples_per_second"], 8)
        e = s["e2e"]
        self.assertAlmostEqual(e["max_rank_window_elapsed_s"], 41)
        self.assertNotAlmostEqual(e["max_rank_window_elapsed_s"], 44)
        self.assertAlmostEqual(sum(e["critical_window_mean_wall_s"].values()), 41 / 20)
        self.assertAlmostEqual(e["global_trajectories_per_s"], 160 / 41)
        self.assertAlmostEqual(e["mean_wall_s"]["other residual"], .35)
        self.assertEqual(e["per_rank"]["1"]["telemetry_direct_wall_s"]["mean"], 0)
        self.assertEqual(e["per_rank"]["0"]["telemetry_direct_wall_s"]["mean"], .3)
        self.assertEqual(s["modules_inclusive"]["rows"][0]["per_call_inclusive_cuda_ms"]["count"], 24)
        self.assertEqual(canvas.normalize(s, "synthetic", "digest")["batch"]["per_rank_trajectories"], 2)

    def test_reject_invalid_and_partial(self):
        bad = copy.deepcopy(self.ranks)
        bad[0]["records"][11]["post_update_wall_s"] = 5
        with self.assertRaises(report.ReportError):
            report.summarize(bad, 0, 35)
        bad = copy.deepcopy(self.ranks)
        bad[1]["completion"]["steady_window_elapsed_s"] = 999
        with self.assertRaises(report.ReportError):
            report.summarize(bad, 0, 35)
        (self.root / "input/rank_03/summary.json").write_text('{"status":"running"}')
        with self.assertRaises(report.ReportError):
            report.load_profile(self.root / "input")

    def test_legacy_and_baseline(self):
        fixture(self.root / "bs1", schema=2, batch=1)
        old = report.summarize(report.load_profile(self.root / "bs1"), 0, 35)
        self.assertIsNone(old["e2e"])
        path = self.root / "summary.json"
        path.write_text(json.dumps(old))
        comparison = report.compare_baseline(self.summary(), path)
        self.assertEqual(comparison["improvement_percent"], 100)
        # Existing schema2 summaries did not have these added fields.
        del old["batch"]
        del old["e2e"]
        self.assertEqual(canvas.normalize(old, path, "digest")["batch"]["global_trajectories"], 4)

    def test_canvas_closure_and_no_overwrite(self):
        s = self.summary()
        s["e2e"]["critical_window_mean_wall_s"]["other residual"] += 1
        with self.assertRaises(canvas.SummaryError):
            canvas.normalize(s, "synthetic", "digest")
        path = self.root / "summary.json"
        path.write_text(json.dumps(self.summary()))
        output = self.root / "test.canvas.tsx"
        self.assertEqual(canvas.main([str(path), "--output", str(output)]), 0)
        original = output.read_text()
        self.assertIn("完整循环互斥耗时", original)
        self.assertEqual(canvas.main([str(path), "--output", str(output)]), 2)
        self.assertEqual(output.read_text(), original)

    def test_full_offline_report(self):
        output = self.root / "report"
        self.assertEqual(report.main([str(self.root / "input"), "--output-dir", str(output)]), 0)
        s = json.loads((output / "summary.json").read_text())
        expected = {"time_share", "memory_share", "total_allocated_share", "loss", "step_time", "rank_memory_peak", "persistent_memory", "operators_self_device", "operators_self_cpu", "e2e_time_share", "e2e_rank_time", "e2e_step_time"}
        self.assertEqual({Path(f).stem for f in s["figures"]}, expected)
        self.assertEqual(len(s["figures"]), 24)
        text = (output / "REPORT.md").read_text()
        for phrase in ("beforeupdate", "ViT visual_only", "bs1 上轮只有 update-only", "worker", "telemetry", "完整20步窗口"):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
