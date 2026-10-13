"""CPU 测试：矩阵、汇总、进程生命周期、续跑和配置冲突。"""
import copy
import csv
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import yaml
import run as runner


class RunnerTests(unittest.TestCase):
    def test_flags_and_results(self):
        self.assertEqual(runner.flags(dict(debug=False, input_masks=["behavior_prompt.state[6,13]"], infer_horizon=None)),
                         ["--no-debug", "--input_masks", "behavior_prompt.state[6,13]"])
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "_result.txt"
            p.write_text("Status: in_progress\nSuccess: 1 / 2\n")
            self.assertIsNone(runner.read_result(p))
            p.write_text("Status: completed\nSuccess: 0 / 2\n")
            self.assertEqual(runner.read_result(p), (0, 2))

    def test_real_matrix(self):
        cfg, robot, out, jobs = runner.plan(Path(__file__).with_name("experiment.yml"))
        self.assertGreater(len(jobs), 0)
        self.assertEqual({j["variant"] for j in jobs}, {"normal", "mask_bp", "wrong_bp"})
        self.assertTrue(all(j["eval"]["ws_port"] == j["args"]["port"] for j in jobs))
        self.assertEqual(jobs[0]["args"]["input_masks"], "")
        changed = copy.deepcopy(jobs[0])
        self.assertEqual(len(changed["tasks"]), 41)

    def test_lifecycle_resume_and_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            robot = root / "robot"
            adapter = robot / "run/ex_plus/eval_batch.py"
            adapter.parent.mkdir(parents=True)
            real_adapter = Path("/vla/workspace/my_robotwin/run/ex_plus/eval_batch.py")
            adapter.write_text(real_adapter.read_text())
            (robot / "run/fast.py").write_text("FULL_TASKS = {'toy': 16}\n")
            (robot / "task_config").mkdir()
            (robot / "task_config/toy.yml").write_text("episode_num: 2\n")
            base = root / "base.yml"
            base.write_text("task_config: toy\n")
            ckpt = root / "checkpoints/000010/pretrained_model"
            ckpt.mkdir(parents=True)
            (ckpt / "config.json").write_text('{"type":"test"}')
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            cfg = dict(name="test", robotwin_root=str(robot), output_root=str(root / "out"),
                       continue_on_error=False,
                       server=dict(python=sys.executable, startup_timeout=10, args=dict(port=port)),
                       client=dict(python=sys.executable, base_config=str(base), tasks={"toy":16}, gpu_ids=[0]),
                       models=[dict(name="m", checkpoints=[str(ckpt)])],
                       scenes=[dict(name="clean")], variants=[dict(name="normal")])
            config = root / "exp.yml"
            config.write_text(yaml.safe_dump(cfg))
            _, _, out, jobs = runner.plan(config)
            fake_server = root / "server.py"
            fake_server.write_text("from http.server import BaseHTTPRequestHandler, HTTPServer\nclass H(BaseHTTPRequestHandler):\n def do_GET(self):\n  self.send_response(200); self.end_headers(); self.wfile.write(b'OK\\n')\nHTTPServer(('127.0.0.1', " + str(port) + "), H).serve_forever()\n")
            fake_client = root / "client.py"
            result = jobs[0]["result_dir"] / "toy/_result.txt"
            fake_client.write_text("from pathlib import Path\np=Path(" + repr(str(result)) + ")\np.parent.mkdir(parents=True,exist_ok=True)\np.write_text('Status: completed\\nSuccess: 1 / 2\\n')\n")
            original_launch = runner.launch
            launches = []
            def fake_launch(cmd, cwd, logfile, extra_env=None):
                launches.append(logfile.name)
                script = fake_server if logfile.name == "server.log" else fake_client
                return original_launch([sys.executable, str(script)], root, logfile)
            with patch.object(sys, "argv", ["run.py", str(config)]), patch.object(runner, "launch", side_effect=fake_launch):
                self.assertEqual(runner.main(), 0)
                self.assertEqual(launches, ["server.log", "client.log"])
                self.assertEqual(runner.ACTIVE, [])
                self.assertEqual(runner.main(), 0)
                self.assertEqual(len(launches), 2)
                with (out / "summary.csv").open(encoding="utf-8-sig") as stream:
                    row = next(csv.DictReader(stream))
                self.assertEqual(row["macro_success_rate"], "0.5")
                self.assertEqual(row["complete"], "True")
                cfg["scenes"][0]["overrides"] = {"instruction_type": "unseen"}
                config.write_text(yaml.safe_dump(cfg))
                with self.assertRaisesRegex(ValueError, "配置发生变化"):
                    runner.main()
            # 服务启动失败时正确记录，并释放其进程。
            cfg["name"] = "failure"
            config.write_text(yaml.safe_dump(cfg))
            with patch.object(sys, "argv", ["run.py", str(config)]), patch.object(runner, "wait_ready", side_effect=RuntimeError("test failure")), patch.object(runner, "launch", side_effect=fake_launch):
                self.assertEqual(runner.main(), 1)
                self.assertEqual(runner.ACTIVE, [])


if __name__ == "__main__":
    unittest.main()
