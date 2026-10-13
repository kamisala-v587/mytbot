#!/usr/bin/env python3
"""配置驱动的批量 checkpoint、场景与 BP 消融测评。"""
import argparse
import copy
import csv
import fcntl
import glob
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import yaml

TBOT = Path(__file__).resolve().parents[2]
ACTIVE = []


def merge(base, extra):
    result = copy.deepcopy(base)
    for k, v in extra.items():
        result[k] = merge(result[k], v) if isinstance(v, dict) and isinstance(result.get(k), dict) else copy.deepcopy(v)
    return result


def path_at(value, root):
    p = Path(os.path.expandvars(str(value))).expanduser()
    return (p if p.is_absolute() else root / p).resolve()


def name(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"名称仅允许英文、数字、下划线和短横线: {value!r}")
    return value


def step_of(p):
    folder = p.parent.name if p.name == "pretrained_model" else p.name
    return int(folder) if folder.isdigit() else None


def read_result(p):
    if not p.is_file():
        return None
    fields = dict(line.split(":", 1) for line in p.read_text().splitlines() if ":" in line)
    if fields.get("Status", "completed").strip() != "completed":
        return None
    try:
        success, total = map(int, fields["Success"].split("/"))
        return (success, total) if 0 <= success <= total and total > 0 else None
    except (KeyError, ValueError):
        return None


def complete(j):
    return all(read_result(j["result_dir"] / t / "_result.txt") for t in j["tasks"])


def plan(config):
    cfg = yaml.safe_load(config.read_text())
    root = config.parent
    robot = path_at(cfg["robotwin_root"], root)
    spec = importlib.util.spec_from_file_location("adapter", robot / "run/ex_plus/eval_batch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    client = cfg["client"]
    tasks = mod.resolve_tasks(mod.load_fast(), client["tasks"])
    if not client["gpu_ids"] or min(int(client.get(k, 5)) for k in ("tasks_per_gpu", "max_retries")) < 1:
        raise ValueError("GPU 列表不能为空；并发和重试次数必须为正数")
    base = yaml.safe_load(path_at(client["base_config"], root).read_text())
    out = path_at(cfg["output_root"], root) / name(cfg["name"])
    jobs = []
    for model in cfg["models"]:
        paths = [path_at(p, root) for p in model.get("checkpoints", [])]
        if model.get("checkpoint_glob"):
            paths += [Path(p).resolve() for p in glob.glob(str(path_at(model["checkpoint_glob"], root)))]
        paths = sorted(set(paths), key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", str(p))])
        if "steps" in model:
            paths = [p for p in paths if step_of(p) in model["steps"]]
        if not paths:
            raise ValueError(f"模型 {model['name']} 未匹配到 checkpoint")
        for ckpt in paths:
            if not (ckpt / "config.json").is_file():
                raise ValueError(f"checkpoint 缺少 config.json: {ckpt}")
            step = step_of(ckpt)
            for scene in cfg["scenes"]:
                for variant in cfg["variants"]:
                    ident = "__".join(map(name, [cfg["name"], model["name"], step if step is not None else ckpt.name, scene["name"], variant["name"]]))
                    args = dict(cfg["server"]["args"])
                    for source in (model, scene, variant):
                        args.update(source.get("server_args", {}))
                    args["ckpt_path"] = str(ckpt)
                    for k in list(args):
                        if args[k] and (k.endswith("_path") or k in ("debug_dir", "bp_dim_config")):
                            args[k] = str(path_at(args[k], root))
                    ev = merge(merge(base, client.get("overrides", {})), scene.get("overrides", {}))
                    ev.update(eval_name=ident, ws_host="127.0.0.1", ws_port=int(args.get("port", 8000)))
                    if ev.get("seed_reuse_file"):
                        ev["seed_reuse_file"] = str(path_at(ev["seed_reuse_file"], root))
                    if args.get("bp_mapping_path"):
                        bp = yaml.safe_load(Path(args["bp_mapping_path"]).read_text())
                        missing = set(tasks) - set(bp.get("tasks", {}))
                        if missing:
                            raise ValueError(f"BP 映射缺少任务: {sorted(missing)}")
                    evidence = dict(args=args, eval=ev, tasks=tasks,
                                    scene=(robot / "task_config" / (ev["task_config"] + ".yml")).read_text(),
                                    ckpt_config=(ckpt / "config.json").read_text())
                    for k in ("bp_mapping_path", "bp_dim_config"):
                        if args.get(k):
                            evidence[k] = Path(args[k]).read_text()
                    if ev.get("seed_reuse_file"):
                        evidence["seeds"] = Path(ev["seed_reuse_file"]).read_text()
                    fingerprint = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
                    jobs.append(dict(id=ident, model=model["name"], checkpoint=str(ckpt), step=step,
                                     scene=scene["name"], variant=variant["name"], args=args, eval=ev, tasks=tasks,
                                     fingerprint=fingerprint, directory=out / ident, result_dir=robot / "eval_result" / ident))
    if not jobs or len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("实验矩阵为空或名称重复；不同训练请使用不同 model name")
    return cfg, robot, out, jobs


def python_cmd(section):
    if section.get("python"):
        return [str(Path(section["python"]).expanduser())]
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        raise ValueError("找不到 conda；可用 server.python / client.python 指定环境解释器")
    return [conda, "run", "--no-capture-output", "-n", section["conda_env"], "python"]


def flags(args):
    result = []
    for k, v in args.items():
        if v is None:
            continue
        if isinstance(v, bool):
            result.append("--" + ("" if v else "no-") + k)
        else:
            if k == "input_masks" and isinstance(v, list):
                v = ",".join(v)
            result += ["--" + k, str(v)]
    return result


def stop(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        except ProcessLookupError:
            pass


def launch(cmd, cwd, logfile, extra_env=None):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.update(extra_env or {})
    with logfile.open("a") as stream:
        stream.write("\nCOMMAND: " + json.dumps(cmd) + "\n")
        stream.flush()
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    ACTIVE.append(proc)
    return proc


def wait_ready(proc, port, timeout):
    deadline = time.monotonic() + timeout
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"服务提前退出，exit={proc.returncode}；请查看 server.log")
        try:
            with opener.open(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                if response.status == 200 and response.read().strip() == b"OK":
                    return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError("服务就绪超时；请检查 server.log 或增大 startup_timeout")


def summarize(out, jobs):
    rows, totals = [], []
    for j in jobs:
        rates, successes, episodes = [], 0, 0
        common = {k: j[k] for k in ("id", "model", "checkpoint", "step", "scene", "variant")}
        for task in j["tasks"]:
            result = read_result(j["result_dir"] / task / "_result.txt")
            success, total = result or (0, 0)
            rate = success / total if total else ""
            rows.append(dict(common, task=task, completed=bool(result), success=success, total=total, success_rate=rate))
            if result:
                rates.append(rate)
                successes += success
                episodes += total
        totals.append(dict(common, completed_tasks=len(rates), expected_tasks=len(j["tasks"]), complete=len(rates) == len(j["tasks"]),
                           macro_success_rate=sum(rates) / len(rates) if rates else "",
                           micro_success_rate=successes / episodes if episodes else "", success=successes, total=episodes))
    for filename, data in (("tasks.csv", rows), ("summary.csv", totals)):
        temp = out / (filename + ".tmp")
        with temp.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        temp.replace(out / filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    opts = parser.parse_args()
    cfg, robot, out, jobs = plan(opts.config.resolve())
    print(f"实验数: {len(jobs)}；每个实验 {len(jobs[0]['tasks'])} 个任务", flush=True)
    for j in jobs:
        print(f"  {j['id']}  ckpt={j['checkpoint']}", flush=True)
    if opts.dry_run:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for j in jobs:
            manifest = j["directory"] / "manifest.json"
            if manifest.exists():
                if json.loads(manifest.read_text())["fingerprint"] != j["fingerprint"]:
                    raise ValueError(f"实验配置发生变化: {j['id']}；请换 name 后测评")
            elif j["result_dir"].exists():
                raise ValueError(f"结果目录存在但没有配置记录: {j['result_dir']}；请换 name")
        if opts.summarize_only:
            summarize(out, jobs)
            return 0
        server_python, client_python = python_cmd(cfg["server"]), python_cmd(cfg["client"])
        failures = 0
        try:
            for index, j in enumerate(jobs, 1):
                d = j["directory"]
                d.mkdir(parents=True, exist_ok=True)
                (d / "manifest.json").write_text(json.dumps({k: v for k, v in j.items() if k not in ("directory", "result_dir")}, indent=2, ensure_ascii=False))
                (d / "eval.yml").write_text(yaml.safe_dump(j["eval"], sort_keys=False))
                job = dict(config=str(d / "eval.yml"), tasks=j["tasks"], gpu_ids=cfg["client"]["gpu_ids"],
                           tasks_per_gpu=cfg["client"].get("tasks_per_gpu", 5), max_retries=cfg["client"].get("max_retries", 5))
                (d / "job.yml").write_text(yaml.safe_dump(job, sort_keys=False))
                print(f"[{index}/{len(jobs)}] {j['id']}", flush=True)
                if complete(j):
                    print("  已完成，跳过", flush=True)
                    summarize(out, jobs)
                    continue
                server = client = None
                try:
                    port = int(j["args"].get("port", 8000))
                    with socket.socket() as probe:
                        if probe.connect_ex(("127.0.0.1", port)) == 0:
                            raise RuntimeError(f"端口 {port} 已占用；请停止原服务或换 port")
                    server = launch(server_python + [str(TBOT / "server/serve_policy.py")] + flags(j["args"]), TBOT, d / "server.log")
                    wait_ready(server, port, float(cfg["server"].get("startup_timeout", 900)))
                    client = launch(client_python + [str(robot / "run/ex_plus/eval_batch.py"), "--job", str(d / "job.yml")], robot, d / "client.log")
                    while client.poll() is None:
                        if server.poll() is not None:
                            raise RuntimeError("测评中模型服务退出")
                        time.sleep(1)
                    if client.returncode or not complete(j):
                        raise RuntimeError(f"客户端失败或结果不完整，exit={client.returncode}；请查看 client.log")
                    (d / "error.txt").unlink(missing_ok=True)
                except (RuntimeError, OSError) as exc:
                    failures += 1
                    print(f"失败: {exc}", file=sys.stderr, flush=True)
                    (d / "error.txt").write_text(str(exc))
                    if not cfg.get("continue_on_error", True):
                        break
                finally:
                    for proc in (client, server):
                        if proc is not None:
                            stop(proc)
                            ACTIVE.remove(proc)
                    summarize(out, jobs)
        finally:
            for proc in reversed(ACTIVE):
                stop(proc)
        print(f"汇总: {out / 'summary.csv'}；失败实验数: {failures}", flush=True)
        return 1 if failures else 0
    finally:
        lock.close()


def interrupted(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("已中断；再次运行同一配置可续评", file=sys.stderr)
        sys.exit(130)
    except (ValueError, KeyError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)
