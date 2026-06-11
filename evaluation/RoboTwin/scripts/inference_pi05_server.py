#!/usr/bin/env python3
"""
RoboTwin 仿真评测客户端 —— 通过 WebSocket 调用远端 PI05 策略服务。

架构
----
本模块是 **环境侧客户端**，不负责加载模型权重。推理在独立的
``my_vla/scripts/serve_pi05_policy.py`` 服务进程中完成，双方通过 WebSocket + msgpack
协议通信。

::

    RoboTwin 仿真 (本脚本)  --WebSocket-->  PI05 Policy Server (my_vla)

主流程（``run_eval``）
---------------------
1. 读取 JSONC 配置，连接 PI05 WebSocket 服务并获取服务端元数据
2. 按 ``task_idx`` 实例化 RoboTwin 任务环境
3. 对每个 episode：
   a. 先采样一个 **规划成功** 的初始场景（与 ``inference.py`` 逻辑一致）
   b. 生成自然语言指令，进入逐步控制循环
   c. 观测 → 转 PI05 frame → WebSocket 推理 → 得到 action chunk
   d. 按 ``infer_horizon`` 逐步执行动作，直至成功或达到步数上限
   e. 保存回放视频与成功率统计
4. 写出 ``summary.json`` / ``summary.txt``

用法::

    cd /vla/my_tbot/evaluation/RoboTwin
    python scripts/inference_pi05_server.py
    python scripts/inference_pi05_server.py pi05_server_eval.jsonc

需先启动 PI05 服务（my_vla 环境）::

    python /vla/my_vla/scripts/serve_pi05_policy.py \\
        --ckpt_path .../pretrained_model --host 0.0.0.0 --port 8000 --infer_horizon 16
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio
import msgpack
import numpy as np
import websockets.sync.client

# ---------------------------------------------------------------------------
# 路径常量与运行时初始化
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROBOTWIN_EVAL_ROOT = SCRIPT_DIR.parent          # evaluation/RoboTwin/
ROOT_PATH = ROBOTWIN_EVAL_ROOT.parents[1]       # my_tbot 项目根
ROBOTWIN_ROOT = ROOT_PATH / "third_party" / "RoboTwin"
DEFAULT_CONFIG = ROBOTWIN_EVAL_ROOT / "pi05_server_eval.jsonc"

# 将 my_tbot / RoboTwin 相关目录加入 sys.path，并切换工作目录到 RoboTwin
for candidate in (str(ROOT_PATH / "src"), str(ROOT_PATH), str(ROBOTWIN_EVAL_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

sys.path.extend(
    [
        str(ROBOTWIN_ROOT),
        str(ROBOTWIN_ROOT / "policy"),
        str(ROBOTWIN_ROOT / "description" / "utils"),
    ]
)

os.chdir(ROBOTWIN_ROOT)

from envs.utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions
from inference import TASK_NAMES, build_task_args, class_decorator, resolve_bool_env
from load_jsonc_config import load_jsonc

logger = logging.getLogger(__name__)

# PI05 服务期望的图像观测 key（与 serve_pi05_policy 协议对齐）
PI05_IMAGE_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


# ---------------------------------------------------------------------------
# JSONC 配置：加载、解析、数据结构
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    """单次 RoboTwin 任务评测的全部运行参数。"""
    ws_host: str
    ws_port: int
    ws_uri: str | None
    task_idx: int
    task_config: str
    instruction_type: str
    seed: int
    test_num: int
    infer_horizon: int | None
    action_horizon_size: int
    action_mode: str
    binarize_gripper: bool
    robot_type: tuple[int, ...]
    video_dir: Path
    fps: int
    log_level: str
    debug: bool


def resolve_config_path(arg: str | None) -> Path:
    """解析 JSONC 配置路径；相对路径相对于 ROBOTWIN_EVAL_ROOT。"""
    path = Path(arg) if arg else DEFAULT_CONFIG
    if not path.is_absolute():
        path = (ROBOTWIN_EVAL_ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    return path


def parse_config(raw: dict[str, Any]) -> EvalConfig:
    """将 JSONC 展平字典转为 EvalConfig，并补全相对路径。"""
    video_dir = Path(raw.get("video_dir", "output_pi05_server/debug"))
    if not video_dir.is_absolute():
        video_dir = (ROBOTWIN_EVAL_ROOT / video_dir).resolve()

    robot_type = raw.get("robot_type", [6, 1, 6, 1])
    if not isinstance(robot_type, (list, tuple)):
        raise ValueError(f"robot_type must be a list, got {type(robot_type)}")

    return EvalConfig(
        ws_host=str(raw.get("ws_host", "127.0.0.1")),
        ws_port=int(raw.get("ws_port", 8000)),
        ws_uri=raw.get("ws_uri"),
        task_idx=int(raw.get("task_idx", 0)),
        task_config=str(raw.get("task_config", "demo_randomized")),
        instruction_type=str(raw.get("instruction_type", "unseen")),
        seed=int(raw.get("seed", 0)),
        test_num=int(raw.get("test_num", 2)),
        infer_horizon=raw.get("infer_horizon"),
        action_horizon_size=int(raw.get("action_horizon_size", 50)),
        action_mode=str(raw.get("action_mode", "abs")),
        binarize_gripper=bool(raw.get("binarize_gripper", True)),
        robot_type=tuple(int(x) for x in robot_type),
        video_dir=video_dir,
        fps=int(raw.get("fps", 30)),
        log_level=str(raw.get("log_level", "WARNING")),
        debug=bool(raw.get("debug", False)),
    )


# ---------------------------------------------------------------------------
# WebSocket 客户端：与 my_vla serve_pi05_policy 的 msgpack 协议
# ---------------------------------------------------------------------------
def _pack_array(obj: Any) -> Any:
    """msgpack 序列化钩子：将 numpy 数组编码为自定义 dict。"""
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype for msgpack: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: Any) -> Any:
    """msgpack 反序列化钩子：还原 numpy 数组。"""
    if isinstance(obj, dict) and b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if isinstance(obj, dict) and b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_packer = msgpack.Packer(default=_pack_array)
_unpack = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class Pi05WebsocketClient:
    """PI05 策略服务的同步 WebSocket 客户端。"""

    def __init__(self, cfg: EvalConfig) -> None:
        if cfg.ws_uri:
            self._uri = cfg.ws_uri
        else:
            self._uri = f"ws://{cfg.ws_host}:{cfg.ws_port}"
        self._ws, self.metadata = self._connect()

    def _connect(self):
        """阻塞重连，直到 PI05 服务可用；首包为服务端 metadata。"""
        while True:
            try:
                ws = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
                return ws, _unpack(ws.recv())
            except OSError as exc:
                logger.info("Waiting for PI05 server at %s (%s)", self._uri, exc)
                time.sleep(2.0)

    def infer_frame(self, frame: dict[str, Any], *, timestep: int, reset: bool) -> dict[str, Any]:
        """发送单帧观测，返回包含 ``actions`` action chunk 的推理结果。"""
        payload = dict(frame)
        payload["timestep"] = int(timestep)
        payload["reset"] = bool(reset)
        self._ws.send(_packer.pack(payload))
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"PI05 server error:\n{resp}")
        return _unpack(resp)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass


def robotwin_obs_to_frame(observation: dict[str, Any], task: str) -> dict[str, Any]:
    """将 RoboTwin 原生观测 dict 转为 PI05 服务期望的 frame 格式。"""
    obs = observation["observation"]
    return {
        PI05_IMAGE_KEYS[0]: obs["head_camera"]["rgb"],
        PI05_IMAGE_KEYS[1]: obs["left_camera"]["rgb"],
        PI05_IMAGE_KEYS[2]: obs["right_camera"]["rgb"],
        "observation.state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
        "task": str(task),
    }


# ---------------------------------------------------------------------------
# 评测主逻辑（整体结构与 evaluation/RoboTwin/inference.py 对齐）
# ---------------------------------------------------------------------------
def apply_action_mode(
    action_pred: np.ndarray,
    observation: dict[str, Any],
    *,
    action_mode: str,
    robot_type: tuple[int, ...],
) -> np.ndarray:
    """按 action_mode 后处理模型输出；delta 模式下将增量叠加到当前关节状态。"""
    if action_mode != "delta":
        return action_pred

    init_action = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    action_dim = min(action_pred.shape[-1], init_action.shape[0])
    left_gripper_idx = sum(robot_type[0:2]) - 1
    right_gripper_idx = sum(robot_type[0:4]) - 1

    out = action_pred.copy()
    init = init_action[None, :action_dim].astype(np.float32, copy=False)
    init[:, left_gripper_idx] = 0.0
    init[:, right_gripper_idx] = 0.0
    out[..., :action_dim] += init
    return out


def write_task_summary(
    cfg: EvalConfig,
    task_name: str,
    success_count: int,
    test_num: int,
    server_metadata: dict[str, Any],
) -> None:
    """将单任务评测结果写入 video_dir 下的 summary.json / summary.txt。"""
    cfg.video_dir.mkdir(parents=True, exist_ok=True)
    success_rate = round((success_count / test_num) * 100, 2) if test_num else 0.0
    summary = {
        "task_idx": cfg.task_idx,
        "task_name": task_name,
        "task_config": cfg.task_config,
        "success_count": int(success_count),
        "test_num": int(test_num),
        "success_rate": success_rate,
        "deployment": "pi05_websocket_server",
        "ws_uri": cfg.ws_uri or f"ws://{cfg.ws_host}:{cfg.ws_port}",
        "server_checkpoint_dir": server_metadata.get("checkpoint_dir"),
        "instruction_type": cfg.instruction_type,
        "action_mode": cfg.action_mode,
        "infer_horizon": server_metadata.get("infer_horizon"),
    }
    (cfg.video_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (cfg.video_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"task_idx: {cfg.task_idx}",
                f"task_name: {task_name}",
                f"success_count: {success_count}/{test_num}",
                f"success_rate: {success_rate:.2f}%",
                f"server_ckpt: {server_metadata.get('checkpoint_dir')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_eval(cfg: EvalConfig) -> None:
    """执行单任务、多 episode 的 RoboTwin 评测主循环。"""
    # ---- 1. 初始化：任务环境 + WebSocket 客户端 ----
    task_name = TASK_NAMES[cfg.task_idx]
    task_args = build_task_args(cfg.task_config, task_name)
    task_env = class_decorator(task_args["task_name"])
    client = Pi05WebsocketClient(cfg)

    # 推理超参与动作维度：优先用本地配置，否则从服务端 metadata 读取
    infer_horizon = int(cfg.infer_horizon or client.metadata.get("infer_horizon", 16))
    action_dim = min(int(client.metadata.get("expected_action_dim", 14)), sum(cfg.robot_type))
    binarize_gripper = resolve_bool_env(cfg.binarize_gripper, "BINARIZE_GRIPPER")

    left_gripper_idx = sum(cfg.robot_type[0:2]) - 1
    right_gripper_idx = sum(cfg.robot_type[0:4]) - 1

    logger.info("task=%s server=%s", task_name, client.metadata.get("checkpoint_dir"))
    logger.info("infer_horizon=%d action_dim=%d", infer_horizon, action_dim)

    # ---- 2. 评测状态变量（与 inference.py 保持一致）----
    task_env.suc = 0
    task_env.test_num = 0
    now_id = 0
    succ_seed = 0
    st_seed = 100000 * (1 + cfg.seed)
    now_seed = st_seed
    clear_cache_freq = task_args["clear_cache_freq"]
    task_args["eval_mode"] = True
    succ_seeds = list(range(st_seed, st_seed * 2))

    try:
        # ---- 3. 外层循环：收集 test_num 个有效 episode ----
        while succ_seed < cfg.test_num:
            render_freq = task_args["render_freq"]
            task_args["render_freq"] = 0

            # 3a. 预演（play_once）：筛选规划成功且物理稳定的初始场景
            try:
                task_env.setup_demo(
                    now_ep_num=now_id, seed=succ_seeds[now_seed - st_seed], is_test=True, **task_args
                )
                episode_info = task_env.play_once()
                task_env.close_env()
            except (UnStableError, Exception):
                task_env.close_env()
                now_seed += 1
                task_args["render_freq"] = render_freq
                continue

            if not (task_env.plan_success and task_env.check_success()):
                now_seed += 1
                task_args["render_freq"] = render_freq
                continue

            succ_seed += 1
            task_args["render_freq"] = render_freq

            # 3b. 用同一 seed 重建场景，注入自然语言指令，开始正式评测
            task_env.setup_demo(
                now_ep_num=now_id, seed=succ_seeds[now_seed - st_seed], is_test=True, **task_args
            )
            instruction = np.random.choice(
                generate_episode_descriptions(task_name, [episode_info["info"]], cfg.test_num)[0][
                    cfg.instruction_type
                ]
            )
            task_env.set_instruction(instruction=instruction)

            # ---- 4. 内层循环：action chunk 推理 + 逐步执行 ----
            succ = False
            action_plan: deque[np.ndarray] = deque([], maxlen=cfg.action_horizon_size)
            replay_images: list[np.ndarray] = []
            replan_idx = 0

            while task_env.take_action_cnt < task_env.step_lim:
                # action_plan 耗尽时向 PI05 服务请求新的 action chunk
                if not action_plan:
                    observation = task_env.get_obs()
                    replay_images.append(observation["observation"]["head_camera"]["rgb"].copy())

                    frame = robotwin_obs_to_frame(observation, task_env.get_instruction())
                    result = client.infer_frame(frame, timestep=replan_idx, reset=(replan_idx == 0))
                    replan_idx += 1

                    action_pred = np.array(result["actions"], dtype=np.float32, copy=True)
                    action_pred = action_pred[:infer_horizon, :action_dim]
                    action_pred = apply_action_mode(
                        action_pred,
                        observation,
                        action_mode=cfg.action_mode,
                        robot_type=cfg.robot_type,
                    )
                    action_plan.extend(row.copy() for row in action_pred)

                # 从 chunk 中取一步动作并下发给仿真器
                action = np.array(action_plan.popleft(), dtype=np.float32, copy=True)
                if binarize_gripper:
                    action[left_gripper_idx] = 0.0 if action[left_gripper_idx] < 0.5 else 1.0
                    action[right_gripper_idx] = 0.0 if action[right_gripper_idx] < 0.5 else 1.0
                task_env.take_action(action, action_type="qpos")

                if task_env.eval_success:
                    succ = True
                    break

            # ---- 5. episode 收尾：统计、录像、清理环境 ----
            print("\033[92mSuccess!\033[0m" if succ else "\033[91mFail!\033[0m")
            if succ:
                task_env.suc += 1

            cfg.video_dir.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(
                cfg.video_dir / f"{'success' if succ else 'failure'}_{succ_seed}.mp4",
                replay_images,
                fps=cfg.fps,
            )

            now_id += 1
            task_env.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
            if task_env.render_freq:
                task_env.viewer.close()
            task_env.test_num += 1

            print(
                f"\033[93m{task_name}\033[0m | success "
                f"\033[96m{task_env.suc}/{task_env.test_num}\033[0m "
                f"(\033[95m{round(task_env.suc / task_env.test_num * 100, 1)}%\033[0m)\n"
            )
            now_seed += 1

        # ---- 6. 全部 episode 完成后写出汇总 ----
        write_task_summary(cfg, task_name, task_env.suc, task_env.test_num, client.metadata)
    finally:
        client.close()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> None:
    """解析命令行 → 加载 JSONC → 配置日志 → 启动评测。"""
    config_path = resolve_config_path(sys.argv[1] if len(sys.argv) > 1 else None)
    cfg = parse_config(load_jsonc(config_path))

    log_levels = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
    logging.basicConfig(
        level=log_levels.get(cfg.log_level.upper(), logging.INFO if cfg.debug else logging.WARNING),
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logging.getLogger("curobo").setLevel(logging.WARNING)

    logger.info("Config: %s", config_path)
    run_eval(cfg)


if __name__ == "__main__":
    main()
