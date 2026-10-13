#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用 LeRobot policy WebSocket 服务。

日常直接改本文件最上方的大写常量。命令行参数只在显式传入时覆盖对应常量。

目前内置 tbot、bpvav2、bpvav3、q35。新增策略时在 ``policy_runtime/`` 下增加模块并导出
``ADAPTER``，把 ``POLICY`` 设为 ``auto`` 后会按 checkpoint 的 type 自动选择。

.. code-block:: bash

   cd /vla/workspace/my_tbot
   python server/serve_policy.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from policy_runtime.common import ServerConfig, prepare_import_path, serve  # noqa: E402

prepare_import_path()

from policy_runtime.registry import create_policy  # noqa: E402


# =============================================================================
# 配置：日常改这里。只有命令行显式传入的参数才会覆盖对应常量。
# =============================================================================

# auto 按 checkpoint config.json 的 type 选择；也可写 tbot / bpvav2 / bpvav3 / q35
POLICY = "auto"

CKPT_PATH = ""

HOST = "0.0.0.0"
PORT = 8000
DEVICE = "auto"  # auto | cuda | cuda:0 | cpu
DTYPE = "bfloat16"  # bfloat16 | float32

DEBUG = True
# 相对路径相对于 server/ 目录。
# 示教视频固定缓存在 debug_logs/bp_video_cache/<adapter>/。
# DEBUG 打开时，每次启动的首次 client 输入写到 debug_logs/<时间戳>/vla 或 bp。
DEBUG_DIR = "debug_logs"

MAX_BATCH_SIZE = 4
BATCH_WAIT_MS = 10.0
QUEUE_SIZE = 64

DEFAULT_PROMPT = "Execute the task."
# None 跟随 checkpoint；也可写 "delta" 或 "abs"
ACTION_MODE = None
# None 使用 checkpoint 的 n_action_steps
INFER_HORIZON = None
STATS_PATH = None
STATS_KEY = None

# tbot / bpvav2：True 时多帧输入必须正好是所需帧数；只给 1 帧时复制补齐。
STRICT_HISTORY = True

RESIZE_SIZE = 224
REQUEST_IMAGE_HEIGHT = 480
REQUEST_IMAGE_WIDTH = 640
NUM_INFERENCE_STEPS = None

BP_MAPPING_PATH = "/vla/workspace/my_tbot/server/policy_runtime/bpva_task_bps_rand.yml"
QWEN3_VL_PATH = "/vla/workspace/models/Qwen3-VL-2B-Instruct"
QWEN3_5_PATH = "/vla/workspace/shiliangchao/models/Qwen35_0_8"

COSMOS_TOKENIZER_PATH = None
DA3_MODEL_PATH = None
DA3_CODE_ROOT = None
LOAD_DEVICE = None
COSMOS_DEVICE = None
DISABLE_3D_TEACHER = True
OMIT_VISUAL_TOKENS = True

# 推理时按路径屏蔽字段或最后一维。空列表表示不屏蔽。
# 客户端字段在组装前处理；behavior_prompt.* 在服务端拼好示教后再处理。
# 例:
#   "behavior_prompt.action"           整段示教动作，同时 action_is_pad=True
#   "behavior_prompt.action[6,13]"     只清 Aloha 左右夹爪
#   "behavior_prompt.state[6,13]"      只清示教 state 夹爪
#   "behavior_prompt"                  整段 BP 置零，chunk mask=False；保留 bp_image_grid_thw
#   "observation.images.cam_left_wrist" 丢掉当前左腕图像
#   "observation.state[6,13]"          清当前 state 夹爪
#   "task"                             丢掉语言指令，回退 DEFAULT_PROMPT
INPUT_MASKS = [
   #  "behavior_prompt.action[6,13]"
]

# 训练配置（json 或 jsonc）。从 dataset.bp_action_use / bp_state_use 裁缓存示教，
# 和训练用同一条规则：false 整段置零并让对应 token 失效，切片只清掉没选中的维度。
# 空字符串则读取 checkpoint 里的 train_config.json；两边都没有时保留完整示教。
BP_DIM_CONFIG = ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="通用 LeRobot policy WebSocket 服务。未传入的参数使用文件顶部常量。",
    )
    parser.add_argument("--policy", default=None, help="auto | tbot | bpvav2 | bpvav3 | q35")
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default=None)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--debug_dir", default=None)
    parser.add_argument("--max_batch_size", type=int, default=None)
    parser.add_argument("--batch_wait_ms", type=float, default=None)
    parser.add_argument("--queue_size", type=int, default=None)
    parser.add_argument("--default_prompt", default=None)
    parser.add_argument("--action_mode", choices=["delta", "abs"], default=None)
    parser.add_argument("--infer_horizon", type=int, default=None)
    parser.add_argument("--stats_path", default=None)
    parser.add_argument("--stats_key", default=None)
    parser.add_argument("--strict_history", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bp_mapping_path", default=None)
    parser.add_argument("--resize_size", type=int, default=None)
    parser.add_argument("--request_image_height", type=int, default=None)
    parser.add_argument("--request_image_width", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--qwen3_vl_path", default=None)
    parser.add_argument("--qwen3_5_path", default=None)
    parser.add_argument("--cosmos_tokenizer_path", default=None)
    parser.add_argument("--da3_model_path", default=None)
    parser.add_argument("--da3_code_root", default=None)
    parser.add_argument("--load_device", default=None)
    parser.add_argument("--cosmos_device", default=None)
    parser.add_argument("--disable_3d_teacher", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--omit_visual_tokens", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--input_masks",
        default=None,
        help="逗号分隔的屏蔽路径，覆盖文件顶部 INPUT_MASKS。例如 behavior_prompt.action,observation.state[6,13]",
    )
    parser.add_argument(
        "--bp_dim_config",
        default=None,
        help="含 dataset.bp_action_use / bp_state_use 的 json 或 jsonc，覆盖 BP_DIM_CONFIG。",
    )
    return parser


def _pick(cli_value, constant):
    return constant if cli_value is None else cli_value


def _parse_input_masks(value) -> list[str]:
    from policy_runtime.input_mask import parse_mask_list, parse_mask_rules

    specs = parse_mask_list(value)
    try:
        parse_mask_rules(specs)
    except ValueError as exc:
        raise SystemExit(f"INPUT_MASKS 无效: {exc}") from exc
    return specs


def _resolve_debug_dir(path: str) -> str:
    debug_dir = Path(path).expanduser()
    if not debug_dir.is_absolute():
        debug_dir = SERVER_DIR / debug_dir
    return str(debug_dir)


def build_config(argv: list[str] | None = None) -> ServerConfig:
    parsed = _build_parser().parse_args(argv)
    config = ServerConfig(
        policy=_pick(parsed.policy, POLICY),
        ckpt_path=_pick(parsed.ckpt_path, CKPT_PATH),
        host=_pick(parsed.host, HOST),
        port=_pick(parsed.port, PORT),
        device=_pick(parsed.device, DEVICE),
        dtype=_pick(parsed.dtype, DTYPE),
        debug=_pick(parsed.debug, DEBUG),
        debug_dir=_resolve_debug_dir(_pick(parsed.debug_dir, DEBUG_DIR)),
        max_batch_size=_pick(parsed.max_batch_size, MAX_BATCH_SIZE),
        batch_wait_ms=_pick(parsed.batch_wait_ms, BATCH_WAIT_MS),
        queue_size=_pick(parsed.queue_size, QUEUE_SIZE),
        default_prompt=_pick(parsed.default_prompt, DEFAULT_PROMPT),
        infer_horizon=_pick(parsed.infer_horizon, INFER_HORIZON),
        stats_key=_pick(parsed.stats_key, STATS_KEY),
        stats_path=_pick(parsed.stats_path, STATS_PATH),
        action_mode=_pick(parsed.action_mode, ACTION_MODE),
        resize_size=_pick(parsed.resize_size, RESIZE_SIZE),
        request_image_height=_pick(parsed.request_image_height, REQUEST_IMAGE_HEIGHT),
        request_image_width=_pick(parsed.request_image_width, REQUEST_IMAGE_WIDTH),
        strict_history=_pick(parsed.strict_history, STRICT_HISTORY),
        bp_mapping_path=_pick(parsed.bp_mapping_path, BP_MAPPING_PATH),
        num_inference_steps=_pick(parsed.num_inference_steps, NUM_INFERENCE_STEPS),
        qwen3_vl_path=_pick(parsed.qwen3_vl_path, QWEN3_VL_PATH) or None,
        qwen3_5_path=_pick(parsed.qwen3_5_path, QWEN3_5_PATH) or None,
        cosmos_tokenizer_path=_pick(parsed.cosmos_tokenizer_path, COSMOS_TOKENIZER_PATH) or None,
        da3_model_path=_pick(parsed.da3_model_path, DA3_MODEL_PATH) or None,
        da3_code_root=_pick(parsed.da3_code_root, DA3_CODE_ROOT) or None,
        load_device=_pick(parsed.load_device, LOAD_DEVICE) or None,
        cosmos_device=_pick(parsed.cosmos_device, COSMOS_DEVICE) or None,
        disable_3d_teacher=_pick(parsed.disable_3d_teacher, DISABLE_3D_TEACHER),
        omit_visual_tokens=_pick(parsed.omit_visual_tokens, OMIT_VISUAL_TOKENS),
        input_masks=_parse_input_masks(_pick(parsed.input_masks, INPUT_MASKS)),
        bp_dim_config=_pick(parsed.bp_dim_config, BP_DIM_CONFIG) or None,
    )
    bp_dim_config = str(config.bp_dim_config or "").strip()
    if bp_dim_config and not Path(bp_dim_config).expanduser().is_file():
        raise SystemExit(f"BP_DIM_CONFIG 不存在: {bp_dim_config}")
    if not str(config.ckpt_path).strip():
        raise SystemExit("请在 serve_policy.py 顶部设置 CKPT_PATH，或传入 --ckpt_path。")
    if config.max_batch_size < 1:
        raise SystemExit("MAX_BATCH_SIZE 必须 >= 1。")
    if config.batch_wait_ms < 0:
        raise SystemExit("BATCH_WAIT_MS 必须 >= 0。")
    if config.queue_size < config.max_batch_size:
        raise SystemExit("QUEUE_SIZE 必须 >= MAX_BATCH_SIZE。")
    if config.dtype not in {"float32", "bfloat16"}:
        raise SystemExit("DTYPE 只能是 float32 或 bfloat16。")
    if config.action_mode not in {None, "delta", "abs"}:
        raise SystemExit("ACTION_MODE 只能是 None、delta 或 abs。")
    if config.debug:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f_%z")
        run_dir = Path(config.debug_dir) / stamp
        run_dir.mkdir(parents=True, exist_ok=False)
        config = replace(config, debug_run_dir=str(run_dir))
        logging.info("Debug 输出目录: %s", run_dir)
    return config


def main(argv: list[str] | None = None) -> None:
    config = build_config(argv)
    logging.info("生效配置:\n%s", json.dumps(asdict(config), indent=2, ensure_ascii=False))
    policy = create_policy(config)
    serve(policy, config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    main()
