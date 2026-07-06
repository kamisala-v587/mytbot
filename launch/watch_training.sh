#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 训练 watchdog：监控 loss.log，异常时钉钉通知
#
# 用法：
#   bash launch/watch_training.sh outputs/TBot_SA1/pretrain_v1/2026-07-03/01-46-07_
#
# 可选环境变量：
#   STALE_THRESHOLD=600   # 多少秒无更新则告警（默认 600）
#   CHECK_INTERVAL=60     # 检查间隔（默认 60）
###############################################################################

OUTPUT_DIR="${1:?用法: bash launch/watch_training.sh <output_dir>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

STALE_THRESHOLD="${STALE_THRESHOLD:-600}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"

exec python -m lerobot.scripts.watch_training \
  --output-dir "${OUTPUT_DIR}" \
  --stale-threshold "${STALE_THRESHOLD}" \
  --check-interval "${CHECK_INTERVAL}"
