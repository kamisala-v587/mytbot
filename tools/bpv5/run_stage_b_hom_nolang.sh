#!/usr/bin/env bash
# Offline Stage B HOM with language padded (isolates Z_demo identity).
# Usage:
#   bash tools/bpv5/run_stage_b_hom_nolang.sh <run_dir> <steps_csv> [max_ep] [out_name]
# Example:
#   bash tools/bpv5/run_stage_b_hom_nolang.sh \
#     /vla/workspace/my_tbot/outputs/BPv5/stage_b/pb3/2026-11-20_10-00 10000
set -euo pipefail
ROOT=/vla/workspace/my_tbot
source "$ROOT/tools/bpv4cdxb/env.local.sh"
export LD_LIBRARY_PATH=/vla/workspace/shiliangchao/env/q35va/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/vla/workspace/shiliangchao/env/q35va/bin/python

RUN="${1:?run dir}"
STEPS="${2:?comma-separated steps}"
MAX_EP="${3:-16}"
OUT_NAME="${4:-heldout_monitor_nolang.jsonl}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

echo "run=$RUN steps=$STEPS max_ep=$MAX_EP out=$OUT_NAME gpu=$GPU"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m lerobot.policies.BPV5StageB.heldout_monitor \
  --run "$RUN" --steps "$STEPS" --max_episodes "$MAX_EP" \
  --pad_language --overwrite --out_name "$OUT_NAME"
