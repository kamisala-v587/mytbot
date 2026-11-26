#!/usr/bin/env bash
# Launch 4 single-GPU Stage A overfit diagnostics in parallel (AE-1 G3 style).
#   bash tools/bpv5/run_stage_a_overfit_diag.sh
# Stops nothing else; assumes GPUs 0-3 free.
set -euo pipefail
REPO=/vla/workspace/my_tbot
cd "$REPO"

launch() {
  local gpu=$1 cfg=$2 port=$3
  echo "GPU $gpu <- $cfg (port $port)"
  CUDA_VISIBLE_DEVICES=$gpu PORT=$port NUM_GPUS=1 \
    bash tools/bpv5/train_stage_a.sh "$cfg"
}

launch 0 configs/bpv5/stage_a_overfit_rt1.jsonc 29670
sleep 2
launch 1 configs/bpv5/stage_a_overfit_rt8.jsonc 29671
sleep 2
launch 2 configs/bpv5/stage_a_overfit_idn32.jsonc 29672
sleep 2
launch 3 configs/bpv5/stage_a_overfit_agi32.jsonc 29673

echo
echo "tail logs:"
echo "  tail -f outputs/BPv5/stage_a/logs/train_stage_a_overfit_rt1_latest.log"
echo "  tail -f outputs/BPv5/stage_a/logs/train_stage_a_overfit_rt8_latest.log"
echo "  tail -f outputs/BPv5/stage_a/logs/train_stage_a_overfit_idn32_latest.log"
echo "  tail -f outputs/BPv5/stage_a/logs/train_stage_a_overfit_agi32_latest.log"
echo "plot later: mytbot python code/stage_a_overfit_diag.ipynb  (or tools/bpv5/plot_stage_a_overfit.py)"
