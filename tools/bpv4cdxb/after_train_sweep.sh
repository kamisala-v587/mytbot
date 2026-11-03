#!/bin/bash
# Wait for the b4_sft_full2 training to exit, then run the 16-task quick sweep:
#   1. run 3 (b4_sft_full2): 10k 20k 30k 40k 50k, all 16 tasks
#   2. run 2 (b4_sft_full) 6k..30k and 3. run 1 (b4_sft) 4k..20k: top-up of the 10 added tasks (old 6 are kept)
#   4. summary (eval_result_bp/_sweep16_summary.md) + figure (figs_b4_bp_sft/sim_success_16.png)
# usage: setsid nohup bash tools/bpv4cdxb/after_train_sweep.sh > <log> 2>&1 &
set -u
REPO=/vla/workspace/my_tbot
cd "$REPO" || exit 1
source tools/bpv4cdxb/env.local.sh >/dev/null 2>&1
ST=$REPO/outputs/BPv4CDX/stage_b
RUN1=$ST/b4_sft/2026-10-29_18-26/2026-10-29_18-27/checkpoints
RUN2=$ST/b4_sft_full/2026-10-30_02-46/2026-10-30_02-47/checkpoints
RUN3=$ST/b4_sft_full2/2026-10-30_13-45/2026-10-30_13-47/checkpoints
TRAIN_PATTERN="generated/b4_sft_full2.json"

echo "$(date '+%F %T') waiting for training ($TRAIN_PATTERN) to exit"
while pgrep -f "$TRAIN_PATTERN" > /dev/null; do sleep 120; done
echo "$(date '+%F %T') training exited; last checkpoints: $(ls "$RUN3" | tail -3 | tr '\n' ' ')"
sleep 90  # let the GPUs release memory
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

ck() { printf "%s/%06d/pretrained_model" "$1" "$2"; }
specs3=()
for s in 10000 20000 30000 40000 50000; do
  [ -f "$(ck "$RUN3" $s)/model.safetensors" ] && specs3+=("full2_$((s / 1000))k:$(ck "$RUN3" $s)")
done
latest=$(ls "$RUN3" | grep -E '^[0-9]{6}$' | sort | tail -1)
if [ "$latest" != "050000" ] && [ -n "$latest" ]; then  # training stopped early: also test the newest checkpoint
  specs3+=("full2_$((10#$latest / 1000))k:$RUN3/$latest/pretrained_model")
fi

echo "$(date '+%F %T') phase 1: run 3, 16 tasks: ${specs3[*]%%:*}"
SWEEP_PORT0=8031 bash tools/bpv4cdxb/sim_sweep_bp.sh b4full2 "${specs3[@]}"

echo "$(date '+%F %T') phase 2: run 2 top-up"
specs2=()
for s in 6000 12000 18000 24000 30000; do specs2+=("full_$((s / 1000))k:$(ck "$RUN2" $s)"); done
SWEEP_PORT0=8041 bash tools/bpv4cdxb/sim_sweep_bp.sh b4full "${specs2[@]}"

echo "$(date '+%F %T') phase 3: run 1 top-up"
specs1=()
for s in 4000 8000 12000 16000 20000; do specs1+=("sft_$((s / 1000))k:$(ck "$RUN1" $s)"); done
SWEEP_PORT0=8051 bash tools/bpv4cdxb/sim_sweep_bp.sh b4sft "${specs1[@]}"

echo "$(date '+%F %T') summary"
python tools/bpv4cdxb/sim_sweep_report.py
echo "$(date '+%F %T') all done"
