#!/bin/bash
# AE-3 evaluation after training: G4 / G5 (protocol v2) / G7 round-trip on the held-out split, several checkpoints in
# parallel (one job per GPU line below). RoboTwin-only jobs are directly comparable with AE-2 (same 750 episodes).
# B200 version (run_ae3_eval.sh is the Pro6k one).
# usage: source tools/bpv4cdx/env.b200.sh; nohup bash tools/bpv4cdx/run_ae3_eval_b200.sh > /dev/null 2>&1 &
# progress: tail -f $OUT/logs/eval_<name>.log ; results: $OUT/g4_g7/<name>.json
set -u
cd "$BPV4CDX_REPO" || exit 1
OUT=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a3
RUN=$OUT/ae3_main/2026-10-23_07-13/checkpoints
mkdir -p "$OUT/logs" "$OUT/g4_g7"

job() {  # job <gpu> <step> <name> [extra args]
  local gpu=$1 step=$2 name=$3
  shift 3
  CUDA_VISIBLE_DEVICES=$gpu python -m lerobot.policies.BPV4CDX.gate_g4_g7 --ckpt "$RUN/$step/pretrained_model" \
    --protocol v2 --split val --out "$OUT/g4_g7" --name "$name" "$@" > "$OUT/logs/eval_$name.log" 2>&1
  echo "$(date '+%F %T') $name exit=$?" >> "$OUT/logs/eval_status.txt"
}

RT=--sources=RoboTwin-LeRobot-v3.0
job 0 0060000 s060k_robotwin $RT &
job 1 0070000 s070k_robotwin $RT &
job 2 0750000 s750k_robotwin $RT &
job 3 0300000 s300k_robotwin $RT &
(job 7 0600000 s600k_robotwin $RT; job 7 0700000 s700k_robotwin $RT) &
job 4 0060000 s060k_all &
job 5 0070000 s070k_all &
job 6 0750000 s750k_all &
wait
echo "$(date '+%F %T') all evaluations finished" >> "$OUT/logs/eval_status.txt"
