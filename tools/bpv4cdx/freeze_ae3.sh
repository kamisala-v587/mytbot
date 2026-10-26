#!/bin/bash
# AE-4 frozen delivery of an AE-3 checkpoint (design doc 4.5), with the G7 two-seed requirement formally waived
# (decision 2026-10-26) and the training-machine gates recorded from its report.
# usage: source tools/bpv4cdx/env.local.sh; bash tools/bpv4cdx/freeze_ae3.sh 0600000 [checkpoint root]
#   -> /vla/workspace/models/BPVA/BPv4CDX/StageA_AE_frozen/<step>/{StageA_AE_frozen.ckpt, manifest.json, pretrained_model/}
set -u
STEP=$1
SRC=${2:-/vla/workspace/models/BPVA/BPv4CDX/A}
cd "$BPV4CDX_REPO" || exit 1
R=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a3_eval/results
s=$((10#$STEP / 10000))w
GATES=()
for g in "g4_g7_robotwin=$R/eval/rt_$s.json" "g4_g7_robotwin_v2=$R/eval/rt_${s}_v2.json" "bp_probe_test=$R/test/probe_$s.json"; do
  [ -f "${g#*=}" ] && GATES+=(--gate "$g")
done
B200="bpv4cdx StageA-AE34 训练机结果.md / bpv4cdx Stage A-AE 实验报告 v3（AE-3）.md"
python -m lerobot.policies.BPV4CDX.freeze_ae --ckpt "$SRC/$STEP" \
  --out /vla/workspace/models/BPVA/BPv4CDX/StageA_AE_frozen/"$STEP" "${GATES[@]}" \
  --require G4,G5,G7_round_trip \
  --waive "G7_seeds=2026-10-26 decision: AE-3 trained with one seed; replaced by the AE-2 two-seed test (S-A / S-U seed 42 vs 43, same direction, final error within 1-8%) and the consistency of AE-3 checkpoints 300k / 600k / 750k (all pass G4 / G5 on all held-out sources)" \
  --external "B200_G1_G2_G3=pass (G3: error 0.00543 vs mean-action 0.258, -97.9%, segment ratio 0.99); $B200" \
  --external "B200_all_heldout_600k=7 sources, 7063 episodes: alternate 3.46 [3.36, 3.56], position v2 3.69, arm 7.07, hard v2 3.99, text drop 1.002, round trip median 2 frames / 53% within 2: G4 / G5 / round trip pass" \
  --external "B200_all_heldout_750k=alternate 3.61, position v2 3.96, arm 7.47, hard v2 4.31: G4 / G5 / round trip pass (700k not evaluated on all held-out; within 3% of 750k on RoboTwin)"
