#!/bin/bash
# G7-S saliency diagnostic for the AE-2 large-scale models (S-A main and S-U), held-out (100 episodes) and bp_probe
# (all 200 episodes, object / place positions); one run per GPU.
# usage: source tools/bpv4cdx/env.local.sh; bash tools/bpv4cdx/run_g7s.sh [out_subdir] [step]
set -u
cd "$BPV4CDX_REPO" || exit 1
OUT=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a/${1:-v2}
STEP=${2:-020000}
R=$OUT/g7s
mkdir -p "$OUT/logs" "$R"
ck() { ls -d "$OUT/$1"/*/checkpoints/$STEP/pretrained_model | tail -1; }
PROBE="--data probe --probe_repo_file configs/ds_ids/pro6k/robo_bpv4_bpen_probe.txt --labels $BPV4CDX_OUT_ROOT/BPv4/probe/labels_probe.json --episodes 0"
S="python -m lerobot.policies.BPV4CDX.saliency_zdemo --out $R --figures 3"
CUDA_VISIBLE_DEVICES=0 $S --ckpt "$(ck ae2l_main)" --data heldout --episodes 100 --name sa_heldout > "$OUT/logs/g7s_sa_heldout.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $S --ckpt "$(ck ae2l_su)" --data heldout --episodes 100 --name su_heldout > "$OUT/logs/g7s_su_heldout.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 $S --ckpt "$(ck ae2l_main)" $PROBE --name sa_probe > "$OUT/logs/g7s_sa_probe.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 $S --ckpt "$(ck ae2l_su)" $PROBE --name su_probe > "$OUT/logs/g7s_su_probe.log" 2>&1 &
wait
echo "done $(date '+%m-%d %H:%M')"
