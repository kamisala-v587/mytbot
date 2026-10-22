#!/bin/bash
# AE-2 large models (step 20000): event-aligned saliency (G7-S v2), attention maps (prefix / suffix groups,
# segment_query -> Z_demo slots) and the demo-encoder inspection (Q-Former attention, Z_demo slot probes).
# usage: source tools/bpv4cdx/env.local.sh; bash tools/bpv4cdx/run_attn_diag.sh [out_subdir]
set -u
cd "$BPV4CDX_REPO" || exit 1
PY=/vla/workspace/shiliangchao/env/q35va/bin/python3.10
OUT=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a/${1:-v2}
mkdir -p "$OUT/logs"
ck() { ls -d "$OUT/$1"/*/checkpoints/020000/pretrained_model | tail -1; }
SA=$(ck ae2l_main)
SU=$(ck ae2l_su)
PROBE="--data probe --probe_repo_file configs/ds_ids/pro6k/robo_bpv4_bpen_probe.txt --labels $BPV4CDX_OUT_ROOT/BPv4/probe/labels_probe.json --episodes 0"
G="$PY -m lerobot.policies.BPV4CDX.saliency_zdemo --out $OUT/g7s_v2 --figures 3"
A="$PY -m lerobot.policies.BPV4CDX.attn_map --out $OUT/attn"
Z="$PY -m lerobot.policies.BPV4CDX.zdemo_inspect --out $OUT/zdemo"
(CUDA_VISIBLE_DEVICES=0 $G --ckpt "$SA" --data heldout --episodes 100 --name sa_heldout &&
 CUDA_VISIBLE_DEVICES=0 $G --ckpt "$SA" $PROBE --name sa_probe) > "$OUT/logs/g7s_v2_sa.log" 2>&1 &
(CUDA_VISIBLE_DEVICES=1 $G --ckpt "$SU" --data heldout --episodes 100 --name su_heldout &&
 CUDA_VISIBLE_DEVICES=1 $G --ckpt "$SU" $PROBE --name su_probe) > "$OUT/logs/g7s_v2_su.log" 2>&1 &
(CUDA_VISIBLE_DEVICES=2 $A --ckpt "$SA" --episodes 100 --name sa_heldout &&
 CUDA_VISIBLE_DEVICES=2 $A --ckpt "$SU" --episodes 100 --name su_heldout &&
 CUDA_VISIBLE_DEVICES=2 $A --ckpt "$SA" --episodes 100 --generic --name sa_heldout_generic) > "$OUT/logs/attn.log" 2>&1 &
(CUDA_VISIBLE_DEVICES=3 $Z --ckpt "$SA" --name sa_heldout &&
 CUDA_VISIBLE_DEVICES=3 $Z --ckpt "$SU" --name su_heldout) > "$OUT/logs/zdemo.log" 2>&1 &
wait
echo "done $(date '+%m-%d %H:%M')"
