#!/bin/bash
# Report-only evaluations of the AE-2 large-scale models (step 20000): independent test (bp_probe, G4 / G5 protocol,
# never used for decisions) and G7-D representation probes (bp_probe + held-out) for S-A and S-U.
# usage: source tools/bpv4cdx/env.local.sh; bash tools/bpv4cdx/run_ae2l_report.sh [out_subdir]
set -u
cd "$BPV4CDX_REPO" || exit 1
OUT=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a/${1:-v2}
mkdir -p "$OUT/logs" "$OUT/ae2l_test" "$OUT/ae2l_g7d"
ck() { ls -d "$OUT/$1"/*/checkpoints/020000/pretrained_model | tail -1; }
PROBE=$BPV4CDX_REPO/configs/ds_ids/pro6k/robo_bpv4_bpen_probe.txt
LABELS=$BPV4CDX_OUT_ROOT/BPv4/probe/labels_probe.json
T=$(ck ae2l_textonly)
EV="python -m lerobot.policies.BPV4CDX.gate_g4_g7 --out $OUT/ae2l_test --textonly_ckpt $T --split all --dataset.repo_id_file=$PROBE --dataset.data_root=null"
PR="python -m lerobot.policies.BPV4CDX.probe_zdemo --out $OUT/ae2l_g7d"
CUDA_VISIBLE_DEVICES=0 $EV --ckpt "$(ck ae2l_main)" --name sa_bp_probe > "$OUT/logs/test_sa.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $EV --ckpt "$(ck ae2l_su)" --name su_bp_probe > "$OUT/logs/test_su.log" 2>&1 &
(CUDA_VISIBLE_DEVICES=2 $PR --ckpt "$(ck ae2l_main)" --data probe --probe_repo_file "$PROBE" --labels "$LABELS" --name sa_probe &&
 CUDA_VISIBLE_DEVICES=2 $PR --ckpt "$(ck ae2l_main)" --data heldout --name sa_heldout) > "$OUT/logs/g7d_sa.log" 2>&1 &
(CUDA_VISIBLE_DEVICES=3 $PR --ckpt "$(ck ae2l_su)" --data probe --probe_repo_file "$PROBE" --labels "$LABELS" --name su_probe &&
 CUDA_VISIBLE_DEVICES=3 $PR --ckpt "$(ck ae2l_su)" --data heldout --name su_heldout) > "$OUT/logs/g7d_su.log" 2>&1 &
wait
echo "done $(date '+%m-%d %H:%M')"
