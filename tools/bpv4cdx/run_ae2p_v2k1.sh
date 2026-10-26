#!/bin/bash
# After run_ae2p.sh: G5 protocol v2 with K = 1 for the three supplement models (same setting as v2_eval/sa42_v2.json and
# su42_v2.json, so the numbers are directly comparable; run_ae2p.sh's K = 3 runs are kept as sensitivity).
# usage: source tools/bpv4cdx/env.local.sh; bash tools/bpv4cdx/run_ae2p_v2k1.sh [out_subdir]
set -u
cd "$BPV4CDX_REPO" || exit 1
OUT=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a/${1:-v2}
R=$OUT/ae2p/eval
until grep -q "^done" "$OUT/run_ae2p.log" 2>/dev/null; do sleep 120; done
ck() { ls -d "$OUT/$1"/*/checkpoints/020000/pretrained_model | tail -1; }
EV="python -m lerobot.policies.BPV4CDX.gate_g4_g7 --out $R --textonly_ckpt $(ck ae2l_textonly) --protocol v2"
echo "v2 K=1 $(date '+%m-%d %H:%M')"
CUDA_VISIBLE_DEVICES=0 $EV --ckpt "$(ck ae2l_suf)" --name suf_v2k1 > "$OUT/logs/ae2p_v2k1_suf.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $EV --ckpt "$(ck ae2l_suf_p64)" --name suf_p64_v2k1 > "$OUT/logs/ae2p_v2k1_suf_p64.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 $EV --ckpt "$(ck ae2l_su_p64)" --name su_p64_v2k1 > "$OUT/logs/ae2p_v2k1_su_p64.log" 2>&1 &
wait
echo "done $(date '+%m-%d %H:%M')"
