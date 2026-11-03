#!/bin/bash
# BP (B-4 350k start) evaluation, stage B: after the training exits, training scenes (16 tasks x 10) for 20k / 50k /
# 80k / 100k (eval_result_bp/b350train_c<k>k) and G9 every 10k. Stage A (new scenes + held-out) is started separately
# with the 3 checkpoints chosen from these results: tools/bpv4cdxb/b350_stage_a.sh c50k c80k c100k
# usage: setsid nohup bash tools/bpv4cdxb/b350_stage_b.sh > <log> 2>&1 &
set -u
REPO=/vla/workspace/my_tbot
cd "$REPO" || exit 1
source tools/bpv4cdxb/env.local.sh >/dev/null 2>&1
RUN=$REPO/outputs/BPv4CDX/stage_b/b4_350k_sft/2026-11-02_12-04/2026-11-02_12-05
# matches the first run and the resumed one (its config path is <run>/checkpoints/010000/.../train_config.json)
echo "$(date '+%F %T') checkpoints: $(ls "$RUN/checkpoints" | tr '\n' ' ')"
# stage C (G9, every 10k) next to the training-scene sweep
bash tools/bpv4cdxb/run_g9_bp.sh "$RUN" "10000 20000 30000 40000 50000 60000 70000 80000 90000 100000" "0 1 2 3" \
  > /tmp/g9_350k_all.log 2>&1 &
SPECS=()
for s in 20000 50000 80000 100000; do
  ck=$(printf "%s/checkpoints/%06d/pretrained_model" "$RUN" $s)
  [ -f "$ck/model.safetensors" ] && SPECS+=("c$((s / 1000))k:$ck") || echo "missing $ck"
done
echo "$(date '+%F %T') stage B: training scenes ${SPECS[*]%%:*}"
SWEEP_SCENES=train SWEEP_EPISODES=10 SWEEP_TASKS_PER_GPU=3 SWEEP_PORT0=8051 \
  bash tools/bpv4cdxb/sim_sweep_bp.sh b350train "${SPECS[@]}"
wait
echo "$(date '+%F %T') stage B done"
