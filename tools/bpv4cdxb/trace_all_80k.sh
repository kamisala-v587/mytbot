#!/bin/bash
# Traced failure-attribution rollouts (10 tasks x 10, new scenes, client trace) for the report:
#   BP 350k 80k + 100k (b350trace_c*), BP 10k 80k (trace_c80k), q35 80k (q35trace_c80k); then failure_modes on each.
# usage: setsid nohup bash tools/bpv4cdxb/trace_all_80k.sh > <log> 2>&1 &
set -u
REPO=/vla/workspace/my_tbot
RT=/vla/workspace/my_robotwin
cd "$REPO" || exit 1
source tools/bpv4cdxb/env.local.sh >/dev/null 2>&1
ST=$REPO/outputs/BPv4CDX/stage_b
B350=$ST/b4_350k_sft/2026-11-02_12-04/2026-11-02_12-05/checkpoints
B10=$ST/b4_sft_full2/2026-10-30_13-45/2026-10-30_13-47/checkpoints
Q35=$ST/q35_bp_baseline/2026-10-31_22-24/2026-10-31_22-25/checkpoints
T=adjust_bottle_bp,place_object_scale_bp,move_stapler_pad_bp,place_a2b_bp,place_object_stand_bp,blocks_ranking_size_bp,place_mouse_pad_bp,stack_blocks_three_bp,click_bell_alarm_bp,place_phone_stand_bp
common() { SWEEP_SCENES=bank SWEEP_TASKS=$T SWEEP_EPISODES=10 SWEEP_TASKS_PER_GPU=3 SWEEP_TRACE=1 "$@"; }
echo "$(date '+%F %T') start"
common env SWEEP_PORT0=8031 bash tools/bpv4cdxb/sim_sweep_bp.sh b350trace "c80k:$B350/080000/pretrained_model" "c100k:$B350/100000/pretrained_model" &
sleep 20
common env SWEEP_PORT0=8041 bash tools/bpv4cdxb/sim_sweep_bp.sh trace "c80k:$B10/030000/pretrained_model" &
sleep 20
common env SWEEP_PORT0=8051 bash tools/bpv4cdxb/sim_sweep_bp.sh q35trace "c80k:$Q35/080000/pretrained_model" &
wait
cd "$RT" || exit 1
for d in b350trace_c80k b350trace_c100k trace_c80k q35trace_c80k; do
  /vla/.conda/miniconda3/envs/myrobotwin/bin/python -m bp_tasks.eval_policy.failure_modes eval_result_bp/$d > /dev/null && echo "failure modes: $d"
done
echo "$(date '+%F %T') all done"
