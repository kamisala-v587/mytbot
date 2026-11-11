#!/bin/bash
# Training scenes (16 x 10) + new scenes (38 x 10) for BP 350k at 0k / 35k and q35 at 0k / 30k / 40k (q35 has no 35k).
# 0k = start of post-training: BP 350k -> B4/350k_lora_merged; q35 -> pt-15w served as type q35 (pt-15w_as_q35).
# Two phases (all ten at once ran out of GPU memory): new scenes, then training scenes. Finished episodes are kept.
set -u
cd /vla/workspace/my_tbot || exit 1
source tools/bpv4cdxb/env.local.sh >/dev/null 2>&1
ST=/vla/workspace/my_tbot/outputs/BPv4CDX/stage_b
B350=$ST/b4_350k_sft/2026-11-02_12-04/2026-11-02_12-05/checkpoints
Q35=$ST/q35_bp_baseline/2026-10-31_22-24/2026-10-31_22-25/checkpoints
BP_SPECS=("c0k:/vla/workspace/models/BPVA/BPv4CDX/B4/350k_lora_merged" "c35k:$B350/035000/pretrained_model")
Q_SPECS=("c0k:/vla/workspace/models/q35/pt-15w_as_q35" "c30k:$Q35/030000/pretrained_model" "c40k:$Q35/040000/pretrained_model")
ALL=$(awk -F/ '{print $(NF-1)}' /vla/workspace/my_robotwin/bp_tasks/splits/v1/train_repos.txt | sort -u | paste -sd,)
echo "$(date '+%F %T') phase 1: new scenes"
SWEEP_SCENES=bank SWEEP_TASKS=$ALL SWEEP_EPISODES=10 SWEEP_TASKS_PER_GPU=3 SWEEP_PORT0=8031 \
  bash tools/bpv4cdxb/sim_sweep_bp.sh b350ood "${BP_SPECS[@]}" &
sleep 20
SWEEP_SCENES=bank SWEEP_TASKS=$ALL SWEEP_EPISODES=10 SWEEP_TASKS_PER_GPU=3 SWEEP_PORT0=8041 \
  bash tools/bpv4cdxb/sim_sweep_bp.sh q35ood "${Q_SPECS[@]}" &
wait
echo "$(date '+%F %T') phase 2: training scenes"
SWEEP_SCENES=train SWEEP_EPISODES=10 SWEEP_TASKS_PER_GPU=3 SWEEP_PORT0=8051 \
  bash tools/bpv4cdxb/sim_sweep_bp.sh b350train "${BP_SPECS[@]}" &
sleep 20
SWEEP_SCENES=train SWEEP_EPISODES=10 SWEEP_TASKS_PER_GPU=3 SWEEP_PORT0=8061 \
  bash tools/bpv4cdxb/sim_sweep_bp.sh q35train "${Q_SPECS[@]}" &
wait
echo "$(date '+%F %T') all done"
