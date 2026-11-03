#!/bin/bash
# Wait for the q35 baseline training (q35_bp_baseline.jsonc) to exit, then evaluate 50k / 80k / 100k exactly like the
# B-4 BP checkpoints and write the BP vs q35 comparison:
#   1. new scenes, all 38 training tasks x 10 (q35ood_*)  ||  held-out sets 19 x 10 + held-out tasks 2 x 20 (q35held_*)
#   2. training scenes, 16 quick-sweep tasks x 10 (q35train_*)  ||  100k with per-step traces, 10 tasks x 10 (q35trace_c100k)
#   3. failure modes, scene-novelty gaps (BP and q35 100k), eval_result_bp/_q35_vs_bp_summary.md
# usage: setsid nohup bash tools/bpv4cdxb/q35_eval_after_train.sh > <log> 2>&1 &
set -u
REPO=/vla/workspace/my_tbot
RT=/vla/workspace/my_robotwin
RT_PY=/vla/.conda/miniconda3/envs/myrobotwin/bin/python
cd "$REPO" || exit 1
source tools/bpv4cdxb/env.local.sh >/dev/null 2>&1
RUNQ=$REPO/outputs/BPv4CDX/stage_b/q35_bp_baseline/2026-10-31_22-24/2026-10-31_22-25/checkpoints
TRAIN_PATTERN="generated/q35_bp_baseline.json"

echo "$(date '+%F %T') waiting for training ($TRAIN_PATTERN) to exit"
while pgrep -f "$TRAIN_PATTERN" > /dev/null; do sleep 120; done
echo "$(date '+%F %T') training exited; checkpoints: $(ls "$RUNQ" | tr '\n' ' ')"
sleep 90
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

SPECS=()
for s in 50000 80000 100000; do
  ck=$(printf "%s/%06d/pretrained_model" "$RUNQ" $s)
  [ -f "$ck/model.safetensors" ] && SPECS+=("c$((s / 1000))k:$ck") || echo "missing $ck"
done
ALL=$(awk -F/ '{print $(NF-1)}' $RT/bp_tasks/splits/v1/train_repos.txt | sort -u | paste -sd,)
HELD=$(python - <<'PY'
import yaml
s = yaml.safe_load(open("/vla/workspace/my_robotwin/bp_tasks/splits/v1.yaml"))
print(",".join(sorted(s["heldout_sets"]) + list(s["heldout_tasks"])))
PY
)
TRACE_TASKS=adjust_bottle_bp,place_object_scale_bp,move_stapler_pad_bp,place_a2b_bp,place_object_stand_bp,blocks_ranking_size_bp,place_mouse_pad_bp,stack_blocks_three_bp,click_bell_alarm_bp,place_phone_stand_bp

echo "$(date '+%F %T') phase 1: new scenes + held-out: ${SPECS[*]%%:*}"
SWEEP_SCENES=bank SWEEP_TASKS=$ALL SWEEP_EPISODES=10 SWEEP_TASKS_PER_GPU=4 SWEEP_PORT0=8031 \
  bash tools/bpv4cdxb/sim_sweep_bp.sh q35ood "${SPECS[@]}" &
p1=$!
sleep 30
SWEEP_SCENES=bank SWEEP_TASKS=$HELD SWEEP_GROUPS="heldout_sets, heldout_tasks" SWEEP_EPISODES=0 \
  SWEEP_HELDOUT_SETS=10 SWEEP_HELDOUT_TASKS=20 SWEEP_TASKS_PER_GPU=3 SWEEP_PORT0=8041 \
  bash tools/bpv4cdxb/sim_sweep_bp.sh q35held "${SPECS[@]}" &
p2=$!
wait $p1 $p2

echo "$(date '+%F %T') phase 2: training scenes + traces"
SWEEP_SCENES=train SWEEP_EPISODES=10 SWEEP_TASKS_PER_GPU=3 SWEEP_PORT0=8051 \
  bash tools/bpv4cdxb/sim_sweep_bp.sh q35train "${SPECS[@]}" &
p3=$!
sleep 30
C100=$(printf "%s/%06d/pretrained_model" "$RUNQ" 100000)
SWEEP_SCENES=bank SWEEP_TASKS=$TRACE_TASKS SWEEP_EPISODES=10 SWEEP_TASKS_PER_GPU=4 SWEEP_TRACE=1 SWEEP_PORT0=8061 \
  bash tools/bpv4cdxb/sim_sweep_bp.sh q35trace "c100k:$C100" &
p4=$!
wait $p3 $p4

echo "$(date '+%F %T') phase 3: analysis"
cd "$RT" || exit 1
$RT_PY -m bp_tasks.eval_policy.failure_modes eval_result_bp/q35trace_c100k > /dev/null
$RT_PY -m bp_tasks.eval_policy.demo_scene_gap eval_result_bp/ood_c100k eval_result_bp/held_c100k \
  --out eval_result_bp/_gap_bp_c100k.jsonl | tail -1
$RT_PY -m bp_tasks.eval_policy.demo_scene_gap eval_result_bp/q35ood_c100k eval_result_bp/q35held_c100k \
  --out eval_result_bp/_gap_q35_c100k.jsonl | tail -1
cd "$REPO" || exit 1
python tools/bpv4cdxb/q35_vs_bp_report.py
echo "$(date '+%F %T') all done"
