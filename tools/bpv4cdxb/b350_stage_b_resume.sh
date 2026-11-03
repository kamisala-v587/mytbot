#!/bin/bash
# Resume stage B (training scenes, 20k/50k/80k/100k) once the given process (stage A held-out sweep) has exited.
# Finished episodes are kept (run_eval resumes per episode uid).
set -u
WAIT_PID=${1:?pid}
cd /vla/workspace/my_tbot || exit 1
source tools/bpv4cdxb/env.local.sh >/dev/null 2>&1
RUN=/vla/workspace/my_tbot/outputs/BPv4CDX/stage_b/b4_350k_sft/2026-11-02_12-04/2026-11-02_12-05
echo "$(date '+%F %T') waiting for pid $WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
echo "$(date '+%F %T') stage B (resume): training scenes"
SPECS=()
for s in 20000 50000 80000 100000; do SPECS+=("c$((s / 1000))k:$(printf "%s/checkpoints/%06d/pretrained_model" "$RUN" $s)"); done
SWEEP_SCENES=train SWEEP_EPISODES=10 SWEEP_TASKS_PER_GPU=3 SWEEP_PORT0=8051 bash tools/bpv4cdxb/sim_sweep_bp.sh b350train "${SPECS[@]}"
echo "$(date '+%F %T') stage B done"
