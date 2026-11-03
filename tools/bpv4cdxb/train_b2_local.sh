#!/bin/bash
# B-2 (Action Expert only, 5e-5 constant, stop at the loss / MSE elbow) on this machine's GPUs.
# usage: source tools/bpv4cdxb/env.local.sh; bash tools/bpv4cdxb/train_b2_local.sh [config.jsonc] [draccus overrides ...]
set -u
CFG=${1:-$BPV4CDXB_REPO/configs/Pro6k/bpv4cdxb_plan/b2_main.jsonc}
shift || true
cd "$BPV4CDXB_REPO" || exit 1
export LD_LIBRARY_PATH=/vla/workspace/shiliangchao/env/q35va/lib:${LD_LIBRARY_PATH:-}
NAME=$(basename "$CFG" .jsonc)
STAMP=$(date +%Y-%m-%d_%H-%M)
GEN=$BPV4CDXB_OUT_ROOT/BPv4CDX/stage_b/generated
mkdir -p "$GEN" "$BPV4CDXB_OUT_ROOT/BPv4CDX/stage_b/logs"
OUT_DIR=$(python - "$CFG" "$GEN/$NAME.json" "$STAMP" <<'EOF'
import json, sys
from lerobot.utils.jsonc import load_jsonc
cfg = load_jsonc(sys.argv[1])
cfg["output_dir"] = f"{cfg['output_dir']}/{sys.argv[3]}"
open(sys.argv[2], "w").write(json.dumps(cfg, indent=1))
print(cfg["output_dir"])
EOF
) || exit 1
NPROC=${BPV4CDXB_TRAIN_GPUS:-$BPV4CDXB_NUM_GPUS}
echo "config $GEN/$NAME.json -> $OUT_DIR ($NPROC GPUs)"
accelerate launch --num_processes="$NPROC" --main_process_port "${BPV4CDXB_PORT:-29660}" \
  -m lerobot.scripts.lerobot_train --config_path="$GEN/$NAME.json" "$@" \
  2>&1 | tee "$BPV4CDXB_OUT_ROOT/BPv4CDX/stage_b/logs/train_${NAME}_${STAMP}.log"
