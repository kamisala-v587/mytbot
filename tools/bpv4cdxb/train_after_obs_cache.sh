#!/bin/bash
# Wait for build_obs_cache.sh, check that every episode of the index has its file, then start the given training.
# usage: setsid nohup bash tools/bpv4cdxb/train_after_obs_cache.sh <config.jsonc> [index] [cache dir] > <log> 2>&1 &
set -u
CFG=${1:?config}
INDEX=${2:-/vla/workspace/my_tbot/outputs/BPv4CDX/stage_b/b4_sft_bp_v1_index.json}
CACHE=${3:-/vla/workspace/cache/bpv4cdxb_obs_vit}
cd /vla/workspace/my_tbot || exit 1
source tools/bpv4cdxb/env.local.sh >/dev/null 2>&1
echo "$(date '+%F %T') waiting for build_obs_cache"
while pgrep -f "lerobot.policies.BPV4CDXB.build_obs_cache" > /dev/null; do sleep 60; done
want=$(python -c "import json; print(len(json.load(open('$INDEX'))['episodes']))")
have=$(find "$CACHE" -name 'ep*.npy' ! -name '*.tmp*' | wc -l)
tmp=$(find "$CACHE" -name '*.tmp*' | wc -l)
echo "$(date '+%F %T') cache files $have / $want (leftover tmp $tmp)"
if [ "$have" -ne "$want" ]; then
  echo "cache incomplete: training NOT started; rerun tools/bpv4cdxb/build_obs_cache.sh (resumable)"; exit 1
fi
sleep 30
echo "$(date '+%F %T') starting training $CFG"
bash tools/bpv4cdxb/train_b2_local.sh "$CFG"
