#!/bin/bash
# 本机 P5 cache 入口（实现复用 build_cache_dist.sh；卡数自动探测）。
#
# 单机 8 卡清剩余:
#   NNODES=1 NODE_RANK=0 bash tools/bpv5/build_cache_local.sh
# 与远端同时（两边卡数需相同）:
#   NNODES=2 NODE_RANK=0 bash tools/bpv5/build_cache_local.sh
set -u
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export PROCS_PER_GPU=${PROCS_PER_GPU:-2}
cd /vla/workspace/my_tbot || exit 1
exec bash tools/bpv5/build_cache_dist.sh
