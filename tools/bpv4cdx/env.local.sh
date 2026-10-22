# bpv4cdx paths of this machine (default training machine, 4 GPUs). `source tools/bpv4cdx/env.local.sh`
# Full caches go to $BPV4CDX_CACHE_ROOT/bpv4cdx_<sampler>; the older partial dev caches (bpv4cdx_dev_*) are not used.

export PATH=/vla/workspace/shiliangchao/env/q35va/bin:$PATH
export PYTHONPATH=/vla/workspace/my_tbot/src${PYTHONPATH:+:$PYTHONPATH}

export BPV4CDX_REPO=/vla/workspace/my_tbot
export BPV4CDX_DATA_ROOT=/vla/workspace/data/robotwin2.0
export BPV4CDX_QWEN_PATH=/vla/workspace/models/q35/base_800m
export BPV4CDX_STATS_ROOT=/vla/workspace/my_tbot/norm_stats
export BPV4CDX_CACHE_ROOT=/vla/workspace/cache
export BPV4CDX_CACHE_PREFIX=bpv4cdx_
export BPV4CDX_OUT_ROOT=/vla/workspace/my_tbot/outputs
# 4 GPUs x 16 trajectories x 4 segments = 256 samples per step (same total batch as bpv4_bpen)
export BPV4CDX_NUM_GPUS=4

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=1
export LEROBOT_TORCHCODEC_CACHE_SIZE=16
