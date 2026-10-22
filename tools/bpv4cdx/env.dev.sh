# Development machine paths (used when evaluating checkpoints copied back from the training machine).
export BPV4CDX_REPO=/vla/workspace/my_tbot
export BPV4CDX_DATA_ROOT=/vla/workspace/data/robotwin2.0
export BPV4CDX_QWEN_PATH=/vla/workspace/models/q35/base_800m
export BPV4CDX_STATS_ROOT=/vla/workspace/my_tbot/norm_stats
export BPV4CDX_CACHE_ROOT=/vla/workspace/cache
export BPV4CDX_CACHE_PREFIX=bpv4cdx_dev_
export BPV4CDX_OUT_ROOT=/vla/workspace/my_tbot/outputs
export BPV4CDX_NUM_GPUS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=1
export LEROBOT_TORCHCODEC_CACHE_SIZE=16
