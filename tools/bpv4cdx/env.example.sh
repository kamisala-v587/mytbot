# bpv4cdx paths for one machine. Copy to tools/bpv4cdx/env.local.sh, edit, then `source tools/bpv4cdx/env.local.sh`.
# Every script under tools/bpv4cdx/ reads these variables; no path is hard-coded in the code.

export BPV4CDX_REPO=/home/jovyan/workspace/mytbot                  # my_tbot checkout on this machine
export BPV4CDX_DATA_ROOT=/path/to/robotwin2.0                      # contains <task>/aloha-agilex_clean_50, <task>/aloha-agilex_randomized_100
export BPV4CDX_QWEN_PATH=/path/to/models/q35/base_800m             # Qwen3.5-0.8B base (no VLA weights)
export BPV4CDX_CACHE_ROOT=/path/to/cache                           # caches go to $BPV4CDX_CACHE_ROOT/bpv4cdx_<sampler>
export BPV4CDX_OUT_ROOT=$BPV4CDX_REPO/outputs                      # outputs go to $BPV4CDX_OUT_ROOT/BPv4CDX/stage_a/<name>
export BPV4CDX_STATS_ROOT=$BPV4CDX_REPO/norm_stats                 # synced with the code
export BPV4CDX_NUM_GPUS=8

# same runtime settings as my_tbot/Readme.md (B200)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=1
export LEROBOT_TORCHCODEC_CACHE_SIZE=64
