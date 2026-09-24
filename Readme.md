### B200环境启动
cd /home/jovyan/workspace/mytbot
conda activate bptbot

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=1
export LEROBOT_TORCHCODEC_CACHE_SIZE=64

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

### pro6k 环境启动
cd /vla/workspace/my_tbot
conda activate bptbot
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=1


### Tbot 训练脚本
accelerate launch --num_processes=8 -m lerobot.scripts.lerobot_train --config_path=/home/jovyan/workspace/mytbot/configs/B200/pretrain_config.jsonc


### BPVA V2 训练脚本
accelerate launch --num_processes=8   -m lerobot.scripts.lerobot_train   --config_path=/home/jovyan/workspace/mytbot/configs/B200/bpvav2_sft_v0.3.jsonc

### Benckmark
accelerate launch --num_processes=8 -m tools.bpva_benchmark.train_benchmark \
  --config-path=/vla/workspace/my_tbot/configs/Pro6k/bpv2.jsonc \
  --warmup-steps=20 \
  --measure-steps=500 \
  --num-workers=12 \
  --output-dir=outputs/bpva_benchmark


accelerate launch --num_processes=4 -m lerobot.scripts.lerobot_train --config_path=

### Q35 环境启动（从 wsa35 复制，transformers 5.14.1，含 qwen3_5）
cd /vla/workspace/my_tbot
conda activate q35va
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=1
export LEROBOT_TORCHCODEC_CACHE_SIZE=32

### Q35 训练脚本
accelerate launch --num_processes=2 -m lerobot.scripts.lerobot_train --config_path=/vla/workspace/my_tbot/configs/Pro6k/q35.jsonc

### bpvav3
accelerate launch --num_processes=8 -m lerobot.scripts.lerobot_train --config_path=/vla/workspace/my_tbot/configs/Pro6k/bpvav3_plus.jsonc