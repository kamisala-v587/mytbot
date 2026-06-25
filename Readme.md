### 预训练Tbot
cd /home/jovyan/vla/workspace/mytbot
source /home/jovyan/.conda/envs/tbot/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --num_processes=4 \
  -m lerobot.scripts.lerobot_train \
  --config_path=/home/jovyan/vla/workspace/mytbot/.配置/pretrain_config_official.jsonc
