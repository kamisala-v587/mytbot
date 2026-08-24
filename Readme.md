### 预训练Tbot - B200
cd /home/jovyan/vla/workspace/mytbot
source /home/jovyan/.conda/envs/tbot/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export LEROBOT_PARALLEL_DATASET_LOAD=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
accelerate launch --num_processes=8 \
  -m lerobot.scripts.lerobot_train \
  --config_path=/home/jovyan/vla/workspace/mytbot/configs/pretrain_config.jsonc

cat /home/jovyan/vla/workspace/mytbot/.config/pretrain_config.jsonc


### 微调Tbot - B200
cd /home/jovyan/vla/workspace/mytbot
source /home/jovyan/.conda/envs/tbot/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

accelerate launch --num_processes=8 \
  -m lerobot.scripts.lerobot_train \
  --config_path=/home/jovyan/vla/workspace/mytbot/.config/finetune_config.jsonc

accelerate launch --num_processes=8 -m lerobot.scripts.lerobot_train --config_path=/home/jovyan/vla/workspace/mytbot/outputs/TBot_SA1/2026-06-26/06-19-44_TBot_SFT_robotwin_v0/checkpoints/154000/pretrained_model/train_config.json


### 防中断（nohup 续训，容器重建后 tmux 不可用）  - B200
```
cd /home/jovyan/vla/workspace/mytbot
mkdir -p logs
source /home/jovyan/.conda/envs/tbot/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=1

CKPT="/home/jovyan/vla/workspace/mytbot/outputs/TBot_SA1/pretrain_v1/2026-07-03/01-46-07_ /checkpoints/010000"
accelerate launch --num_processes=8 \
  -m lerobot.scripts.lerobot_train \
  --resume=true \
  --config_path="${CKPT}/pretrained_model/train_config.json"

nohup accelerate launch --num_processes=8 \
  -m lerobot.scripts.lerobot_train \
  --resume=true \
  --config_path="${CKPT}/pretrained_model/train_config.json" \
  > logs/resume_$(date +%F_%H%M).log 2>&1 &

# [1] 297 PID
tail -f /home/jovyan/vla/workspace/mytbot/logs/resume_2026-07-03_1014.log

kill PID # 1816

```
--resume=true会覆盖，config_path设定为断点的ckpt即可
resume=true 时代码自动设pretrained_path为 .../008000/pretrained_model/



### 监控

cd /home/jovyan/vla/workspace/mytbot
bash /home/jovyan/vla/workspace/mytbot/launch/watch_training.sh \
  "outputs/TBot_SA1/pretrain_v1/2026-07-03/01-46-07_ "


### Tbot训练 - H100
```bash
cd /vla/workspace/my_tbot
conda activate mytbot
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=0

CUDA_VISIBLE_DEVICES=0 
accelerate launch --num_processes=4 \
  -m lerobot.scripts.lerobot_train \
  --config_path=/vla/workspace/my_tbot/configs/tbot_sft_h100.jsonc
```

### BPVA 初始化模型
运行 `.code/LetMeSeeSee/初始化模型.ipynb`，输出 `/vla/workspace/models/bpva_init_c10_a50`。

### BPVA smoke 训练
```bash
cd /vla/workspace/my_tbot
conda activate mytbot
export HF_HOME=/vla/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=0

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 \
  -m lerobot.scripts.lerobot_train \
  --config_path=/vla/workspace/my_tbot/configs/bpva_train.jsonc \
  --batch_size=1 \
  --num_workers=0 \
  --steps=1 \
  --eval_freq=0 \
  --save_checkpoint=false
```

### BPVA 正式训练
```bash
cd /vla/workspace/my_tbot
conda activate mytbot
export HF_HOME=/vla/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=0

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 \
  -m lerobot.scripts.lerobot_train \
  --config_path=/vla/workspace/my_tbot/configs/bpva_train.jsonc
```

### BPVA 恢复训练
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 \
  -m lerobot.scripts.lerobot_train \
  --resume=true \
  --config_path=<checkpoint>/pretrained_model/train_config.json
```

#### PRO 6 K
cd /vla/workspace/my_tbot
source /vla/.conda/miniconda3/etc/profile.d/conda.sh
conda activate mytbot

export HF_HOME=/vla/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=0
export PYTHONPATH=/vla/workspace/my_tbot/src:${PYTHONPATH}

export NCCL_CUMEM_ENABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes=2 \
  -m lerobot.scripts.lerobot_train \
  --config_path=/vla/workspace/my_tbot/configs/tbot_sft_h100.jsonc


### P6k bpva的 多卡训练命令
cd /vla/workspace/my_tbot
source /vla/.conda/miniconda3/etc/profile.d/conda.sh
conda activate bptbot

export LEROBOT_PARALLEL_DATASET_LOAD=1
export LEROBOT_DDP_TIMEOUT_SEC=1800
export LEROBOT_LOG_RANK_DEVICE_MAP=1

unset NCCL_DEBUG
unset NCCL_SOCKET_IFNAME
unset NCCL_IB_DISABLE

accelerate launch --num_processes=4 \
  -m lerobot.scripts.lerobot_train \
  --config_path=/vla/workspace/my_tbot/configs/baiduyun/bpva_pretrain.jsonc

**需要配置的环境变量**

+ HF_HOME /vla/.cache/huggingface
+ HF_HUB_OFFLINE 1
+ LEROBOT_PARALLEL_DATASET_LOAD 0
+ PYTHONUNBUFFERED 1
+ TOKENIZERS_PARALLELISM false
+ TRANSFORMERS_OFFLINE 1
+ WANDB_MODE offline
