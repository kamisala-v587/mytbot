# Server

```bash
cd /vla/workspace/my_tbot
conda activate bptbot
export PYTHONPATH=/vla/workspace/my_tbot/src
```

## LeRobot Policy

```bash
python server/serve_lerobot_policy.py \
  --ckpt_path /path/to/checkpoint \
  --host 0.0.0.0 \
  --port 8000

CKPT=/vla/workspace/models/tbot_ckpts/sftByOfficial/clean-e9/pretrained_model
python server/serve_lerobot_policy_batch.py \
  --ckpt_path $CKPT \
  --max_batch_size 4 \
  --batch_wait_ms 10 \
  --queue_size 64 \
  --host 0.0.0.0 \
  --port 8000
```

## BPVA
```bash
python server/bpva_serve.py \
  --ckpt_path /path/to/bpva/checkpoint \
  --bp_mapping_path .config/bpva_task_bps.yml \
  --stats_path norm_stats/robotwin_delta/stats.json \
  --action_mode delta \
  --host 0.0.0.0 \
  --port 8000
```

## BPVA Debug（记录 BP 主视角 MP4）

```bash
python server/bpva_serve_debug.py \
  --ckpt_path /path/to/bpva/checkpoint \
  --bp_mapping_path .config/bpva_task_bps.yml \
  --stats_path norm_stats/robotwin_delta/stats.json \
  --action_mode delta \
  --bplogs_dir server/bplogs \
  --host 0.0.0.0 \
  --port 8000
```

## 消融实验
```bash
BPVA_CKPT=/vla/workspace/my_tbot/outputs/ckpts/bpva/bpva-clean-robotwin-v1.1/035000/pretrained_model 
python server/bpva_serve_mask_batch.py \
  --ckpt_path $BPVA_CKPT \
  --bp_mapping_path .config/bpva_task_bps.yml \
  --stats_path norm_stats/robotwin_delta/stats.json \
  --action_mode delta \
  --max_batch_size 4 \
  --batch_wait_ms 10 \
  --queue_size 64 \
  --host 0.0.0.0 \
  --port 8000
```

## CUDA 环境压测

用于验证 CUDA/PyTorch 环境、显存分配和矩阵计算负载，会在指定时长后自动释放显存并退出。

```bash
cd /vla/workspace/my_tbot
conda activate bptbot

# 多卡：在 0/1/2/3 号卡上分别占用当前空闲显存的 50%，同时运行计算负载
python server/gpu_cuda_stress_test.py \
  --device 0,1,2,3 \
  --duration_sec 60000 \
  --memory_fraction 0.8 \
  --matrix_size 4096
```


## 实用

```bash
cd /vla/workspace/my_tbot
conda activate bptbot 

LEROBOT_CKPT=/vla/workspace/my_tbot/outputs/ckpts/tbots/tbot-robotwin-sft-clean-v1.1/pretrained_model
python server/serve_lerobot_policy_batch.py \
  --ckpt_path $LEROBOT_CKPT \
  --max_batch_size 4 \
  --batch_wait_ms 10 \
  --queue_size 64 \
  --host 0.0.0.0 \
  --port 8000

BPVA_CKPT=/vla/workspace/my_tbot/outputs/ckpts/bpva/bpva-clean-robotwin-v1.1/035000/pretrained_model 

python server/bpva_serve_debug.py \
  --ckpt_path $BPVA_CKPT \
  --bp_mapping_path .config/bpva_task_bps.yml \
  --stats_path norm_stats/robotwin_delta/stats.json \
  --action_mode delta \
  --bplogs_dir server/bplogs \
  --host 0.0.0.0 \
  --port 8000

# 动态 batch debug server（带行为提示 MP4 日志）
python server/bpva_serve_debug_batch.py \
  --ckpt_path $BPVA_CKPT \
  --bp_mapping_path .config/bpva_task_bps.yml \
  --stats_path norm_stats/robotwin_delta/stats.json \
  --action_mode delta \
  --bplogs_dir server/bplogs \
  --max_batch_size 4 \
  --batch_wait_ms 10 \
  --queue_size 64 \
  --host 0.0.0.0 \
  --port 8000
```