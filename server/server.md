###  环境启动
**通用**
cd /vla/workspace/my_tbot
conda activate bptbot

**q35**
conda activate q35va
cd /vla/workspace/my_tbot

### 通用启动
```bash
python server/serve_policy.py

python server/serve_policy.py \
  --ckpt_path $CKPT \
  --policy auto \
  --device cuda:0 \
  --max_batch_size 4 \
  --batch_wait_ms 10 \
  --queue_size 64 \
  --host 0.0.0.0 \
  --port 8000
  
python server/serve_policy.py \
  --ckpt_path $CKPT \
  --policy auto \
  --device cuda:1 \
  --max_batch_size 4 \
  --batch_wait_ms 5 \
  --queue_size 64 \
  --host 0.0.0.0 \
  --port 8001

```
可追加：--bp_mapping_path /vla/workspace/my_tbot/server/policy_runtime/bpva_task_bps_clean.yml

### CKPT
**q35**
CKPT=/vla/workspace/my_tbot/outputs/Q35/pretrain/2026-09-22/17-33-51_q35-pretrain/checkpoints/020000/pretrained_model

CKPT=/vla/workspace/my_tbot/outputs/Q35/pretrain/2026-09-22/17-33-51_q35-pretrain/checkpoints/010000/pretrained_model

**bpvav3**
CKPT=/vla/workspace/my_tbot/outputs/BPv3/q35/bpv3-clean-stage2/2026-10-09_18-32/checkpoints/007000/pretrained_model

### 服务器占用 48小时
```
cd /home/jovyan/workspace/mytbot/server && mkdir -p logs && setsid nohup /home/jovyan/conda-envs/q35va/bin/python gpu_cuda_stress_test.py --device 0,1,2,3,4,5,6,7 --duration_sec 172800 --util_range 63 77 --memory_range_gib 41 54 --retarget_interval_sec 60 --matrix_size 8192 --log_interval_sec 300 > logs/gpu_stress_$(date +%Y%m%d_%H%M).log 2>&1 < /dev/null &
```

pkill -TERM -f 'gpu_cuda_stress_test.py --device'