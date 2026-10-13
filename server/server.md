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

