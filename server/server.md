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
  --max_batch_size 4 \
  --batch_wait_ms 10 \
  --queue_size 64 \
  --host 0.0.0.0 \
  --port 8000
```

### CKPT
**q35**
CKPT=/vla/workspace/my_tbot/outputs/Q35/pretrain/2026-09-22/17-33-51_q35-pretrain/checkpoints/020000/pretrained_model
