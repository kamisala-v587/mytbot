### 新任务，跑通TBot
```
cd /vla/my_tbot
conda activate tbot_sa1

```

**nrom stats 计算**
```
python tools/compute_norm_stats_single.py \
  --repo_id /vla/.data/adjust_bottle \
  --action_mode delta \
  --chunk_size 50 \
  --output_dir norm_stats
```

**训练**
```
POLICY_INIT_PATH=zaleni/TBot-SA1-Base \
DATASET_REPO_ID=/vla/.data/adjust_bottle \
ACTION_TYPE=delta \
USE_EXTERNAL_STATS=true \
bash launch/tbot_sa1_finetune.sh
```