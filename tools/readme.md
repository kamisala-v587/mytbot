# 首次全量
python tools/run_norm_stats.py init

# 增量更新（pretrain_data.txt 追加后）
python tools/run_norm_stats.py update

# 强制重算所有 per-dataset 缓存
python tools/run_norm_stats.py init --force

# 预览计划
python tools/run_norm_stats.py init --dry-run --limit 10

# 并行度（环境变量或参数均可）
NUM_WORKERS=8 python tools/run_norm_stats.py init
python tools/run_norm_stats.py init --num-workers 8
