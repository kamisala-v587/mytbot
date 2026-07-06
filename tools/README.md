# Tools

本目录包含预训练数据检查、格式转换与 norm stats 工作流脚本。

## 数据集工具

| 脚本 | 作用 |
|------|------|
| `check_lerobot_v3_integrity.py` | LeRobot v3 数据集基础完整性检查 |
| `check_pretrain_dataset_pipeline.py` | pretrain 数据管线检查 |
| `convert_egodex_to_lerobot.py` | EgoDex 格式 → LeRobot v3 |

## Norm stats — 官方全量脚本（勿改）

| 脚本 | 作用 |
|------|------|
| `compute_norm_stats_single.py` | 单个 dataset 的 state/action 统计 |
| `compute_norm_stats_multi.py` | 同 robot_type 多 dataset 一次性全量扫描聚合 |

## Norm stats — 增量工作流（推荐）

策略文件：`.config/norm_stats_policy.json`

| 脚本 | 作用 |
|------|------|
| `run_norm_stats.py` | CLI 入口：`init` / `update` |
| `norm_stats_pipeline.py` | Step1 per-dataset 缓存 + Step2 robot_type 聚合 |
| `norm_stats_lib.py` | policy、manifest、merge、进度条等共享逻辑 |

**两阶段流程**

1. **Step1**：每个 dataset 写 `norm_stats_cache/datasets/{sha16}/stats.json`
2. **Step2**：按 robot_type merge（或 import 外部 stats）→ `norm_stats/{robot_type}/delta/stats.json`

**policy 语义**

- `placeholder_robot_types`（如 `egodex_v`）：无真实 action，写 metadata 占位 stats
- `import_group_stats_*`（如 `aloha`）：跳过 Step1，Step2 直接复制外部 group stats
- `FORCE=1` / `--force`：仅强制重算**单个 dataset** 的 per-dataset 缓存，不影响 import 类型

**常用命令**

```bash
# 首次全量（默认 8 并行 worker，可用 NUM_WORKERS 或 --num-workers 覆盖）
python tools/run_norm_stats.py init

# pretrain_data.txt 追加路径后增量更新
python tools/run_norm_stats.py update

# 强制重算所有 per-dataset 缓存
python tools/run_norm_stats.py init --force

# 预览计划，不写入
python tools/run_norm_stats.py init --dry-run --limit 10
```

**训练配置**

```json
"use_external_stats": true,
"external_stats_root": "/home/jovyan/vla/workspace/mytbot/norm_stats",
"action_mode": "delta"
```

## 其他

| 脚本 | 作用 |
|------|------|
| `launch_pretrain.sh` | 启动 pretrain 训练 |
