# BP cache 生成与训练使用

BP cache 是从原始 LeRobot v3 数据集抽样出的本地、完整 LeRobot v3 prompt 数据集。默认 `SAMPLING_SCOPE="dataset"`、`SAMPLE_RATIO=0.1`：例如 50 个 episode 全局选择 `max(1, ceil(50*0.1))=5` 个。训练只读取映射 YAML 指向的 cache，不依赖源数据集目录中的 `.cache` 或其他临时产物。

## 命令

脚本入口：

```bash
python tools/run_bp_cache.py \
  --repo-id-file configs/ds_ids/B200/RoboTwin-LeRobot-v3.0.txt \
  --bp-cache-root /absolute/path/to/bp_cache \
  --mapping-output /home/jovyan/workspace/mytbot/configs/B200/bp_cache_roboTwin.yaml \
  --sample-ratio 0.1 \
  --sampling-scope dataset \
  --sampling-mode random
```

等价的模块入口（从仓库根目录运行）：

```bash
python -m tools.bp_cache.generate \
  --repo-id-file configs/ds_ids/B200/RoboTwin-LeRobot-v3.0.txt \
  --bp-cache-root /absolute/path/to/bp_cache \
  --mapping-output /home/jovyan/workspace/mytbot/configs/B200/bp_cache_roboTwin.yaml \
  --sample-ratio 0.1 \
  --sampling-scope dataset \
  --sampling-mode random
```

两个入口共用 `tools/run_bp_cache.py` 中的 `main` 和参数配置。`tools/run_bp_cache.py` 顶部提供全部默认常量，可直接编辑；命令行参数优先。关键常量包括 repo 列表、cache 根目录、mapping 输出、抽样比例和模式、随机种子、canonical camera keys，以及视频文件大小阈值。

## 抽样

默认 dataset 抽样以整个 repo 的完整 episode 为范围，选择 `max(1, ceil(total * ratio))` 个；`random` 使用 seed 与 source identity 得到稳定结果，`first` 选择全局前 N 个。RoboTwin 一个 repo/dataset 代表一个任务，episode 中不同 task 文本只是描述变化，不参与 cache 覆盖要求；manifest 仍记录每个 selected episode 的原始 task 描述。可用 `--sampling-scope task` 保留旧的逐 task 抽样兼容行为。

## repo 列表与 identity

repo 列表每行一个 Hub ID 或路径。Hub ID 保持原样；绝对路径以及 `~`、`./`、`../` 开头的明确本地路径会 expanduser 并 resolve。列表在规范化后不允许重复，manifest 和 mapping key 使用同一规范 identity。当前本地 repo 列表推荐写绝对路径。

## mapping YAML

YAML 是 `source identity -> cache 绝对目录`：

```yaml
org/dataset: /data/bp_cache/dataset-abcd1234
/home/user/data/local_dataset: /data/bp_cache/local_dataset-1234abcd
```

key/value 必须是非空字符串，cache 路径必须为绝对路径，不允许重复 key、规范化后重复 source 或重复 cache root。解析使用安全 loader，并在任意 mapping 层级拒绝重复 key。

## manifest

每个 cache 根目录含 `bp_cache_manifest.json`。稳定契约包括：

- `schema_version: 1`
- `algorithm_version: "bp-cache-v2"`
- `config.sampling_scope: "dataset"`（或兼容的 `"task"`）
- `complete: true`（严格布尔值）
- 规范化后的 `source_repo`
- 非空 `cache_repo_id`
- 抽样 episode/task、source→cache episode/task 映射
- `actual_camera_keys`、fingerprint、配置、视频与 fallback 记录

训练会验证 manifest、v3 metadata、fps、robot type、state/action schema、camera 实际映射、非空 parquet/video；dataset scope 不要求 selected_by_task 覆盖当前数据集的所有 task name，task 文本不同也不会导致 cache prompt 失败。手工修改 manifest 或 mapping 通常会使训练快速失败。

## 动态 chunk

cache 只缩小同 task 的候选轨迹集，不预先固定训练 prompt 关键帧。训练时仍根据轨迹长度动态选择最多 `bp_num_chunks` 个关键帧，并读取每个关键帧后续 action chunk。dataset 与 policy 的 `bp_num_chunks`、`max_state_dim`、`max_action_dim` 必须一致。

cache prompt 使用自身 fps 和 schema 构造 delta timestamps。普通数据集加载完整 parquet 列；EgoDex 视觉 prompt 仅加载必要索引/task 列，state/action 由后续占位逻辑生成。本地 cache 固定 `allow_hub_download=False`。

## 视频大小阈值

默认值为：目标文件 `5 MB`、单 episode soft warning `10 MB`、单 episode hard warning `20 MB`。须满足 `0 < target <= soft <= hard`。单 episode 无法进一步跨文件拆分时会记录 warning/fallback；具体编码与文件记录写入 manifest。

## 恢复与覆盖

- `--resume`：仅当已存在 cache 的 manifest fingerprint 完全命中时复用。
- `--overwrite`：只覆盖由本工具生成且 source identity 匹配的目录。旧 `bp-cache-v1` cache 必须 overwrite 重建：`--overwrite --sampling-scope dataset --sample-ratio 0.1`。
- 两者不能同时使用。

覆盖不会先删除 final。building 完成并验证后，旧 final 原子 rename 到唯一 backup，再将 building 原子 rename 为 final；发布失败会恢复 backup，成功后才删除 backup并更新 mapping。构建失败或发布失败时，旧 final 与旧 mapping 保留。遗留 `.building` 会在下一次该 repo 构建前清理。

## 退出码

- `0`：全部成功（或 dry-run 成功）
- `1`：命令行/配置/启动级 fatal 错误
- `2`：至少一个 repo 构建失败；其他 repo 仍继续处理
- `130`：KeyboardInterrupt

每次非 dry-run 执行会在 `<bp-cache-root>/_bp_cache_runs/` 写 effective config、逐 repo result/failure 和 summary。

## 训练配置

BPVAv2 cache 模式：

```jsonc
"bp_prompt_source": "cache",
"bp_cache_root_file": "/absolute/path/to/mapping.yaml",
"bp_same_episode_policy": "allow"
```

cache 与 source episode 编号空间不同，推荐 `allow`。`avoid` 会退化为 allow 并 warning；`neighbor` 会退化为同 task 随机并 warning；`forbid` 会在配置阶段拒绝。
