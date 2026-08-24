# Tools

本目录的 norm stats 只提供一个 Python 入口：`tools/run_norm_stats.py`。实现位于
`tools/norm_stats/` 子包，不应直接作为命令调用。

## 快速开始

repo 列表每行一个 repo id 或本地数据集路径；空行会跳过。相对 repo id 默认在
`HF_LEROBOT_HOME` 下解析，也可通过 `--root` 指定共同根目录。解析后的 canonical 路径
不得重复。

### TBot 训练目录（默认）

```bash
python tools/run_norm_stats.py \
  --repo-id-file .config/ds_ids/pretrain_data.txt \
  --output-format tbot \
  --output-root /path/to/norm_stats \
  --action-mode delta \
  --chunk-size 50 \
  --num-workers 16
```
python tools/run_norm_stats.py \
  --repo-id-file /vla/workspace/my_tbot/configs/ds_ids/Baidunyun/pretrain_data_ids_baiduyun.txt \
  --output-root /vla/workspace/my_tbot/norm_stats \
  --action-mode delta \
  --chunk-size 50 \
  --num-workers 16

python tools/run_norm_stats.py \
  --repo-id-file /vla/workspace/my_tbot/configs/ds_ids/Baidunyun/pretrain_data_ids_baiduyun.txt \
  --output-root /vla/workspace/my_tbot/norm_stats \
  --action-mode delta \
  --chunk-size 50 \
  --num-workers 48
它按 `infer_embodiment_variant` 的 `resolved_robot_type` 分组，写入：

```text
/path/to/norm_stats/<resolved_robot_type>/<action_mode>/stats.json
/path/to/norm_stats/<resolved_robot_type>/<action_mode>/manifest.json
```

此布局与训练配置直接匹配：

```json
{
  "use_external_stats": true,
  "external_stats_root": "/path/to/norm_stats",
  "action_mode": "delta"
}
```

### 合并成一个 stats.json

```bash
python tools/run_norm_stats.py \
  --repo-id-file/vla/workspace/my_tbot/.config/ds_ids/robotwin_h100_ids.txt \
  --output-format default \
  --output-path /vla/workspace/my_tbot/norm_stats/robotwin_delta/stats.json --num-workers 16
```

`default` 不按机械臂分组，只写 `--output-path` 指定的单个 `stats.json`，不会在旁边
生成 manifest。聚合前会强制校验全部非视觉 feature key、shape、feature mapping 和 mask
一致；不一致即报错。`--output-path` 在此模式下必填。

## 计算语义

- `--action-mode` 默认 `delta`。
- 非 action feature 始终统计 episode 的全部帧，包括短 episode。
- `abs` action 始终统计全部帧，不受 `chunk-size` 或采样上限影响。
- `delta` action 只统计长度至少为 `chunk-size` 的有效滑动窗口。
- `--max-chunks-per-episode`、`--max-chunks-per-repo` 只抽样 delta action 窗口，
  `--sample-seed` 保证可复现。
- `--skip-action-robot-types` 可跳过指定原始或 resolved robot type 的 action；被跳过的
  action 不写虚构 count，聚合时仅合并真实存在的 action 样本。
- 内部以 float64 累积 count/mean/mean_sq/min/max，跨 repo 按每个 feature 自身 count
  加权，最后计算 std；输出 JSON 的 count 保持 `[N]`。
- 外部 `stats.json` 完全排除 video/image key；视觉 shape 和数值不参与跨 repo 校验。图像统计由
  各数据集自身 `meta.stats` 保留，或由训练时启用的 ImageNet stats 配置覆盖。

## 增量缓存

默认缓存目录为 `norm_stats_cache`，新结构为：

```text
<cache-root>/datasets/<dataset_id>/metadata.json
<cache-root>/datasets/<dataset_id>/stats_payload.json
```

快速指纹跟踪 `meta/info.json`、`meta/episodes*`、`data/**/*.parquet` 的相对路径、
文件大小和 `mtime_ns`，并包含 info 内容 SHA256、算法/缓存版本以及全部影响统计的参数。
因此文件增删改或有效计算配置变化都会产生新的 dataset id。`abs` 始终全帧计算，故其
指纹会忽略无效的 chunk size、采样上限和 seed；`delta` 继续纳入这些参数。项目不依赖 xxhash。

写缓存与最终输出都采用同目录临时文件后原子替换；缓存 metadata 最后写入并校验
payload SHA256，因此两文件半写或错配会安全视为 miss。程序也会扫描旧缓存目录，读取旧
`{fingerprint,result}` schema，并依据 payload 内的 action/config/data fingerprint 校验，
而不是信任旧目录名；命中后自动迁移到新结构。

## 规划与进度

```bash
python tools/run_norm_stats.py --repo-id-file repos.txt --dry-run
```

`dry-run` 只展示 hit/miss 和 miss 的预计 frame 工作量，不计算且不写任何文件。正常运行
按 repo 使用 spawn 多进程并实时接收完成项；每完成一仓立即原子写缓存，并打印完成进度、
耗时和基于 miss `info.total_frames` 吞吐估算的 ETA。

## 参数

所有参数都支持下划线旧拼法（例如 `--repo_id_file`），推荐连字符形式：

- `--repo-id-file`（必填）、`--cache-root`、`--root`
- `--output-format {tbot,default}`、`--output-root`、`--output-path`
- `--action-mode {delta,abs}`、`--chunk-size`、`--num-workers`
- `--max-chunks-per-episode`、`--max-chunks-per-repo`、`--sample-seed`
- `--skip-action-robot-types [TYPE ...]`、`--dry-run`

## 轻量验证

```bash
python -m compileall -q tools/run_norm_stats.py tools/norm_stats
pytest -q tools/norm_stats/tests
```

## 其他工具

- `check_lerobot_v3_integrity.py`：LeRobot v3 数据集基础完整性检查。
- `check_pretrain_dataset_pipeline.py`：pretrain 数据管线检查。
- `convert_egodex_to_lerobot.py`：EgoDex 转 LeRobot v3。
