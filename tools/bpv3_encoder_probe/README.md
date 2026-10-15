# BPV3 Encoder Probe

这个工具固定 clean 数据、正确任务 BP、episode manifest 和 BP 预处理，用于比较多个 BPV3 checkpoint 的 encoder 表征。它不会训练模型，也不会修改 checkpoint。

## 运行

```bash
cd /vla/workspace/my_tbot
 bash tools/bpv3_encoder_probe/run.sh --dry-run
 bash tools/bpv3_encoder_probe/run.sh
```

首次运行会生成固定样本清单。之后所有 checkpoint 复用同一个 `manifest.jsonl`。结果写入 `tools/bpv3_encoder_probe/results/<name>/`：

- `manifest.jsonl`：固定的 clean prompt episode 来源。
- `shards/<ckpt>.pt`：每个 checkpoint 单独的推理结果；签名匹配时自动跳过，增删 checkpoint 只需补跑新增部分。
- `logs/<ckpt>.log`：多卡并行时每个 checkpoint 子进程的日志。
- `embeddings.pt`：按 config 顺序合并全部 shard，含 raw token、chunk mean、global mean 和 metadata。
- `metrics.json`：intra/inter cosine distance、silhouette、nearest-center accuracy、norm、PCA effective rank、最近邻同任务/同长度，以及按有效 chunk 数分层（`length_stratified_*`）的中心准确率和轮廓系数。默认只计算 `chunk_flat`（每个 chunk 内 token 取平均后按顺序拼接），可在 config 中用 `features: [chunk_flat, global_mean]` 增加特征。
- `plots/`：均基于 `features` 中的第一个特征。每个 checkpoint 的 PCA（`pca_*`）和任务中心距离热力图（`center_distance_*`）；`task_distance.png` 为 intra/inter 曲线；`metrics_summary.png` 汇总全部表格指标随 checkpoint 的变化（中心准确率、轮廓系数含全量与按长度分层两版）。

## 分阶段运行

```bash
# 只生成并检查样本清单
bash tools/bpv3_encoder_probe/run.sh --manifest-only

# 只做 dry-run，不加载模型
bash tools/bpv3_encoder_probe/run.sh --dry-run

# 已有 embeddings 时只重算指标和图片
bash tools/bpv3_encoder_probe/run.sh --analyze-only
```

## 多卡并行

```bash
# 每张卡一次跑一个 checkpoint，空出来的卡自动领取下一个；全部完成后合并 shard 再分析
bash tools/bpv3_encoder_probe/run.sh --gpus 0,1,2,3
# 等价写法
BPV3_PROBE_GPUS=0,1,2,3 bash tools/bpv3_encoder_probe/run.sh
```

不传 `--gpus`（或只给一张卡）时在当前进程串行执行。任一 checkpoint 失败时不会合并，查看 `logs/<ckpt>.log` 后重跑即可，已完成的 shard 会被复用。

脚本使用当前仓库源码和训练时的 BP transform。默认输入与 ex01 clean 实验一致：单主视角、8 chunks、BP state、关闭 BP action。

## 其他配置

`run.sh` 默认读 `config.yaml`，可用 `BPV3_PROBE_CONFIG` 指定其他配置。stage1 与 stage1.5 训练任务不同时，应各用一份配置（各自的 `data_repo_file` 和 `output_dir`），例如 BPV3Plan/q35/v1：

```bash
BPV3_PROBE_CONFIG=tools/bpv3_encoder_probe/config_q35v1_stage1.yaml bash tools/bpv3_encoder_probe/run.sh --gpus 0,1
BPV3_PROBE_CONFIG=tools/bpv3_encoder_probe/config_q35v1_stage15.yaml bash tools/bpv3_encoder_probe/run.sh --gpus 2,3
```

`extra_analysis.py --config <cfg>` 在已有 `embeddings.pt` 上补充计算：去掉 chunk 位置编码后的指标、PC1 与长度相关、任务内跨长度距离比、按长度分层的各任务准确率与混淆对，结果写入 `extra_metrics.json`。

## 尝试启动
source /root/miniforge/etc/profile.d/conda.sh && conda activate q35va
cd /vla/workspace/my_tbot

HF_DATASETS_CACHE=/tmp/bpv3_probe_hf_datasets \
HF_HOME=/vla/tmp/bpv3_probe_hf_home \
PYTHONPYCACHEPREFIX=/tmp/bpv3_probe_pycache \
bash tools/bpv3_encoder_probe/run.sh --gpus 0,1,2,3