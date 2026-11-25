# 训练运行时优化：GPU 遥测与持久 DataLoader workers

## 正式默认

未显式指定配置的新训练（以及缺少这些字段的旧配置）使用以下默认值，所有策略共用，不改变 short/full 模型语义：

- `gpu_stats_freq=10`：仅主进程在完成第 10、20、30… 个 optimizer update 后采集 GPU 遥测。梯度累计的 micro-batch 不另计步，恢复训练沿用全局 optimizer step。采样不依赖 `log_freq`。减少遥测中的 CUDA 同步和 `nvidia-smi` 子进程调用。
- `dataloader_persistent_workers=true`：在 DataLoader 迭代器/epoch 切换时保留 workers，减少重复创建进程及加载 worker 内状态的成本。以最终实际 `num_workers` 为准；为 0 时传给 PyTorch 的 `persistent_workers` 自动为 false。worker 内缓存/状态会跨 epoch 保留，需 epoch 重新初始化的自定义 dataset 应显式关闭此项。

此提交不包含 DDP `find_unused_parameters` 优化，保持原来的环境变量行为及安全默认 true，也不包含 BPv5 模型、数据工厂或训练语义调整。

## 遥测与日志语义

未采样的 optimizer step 不更新 GPU meter，不把 0 或旧值重复计入样本数、均值。每个日志窗口有采样时，日志显示窗口内实际采样值的均值；进度条非日志步显示最近采样值。窗口没有采样时（例如 `log_freq < gpu_stats_freq`），日志与进度条沿用该指标最近一次有效值，不能将其理解成该步实时测量。首次有效采样前为 NaN，采集失败/部分指标缺失也不伪造 0。日志重置清空窗口的 count/sum，但 GPU 表保留最新值；其他训练指标重置行为不变。显存 peak 仍为 PyTorch 进程累计峰值，不在采样时重置。

`gpu_stats_freq=0` 完全禁用自动 GPU 遥测与对应指标注册；无 CUDA 时也不注册 GPU 指标。训练 loss 等统计不受影响。

## 配置与回退

在正式训练 JSON/JSONC 顶层指定即可覆盖默认值：

```json
{
  "gpu_stats_freq": 1,
  "dataloader_persistent_workers": false
}
```

`gpu_stats_freq=1` 恢复每个 optimizer update 采样，`0` 禁用，正整数 N 每 N 次更新采样，负值在配置校验时报错。`dataloader_persistent_workers=false` 恢复非持久 workers，便于排查 worker 状态、内存或数据初始化问题。现有显式配置优先于新默认，不改实验配置或用户的 short/full 设置。

## 收益边界与验证

既有 **40 步、单次实验**中，单独降低 GPU 遥测频率和单独启用 persistent workers 的观测收益分别为 **2.59%** 与 **8.68%**。这些数值不是稳定基准，不可相加，也不能代表两项一起启用的收益。三项组合曾有的 16.61% 包含额外 DDP 变化，不是本次两项的收益；本提交不启用该 DDP 变化，也不声称已测得两项组合收益。

独立轻量测试（无需长训练或 GPU）：

```bash
/vla/.conda/miniconda3/envs/q35va/bin/python -m pytest -q tests/test_training_runtime_optimizations.py
```

覆盖默认字段与覆盖值、负频率校验、0 workers 安全行为、真实 DataLoader 迭代器复用、10/1/0 采样频率、主进程/CUDA 门控、梯度累计只按完成的 optimizer update 计数，以及 GPU 样本均值、日志重置、NaN、失败采集和 loss.log 的行为。为避免加载所有策略后端，训练脚本相关测试通过 AST 提取生产代码并执行；未替代真实 GPU 性能基准或完整分布式训练。
