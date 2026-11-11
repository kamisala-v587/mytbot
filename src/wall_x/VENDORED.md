# wall_x（vendored）

来源：`https://github.com/StarrickLiu/wall-x.git` 分支 `starrick/oss-training`，commit `b18112d`（X-Square-Robot/wall-x PR #128 的 Qwen3.5 发布 + 训练代码），Apache-2.0（见 `LICENSE`）。
本地完整仓库：`/vla/workspace/code/repos/wall-x`。

供 `lerobot.policies.wall_oss` 使用（加载 `wall-oss-1.0-lite` 等 Qwen3.5 ckpt）。未编译 CUDA 算子（`model/core/ops/csrc`），所有算子走 PyTorch 回退实现。

相对上游的改动（运行在 my_tbot 的 q35va 环境：torch 2.8 / transformers 5.14）：

1. `model/core/attention/selector.py`：`_check_and_adjust_attn_implementation` 增加 `**kwargs`（transformers 5.14 多传了 `allow_all_kernels`）。
2. `model/qact/qwen3_5/qwen3_5_inference_mixin.py`：`torchdiffeq` 改为在用到的地方导入（训练不需要它）。

同步上游时：用新 commit 的 `wall_x/` 覆盖本目录（删去 `*.so`），再重新应用以上两处改动。
