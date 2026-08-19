# TBot-SA1 源码架构

> 截至/最后核验：2026-08-19
>
> 时效与代码优先声明：官方仓库状态可能变化，本文不据此判断维护状态；本地行为以 `/vla/workspace/my_tbot` 当前代码为准。若本文、官方 README 与代码不符，以当前代码为准并更新本文；如代码与文档不符，以当前代码为准并更新本文。

## 上游定位

- 【公开资料】上游 TBot-SA1 自述为 World-Spatial-Action 模型，统一 instruction-aligned 2D visual planning、action-conditioned 3D world modeling 与 3D-aware action generation；截至核验时公开页面可访问；本文不对其是否归档作断言：[zaleni/TBot-SA1](https://github.com/zaleni/TBot-SA1)。
- 【推断】本地代码在 InternVLA-A1 风格三专家上加入 DA3 蒸馏，构成当前“视觉理解 + 2D 未来 latent + 3D messenger/query + action flow”实现；不要用上游 README 替代本地逐行事实。

## current / middle / suffix 与三专家

- 【代码事实】三段与三专家映射为：prefix/current → und expert（Qwen3-VL 语言模型与视觉理解），middle → gen expert（2D/Cosmos 与 3D query），suffix → act expert（state + noisy action）：`src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:394-402`。
- 【代码事实】und 从 Qwen3-VL pretrained 初始化；gen/act 是 Qwen3-VL text config 构建、移除 token embedding/lm head 的独立 transformer：`src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:404-467`。
- 【代码事实】三流联合执行时，每层把三专家送入统一 attention 计算，再分别回写；最终各自 norm 并返回 prefix/middle/suffix：`src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:542-613`。
- 【代码事实】current 图像经 Qwen3-VL visual encoder 替换 image placeholder token：`src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:1013-1033`。

## Cosmos 2D 视觉前瞻

- 【代码事实】Cosmos tokenizer 将图像 resize 到 256×256、归一化到 [-1,1] 后编码：`src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:1035-1044`。
- 【代码事实】middle 包含 Cosmos visual tokens；训练用 gen 输出解码并拟合未来帧 Cosmos latent，权重由 `lambda_gen` 控制：`src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:1834-1845`；配置见 `src/lerobot/policies/TBot_SA1/configuration_tbot_sa1.py:273-276`。
- 【公开资料】这继承 InternVLA-A1 的视觉前瞻动机：[InternVLA-A1 §3.2](https://arxiv.org/html/2601.02456v2#S3.SS2)。

## DA3 3D 路径

- 【代码事实】可学习 future 3D queries 被追加到 middle；Cosmos visual 子块双向，3D query 可读取 visual 且 query 内双向：`src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:1046-1078`。
- 【代码事实】默认 view-aware mask 限制每组 3D query 只读对应视角 visual token，同时 query 组互通：`src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:1080-1131`。
- 【代码事实】训练从 gen 指定层取中间状态，对齐冻结 DA3 teacher 多层特征；配置包括 query 层、teacher 层、每视角 token 数和 `lambda_3d`：`src/lerobot/policies/TBot_SA1/configuration_tbot_sa1.py:278-297`。
- 【代码事实】DA3 teacher 和 Cosmos 总是 eval 且不训练：`src/lerobot/policies/BPVA/modeling_bpva.py:925-934`（BPVA 复制的主干冻结逻辑；TBot 同类实现应回查对应文件）。

## Flow matching 动作路径

- 【代码事实】suffix 由 state、带噪 action 与时间编码组成；模型一次预测 `chunk_size` 动作，默认 50，推理默认 10 次去噪：`src/lerobot/policies/TBot_SA1/configuration_tbot_sa1.py:200-218`。
- 【代码事实】BPVA 保留的 TBot 训练目标为 `x_t=t*noise+(1-t)*actions`、目标速度 `noise-actions`，action head 对 suffix 输出做 MSE：`src/lerobot/policies/BPVA/modeling_bpva.py:1844-1857`、`1937-1946`。
- 【公开资料】flow matching 的原始解释与 ODE 采样见 [InternVLA-A1 §3.2](https://arxiv.org/html/2601.02456v2#S3.SS2)。

## 参数训练边界

- 【代码事实】TBot 配置允许冻结 Qwen vision、只训 gen/act、只训 und 或按专家 LoRA；默认并非全部冻结：`src/lerobot/policies/TBot_SA1/configuration_tbot_sa1.py:256-271`。
- 【代码事实】本地 BPVA v3 配置设置 `freeze_vision_encoder=false`、`train_expert_only=false`、`train_vlm_only=false`，并保留 `lambda_gen=lambda_3d=0.01`：`configs/bpva_train_v3.jsonc:86-115`。
- 【结论/代码事实】因此 BPVA 仍保留并训练 TBot 主干及视觉/世界/动作路径；不能描述成只训练一个 BP encoder 或完全不依赖 VLM。
