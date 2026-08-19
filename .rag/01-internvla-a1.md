# InternVLA-A1 基础知识

> 截至/最后核验：2026-08-19
>
> 时效与代码优先声明：公开资料描述上游设计，本仓库行为以当前源码和运行配置为准；若两者不一致，以代码为准并更新本文；如代码与文档不符，以当前代码为准并更新本文。

## 名称边界：不能把 InternVL 当作 InternVLA-A1

- 【公开资料】InternVLA-A1 是“理解—视觉前瞻—动作”统一机器人模型，论文为 *InternVLA-A1: Unifying Understanding, Generation and Action for Robotic Manipulation*：[arXiv 2601.02456v2](https://arxiv.org/html/2601.02456v2)，官方代码的 InternVLA-A1 分支：[GitHub](https://github.com/InternRobotics/InternVLA-A-series/tree/InternVLA-A1)。
- 【公开资料】InternVL3 只是 InternVLA-A1 2B 版本采用的 understanding expert/MLLM backbone；3B 版本采用 Qwen3-VL。论文明确列出 2B 为 InternVL3 + Qwen2.5 gen/act，3B 为 Qwen3-VL + Qwen3 gen/act：[论文 §3.4](https://arxiv.org/html/2601.02456v2#S3.SS4)。所以“InternVL”与“InternVLA-A1”不是同一模型名。
- 【代码事实】本地 TBot/BPVA 默认走 Qwen3-VL understanding expert，不是 InternVL：`src/lerobot/policies/TBot_SA1/configuration_tbot_sa1.py:194-198`；BPVA 的本地预训练路径见 `src/lerobot/policies/BPVA/configuration_bpva.py:138-145`。

## 论文架构

- 【公开资料】InternVLA-A1 使用 Mixture-of-Transformers（MoT），由 understanding、generation、action 三专家组成，并通过统一 masked self-attention 交互：[论文 Figure 2 与 §3.1](https://arxiv.org/html/2601.02456v2#S3.SS1)。
- 【公开资料】understanding expert 接收多视角当前观测与 instruction 文本，形成语义上下文；generation expert 使用三视角、历史/当前共六张图，借助 Cosmos CI8×8 tokenizer 保留细粒度空间信息；action expert 加入 proprioceptive state，以 flow matching 预测连续 action chunk：[论文 §3.1](https://arxiv.org/html/2601.02456v2#S3.SS1)。
- 【公开资料】generation 分支把每张图压到 4×4=16 tokens，六张图合计 96 tokens，并监督未来帧 Cosmos latent；action 分支学习噪声到示范动作的速度场，推理通过 ODE/Euler 迭代采样：[论文 §3.2](https://arxiv.org/html/2601.02456v2#S3.SS2)。
- 【公开资料】论文的信息流是 understanding → generation → action 的块级因果关系，块内存在双向注意力：[论文 Attention Mechanism](https://arxiv.org/html/2601.02456v2#S3.SS1)。

## 数据与结果的正确使用方式

- 【公开资料】官方称预训练覆盖真实机器人、仿真与人类视频，共超过 692M frames，并在 12 个真实任务和 RoboTwin 2.0 评估：[论文摘要](https://arxiv.org/html/2601.02456v2)。
- 【公开资料】官方仓库公布 RoboTwin 2.0 50 任务结果并提供训练/评估教程：[官方仓库](https://github.com/InternRobotics/InternVLA-A-series/tree/InternVLA-A1)。
- 【推断】这些论文结果不能直接归因于本地 TBot/BPVA。后者虽然沿用三专家、Cosmos、flow matching 等思想，但增加 DA3 与 BP prefix，并存在本地注意力和数据路径差异。
- 【待验证】若要声称“BPVA 优于 InternVLA-A1/TBot-SA1”，必须在同一数据划分、动作表示、观测历史、推理步数与评测脚本下做 checkpoint 对照；仓库静态阅读不能支持性能结论。

## 与 BPVA 的关系

- 【代码事实】BPVA 保留 Qwen3-VL understanding、gen、act 三专家结构：`src/lerobot/policies/BPVA/modeling_bpva.py:395-465`。
- 【代码事实】BPVA 的变化是将当前无 instruction 文本的 Qwen 图像 prefix 与 K 个行为轨迹 token 拼接，而不是把文本换成几句 prompt：`src/lerobot/transforms/core_bp.py:236-270`；`src/lerobot/policies/BPVA/modeling_bpva.py:1097-1147`。
- 【推断】BP 可视为 task-conditioned demonstration memory，但它并不等价于论文原始 instruction 通道；它携带动作答案模式，也因此引入更强的数据泄漏与捷径风险。
