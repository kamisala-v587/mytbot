# InternVLA 相关演进参考：A1.5、M1、N1

> 截至/最后核验：2026-08-19
>
> 时效与代码优先声明：本篇仅记录官方公开定位，用于避免名称混淆；不表示这些模型已集成本仓库。涉及本地 TBot/BPVA 时以当前代码为准，若代码与文档不符，以当前代码为准并更新本文；如代码与文档不符，以当前代码为准并更新本文。

## A1.5：latent foresight 的操控 VLA

- 【公开资料】InternVLA-A1.5 将 Qwen3.5-2B VLM 与轻量 action expert 连接，通过 shared full-attention 与 modality-specific Gated DeltaNet 处理理解和动作：[官方仓库](https://github.com/InternRobotics/InternVLA-A-series)。
- 【公开资料】训练期使用 learnable foresight tokens，并由冻结 WAN2.2-5B 视频生成模型监督 task-relevant future dynamics；标准 action-only 推理会丢弃/跳过 WAN 视频分支，再以 flow matching 输出连续 action chunk：[官方仓库](https://github.com/InternRobotics/InternVLA-A-series)。
- 【边界】A1.5 的 WAN latent foresight 不是 A1/TBot 的 Cosmos future-image latent，也不是 BPVA 的行为轨迹 prefix。当前 TBot/BPVA 仍有 Cosmos 与 DA3 路径，不能套用“A1.5 推理丢弃 WAN”描述。

## M1：空间 grounding 双系统操控框架

- 【公开资料】InternVLA-M1 定位为 spatially guided generalist robot policy，整合 language head 与 action head，以 dual-system、dual-supervision 协同训练，并支持 image QA / spatial grounding：[官方仓库](https://github.com/InternRobotics/InternVLA-M1)。
- 【边界】M1 的“空间 prompt/grounding + language/action 双系统”不是 TBot 的 und/gen/act 三专家，也不是 BPVA 的三相机+state+action 行为示范。它是相关空间推理参考，不是当前实现谱系证据。

## N1：导航，不是机械臂操控

- 【公开资料】InternVLA-N1 是 InternNav 体系中的双系统 navigation foundation model，面向 embodied navigation、VLN、路径规划与真实世界零样本泛化：[InternNav 官方仓库](https://github.com/InternRobotics/InternNav)；[官方技术报告入口](https://internrobotics.github.io/internvla-n1.github.io/static/pdfs/InternVLA_N1.pdf)。
- 【边界】N1 的 System-1/System-2 导航栈不等于 TBot/BPVA 的 manipulation action chunk 策略。不得以 N1 的导航结果证明 BPVA 操控能力，也不得把 N1 当作 A1/A1.5 的直接操控版本升级。

## 与当前仓库的关系

- 【代码事实】当前仓库明确包含 `InternVLA_A1_2B`、`InternVLA_A1_3B`、`TBot_SA1`、`BPVA` 策略目录；没有因公开名称相近而自动获得 A1.5/M1/N1 的模块或权重。
- 【高置信推断】A1.5 可启发训练期 teacher-only foresight，M1 可启发 instruction/grounding 与动作双监督，N1 可启发双系统规划；这些只是研究参照，采用前须另做接口、许可与实验验证。
