# 后续 AI 工作提示词

> 截至/最后核验：2026-08-19
>
> 时效与代码优先声明：先读知识库，但实现判断以当前源码、配置与 checkpoint 元数据为准；若代码与文档不符，以当前代码为准并同步更新 `.rag/`；如代码与文档不符，以当前代码为准并更新本文。

```text
你正在维护 /vla/workspace/my_tbot。先阅读 .rag/README.md；涉及效果判断时优先检查 .rag/07-experiment-evidence.md，并区分【用户实验】与【代码事实】；再按索引阅读相关专题，最后回查当前源码与实际配置。回答和修改时区分【代码事实】【公开资料】【推断】【待验证】，代码事实给精确路径+行号，公开资料给原始 URL；不确定就标待验证。

关键约束：不要混淆 InternVL/InternVL3 与 InternVLA-A1。BP 是行为示范轨迹：每块三相机当前帧 + state + 未来 action chunk，不是普通文本 prompt。训练 BP 同任务优先、尽量不同 episode；推理按 task_type 映射数据集并确定性缓存首个可读 episode。BP 编码为 K 个 2048 维 prefix token，与无 instruction 文本但仍经 Qwen3-VL 视觉编码的当前图像 tokens 拼接。BPVA 保留 TBot 的 und/gen/act、Cosmos、DA3、flow matching 及视觉/世界/动作训练路径；不得声称完全不依赖 language/VLM。

涉及效果必须检查泄漏/捷径、训练推理 BP 分布差异、压缩瓶颈、选择鲁棒性，并包含跨任务、错配、无 BP 消融；任何“有效、提升、泛化”效果声明都必须有固定当前观测换 BP 等反事实证据。代码改变上述事实时，同一变更中更新滞后的 .rag 文档；文档冲突时以当前代码为准。
```
