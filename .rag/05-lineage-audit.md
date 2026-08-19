# TBot / InternVLA / BPVA 代码谱系审计

> 截至/最后核验：2026-08-19
>
> 时效与代码优先声明：本文是只读静态比对记录，不是版本控制、著作权或许可证鉴定。仓库行为以当前源码、实际配置和 checkpoint 元数据为准；如代码与本文不符，以当前代码为准并更新本文；如代码与文档不符，以当前代码为准并更新本文。

## 结论边界

- 【代码事实】本仓库同时存在本地 `InternVLA_A1_3B`、`TBot_SA1` 与 `BPVA` 三套 modeling 文件：`src/lerobot/policies/InternVLA_A1_3B/modeling_internvla_a1.py`、`src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py`、`src/lerobot/policies/BPVA/modeling_bpva.py`。
- 【高置信推断】TBot 与本地 InternVLA_A1_3B 在三专家逐层联合 attention、Qwen current prefix、Cosmos middle、state/noisy-action suffix、flow-matching action loss 与采样框架上存在大量逐行同构，TBot 是在该骨架上扩展 3D/DA3 等能力的高置信结构性判断。
- 【待验证】没有精确上游 commit、fork base、patch series、作者声明或完整 git ancestry 可把这些相似行绑定到唯一来源。因此不得据此作“复制自某精确 commit”、原创权归属、许可证兼容或法律级来源断言。

## 逐行同构证据

- 【代码事实】两者都定义三专家逐层联合计算入口：InternVLA `src/lerobot/policies/InternVLA_A1_3B/modeling_internvla_a1.py:131-222`；TBot `src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:281-352`。
- 【代码事实】两者都用 Qwen3-VL understanding expert，加独立 gen/act text transformers，并在联合路径逐层计算、分别 final norm：InternVLA `src/lerobot/policies/InternVLA_A1_3B/modeling_internvla_a1.py:242-454`；TBot `src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:394-613`。
- 【代码事实】两者的 current prefix 都用 Qwen visual embedding 替换 image placeholder：InternVLA `src/lerobot/policies/InternVLA_A1_3B/modeling_internvla_a1.py:619-638`；TBot `src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:1013-1033`。
- 【代码事实】两者都有 Cosmos feature encoding、middle visual token 与未来 latent `loss_gen`：InternVLA `src/lerobot/policies/InternVLA_A1_3B/modeling_internvla_a1.py:640-674`、`815-825`；TBot `src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:1035-1044`、`1551-1575`、`1834-1845`。
- 【代码事实】两者都把 state、noisy action 与 timestep 编成 suffix，并用 flow matching 训练/迭代采样：InternVLA `src/lerobot/policies/InternVLA_A1_3B/modeling_internvla_a1.py:675-837`、`839-910`；TBot `src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:1671-1864`。
- 【代码事实】TBot 在同一骨架上新增 future 3D queries、view-aware/causal attention 和 DA3 teacher 对齐：`src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:1046-1227`、`1371-1466`。
- 【代码事实】BPVA 又在 TBot current prefix 上加入 K 个 BP tokens，同时保留 middle/suffix 与三类主 loss：`src/lerobot/policies/BPVA/modeling_bpva.py:1097-1147`、`1844-1953`。

## 机械相似度记录与局限

- 【代码事实】收到的只读源码报告记录“约 912 匹配行、`SequenceMatcher≈0.4659`”。
- 【代码事实】本次以 Python `difflib.SequenceMatcher(None, lines_a, lines_b, autojunk=False)` 对当前 TBot 与本地 InternVLA_A1_3B modeling 文件按整行重跑，得到 2714 对 1201 行、910 个 matching lines、ratio≈0.464879、163 个非空匹配块。与报告近似但非完全相同，可能由文件版本、换行、autojunk 或计数口径造成。
- 【方法局限】SequenceMatcher 不理解 AST、重命名、格式化、移动代码、第三方共同模板或独立同构实现；ratio 受文件长度和新增 DA3 大段代码影响。匹配行数也不等于独立原创行数，更不证明传播方向。
- 【待验证】要做工程级溯源，应固定文件 hash，获取双方完整 git 历史，用 commit-aware blame、AST/语义 diff 和许可证清单复核；法律判断应交由具备资质的人员。

## 公开背景

- 【公开资料】InternVLA-A1 官方分支现位于 [InternVLA-A-series / InternVLA-A1](https://github.com/InternRobotics/InternVLA-A-series/tree/InternVLA-A1)。
- 【公开资料】TBot-SA1 公开仓库为 [zaleni/TBot-SA1](https://github.com/zaleni/TBot-SA1)。截至本次实时页面核验仅确认仓库可访问，不声称其归档或活跃状态。
