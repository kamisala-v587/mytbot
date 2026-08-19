import {
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasAction,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

type View = "architecture" | "risks" | "roadmap";

const riskRows = [
  ["泄漏 / episode 记忆", "高", "avoid 会在无候选时回退同 episode；必须严格 split + forbid"],
  ["task_type 捷径", "高", "task_type→固定数据集/BP，本身可成为强任务标签"],
  ["训练—推理分布差异", "高", "训练随机同任务 episode；推理固定首个可读 episode"],
  ["单 token 压缩瓶颈", "中高", "3 图像 + state + 50 步 action 被压为一个 2048D token"],
  ["首 episode 鲁棒性", "中高", "可读不等于匹配、成功或高质量"],
  ["缺少 BP-specific loss", "高", "总 loss 仅 action + Cosmos + DA3，不能保证模型使用 BP"],
  ["task_index 静默回退", "高", "缺失时按 0；无候选再回退全部 episodes，可能跨任务"],
  ["raw cache 误读", "中", "服务缓存 BP 张量，不是 ViT/BP encoder 特征"],
  ["“无文本”误读", "中", "没有 instruction 文本，但仍依赖 task_type 与 Qwen3-VL"],
];

const roadmapRows = [
  ["P0", "可信评估", "严格 episode 隔离；null/错配/跨任务 BP；多 seed；记录 provenance"],
  ["P1", "选择鲁棒", "相似度 top-k；forbid；BP dropout；ViT 特征缓存；多候选 gating"],
  ["P2", "缓解压缩", "action 子段 token；spatial resampler；temporal encoder；分组 LR"],
  ["P3", "长期泛化", "BP-action 对比目标；instruction+BP；置信回退；联合优化"],
];

function FlowBox({ title, body, accent = false }: { title: string; body: string; accent?: boolean }) {
  const theme = useHostTheme();
  return (
    <div
      style={{
        border: `1px solid ${accent ? theme.accent.primary : theme.stroke.secondary}`,
        borderRadius: 8,
        padding: 12,
        background: accent ? theme.fill.secondary : theme.bg.elevated,
        minHeight: 92,
      }}
    >
      <Text weight="semibold" style={{ color: accent ? theme.accent.primary : theme.text.primary }}>{title}</Text>
      <Text size="small" tone="secondary">{body}</Text>
    </div>
  );
}

function Arrow() {
  const theme = useHostTheme();
  return <Text weight="bold" style={{ color: theme.text.tertiary, textAlign: "center" }}>→</Text>;
}

function Architecture() {
  return (
    <Stack gap={18}>
      <Callout tone="info" title="关键判断">
        BP 不是文本 prompt，而是 K 个行为示范块；每块由三相机当前帧、state、未来 action chunk 组成并压成一个 2048 维 prefix token。
      </Callout>

      <H2>数据选择与编码</H2>
      <Grid columns="1fr 28px 1fr 28px 1fr" gap={8} align="center">
        <FlowBox title="训练 BP" body="同 task_index 优先；默认尽量不同 episode；seeded RNG 选择" />
        <Arrow />
        <FlowBox title="K 个示范块" body="每块：3 cameras + state + future action[50×D]；短轨迹 pad" accent />
        <Arrow />
        <FlowBox title="BPObsEncoder" body="5×768 modality tokens → MLP fusion → 每块 1×2048D" />
      </Grid>
      <Text size="small" tone="tertiary">推理路径不同：task_type → YAML dataset → 从 episode 0 扫描 → 缓存首个可读 raw BP。它不是 ViT 特征缓存；训练中 task_index 缺失会静默回退。</Text>

      <H2>MoT 信息流</H2>
      <Grid columns="1fr 28px 1fr 28px 1fr" gap={8} align="center">
        <FlowBox title="Current / Prefix / und" body="当前三相机无 instruction 文本，但仍由 Qwen3-VL visual 编码；其 tokens 与 K 个 BP tokens 拼接" accent />
        <Arrow />
        <FlowBox title="Middle / gen" body="历史+当前 Cosmos visual tokens；3D messenger queries；DA3 多层蒸馏" />
        <Arrow />
        <FlowBox title="Suffix / act" body="current state + noisy action + time；flow matching 预测 action velocity" />
      </Grid>
      <Grid columns={3} gap={12}>
        <Stat value="K × 2048" label="BP prefix token 形状" />
        <Stat value="3" label="und / gen / act 专家" />
        <Stat value="0" label="BP-specific loss" />
      </Grid>

      <Divider />
      <H3>训练边界</H3>
      <Text>当前 v3 配置仅冻结 BP 的预训练 ViT；Qwen vision、und/gen/act 与动作、2D 世界、3D 世界路径仍参与训练。Cosmos tokenizer 与 DA3 teacher 冻结。总 loss 只有 action、Cosmos、DA3，没有专门 BP loss。</Text>
      <Callout tone="warning" title="措辞边界">
        可以说“当前数据变换不注入 instruction 文本”；不能说“完全不依赖 language/VLM”。task_type 选择 BP，当前图像仍经过 Qwen3-VL，understanding expert 仍保留。
      </Callout>
    </Stack>
  );
}

function Risks() {
  return (
    <Stack gap={16}>
      <H2>风险登记</H2>
      <Table
        headers={["风险", "等级", "证据与处置"]}
        rows={riskRows}
        rowTone={["danger", "danger", "danger", "warning", "warning", "danger", "danger", "warning", "info"]}
        striped
      />
      <H2>判定 BP 是否真的有效</H2>
      <Grid columns={2} gap={12}>
        <FlowBox title="已有入口，无结果" body="mask server 支持 mask=False 无 BP 与 mask=True 零值有效 BP；仓库未见成功率结论" />
        <FlowBox title="必须敏感但不脆弱" body="正确 BP 优于跨任务与错配 BP；多个合法 BP 的方差可控" />
        <FlowBox title="必须排除泄漏" body="query 与 BP episode 严格隔离，记录 episode hash、split 与 provenance" />
        <FlowBox title="必须覆盖 OOD" body="新场景、新对象、跨 embodiment、无 BP、错误 BP 与相似任务 BP" />
      </Grid>
      <Callout tone="danger" title="发布结论门槛">
        只有严格 split 下正确 BP 显著优于 null BP、错配显著更差但不灾难、合法 prompt 方差可控，才能支持“模型利用可迁移行为信息”，而非任务标签或 episode 记忆。
      </Callout>
    </Stack>
  );
}

function Roadmap() {
  const dispatch = useCanvasAction();
  return (
    <Stack gap={16}>
      <H2>分优先级优化路线</H2>
      <Table
        headers={["优先级", "目标", "交付内容"]}
        rows={roadmapRows}
        rowTone={["danger", "warning", "info", "neutral"]}
        striped
      />
      <H2>P0 反事实矩阵</H2>
      <Grid columns="1fr 1fr 1fr" gap={12}>
        <Card>
          <CardHeader trailing={<Pill size="sm" active>核心</Pill>}>有无与匹配</CardHeader>
          <CardBody><Text size="small">正确 / mask=False / 零值有效 / 错任务 / 同任务不同 episode</Text></CardBody>
        </Card>
        <Card>
          <CardHeader>单模态反事实</CardHeader>
          <CardBody><Text size="small">只乱 action / 只乱 image / chunk 反序；固定当前观测与环境 seed</Text></CardBody>
        </Card>
        <Card>
          <CardHeader>因果敏感性</CardHeader>
          <CardBody><Text size="small">固定当前观测只换 BP；记录动作变化、成功率、多个合法 BP 方差</Text></CardBody>
        </Card>
      </Grid>
      <Row gap={8}>
        <Button variant="primary" onClick={() => dispatch({ type: "openFile", path: ".rag/04-bpva-assessment-roadmap.md" })}>打开完整评估路线</Button>
        <Button variant="secondary" onClick={() => dispatch({ type: "openFile", path: ".rag/03-bpva-current-implementation.md" })}>打开实现事实</Button>
      </Row>
    </Stack>
  );
}

export default function BPVAArchitectureAssessment() {
  const [view, setView] = useCanvasState<View>("bpva-assessment-view", "architecture");
  return (
    <Stack gap={20} style={{ padding: 20, maxWidth: 1180, margin: "0 auto" }}>
      <Row justify="space-between" align="start" wrap>
        <Stack gap={4}>
          <H1>BPVA 架构评估</H1>
          <Text tone="secondary">行为示范 prefix × TBot 三专家 × Cosmos / DA3 / Flow Matching</Text>
          <Text size="small" tone="tertiary">截至/最后核验：2026-08-19 · 本地当前代码优先</Text>
        </Stack>
        <Row gap={8} wrap>
          <Pill active={view === "architecture"} onClick={() => setView("architecture")}>架构与信息流</Pill>
          <Pill active={view === "risks"} onClick={() => setView("risks")}>风险与判断</Pill>
          <Pill active={view === "roadmap"} onClick={() => setView("roadmap")}>优化路线</Pill>
        </Row>
      </Row>
      {view === "architecture" && <Architecture />}
      {view === "risks" && <Risks />}
      {view === "roadmap" && <Roadmap />}
      <Divider />
      <Text size="small" tone="tertiary">证据源：本仓库 `.rag/` 与源码；公开架构背景来自 InternVLA-A1 论文和官方 TBot-SA1 仓库。Canvas 不访问网络。</Text>
    </Stack>
  );
}
