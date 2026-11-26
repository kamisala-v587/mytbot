#!/usr/bin/env python3
"""Convert a completed report summary to an offline Cursor canvas; no training IO."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

OUTPUT = Path('/root/.cursor/projects/vla-workspace/canvases/BPv5-stage-b-v2-baseline.canvas.tsx')
E2E_OUTPUT = OUTPUT.with_name('BPv5-stage-b-v2-bs2-e2e.canvas.tsx')
PHASES = ('forward', 'backward_ddp', 'clip', 'optimizer', 'zero_grad', 'scheduler', 'other residual')
CATEGORIES = ('parameter', 'optimizer_state', 'gradient', 'saved_activation', 'other_residual')


class SummaryError(ValueError):
    """Unsupported or inconsistent report summary."""


def require(condition, message):
    if not condition:
        raise SummaryError(message)


def obj(value, label):
    require(isinstance(value, dict), f'{label} 必须为对象')
    return value


def number(value, label, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f'{label} 必须为有限非负数（分母须为正）')
    return value


def stats(value, label):
    value = obj(value, label)
    require(type(value.get('count')) is int and value['count'] == 20, f'{label}.count 必须为 20')
    for key in ('mean', 'p50', 'p95', 'min', 'max'):
        number(value.get(key), f'{label}.{key}', True)
    require(value['min'] <= value['p50'] <= value['p95'] <= value['max']
            and value['min'] <= value['mean'] <= value['max'], f'{label} 统计范围不一致')
    return value


def parts(value, label, keys=None):
    value = obj(value, label)
    if keys is not None:
        require(set(value) == set(keys), f'{label} 分类须为 {list(keys)}')
    for key, size in value.items():
        require(isinstance(key, str) and bool(key), f'{label} 名称无效')
        number(size, f'{label}.{key}')
    return [{'label': k, 'value': v} for k, v in value.items() if v > 0]


def normalize(summary, source, digest):
    summary = obj(summary, 'summary')
    require(type(summary.get('schema_version')) is int and summary['schema_version'] in (2, 3),
            '只支持 report_stage_b_v2_profile.py summary schema_version=2/3')
    require(summary.get('status') == 'complete', 'status 必须为 complete，拒绝部分报告')
    require(summary.get('steady_steps') == [11, 30], 'steady_steps 必须为 [11, 30]')
    ranks = obj(summary.get('per_rank'), 'per_rank')
    require(set(ranks) == {'0', '1', '2', '3'}, 'per_rank 必须恰含四卡 0–3')
    peaks = []
    for rank in ('0', '1', '2', '3'):
        data = obj(ranks[rank], f'rank {rank}')
        stats(data.get('steady_update_wall_s'), f'rank {rank} wall')
        allocated = stats(data.get('steady_peak_allocated_bytes'), f'rank {rank} allocated')['max']
        reserved = stats(data.get('steady_peak_reserved_bytes'), f'rank {rank} reserved')['max']
        require(reserved >= allocated, f'rank {rank} reserved 峰值小于 allocated')
        peaks.append({'rank': rank, 'allocated': allocated, 'reserved': reserved})
    throughput = obj(summary.get('synchronized_throughput'), 'synchronized_throughput')
    batch = summary.get('batch') or {'per_rank_trajectories': 1, 'global_trajectories': 4}
    local = batch.get('per_rank_trajectories')
    require(type(local) is int and local > 0, 'per_rank_trajectories 须为正整数')
    require(throughput.get('global_samples_per_update') == local * 4 == batch.get('global_trajectories'), '实际batch口径不一致')
    result = {'source': str(source), 'sha256': digest, 'peaks': peaks,
              'steady': stats(throughput.get('steady_max_rank_update_wall_s'), 'steady_max_rank_update_wall_s'),
              'time': [], 'allocated': [], 'saved': [], 'snapshot': None,
              'batch': batch, 'e2e': None, 'e2eParts': [], 'modules': [],
              'comparison': summary.get('baseline_comparison'),
              'updateTrajectories': number(throughput.get('update_only_samples_per_second'), 'update trajectories/s', True),
              'schema': summary['schema_version']}
    if summary['schema_version'] == 3:
        e2e = obj(summary.get('e2e'), 'e2e')
        require(e2e.get('steps') == [11, 30] and e2e.get('samples') == 80, 'E2E须为四卡20步')
        elapsed = number(e2e.get('max_rank_window_elapsed_s'), 'max_rank_window_elapsed_s', True)
        per_rank = obj(e2e.get('per_rank'), 'e2e.per_rank')
        require(set(per_rank) == {'0', '1', '2', '3'}, 'E2E需四rank')
        require(math.isclose(elapsed, max(number(r.get('steady_window_elapsed_s'), 'rank window', True) for r in per_rank.values())), 'E2E max窗口不一致')
        require(math.isclose(number(e2e.get('global_trajectories_per_s'), 'E2E trajectories/s', True), 20 * local * 4 / elapsed), 'E2E轨迹吞吐分母不一致')
        segments = number(batch.get('global_segments'), 'global_segments', True)
        require(math.isclose(number(e2e.get('global_segments_per_s'), 'E2E segments/s', True), 20 * segments / elapsed), 'E2E segment吞吐不一致')
        result['e2eParts'] = parts(e2e.get('critical_window_mean_wall_s'), 'E2E互斥分类', ('dataloader_next', 'update', 'postupdate', 'other residual'))
        require(math.isclose(sum(d['value'] for d in result['e2eParts']), elapsed / 20, rel_tol=1e-7, abs_tol=1e-7), 'E2E分类不闭合到完整窗口')
        result['e2e'] = e2e
    for module in summary.get('modules_inclusive', {}).get('rows', []):
        if module.get('name') in ('function/vision/visual_only', 'function/prefix/vision', 'function/prefix/build_prefix'):
            stat = module['per_call_inclusive_cuda_ms']
            result['modules'].append({'name': module['name'], 'mean': number(stat['mean'], 'module mean'), 'p95': number(stat['p95'], 'module p95'), 'count': number(stat['count'], 'module calls', True)})
    if summary.get('time_share') is not None:
        time = obj(summary['time_share'], 'time_share')
        require(time.get('steps') == [31, 34] and time.get('ranks') == [0, 1, 2, 3]
                and time.get('samples') == 16, 'time_share 须为 steps 31–34 × 四卡，16 样本')
        result['time'] = parts(time.get('mean_wall_s'), 'mean_wall_s', PHASES)
    if summary.get('memory_share') is not None:
        memory = obj(summary['memory_share'], 'memory_share')
        require(memory.get('kind') == 'saved_activation_module_share_only', 'memory_share.kind 不匹配')
        rank, step = memory.get('rank'), memory.get('step')
        require(type(rank) is int and rank in range(4) and type(step) is int and step in range(35, 39),
                'memory_share 须选择 rank 0–3 / step 35–38')
        allocated = number(memory.get('forward_end_allocated_bytes'), 'forward_end_allocated_bytes', True)
        saved = obj(memory.get('saved_snapshot'), 'saved_snapshot')
        groups = parts(saved.get('groups_storage_bytes'), 'groups_storage_bytes')
        total = number(saved.get('total_storage_bytes'), 'total_storage_bytes')
        require(sum(d['value'] for d in groups) == total <= allocated, 'saved 模块和 != total 或 total > allocated')
        require(memory.get('saved_le_allocated_check') is True, '缺少 saved_le_allocated_check=true')
        result['saved'] = sorted(groups, key=lambda d: d['value'], reverse=True)
        result['snapshot'] = {'rank': rank, 'step': step, 'savedTotal': total, 'allocated': allocated}
        decomposition = memory.get('total_allocated_decomposition')
        if decomposition is not None:
            decomposition = obj(decomposition, 'total_allocated_decomposition')
            require(decomposition.get('allocated_bytes') == allocated, '总分解 allocated 与同刻分母不一致')
            require(decomposition.get('disjoint_storage_check') is True
                    and decomposition.get('sum_equals_allocated_check') is True,
                    '总分解须通过 storage 无交叉 / 分类和等于 allocated 检查')
            require(isinstance(decomposition.get('device'), str)
                    and decomposition['device'].startswith('cuda'), '总分解缺少明确 CUDA device')
            result['allocated'] = parts(decomposition.get('categories_bytes'), 'categories_bytes', CATEGORIES)
            require(sum(d['value'] for d in result['allocated']) == allocated, '分类之和不等于 allocated')
            result['snapshot']['device'] = decomposition['device']
    return result


TEMPLATE = r'''import { H1, H2, Text, Stack, Row, Grid, Divider, Table, useHostTheme } from "cursor/canvas";
type Stat = { count: number; mean: number; p50: number; p95: number; min: number; max: number };
type E2E = { mean_step_wall_s: number; global_trajectories_per_s: number; global_segments_per_s: number; max_rank_window_elapsed_s: number; slowest_window_rank: number; per_rank: Record<string, { mean_wall_s: Record<string, number>; telemetry_direct_wall_s: Stat; h2d_enqueue_wall_s: Stat; worker_timing: Record<string, Stat> }> };
type Report = { source: string; sha256: string; schema: number; peaks: {rank: string; allocated: number; reserved: number}[]; steady: Stat; time: Part[]; allocated: Part[]; saved: Part[]; snapshot: null | {rank: number; step: number; savedTotal: number; allocated: number; device?: string}; batch: {per_rank_trajectories: number; global_trajectories: number}; e2e: E2E | null; e2eParts: Part[]; modules: {name: string; mean: number; p95: number; count: number}[]; comparison: null | {ratio: number; improvement_percent: number}; updateTrajectories: number };
const report = __REPORT__ as unknown as Report;
type Part = { label: string; value: number };
const gib = (n: number) => (n / 2 ** 30).toFixed(3);
const snapshot = report.snapshot as null | { rank: number; step: number; savedTotal: number; allocated: number; device?: string };

// Single token hue with numbered slices avoids a rainbow palette.
function Pie({ title, data, unit, caption }: { title: string; data: Part[]; unit: "ms" | "GiB"; caption: string }) {
  const theme = useHostTheme();
  const total = data.reduce((s, d) => s + d.value, 0);
  let offset = 0;
  const format = (n: number) => unit === "ms" ? (n * 1000).toFixed(2) : gib(n);
  const opacity = (i: number) => .22 + .78 * (i + 1) / data.length;
  return <Stack gap={12} style={{ minWidth: 0 }}>
    <H2>{title}</H2>
    <Row gap={20} wrap align="center">
      <svg viewBox="0 0 220 220" width={220} height={220} role="img" aria-label={`${title}，合计 ${format(total)} ${unit}`}>
        {data.map((d, i) => {
          const start = offset; offset += d.value / total * 360;
          const point = (angle: number) => [110 + 100 * Math.cos((angle - 90) * Math.PI / 180), 110 + 100 * Math.sin((angle - 90) * Math.PI / 180)];
          const a = point(start), b = point(offset), mid = (start + offset) / 2;
          return <g key={d.label}>
            <title>{d.label}: {format(d.value)} {unit} ({(100 * d.value / total).toFixed(1)}%)</title>
            {data.length === 1 ? <circle cx={110} cy={110} r={100} fill={theme.accent.primary} /> :
              <path d={`M 110 110 L ${a[0]} ${a[1]} A 100 100 0 ${offset - start > 180 ? 1 : 0} 1 ${b[0]} ${b[1]} Z`} fill={theme.accent.primary} fillOpacity={opacity(i)} stroke={theme.bg.editor} strokeWidth={1} />}
            {d.value / total >= .045 && <text x={110 + 70 * Math.cos((mid - 90) * Math.PI / 180)} y={114 + 70 * Math.sin((mid - 90) * Math.PI / 180)} textAnchor="middle" fill={theme.text.primary} fontSize={12}>{i + 1}</text>}
          </g>;
        })}
      </svg>
      <Stack gap={7} style={{ flex: 1, minWidth: 220 }}>
        {data.map((d, i) => <Row key={d.label} gap={8} align="center">
          <span style={{ width: 10, height: 10, flexShrink: 0, background: theme.accent.primary, opacity: opacity(i) }} />
          <Text size="small" style={{ overflowWrap: "anywhere" }}>{i + 1}. {d.label} · {format(d.value)} {unit} · {(100 * d.value / total).toFixed(1)}%</Text>
        </Row>)}
      </Stack>
    </Row>
    <Text size="small" tone="secondary">{caption} 分母合计 {format(total)} {unit}；面积表示占比，无坐标轴。</Text>
  </Stack>;
}

export default function ProfileCanvas() {
  const theme = useHostTheme();
  return <Stack gap={24} style={{ padding: 24, maxWidth: 1180, margin: "0 auto", color: theme.text.primary }}>
    <Stack gap={8}>
      <H1>BPv5 · Stage B v2 · {report.e2e ? "bs2 E2E" : "update-only baseline"}</H1>
      <Text>训练 short · 40 updates · 四卡 DDP · 每卡 bs{report.batch.per_rank_trajectories} · 全局 batch {report.batch.global_trajectories} · GC off（activation gradient checkpointing 关闭）· 无梯度累积（accum=1）</Text>
      <Text size="small" tone="secondary">实际batch来自离线报告已核验的运行metadata/config；画布未额外读取采集文件。短程 profiling 不代表收敛或泛化。</Text>
    </Stack>
    <Grid columns="repeat(auto-fit, minmax(220px, 1fr))" gap={24}>
      <Stack gap={6}>
        <Text size="small" tone="secondary">update-only step seconds · steps 11–30</Text>
        <Text weight="bold" style={{ fontSize: 24, color: theme.accent.primary }}>{report.steady.mean.toFixed(4)} s / update</Text>
        <Text size="small">p50 {report.steady.p50.toFixed(4)} s · p95 {report.steady.p95.toFixed(4)} s</Text>
      </Stack>
      <Stack gap={6}>
        <Text size="small" tone="secondary">稳态单卡 allocated 峰值 · 四卡取 max，不求和</Text>
        <Text weight="bold" style={{ fontSize: 24 }}>{gib(Math.max(...report.peaks.map(p => p.allocated)))} GiB</Text>
        <Text size="small">reserved max {gib(Math.max(...report.peaks.map(p => p.reserved)))} GiB；两个 max 未必同刻</Text>
      </Stack>
    </Grid>
    <Text size="small" tone="secondary">来源：summary.synchronized_throughput / per_rank。20 步逐步取四卡 update 墙时 max 再统计均值；仅 update_policy，不含 loader、前置 H2D、后置日志。p50=median，p95=nearest-rank。显存为各步 allocator peak 的区间 max，不是四卡总和，也非全 40 步 max。</Text>
    {report.e2e && <Stack gap={12}>
      <H2>完整20步 E2E 窗口 · 与 update-only 区分</H2>
      <Text weight="bold">{report.e2e.mean_step_wall_s.toFixed(4)} s/step · {report.e2e.global_trajectories_per_s.toFixed(3)} trajectories/s · {report.e2e.global_segments_per_s.toFixed(3)} segments/s</Text>
      <Text size="small" tone="secondary">来源：summary.e2e；steps 11–30。各rank连续20步 elapsed 取 max={report.e2e.max_rank_window_elapsed_s.toFixed(4)} s（rank {report.e2e.slowest_window_rank}）；非逐步max之和。update-only={report.updateTrajectories.toFixed(3)} trajectories/s。bs1上轮只有update-only，不可比较E2E。</Text>
      <Pie title="完整循环互斥耗时 · 最慢窗口 rank 的20步均值" data={report.e2eParts} unit="ms" caption="来源：summary.e2e.critical_window_mean_wall_s。next含Accelerate implicit H2D；postupdate含遥测/日志/跨rank scalar reduce。residual=iteration−next−update−postupdate，包含beforeupdate同步、worker计时提取及诊断wrapper/export等未入update开销；不漏记诊断开始前同步。H2D/telemetry是子项，不再相加。" />
      <Table headers={["Rank", "next ms", "update ms", "postupdate ms", "other ms", "telemetry direct ms", "H2D enqueue ms"]} rows={Object.entries(report.e2e.per_rank).map(([r, d]) => [r, ...Object.values(d.mean_wall_s).map(v => (v * 1000).toFixed(3)), (d.telemetry_direct_wall_s.mean * 1000).toFixed(3), (d.h2d_enqueue_wall_s.mean * 1000).toFixed(3)])} />
      <Text size="small" tone="secondary">上述next/postupdate为稳态均值；telemetry仅rank0直接调用，其余rank可在scalar reduce或DDP间接等待，但未独立测量，不能把其全部postupdate归因遥测。worker getitem/cache/frames/xform在预取worker并行，与训练重叠，不可与next或E2E相加。H2D enqueue非GPU传输完成时间；诊断31–40同步H2D不可冒充稳态。</Text>
      {Object.values(report.e2e.per_rank).some(d => Object.keys(d.worker_timing).length > 0) && <Table headers={["Rank", "worker getitem mean ms", "cache mean ms", "frames mean ms", "xform mean ms"]} rows={Object.entries(report.e2e.per_rank).filter(([, d]) => Object.keys(d.worker_timing).length > 0).map(([r, d]) => [r, ...["getitem_s", "cache_s", "frames_s", "xform_s"].map(k => d.worker_timing[k] ? (d.worker_timing[k].mean * 1000).toFixed(3) : "未采集")])} />}
    </Stack>}
    {report.modules.length > 0 && <Stack gap={10}>
      <H2>ViT visual_only 与 prefix 范围分别列出</H2>
      <Table headers={["范围", "per-call mean ms", "p95 ms", "calls"]} rows={report.modules.map(m => [m.name, m.mean.toFixed(3), m.p95.toFixed(3), m.count])} />
      <Text size="small" tone="secondary">来源：summary.modules_inclusive，schema3诊断35–40步、四rank。visual_only只包vlm_visual.forward；prefix/vision包括_vision周边处理；build_prefix包含完整prefix。inclusive CUDA流区间有父子重叠，不相加、不是E2E exclusive kernel耗时。</Text>
    </Stack>}
    <Text size="small">bs1仅update-only可比；{report.comparison ? `本次/上轮轨迹吞吐比=${report.comparison.ratio.toFixed(4)}，提升=${report.comparison.improvement_percent.toFixed(2)}%（非E2E）` : "未提供bs1完成报告，不推算提升百分比。"}</Text>
    <Table headers={["Rank", "稳态 peak allocated (GiB)", "稳态 peak reserved (GiB)"]} rows={report.peaks.map(p => [p.rank, gib(p.allocated), gib(p.reserved)])} columnAlign={["left", "right", "right"]} />
    {(report.time.length > 0 || report.allocated.length > 0) && <Grid columns="repeat(auto-fit, minmax(360px, 1fr))" gap={32}>
      {report.time.length > 0 && <Pie title="诊断阶段时间占比（非稳态分解）" data={report.time} unit="ms" caption="来源：summary.time_share；steps 31–34 × 四卡，16 个 rank-step 等权均值。同步 sibling 阶段互不重叠；other residual = update wall − 阶段和。阶段前后 CUDA synchronize，含末端同步/DDP 等待；instrumentation 有扰动，不可直接推导加速比。" />}
      {report.allocated.length > 0 && snapshot && <Pie title="同刻 total CUDA allocated 分解" data={report.allocated} unit="GiB" caption={`来源：summary.memory_share.total_allocated_decomposition；rank ${snapshot.rank} / step ${snapshot.step} / ${snapshot.device} 的 forward-end 单快照。storage 无交叉，分类和等于 allocated；非 reserved、非全步峰值。other_residual 含 buffer、batch、未保存 activation、DDP bucket、workspace、allocator rounding 等，不强行归因。`} />}
    </Grid>}
    {report.saved.length > 0 && snapshot && <Stack gap={12}>
      <H2>前向末端 saved activation · 模块 storage 归因</H2>
      <Text size="small" tone="secondary">来源：summary.memory_share.saved_snapshot；rank {snapshot.rank} / step {snapshot.step} 单一 forward-end 快照。横轴：saved storage（GiB）；纵轴：原始模块名；单系列：saved storage。占比分母仅 saved total {gib(snapshot.savedTotal)} GiB，不是总显存/峰值。saved 总量 ≤ 同刻 allocated {gib(snapshot.allocated)} GiB。</Text>
      {report.saved.map(d => <Grid key={d.label} columns="minmax(130px, 2fr) minmax(80px, 3fr) 130px" gap={12} align="center">
        <Text size="small" style={{ overflowWrap: "anywhere" }}>{d.label}</Text>
        <div style={{ height: 10, background: theme.fill.tertiary }}><div style={{ height: 10, width: `${100 * d.value / report.saved[0].value}%`, background: theme.accent.primary }} /></div>
        <Text size="small" style={{ textAlign: "right" }}>{gib(d.value)} GiB · {(100 * d.value / snapshot.savedTotal).toFixed(1)}%</Text>
      </Grid>)}
      <Text size="small" tone="secondary">条长以最大模块为上限；右侧显示真实 GiB 与 saved 内占比。排除参数/buffer/batch，storage 去重，归首个存活 save 的最深函数/模块；弱引用 wrapper 失效可能低估 saved。非模块时间，不求和父子 inclusive CUDA 时间；总分解只计选定 CUDA device，saved 图保留报告原始归因口径。</Text>
    </Stack>}
    <Divider />
    <Stack gap={6}>
      <Text size="small" tone="secondary">离线来源：{report.source}</Text>
      <Text size="small" tone="tertiary" style={{ overflowWrap: "anywhere" }}>summary SHA-256：{report.sha256} · 内嵌数据，无网络请求。未读取采集目录补充信息；无正值的可选图省略，不补零、不推算旧采集 schema 的同刻分解。</Text>
    </Stack>
  </Stack>;
}
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description='将已完成的 report summary 转为离线 Cursor canvas')
    parser.add_argument('summary', type=Path, help='已完成报告的 summary.json，不接受采集目录')
    parser.add_argument('--output', type=Path, help='直接写入 .canvas.tsx（不创建目录）')
    args = parser.parse_args(argv)
    try:
        source = args.summary.expanduser().resolve()
        require(source.name == 'summary.json', '输入必须为 report summary.json')
        raw = source.read_bytes()
        payload = normalize(json.loads(raw), source, hashlib.sha256(raw).hexdigest())
        output = (args.output or (E2E_OUTPUT if payload['schema'] == 3 else OUTPUT)).expanduser().resolve()
        require(str(output).endswith('.canvas.tsx'), '输出须以 .canvas.tsx 结尾')
        require(not output.exists(), f'输出已存在，保留旧画布、拒绝覆盖：{output}')
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        encoded = encoded.replace('<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
        output.write_text(TEMPLATE.replace('__REPORT__', encoded), encoding='utf-8')
        print(f'已生成离线画布：{output}')
        return 0
    except (SummaryError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f'错误：summary 架构不匹配或文件不可用：{exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
