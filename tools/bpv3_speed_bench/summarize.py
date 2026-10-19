"""把 results/k*.json 汇总成 markdown 表格。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "results")
runs = sorted((json.loads(p.read_text()) for p in root.glob("k*.json")), key=lambda r: r["num_chunks"])


def ratio(raw, comp):
    return f"{raw / comp:.2f}x"


print("## inference e2e (wall, ms)")
for batch in (1, 8, 32):
    print(f"\n### B={batch}")
    print("| K | BP tokens comp/raw | prefix comp/raw | comp mean | comp p90 | raw mean | raw p90 | raw/comp | "
          "GPU comp | GPU raw | GPU raw/comp | mem comp | mem raw |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        c = r["inference"].get(f"compressed_b{batch}")
        w = r["inference"].get(f"raw_b{batch}")
        if not c or not w:
            continue
        print(
            f"| {r['num_chunks']} | {c['bp_tokens']}/{w['bp_tokens']} | {c['prefix_len']}/{w['prefix_len']} | "
            f"{c['e2e_ms']['mean']:.1f} | {c['e2e_ms']['p90']:.1f} | {w['e2e_ms']['mean']:.1f} | {w['e2e_ms']['p90']:.1f} | "
            f"{ratio(w['e2e_ms']['mean'], c['e2e_ms']['mean'])} | {c['gpu_kernel_ms']['total']:.1f} | "
            f"{w['gpu_kernel_ms']['total']:.1f} | {ratio(w['gpu_kernel_ms']['total'], c['gpu_kernel_ms']['total'])} | "
            f"{c['peak_mem_gb']:.2f} | {w['peak_mem_gb']:.2f} |"
        )

print("\n## inference stage breakdown (wall ms with sync / GPU kernel ms)")
for batch in (1, 8, 32):
    print(f"\n### B={batch}")
    print("| K | mode | ViT | compressor | VLM prefill | denoise x10 | GPU ViT | GPU comp | GPU prefill | GPU denoise |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        for mode in ("compressed", "raw"):
            x = r["inference"].get(f"{mode}_b{batch}")
            if not x:
                continue
            s, g = x["stages_ms"], x["gpu_kernel_ms"]
            get = lambda d, k: f"{d[k]['mean']:.1f}" if k in d else "-"  # noqa: E731
            gg = lambda k: f"{g[k]:.1f}" if k in g else "-"  # noqa: E731
            print(
                f"| {r['num_chunks']} | {mode} | {get(s, 'vit')} | {get(s, 'bp_compressor')} | {get(s, 'vlm_prefill')} | "
                f"{get(s, 'denoise_10steps')} | {gg('vit')} | {gg('bp_compressor')} | {gg('vlm_prefill')} | "
                f"{gg('denoise_10steps')} |"
            )

print("\n## training fwd+bwd")
print("| K | B | comp step ms | raw step ms | raw/comp | GPU comp | GPU raw | GPU raw/comp | "
      "GPU fwd ViT comp/raw | GPU fwd VLM+expert comp/raw | mem comp | mem raw |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in runs:
    for batch in (8, 16, 32):
        c = r["training"].get(f"compressed_b{batch}")
        w = r["training"].get(f"raw_b{batch}")
        if not c or not w:
            continue
        if c.get("oom") or w.get("oom"):
            fmt = lambda x: "OOM" if x.get("oom") else f"{x['step_ms']['mean']:.0f}"  # noqa: E731
            print(f"| {r['num_chunks']} | {batch} | {fmt(c)} | {fmt(w)} | - | - | - | - | - | - | - | - |")
            continue
        gc, gw = c["gpu_kernel_ms"], w["gpu_kernel_ms"]
        print(
            f"| {r['num_chunks']} | {batch} | {c['step_ms']['mean']:.0f} | {w['step_ms']['mean']:.0f} | "
            f"{ratio(w['step_ms']['mean'], c['step_ms']['mean'])} | {gc['total']:.0f} | {gw['total']:.0f} | "
            f"{ratio(gw['total'], gc['total'])} | {gc.get('vit', 0):.0f}/{gw.get('vit', 0):.0f} | "
            f"{gc.get('train_vlm_expert_fwd', 0):.0f}/{gw.get('train_vlm_expert_fwd', 0):.0f} | "
            f"{c['peak_mem_gb']:.1f} | {w['peak_mem_gb']:.1f} |"
        )
