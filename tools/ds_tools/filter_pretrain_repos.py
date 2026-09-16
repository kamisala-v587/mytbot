#!/usr/bin/env python3
"""按 episode 时长 / 打包 mp4 跨度 / bench slow_samples 过滤 LeRobot repo-id 列表。

修改下方大写常量后运行：

  cd /home/jovyan/workspace/mytbot
  /home/jovyan/conda-envs/bptbot/bin/python tools/ds_tools/filter_pretrain_repos.py

只要命中任意一条已启用规则（OR），该 repo 就会被剔除。将阈值设为 None /
路径设为空字符串可关闭对应规则。

mean_ep_s  = total_frames / total_episodes / fps          (meta/info.json)
max_mp4_s  = 在 meta/episodes/*.parquet 中，对各 video file_index 取 (to_ts - from_ts) 的最大值
slow_n     = 该 repo_id 在 bpva_benchmark 的 slow_samples.jsonl 中出现的次数

cd /home/jovyan/workspace/mytbot
/home/jovyan/conda-envs/bptbot/bin/python tools/ds_tools/filter_pretrain_repos.py
"""

from __future__ import annotations

import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# =========================
# 可编辑的过滤设置
# =========================
# 输入 / 输出（每行一个 repo 路径；空行与 # 开头行会被忽略）
INPUT_TXT = Path("/home/jovyan/workspace/mytbot/configs/ds_ids/B200/pretrain-data1.txt")
OUTPUT_TXT = Path("/home/jovyan/workspace/mytbot/configs/ds_ids/B200/pretrain-data1_filteredVideo.txt")
# 若为 None，则写在 OUTPUT_TXT 同目录下，文件名为 "<stem>.removed.txt"
REMOVED_TXT: Path | None = None

# --- 全局规则（平均 episode 时长 >= 阈值则剔除）---
# 示例：data3 用 30；data3.1 用 40。设为 None 可关闭。
MEAN_EP_S_MIN: float | None = 55.0

# --- 按 domain 的 mean_ep 规则 ---
# egodex：mean_ep >= 该值则剔除（data3: 15；data3.1: 25）。None=关闭。
EGODEX_MEAN_EP_S_MIN: float | None = 55.0
# InternData：mean_ep >= 该值则剔除（data3: 25；data3.1: 40）。None=关闭。
INTERNDATA_MEAN_EP_S_MIN: float | None = 55.0

# --- InternData 打包 mp4 跨度（秒）---
# InternData：最大打包 mp4 时长 >= 该值则剔除（data3: 1500；data3.1: 4000）。
# 仅对 InternData 读取 episode parquet。None=关闭。
INTERNDATA_MAX_MP4_S_MIN: float | None = 1500.0

# --- 可选：来自 bpva_benchmark 的经验 denylist ---
# prove_* 运行产物中的 slow_samples.jsonl 路径；空 / None 则关闭。
SLOW_SAMPLES_JSONL: Path | str | None = (
    "/home/jovyan/workspace/mytbot/outputs/bpva_benchmark/"
    "prove_bp_encode_pretrained32/2026-09-12/14-21-22-834279/slow_samples.jsonl"
)
# 该 repo 在 slow_samples 中出现次数 >= N 则剔除（data3: 1；data3.1: 20）。
SLOW_N_MIN: int | None = 20

# meta / parquet 扫描的并行度
MAX_WORKERS = min(32, os.cpu_count() or 8)

# 为 True 时只打印计划，不写 OUTPUT_TXT / REMOVED_TXT
DRY_RUN = False

# 可选：按 repo 缓存统计结果的 JSON（加速重复运行）。空 / None 则关闭。
STATS_CACHE_JSON: Path | str | None = "/tmp/pretrain_repo_filter_stats.json"


# ---------------------------------------------------------------------------
# 实现
# ---------------------------------------------------------------------------


@dataclass
class RepoStats:
    mean_ep_s: float | None
    max_mp4_s: float | None
    mean_mp4_s: float | None
    total_frames: int
    total_episodes: int
    fps: float
    err: str | None = None


def _read_repo_list(path: Path) -> list[str]:
    repos: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        repos.append(line)
    return repos


def _domain(repo: str) -> str:
    if "egodex" in repo:
        return "egodex"
    if "InternData" in repo:
        return "InternData"
    if "agibot" in repo:
        return "agibot"
    if "RoboChallenge" in repo:
        return "RoboChallenge"
    if "RoboTwin" in repo:
        return "RoboTwin"
    return "other"


def _need_mp4_stats() -> bool:
    return INTERNDATA_MAX_MP4_S_MIN is not None


def _mp4_span_from_episodes(repo: Path) -> tuple[float | None, float | None, str | None]:
    """返回所有相机打包 mp4 跨度上的 (max_file_s, mean_file_s, err)。

    对每个 ``videos/<cam>/file_index``，按时长 ``max(to_ts)-min(from_ts)``
    计算每个 file。repo 级 max 取最差相机/文件（与此前过滤运行一致）。
    """
    try:
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        return None, None, f"parquet_deps:{exc}"

    ep_files = list((repo / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        return None, None, "no_episodes"

    schema_cols = pq.read_schema(ep_files[0]).names
    file_cols = [c for c in schema_cols if c.startswith("videos/") and c.endswith("/file_index")]
    if not file_cols:
        return None, None, "no_video_cols"

    need = ["episode_index"]
    cam_specs: list[tuple[str, str, str]] = []
    for fic_col in file_cols:
        key = fic_col.removesuffix("/file_index")
        fr_col = f"{key}/from_timestamp"
        to_col = f"{key}/to_timestamp"
        if fr_col in schema_cols and to_col in schema_cols:
            cam_specs.append((fic_col, fr_col, to_col))
            need.extend([fic_col, fr_col, to_col])
    if not cam_specs:
        return None, None, "missing_ts_cols"

    df = pd.concat([pd.read_parquet(e, columns=need) for e in ep_files], ignore_index=True)
    max_file_s = 0.0
    mean_list: list[float] = []
    for fic_col, fr_col, to_col in cam_specs:
        g = df.groupby(fic_col).agg(t0=(fr_col, "min"), t1=(to_col, "max"))
        dur = (g["t1"] - g["t0"]).astype(float)
        if len(dur) == 0:
            continue
        max_file_s = max(max_file_s, float(dur.max()))
        mean_list.append(float(dur.mean()))
    if not mean_list:
        return None, None, "empty_files"
    return max_file_s, float(sum(mean_list) / len(mean_list)), None


def _compute_repo_stats(repo: str, want_mp4: bool = False) -> tuple[str, dict[str, Any]]:
    p = Path(repo)
    info_path = p / "meta" / "info.json"
    if not info_path.is_file():
        st = RepoStats(None, None, None, 0, 0, 0.0, err="missing_info")
        return repo, asdict(st)

    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info.get("fps") or 0.0)
    ep = int(info.get("total_episodes") or 0)
    fr = int(info.get("total_frames") or 0)
    mean_ep = (fr / ep / fps) if fps > 0 and ep > 0 else None

    max_mp4 = mean_mp4 = None
    err = None
    if want_mp4 and "InternData" in repo:
        max_mp4, mean_mp4, err = _mp4_span_from_episodes(p)

    st = RepoStats(
        mean_ep_s=mean_ep,
        max_mp4_s=max_mp4,
        mean_mp4_s=mean_mp4,
        total_frames=fr,
        total_episodes=ep,
        fps=fps,
        err=err,
    )
    return repo, asdict(st)


def _load_slow_counts(path: Path | str | None) -> Counter[str]:
    if not path:
        return Counter()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"SLOW_SAMPLES_JSONL not found: {p}")
    counts: Counter[str] = Counter()
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            repo = obj.get("repo_id")
            if repo:
                counts[repo] += 1
    return counts


def _match_reasons(repo: str, st: RepoStats, slow_n: int) -> list[str]:
    reasons: list[str] = []
    mean_ep = st.mean_ep_s

    if SLOW_N_MIN is not None and slow_n >= SLOW_N_MIN:
        reasons.append(f"slow_n={slow_n}>={SLOW_N_MIN}")

    if MEAN_EP_S_MIN is not None and mean_ep is not None and mean_ep >= MEAN_EP_S_MIN:
        reasons.append(f"mean_ep>={MEAN_EP_S_MIN:g}")

    if (
        EGODEX_MEAN_EP_S_MIN is not None
        and "egodex" in repo
        and mean_ep is not None
        and mean_ep >= EGODEX_MEAN_EP_S_MIN
    ):
        reasons.append(f"egodex_mean_ep>={EGODEX_MEAN_EP_S_MIN:g}")

    if (
        INTERNDATA_MEAN_EP_S_MIN is not None
        and "InternData" in repo
        and mean_ep is not None
        and mean_ep >= INTERNDATA_MEAN_EP_S_MIN
    ):
        reasons.append(f"InternData_mean_ep>={INTERNDATA_MEAN_EP_S_MIN:g}")

    if (
        INTERNDATA_MAX_MP4_S_MIN is not None
        and "InternData" in repo
        and st.max_mp4_s is not None
        and st.max_mp4_s >= INTERNDATA_MAX_MP4_S_MIN
    ):
        reasons.append(f"InternData_max_mp4>={INTERNDATA_MAX_MP4_S_MIN:g}")

    return reasons


def _gather_stats(repos: list[str]) -> dict[str, RepoStats]:
    want_mp4 = _need_mp4_stats()
    cache_path = Path(STATS_CACHE_JSON) if STATS_CACHE_JSON else None
    cached: dict[str, Any] = {}
    if cache_path and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))

    out: dict[str, RepoStats] = {}
    todo: list[str] = []
    for repo in repos:
        hit = cached.get(repo)
        cache_ok = hit is not None and (
            not want_mp4
            or "InternData" not in repo
            or hit.get("max_mp4_s") is not None
            or hit.get("err")
            in {"no_episodes", "no_video_cols", "missing_ts_cols", "empty_files"}
        )
        if cache_ok:
            out[repo] = RepoStats(
                mean_ep_s=hit.get("mean_ep_s"),
                max_mp4_s=hit.get("max_mp4_s"),
                mean_mp4_s=hit.get("mean_mp4_s"),
                total_frames=int(hit.get("total_frames") or 0),
                total_episodes=int(hit.get("total_episodes") or 0),
                fps=float(hit.get("fps") or 0.0),
                err=hit.get("err"),
            )
        else:
            todo.append(repo)

    if todo:
        print(f"Computing stats for {len(todo)}/{len(repos)} repos (want_mp4={want_mp4})...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(_compute_repo_stats, r, want_mp4) for r in todo]
            done = 0
            for fut in as_completed(futs):
                repo, payload = fut.result()
                out[repo] = RepoStats(**payload)
                cached[repo] = payload
                done += 1
                if done % 200 == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached), encoding="utf-8")
        print(f"Wrote stats cache: {cache_path}")

    return out


def main() -> None:
    input_txt = Path(INPUT_TXT)
    output_txt = Path(OUTPUT_TXT)
    removed_txt = (
        Path(REMOVED_TXT)
        if REMOVED_TXT is not None
        else output_txt.with_name(output_txt.stem + ".removed.txt")
    )

    repos = _read_repo_list(input_txt)
    slow_counts = _load_slow_counts(SLOW_SAMPLES_JSONL)
    stats = _gather_stats(repos)

    kept: list[str] = []
    removed_rows: list[tuple[str, RepoStats, list[str], int]] = []
    for repo in repos:
        st = stats[repo]
        n = int(slow_counts.get(repo, 0))
        reasons = _match_reasons(repo, st, n)
        if reasons:
            removed_rows.append((repo, st, reasons, n))
        else:
            kept.append(repo)

    kept_frames = sum(stats[r].total_frames for r in kept)
    removed_frames = sum(st.total_frames for _, st, _, _ in removed_rows)
    in_frames = kept_frames + removed_frames
    rem_dom = Counter(_domain(r) for r, _, _, _ in removed_rows)
    keep_dom = Counter(_domain(r) for r in kept)

    print(
        f"\n{input_txt.name}: {len(repos)} repos, {in_frames:,} frames\n"
        f"  kept={len(kept)} ({kept_frames:,} frames, {kept_frames / 1e4:.1f} 万)\n"
        f"  removed={len(removed_rows)} ({removed_frames:,} frames)\n"
        f"  kept_by_domain={dict(keep_dom)}\n"
        f"  removed_by_domain={dict(rem_dom)}"
    )

    header = [
        f"# Filtered by tools/ds_tools/filter_pretrain_repos.py",
        f"# input: {input_txt}",
        f"# output: {output_txt}",
        f"# removed_at: {datetime.now().isoformat(timespec='seconds')}",
        f"# RULES (OR):",
        f"#   MEAN_EP_S_MIN={MEAN_EP_S_MIN}",
        f"#   EGODEX_MEAN_EP_S_MIN={EGODEX_MEAN_EP_S_MIN}",
        f"#   INTERNDATA_MEAN_EP_S_MIN={INTERNDATA_MEAN_EP_S_MIN}",
        f"#   INTERNDATA_MAX_MP4_S_MIN={INTERNDATA_MAX_MP4_S_MIN}",
        f"#   SLOW_SAMPLES_JSONL={SLOW_SAMPLES_JSONL}",
        f"#   SLOW_N_MIN={SLOW_N_MIN}",
        f"# count_removed: {len(removed_rows)}  count_kept: {len(kept)}",
        f"# frames_kept: {kept_frames}  frames_removed: {removed_frames}",
        f"# removed_by_domain: {dict(rem_dom)}",
        f"# kept_by_domain: {dict(keep_dom)}",
        "# format: slow_n\\tmean_ep_s\\tmax_mp4_s\\treasons\\tpath",
    ]
    body = []
    for repo, st, reasons, n in sorted(
        removed_rows, key=lambda x: (-x[3], -(x[1].mean_ep_s or 0.0))
    ):
        mean_ep = f"{st.mean_ep_s:.2f}" if st.mean_ep_s is not None else "na"
        max_mp4 = f"{st.max_mp4_s:.1f}" if st.max_mp4_s is not None else "na"
        body.append(f"{n}\t{mean_ep}\t{max_mp4}\t{'|'.join(reasons)}\t{repo}")

    if DRY_RUN:
        print("DRY_RUN=True: not writing files.")
        return

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    removed_txt.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    print(f"Wrote kept:    {output_txt}")
    print(f"Wrote removed: {removed_txt}")


if __name__ == "__main__":
    main()
