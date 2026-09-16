#!/usr/bin/env python3
"""统计 LeRobot v3.0 数据集的帧数、轨迹、视频与任务信息。

从 configs/ds_ids 下的 txt（每行一个数据集路径）读取列表，将每个数据集的
统计结果写入 pandas DataFrame，再导出 CSV，并在终端打印全局汇总。

CSV 列按「筛选长尾」优先排序：帧数 / 轨迹长度 / 平均视频时长与大小。
可通过 WEIGHT_RULES_YAML 为每个 repo 匹配所属组（与训练 weight_rules 一致）。

Usage:
  python tools/ds_tools/compute_meta_repos.py
  # 或先改脚本顶部大写常量，再直接运行。
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# 配置（优先改这里；命令行未接入，保证一处配置即可跑）
# ---------------------------------------------------------------------------
DS_IDS_FILE = Path(
    "/home/jovyan/workspace/mytbot/configs/ds_ids/B200/pretrain-data.txt"
)
# 相对路径会拼到该 root 下；绝对路径不受影响。设为 None 表示不做拼接。
REPO_ROOT: Path | None = None
OUTPUT_CSV = Path("/home/jovyan/workspace/mytbot/configs/ds_ids/B200/pretrain-data.csv")
# repo 分组规则（与训练 weight_rules 一致；None 则不填 group 列）
WEIGHT_RULES_YAML = Path("/home/jovyan/workspace/mytbot/configs/ds_ids/weight_rules.yaml")
# 是否用 PyAV 打开每个 mp4 读取真实时长（大库会慢；False 则时长相关列留空）
PROBE_MP4_DURATION = True
MP4_PROBE_WORKERS = 16
# 只统计 videos/ 下的 mp4；若数据集无 videos 目录则记 0
VIDEO_GLOB = "videos/**/*.mp4"
LOG_LEVEL = logging.INFO

# CSV 列顺序：核心筛选列靠前；fps 放末尾
CSV_COLUMNS = [
    # 核心列
    "repo",
    "group",
    "total_frames",
    "total_episodes",
    "avg_ep_duration_s",
    "median_ep_duration_s",
    "p95_ep_duration_s",
    "avg_mp4_size_mb",
    "total_tasks",
    # 轨迹帧数
    "avg_ep_frames",
    "median_ep_frames",
    "p95_ep_frames",
    "min_ep_frames",
    "max_ep_frames",
    # 轨迹时长（秒）
    "min_ep_duration_s",
    "max_ep_duration_s",
    # 视频文件
    "mp4_count",
    "n_video_keys",
    "avg_mp4_duration_s",
    "median_mp4_duration_s",
    "p95_mp4_duration_s",
    "max_mp4_duration_s",
    "median_mp4_size_mb",
    "p95_mp4_size_mb",
    "max_mp4_size_mb",
    "mp4_total_gb",
    "mp4_total_duration_h",
    # 末尾元信息
    "fps",
    "codebase_version",
    "resolved_path",
]

# 导出 CSV 时的数值格式（避免过长小数）
INT_COLUMNS = {
    "total_frames",
    "total_episodes",
    "total_tasks",
    "mp4_count",
    "n_video_keys",
}
DURATION_COLUMNS = {
    "avg_ep_duration_s",
    "median_ep_duration_s",
    "p95_ep_duration_s",
    "min_ep_duration_s",
    "max_ep_duration_s",
    "avg_mp4_duration_s",
    "median_mp4_duration_s",
    "p95_mp4_duration_s",
    "max_mp4_duration_s",
}
SIZE_MB_COLUMNS = {
    "avg_mp4_size_mb",
    "median_mp4_size_mb",
    "p95_mp4_size_mb",
    "max_mp4_size_mb",
}
FRAME_COLUMNS = {
    "avg_ep_frames",
    "median_ep_frames",
    "p95_ep_frames",
    "min_ep_frames",
    "max_ep_frames",
}

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stats_lerobot_v30")


def read_repo_list(path: Path) -> list[str]:
    repos: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        repos.append(line)
    return repos


def resolve_repo(repo: str, root: Path | None) -> Path:
    repo_path = Path(repo).expanduser()
    if not repo_path.is_absolute() and root is not None:
        repo_path = root.expanduser() / repo_path
    return repo_path


def _load_weight_rule_groups(path: Path | None) -> list[tuple[str, str]]:
    """读取 weight_rules.yaml，返回 [(group_name, regex), ...]。"""
    if path is None:
        return []
    yaml_path = path.expanduser()
    if not yaml_path.is_file():
        logger.warning("WEIGHT_RULES_YAML not found: %s (group 列留空)", yaml_path)
        return []

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    groups = cfg.get("groups") or []
    rule_groups: list[tuple[str, str]] = []
    for item in groups:
        name = str(item.get("name", "")).strip()
        pattern = str(item.get("match", "")).strip()
        if name and pattern:
            rule_groups.append((name, pattern))
    return rule_groups


def _match_repo_group(repo: str, rule_groups: list[tuple[str, str]]) -> str:
    """按 weight_rules 顺序匹配 repo；未命中则返回 default。"""
    for name, pattern in rule_groups:
        if re.search(pattern, repo):
            return name
    return "default"


def build_repo_group_map(repos: list[str], rules_path: Path | None) -> dict[str, str]:
    rule_groups = _load_weight_rule_groups(rules_path)
    if not rule_groups:
        return {}
    return {repo: _match_repo_group(repo, rule_groups) for repo in repos}


def _fmt_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def _fmt_hours(seconds: float) -> str:
    hours = seconds / 3600.0
    return f"{hours:.3f} h ({hours * 60.0:.1f} min)"


def _round_value(col: str, value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if col in INT_COLUMNS:
        return int(round(float(value)))
    if col in DURATION_COLUMNS:
        return round(float(value), 1)
    if col in SIZE_MB_COLUMNS:
        return round(float(value), 1)
    if col in FRAME_COLUMNS:
        return int(round(float(value)))
    if col == "mp4_total_gb":
        return round(float(value), 2)
    if col == "mp4_total_duration_h":
        return round(float(value), 2)
    if col == "fps":
        fps = float(value)
        return int(fps) if abs(fps - round(fps)) < 1e-6 else round(fps, 1)
    return value


def _format_row_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    return {col: _round_value(col, row.get(col)) for col in CSV_COLUMNS}


def _series_stats(values: list[float] | pd.Series) -> dict[str, float | None]:
    """返回 avg / median / p95 / min / max；空则全 None。"""
    if values is None:
        return {k: None for k in ("avg", "median", "p95", "min", "max")}
    s = values if isinstance(values, pd.Series) else pd.Series(values, dtype="float64")
    s = s.dropna()
    if len(s) == 0:
        return {k: None for k in ("avg", "median", "p95", "min", "max")}
    return {
        "avg": float(s.mean()),
        "median": float(s.median()),
        "p95": float(s.quantile(0.95)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def _probe_mp4_duration(path: Path) -> float:
    import av

    with av.open(str(path)) as container:
        video_stream = container.streams.video[0]
        if video_stream.duration is not None and video_stream.time_base is not None:
            return float(video_stream.duration * video_stream.time_base)
        if container.duration is not None:
            return float(container.duration / av.time_base)
    return 0.0


def _sum_mp4_stats(
    repo_path: Path, probe_duration: bool, workers: int
) -> tuple[int, list[int], list[float] | None]:
    """返回 (mp4_count, sizes_bytes[], durations_s[]|None)."""
    videos_root = repo_path / "videos"
    if not videos_root.is_dir():
        return 0, [], [] if probe_duration else None

    mp4_files = sorted(videos_root.rglob("*.mp4"))
    count = len(mp4_files)
    sizes = [int(p.stat().st_size) for p in mp4_files]

    if not probe_duration:
        return count, sizes, None
    if count == 0:
        return 0, [], []

    durations: list[float] = [0.0] * count
    workers = max(1, int(workers))
    if workers == 1 or count == 1:
        for i, p in enumerate(mp4_files):
            durations[i] = _probe_mp4_duration(p)
        return count, sizes, durations

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_probe_mp4_duration, p): i for i, p in enumerate(mp4_files)}
        for fut in as_completed(futures):
            durations[futures[fut]] = float(fut.result())
    return count, sizes, durations


def _load_episode_lengths(repo_path: Path) -> pd.Series | None:
    ep_dir = repo_path / "meta" / "episodes"
    if not ep_dir.is_dir():
        return None
    frames: list[pd.Series] = []
    for parquet_path in sorted(ep_dir.rglob("*.parquet")):
        try:
            part = pd.read_parquet(parquet_path, columns=["length"])
        except Exception:
            # 个别旧文件可能缺 length，整表读再取
            part = pd.read_parquet(parquet_path)
            if "length" not in part.columns:
                continue
            part = part[["length"]]
        frames.append(part["length"])
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _count_tasks(repo_path: Path, info: dict[str, Any]) -> int:
    tasks_path = repo_path / "meta" / "tasks.parquet"
    if tasks_path.is_file():
        try:
            tasks = pd.read_parquet(tasks_path)
            return int(len(tasks))
        except Exception:
            pass
    return int(info.get("total_tasks", 0) or 0)


def _failed_row(repo: str) -> dict[str, Any]:
    """失败 repo 只保留 repo，其余列留空。"""
    return {"repo": repo, "_ok": False, "_error": None}


def _empty_ok_row(repo: str, repo_path: Path) -> dict[str, Any]:
    return {col: None for col in CSV_COLUMNS} | {
        "repo": repo,
        "resolved_path": str(repo_path),
        "_ok": True,
        "_error": None,
    }


def stats_one_repo(repo: str, root: Path | None) -> dict[str, Any]:
    repo_path = resolve_repo(repo, root)

    info_path = repo_path / "meta" / "info.json"
    if not info_path.is_file():
        return _failed_row(repo) | {
            "_error": f"missing meta/info.json: {info_path}",
        }

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _failed_row(repo) | {"_error": f"failed to read info.json: {exc}"}

    row = _empty_ok_row(repo, repo_path)

    version = str(info.get("codebase_version", "") or "")
    row["codebase_version"] = version
    if version and not version.lstrip("v").startswith("3"):
        logger.warning("%s codebase_version=%s (期望 v3.x)", repo, version)

    fps = float(info.get("fps", 0) or 0)
    total_frames = int(info.get("total_frames", 0) or 0)
    total_episodes = int(info.get("total_episodes", 0) or 0)
    features = info.get("features") or {}
    video_keys = [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"]

    lengths = _load_episode_lengths(repo_path)
    if lengths is not None and len(lengths) > 0:
        ep_frame_stats = _series_stats(lengths.astype(float))
        if total_episodes <= 0:
            total_episodes = int(len(lengths))
        if total_frames <= 0:
            total_frames = int(lengths.sum())
        if fps > 0:
            ep_dur_stats = _series_stats(lengths.astype(float) / fps)
        else:
            ep_dur_stats = _series_stats([])
    else:
        avg_len = (total_frames / total_episodes) if total_episodes > 0 else None
        ep_frame_stats = {
            "avg": avg_len,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
        if avg_len is not None and fps > 0:
            ep_dur_stats = {
                "avg": avg_len / fps,
                "median": None,
                "p95": None,
                "min": None,
                "max": None,
            }
        else:
            ep_dur_stats = _series_stats([])

    try:
        mp4_count, sizes, durations = _sum_mp4_stats(
            repo_path, PROBE_MP4_DURATION, MP4_PROBE_WORKERS
        )
    except Exception as exc:
        return _failed_row(repo) | {"_error": f"mp4 stats failed: {exc}"}

    size_stats = _series_stats([b / (1024.0**2) for b in sizes])  # MB
    dur_stats = _series_stats(durations) if durations is not None else _series_stats([])
    total_bytes = int(sum(sizes)) if sizes else 0
    total_dur_s = float(sum(durations)) if durations is not None else None

    row.update(
        {
            "_ok": True,
            "fps": fps,
            "total_frames": total_frames,
            "total_episodes": total_episodes,
            "avg_ep_frames": ep_frame_stats["avg"],
            "median_ep_frames": ep_frame_stats["median"],
            "p95_ep_frames": ep_frame_stats["p95"],
            "min_ep_frames": ep_frame_stats["min"],
            "max_ep_frames": ep_frame_stats["max"],
            "avg_ep_duration_s": ep_dur_stats["avg"],
            "median_ep_duration_s": ep_dur_stats["median"],
            "p95_ep_duration_s": ep_dur_stats["p95"],
            "min_ep_duration_s": ep_dur_stats["min"],
            "max_ep_duration_s": ep_dur_stats["max"],
            "total_tasks": _count_tasks(repo_path, info),
            "n_video_keys": len(video_keys),
            "mp4_count": mp4_count,
            "avg_mp4_duration_s": dur_stats["avg"],
            "median_mp4_duration_s": dur_stats["median"],
            "p95_mp4_duration_s": dur_stats["p95"],
            "max_mp4_duration_s": dur_stats["max"],
            "avg_mp4_size_mb": size_stats["avg"],
            "median_mp4_size_mb": size_stats["median"],
            "p95_mp4_size_mb": size_stats["p95"],
            "max_mp4_size_mb": size_stats["max"],
            "mp4_total_gb": total_bytes / (1024.0**3),
            "mp4_total_duration_h": (total_dur_s / 3600.0) if total_dur_s is not None else None,
        }
    )
    return row


def build_dataframe(
    repos: list[str],
    root: Path | None,
    repo_to_group: dict[str, str],
) -> tuple[pd.DataFrame, list[tuple[str, str | None]]]:
    ok_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    failed_details: list[tuple[str, str | None]] = []
    for i, repo in enumerate(repos, start=1):
        logger.info("[%d/%d] %s", i, len(repos), repo)
        raw = stats_one_repo(repo, root)
        if raw.get("_ok"):
            row = _format_row_for_csv(raw)
            row["group"] = repo_to_group.get(repo)
            ok_rows.append(row)
        else:
            failed_rows.append({"repo": repo})
            failed_details.append((repo, raw.get("_error")))

    df_ok = pd.DataFrame(ok_rows) if ok_rows else pd.DataFrame(columns=CSV_COLUMNS)
    df_failed = pd.DataFrame(failed_rows) if failed_rows else pd.DataFrame(columns=["repo"])

    for col in CSV_COLUMNS:
        if col not in df_ok.columns:
            df_ok[col] = None
        if col not in df_failed.columns:
            df_failed[col] = None

    df = pd.concat([df_ok, df_failed], ignore_index=True)
    return df[CSV_COLUMNS], failed_details


def print_summary(
    df: pd.DataFrame,
    source_txt: Path,
    failed_details: list[tuple[str, str | None]],
) -> None:
    ok = df[df["total_frames"].notna()]

    total_frames = int(ok["total_frames"].sum()) if len(ok) else 0
    total_episodes = int(ok["total_episodes"].sum()) if len(ok) else 0
    total_tasks = int(ok["total_tasks"].sum()) if len(ok) else 0
    total_mp4 = int(ok["mp4_count"].sum()) if len(ok) else 0
    total_gb = float(ok["mp4_total_gb"].sum()) if len(ok) else 0.0

    weighted_avg_frames = (total_frames / total_episodes) if total_episodes > 0 else float("nan")

    # 按 episode 数加权的平均轨迹时长
    if len(ok) and ok["avg_ep_duration_s"].notna().any() and total_episodes > 0:
        weighted_avg_ep_s = float(
            (ok["avg_ep_duration_s"].fillna(0) * ok["total_episodes"]).sum() / total_episodes
        )
    else:
        weighted_avg_ep_s = float("nan")

    mp4_hours = None
    if len(ok) and ok["mp4_total_duration_h"].notna().any():
        mp4_hours = float(ok["mp4_total_duration_h"].fillna(0).sum())

    # 全局平均单个 mp4 大小 / 时长（按文件数加权）
    if len(ok) and total_mp4 > 0:
        global_avg_mp4_mb = float(
            (ok["avg_mp4_size_mb"].fillna(0) * ok["mp4_count"]).sum() / total_mp4
        )
    else:
        global_avg_mp4_mb = float("nan")
    if len(ok) and ok["avg_mp4_duration_s"].notna().any() and total_mp4 > 0:
        global_avg_mp4_s = float(
            (ok["avg_mp4_duration_s"].fillna(0) * ok["mp4_count"]).sum() / total_mp4
        )
    else:
        global_avg_mp4_s = float("nan")

    fps_values = sorted({float(x) for x in ok["fps"].dropna().unique()}) if len(ok) else []

    print("=" * 60)
    print("LeRobot v3.0 数据集汇总（长尾筛选视角）")
    print("=" * 60)
    print(f"Source txt              : {source_txt}")
    print(f"Repo count              : {len(df)} (ok={len(ok)}, failed={len(failed_details)})")
    print(f"Total frames            : {total_frames:,}")
    print(f"Total trajectories      : {total_episodes:,}")
    print(f"Avg traj frames         : {weighted_avg_frames:,.2f}")
    print(f"Avg traj duration       : {weighted_avg_ep_s:,.2f} s")
    print(f"Total tasks (sum)       : {total_tasks:,}")
    if len(fps_values) == 1:
        print(f"FPS                     : {fps_values[0]:g}")
    elif fps_values:
        print(f"FPS values              : {fps_values}")
    print(f"MP4 files               : {total_mp4:,}")
    print(f"MP4 total size          : {total_gb:.3f} GB ({_fmt_bytes(total_gb * 1024**3)})")
    print(f"Avg mp4 size            : {global_avg_mp4_mb:,.2f} MB")
    print(f"Avg mp4 duration        : {global_avg_mp4_s:,.2f} s")
    if mp4_hours is not None:
        print(f"MP4 total duration      : {_fmt_hours(mp4_hours * 3600.0)}")
    else:
        print("MP4 total duration      : (skipped; set PROBE_MP4_DURATION=True)")

    if len(ok) and ok["group"].notna().any():
        print("\nBy group:")
        grouped = (
            ok.groupby("group", dropna=False)
            .agg(repos=("repo", "count"), frames=("total_frames", "sum"))
            .sort_values("frames", ascending=False)
        )
        for group_name, row in grouped.iterrows():
            print(f"  {group_name:16s} repos={int(row['repos']):4d}  frames={int(row['frames']):,}")

    if len(ok):
        # 方便快速看长尾：按 max_ep_duration_s / max_mp4_duration_s 排前几
        print("\nTop-5 by max episode duration (s):")
        top_ep = ok.nlargest(5, "max_ep_duration_s", keep="all")[
            ["group", "repo", "avg_ep_duration_s", "p95_ep_duration_s", "max_ep_duration_s", "total_frames"]
        ]
        print(top_ep.to_string(index=False))
        if ok["max_mp4_duration_s"].notna().any():
            print("\nTop-5 by max mp4 duration (s):")
            top_mp4 = ok.nlargest(5, "max_mp4_duration_s", keep="all")[
                [
                    "group",
                    "repo",
                    "avg_mp4_duration_s",
                    "p95_mp4_duration_s",
                    "max_mp4_duration_s",
                    "avg_mp4_size_mb",
                ]
            ]
            print(top_mp4.to_string(index=False))

    if failed_details:
        print(f"\nFailed repos ({len(failed_details)}):")
        for repo, err in failed_details:
            print(f"  - {repo}: {err or 'unknown error'}")
    print("=" * 60)


def main() -> None:
    txt_path = DS_IDS_FILE.expanduser()
    if not txt_path.is_file():
        raise FileNotFoundError(f"DS_IDS_FILE not found: {txt_path}")

    repos = read_repo_list(txt_path)
    if not repos:
        raise ValueError(f"empty repo list: {txt_path}")

    logger.info(
        "repos=%d probe_mp4=%s weight_rules=%s -> %s",
        len(repos),
        PROBE_MP4_DURATION,
        WEIGHT_RULES_YAML,
        OUTPUT_CSV,
    )
    repo_to_group = build_repo_group_map(repos, WEIGHT_RULES_YAML)
    df, failed_details = build_dataframe(repos, REPO_ROOT, repo_to_group)

    out = OUTPUT_CSV.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("wrote csv: %s (%d rows)", out, len(df))

    print_summary(df, txt_path, failed_details)
    print(f"Per-dataset CSV saved to: {out}")


if __name__ == "__main__":
    main()
