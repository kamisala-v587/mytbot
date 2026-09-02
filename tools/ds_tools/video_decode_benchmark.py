#!/usr/bin/env python3
# 启动步骤（可直接复制到 shell；反斜杠表示命令在下一行继续）：
#   cd /vla/workspace/my_tbot
#   conda activate mytbot
#   python .code/LetMeSeeSee/vdecode_benchmark.py
#   python .code/LetMeSeeSee/vdecode_benchmark.py --help
#   python .code/LetMeSeeSee/vdecode_benchmark.py --resume-output-dir /path/to/video_decode_result_xxx
#   python .code/LetMeSeeSee/vdecode_benchmark.py \
#       --gpu-count 1 \
#       --parallel-repos-per-gpu 1 \
#       --workers-per-repo 1
"""LeRobot 视频解码并发基准：模拟 BPVA 训练读取视频时的 CPU 解码压力。"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import random
import re
import signal
import statistics
import sys
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# ============================== 默认配置（可被 CLI 覆盖） ==============================
# repo 清单的默认路径（单位不适用）；清单越大总运行时间越长，但不改变瞬时并发数。
DEFAULT_REPO_LIST = Path(
    "/vla/workspace/my_tbot/configs/ds_ids/Baidunyun/pretrain_data_ids_baiduyun.txt"
)
# 相对 repo ID 拼接到这个根目录（单位不适用）；设得过宽可能扫描到非预期目录。
DEFAULT_DATA_ROOT = Path("/")
# 未指定输出目录时使用的父目录（单位不适用）；大量运行会占用更多磁盘空间。
DEFAULT_OUTPUT_PARENT = Path.cwd()
# 自动创建的输出目录名前缀（单位不适用）；只影响命名，调长会让路径更难阅读。
DEFAULT_OUTPUT_DIR_PREFIX = "video_decode_result"
# 断点恢复输出目录；None 表示新建输出目录。填已有目录或用 CLI 指定时，会跳过已完成 repo 并继续追加。
RESUME_OUTPUT_DIR = None
# 模拟的 GPU 分组数（个）；默认保守为 1，确认机器稳定后再通过 CLI 逐步调大。
GPU_COUNT = 2
# 每个 GPU 分组并行处理的 repo 数（个）；默认保守为 1，调大会放大外层并发压力。
PARALLEL_REPOS_PER_GPU = 1
# 每个 repo 内同时解码的线程数（个）；默认保守为 1，调大会增加 CPU、内存和磁盘压力。
WORKERS_PER_REPO = 1
# 每个 repo 选择的视频数（个）；值为 1 时固定选择排序后的第一个视频，不进行随机抽样。
VIDEOS_PER_REPO = 1
# 每个选中视频独立解码的次数（次）；不改变线程上限，调大会增加总调用数和运行时间。
REPEATS_PER_VIDEO = 2
# 每次解码请求的时间戳数量（帧位置个数）；调大会增加单次 CPU、内存和解码耗时。
TIMESTAMPS_PER_CALL = 10
# 传给 LeRobot 的解码后端名称（单位不适用）；更换后端可能改变依赖、性能和可用性。
BACKEND = "pyav"
# 多选视频时的随机抽样种子（单位不适用）；值为 1 时不使用，更换不改变并发量。
SEED = 20260831
# 默认淘汰最终慢速分数最高的 20% repo；0 表示不自动淘汰，100 表示淘汰全部有有效分数的 repo。
SLOWEST_PERCENT_TO_FILTER = 20.0
# 兼容旧参数：当前默认百分比规则不再使用下面三个阈值判慢。
SLOW_ABSOLUTE_P95_SECONDS = 2
SLOW_RELATIVE_MEDIAN_MULTIPLIER = 2.0
MIN_SUCCESSFUL_CALLS_FOR_SLOW = 2
# 时间戳覆盖区间的起点比例（0~1）；调大可能缩短跨度，设得不合理会降低模拟代表性。
TIMESTAMP_START_FRACTION = 0.05
# 时间戳覆盖区间的终点比例（0~1）；调大更靠近结尾，可能增加边界帧读取失败风险。
TIMESTAMP_END_FRACTION = 0.95
# 单次 decode 调用允许覆盖的最大时间跨度（秒）；防止 PyAV 一次解码几百秒视频造成内存峰值。
MAX_TIMESTAMP_SPAN_SECONDS = 60 # 通常轨迹在30 -60秒内
# 请求时间戳与实际最近帧允许的误差（秒）；调大更宽松，但取到的帧可能离目标更远。
DEFAULT_TOLERANCE_SECONDS = 0.1
# 默认跳过 repo 的正则表达式（单位不适用）；规则过宽会跳过本应测试的数据。
EXCLUDE_REPO_REGEX = ""
# 默认跳过视频路径的正则表达式（单位不适用）；规则过宽会减少样本甚至使 repo 无视频可测。
EXCLUDE_VIDEO_REGEX = ""

# 全局停止标记（单位不适用）；SIGINT 会设置它，让两层线程池停止启动后续工作并安全收尾。
STOP_EVENT = threading.Event()
# 控制台打印锁（单位不适用）；避免并发线程输出互相穿插，不参与解码并发公式。
PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class Config:
    """保存一次基准运行的完整配置，包括输入、输出、并发和判慢参数。"""

    repo_list: Path
    data_root: Path
    output_dir: Path
    resume_output_dir: Path | None
    gpu_count: int
    parallel_repos_per_gpu: int
    workers_per_repo: int
    videos_per_repo: int
    repeats_per_video: int
    timestamps_per_call: int
    max_timestamp_span_seconds: float
    backend: str
    seed: int
    slowest_percent_to_filter: float
    slow_threshold_s: float
    slow_relative_multiplier: float
    min_successful_calls: int
    tolerance_s: float
    exclude_repo_regex: str
    exclude_video_regex: str

    @property
    def concurrent_repos(self) -> int:
        """返回外层可同时处理的 repo 数，即 GPU 分组数乘以每组 repo 数。"""
        return self.gpu_count * self.parallel_repos_per_gpu

    @property
    def potential_decode_threads(self) -> int:
        """返回理论最大解码线程数，即并发 repo 数乘以每 repo worker 数。"""
        return self.concurrent_repos * self.workers_per_repo


@dataclass(frozen=True)
class RepoEntry:
    """表示输入清单中的一个 repo，保留原始顺序、ID 和解析后的本地路径。"""

    index: int
    repo_id: str
    repo_path: Path


@dataclass
class RepoResult:
    """累计单个 repo 的运行状态、成功/失败次数、耗时统计和判慢结果。"""

    index: int
    repo_id: str
    repo_path: str
    gpu_group: int
    status: str = "failed"
    error: str | None = None
    discovered_videos: int = 0
    sampled_videos: int = 0
    planned_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    metadata_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    decode_seconds: list[float] = field(default_factory=list)
    p50_seconds: float | None = None
    p95_seconds: float | None = None
    mean_seconds: float | None = None
    max_seconds: float | None = None
    slow_score_seconds: float | None = None
    slow_candidate: bool = False
    slow_reason: str | None = None

    def public_dict(self) -> dict[str, Any]:
        """返回适合写入结果文件的字典，并去掉仅用于内部统计的逐次耗时列表。"""
        result = asdict(self)
        result.pop("decode_seconds", None)
        return result


class JsonlWriter:
    """线程安全的逐行 JSON 写入器；每次写入立即 flush，尽量保住中断前结果。"""

    def __init__(self, path: Path) -> None:
        """打开 ``path`` 指定的 JSONL 文件，供多个线程安全追加。"""
        self.path = path
        self._lock = threading.Lock()
        self._file = path.open("a", encoding="utf-8", buffering=1)

    def append(self, value: dict[str, Any]) -> None:
        """把一个字典编码成一行 JSON，写入后立即刷新到操作系统。"""
        line = json.dumps(value, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._file.write(line + "\n")
            # 每条记录立刻 flush；即使进程崩溃或被 Ctrl+C，中途结果也更可能已落盘。
            self._file.flush()

    def close(self) -> None:
        """刷新剩余内容并关闭文件；应在最终汇总前调用。"""
        with self._lock:
            self._file.flush()
            self._file.close()


class Progress:
    """线程安全地统计已完成 repo，并打印成功数、失败数和预计剩余时间。"""

    def __init__(self, total: int) -> None:
        """创建进度计数器；``total`` 是计划测试的 repo 总数。"""
        self.total = total
        self.done = 0
        self.success = 0
        self.failed = 0
        self.started = time.perf_counter()
        self._lock = threading.Lock()

    def finish(self, ok: bool, repo_id: str) -> None:
        """记录一个 repo 完成；``ok`` 表示结果可用，``repo_id`` 用于进度提示。"""
        with self._lock:
            self.done += 1
            self.success += int(ok)
            self.failed += int(not ok)
            elapsed = time.perf_counter() - self.started
            rate = self.done / elapsed if elapsed > 0 else 0.0
            eta = (self.total - self.done) / rate if rate > 0 else math.inf
            percent = 100.0 * self.done / self.total if self.total else 100.0
            eta_text = format_duration(eta) if math.isfinite(eta) else "未知"
            message = (
                f"[进度] {self.done}/{self.total} ({percent:5.1f}%) | "
                f"成功 {self.success} | 失败 {self.failed} | "
                f"耗时 {format_duration(elapsed)} | ETA {eta_text} | {repo_id}"
            )
            with PRINT_LOCK:
                print(message, flush=True)


def format_duration(seconds: float) -> str:
    """把秒数转换成便于阅读的中文时、分、秒文本。"""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}时{minutes:02d}分{secs:02d}秒"
    if minutes:
        return f"{minutes}分{secs:02d}秒"
    return f"{secs}秒"


def percentile(values: Sequence[float], q: float) -> float | None:
    """用线性插值计算 ``values`` 的 q 分位数；空输入返回 ``None``。"""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compute_slow_score_seconds(result: RepoResult) -> float | None:
    """返回最终慢速分数：优先 p95，其次 mean，再其次 max；单位秒，越大越慢。"""
    for value in (result.p95_seconds, result.mean_seconds, result.max_seconds):
        if value is not None:
            return value
    return None


def make_parser() -> argparse.ArgumentParser:
    """创建并返回命令行参数解析器，包含全部默认值和帮助文字。"""
    description = (
        "模拟 BPVA 训练并发读取 LeRobot MP4，并用最终慢速分数找出偏慢的数据集。"
        "slow_score_seconds 是最终依据，单位秒，越大越慢；"
        "p95 只是内部优先采用的偏保守耗时统计，用户主要看最终分数即可。"
        "GPU 仅用于把 repo 分组；PyAV/torchvision 实际在 CPU 解码，不要求 CUDA。"
    )
    epilog = (
        "并发怎么算：并发 repo 数 = GPU 数 × 每 GPU 并行 repo 数；"
        "潜在总解码线程 = GPU 数 × 每 GPU 并行 repo 数 × 每 repo worker 数。\n"
        "本脚本面向用户的速度表只需要看 slow_score_seconds 和 is_filtered："
        "slow_score_seconds 是最终慢速分数，单位秒，越大越慢；"
        "p95 可以理解为‘多数情况下接近最慢的一次耗时’，用于避免只看偶发最大值。\n"
        "安全起步示例：python benchmark_lerobot_video_decode.py --gpu-count 1 "
        "--parallel-repos-per-gpu 1 --workers-per-repo 1；稳定后再逐步调大并发。"
    )
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo-list", type=Path, default=DEFAULT_REPO_LIST,
                        help="repo 列表文本；每行一个路径/ID，空行和 # 注释会忽略。")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="相对 repo ID 的根目录；绝对路径不受它影响。默认：/")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="结果目录；默认在当前工作目录创建带时间的易懂子目录。")
    parser.add_argument(
        "--resume-output-dir",
        type=Path,
        default=RESUME_OUTPUT_DIR,
        help=("填上一次被 Killed 的输出目录，脚本会读取 repo_results.jsonl，"
              "跳过已完成的 repo，继续测剩下的。注意仍会追加到同一目录。"),
    )
    parser.add_argument("--gpu-count", type=int, default=GPU_COUNT,
                        help="模拟训练使用的 GPU 分组数；不检查 CUDA，也不在 GPU 解码。")
    parser.add_argument("--parallel-repos-per-gpu", type=int, default=PARALLEL_REPOS_PER_GPU,
                        help="每个 GPU 分组同时处理多少个 repo。")
    parser.add_argument("--workers-per-repo", type=int, default=WORKERS_PER_REPO,
                        help="每个 repo 内并发解码线程数；过大可能让 CPU/磁盘过载。")
    parser.add_argument("--videos-per-repo", type=int, default=VIDEOS_PER_REPO,
                        help="每个 repo 选择多少个 MP4；值为 1 时固定选择排序首项，大于 1 时固定 seed 随机抽样。")
    parser.add_argument("--repeats-per-video", type=int, default=REPEATS_PER_VIDEO,
                        help="每个选中视频重复解码多少次；每次都会独立打开/关闭。")
    parser.add_argument("--timestamps-per-call", type=int, default=TIMESTAMPS_PER_CALL,
                        help="每次解码取多少帧；先选视频 5%% 到 95%% 区间，再按最大跨度裁剪。")
    parser.add_argument("--max-timestamp-span-seconds", type=float, default=MAX_TIMESTAMP_SPAN_SECONDS,
                        help=("单次 decode 调用最多覆盖多少秒；默认限制较短以避免内存暴涨。"
                              "设为 0 表示不限制，很危险，长视频可能让进程被系统 Killed。"))
    parser.add_argument("--backend", choices=("pyav", "video_reader", "torchcodec"), default=BACKEND,
                        help="传给 LeRobot 真实解码函数的 backend。默认 pyav（CPU）。")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="多选视频时的抽样随机种子；值为 1 时不使用。")
    parser.add_argument("--tolerance-seconds", type=float, default=DEFAULT_TOLERANCE_SECONDS,
                        help="请求时间戳允许偏离最近视频帧的秒数。")
    parser.add_argument("--slowest-percent-to-filter", type=float, default=SLOWEST_PERCENT_TO_FILTER,
                        help=("按最终慢速分数从慢到快自动淘汰最慢百分比，范围 [0,100]；"
                              "0 表示不自动淘汰，100 表示淘汰全部有有效分数的 repo。默认：20。"))
    parser.add_argument("--slow-threshold-seconds", type=float, default=SLOW_ABSOLUTE_P95_SECONDS,
                        help="兼容旧参数，当前默认百分比规则不使用：旧版慢候选的 repo p95 绝对下限（秒）。")
    parser.add_argument("--slow-relative-multiplier", type=float,
                        default=SLOW_RELATIVE_MEDIAN_MULTIPLIER,
                        help="兼容旧参数，当前默认百分比规则不使用：旧版 p95 中位数倍率门槛。")
    parser.add_argument("--min-successful-calls", type=int,
                        default=MIN_SUCCESSFUL_CALLS_FOR_SLOW,
                        help="兼容旧参数，当前默认百分比规则不使用：旧版判慢前要求的最少成功次数。")
    parser.add_argument("--exclude-repo-regex", default=EXCLUDE_REPO_REGEX,
                        help="跳过匹配此正则的 repo（留在 filtered 列表，不武断删除）。")
    parser.add_argument("--exclude-video-regex", default=EXCLUDE_VIDEO_REGEX,
                        help="扫描时跳过路径匹配此正则的 MP4；空字符串表示不排除。")
    return parser


def positive(name: str, value: int) -> None:
    """检查整数参数必须大于零；不满足时用参数名 ``name`` 报错。"""
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0，当前是 {value}")


def config_from_args(args: argparse.Namespace) -> Config:
    """校验命令行参数并转换为不可变的 ``Config``；无效输入会抛出 ``ValueError``。"""
    for name in (
        "gpu_count", "parallel_repos_per_gpu", "workers_per_repo", "videos_per_repo",
        "repeats_per_video", "timestamps_per_call", "min_successful_calls",
    ):
        positive("--" + name.replace("_", "-"), getattr(args, name))
    if args.tolerance_seconds <= 0:
        raise ValueError("--tolerance-seconds 必须大于 0")
    if args.max_timestamp_span_seconds < 0:
        raise ValueError("--max-timestamp-span-seconds 必须大于等于 0")
    if not 0 <= args.slowest_percent_to_filter <= 100:
        raise ValueError("--slowest-percent-to-filter 必须在 [0, 100] 范围内")
    if args.slow_threshold_seconds < 0 or args.slow_relative_multiplier <= 0:
        raise ValueError("慢阈值必须非负，慢倍率必须大于 0")
    for label, pattern in (("repo", args.exclude_repo_regex), ("video", args.exclude_video_regex)):
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"排除 {label} 的正则无效：{exc}") from exc
    resume_output_dir = (
        Path(args.resume_output_dir) if args.resume_output_dir is not None else None
    )
    if resume_output_dir is not None:
        output_dir = resume_output_dir
    elif args.output_dir is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = DEFAULT_OUTPUT_PARENT / f"{DEFAULT_OUTPUT_DIR_PREFIX}_{stamp}"
    else:
        output_dir = args.output_dir
    return Config(
        repo_list=args.repo_list.expanduser().resolve(),
        data_root=args.data_root.expanduser().resolve(),
        output_dir=output_dir.expanduser().resolve(),
        resume_output_dir=(
            resume_output_dir.expanduser().resolve() if resume_output_dir is not None else None
        ),
        gpu_count=args.gpu_count,
        parallel_repos_per_gpu=args.parallel_repos_per_gpu,
        workers_per_repo=args.workers_per_repo,
        videos_per_repo=args.videos_per_repo,
        repeats_per_video=args.repeats_per_video,
        timestamps_per_call=args.timestamps_per_call,
        max_timestamp_span_seconds=args.max_timestamp_span_seconds,
        backend=args.backend,
        seed=args.seed,
        slowest_percent_to_filter=args.slowest_percent_to_filter,
        slow_threshold_s=args.slow_threshold_seconds,
        slow_relative_multiplier=args.slow_relative_multiplier,
        min_successful_calls=args.min_successful_calls,
        tolerance_s=args.tolerance_seconds,
        exclude_repo_regex=args.exclude_repo_regex,
        exclude_video_regex=args.exclude_video_regex,
    )


def read_repo_entries(config: Config) -> list[RepoEntry]:
    """读取 repo 清单、忽略空行和注释，并返回保留原顺序的条目列表。"""
    if not config.repo_list.is_file():
        raise FileNotFoundError(f"repo 列表不存在：{config.repo_list}")
    entries: list[RepoEntry] = []
    for raw in config.repo_list.read_text(encoding="utf-8").splitlines():
        value = raw.split("#", 1)[0].strip()
        if not value:
            continue
        repo_path = Path(value).expanduser()
        if not repo_path.is_absolute():
            repo_path = config.data_root / repo_path
        entries.append(RepoEntry(len(entries), value, repo_path.resolve()))
    if not entries:
        raise ValueError(f"repo 列表没有有效内容：{config.repo_list}")
    return entries


def repo_result_from_dict(data: dict[str, Any], entry: RepoEntry) -> RepoResult:
    """把历史 JSONL 行恢复成 ``RepoResult``，缺失字段使用当前清单和默认值补齐。"""
    field_names = set(RepoResult.__dataclass_fields__)
    values = {key: value for key, value in data.items() if key in field_names}
    values["index"] = entry.index
    values["repo_id"] = entry.repo_id
    values["repo_path"] = str(entry.repo_path)
    values.setdefault("gpu_group", int(data.get("gpu_group", 0)))
    result = RepoResult(**values)
    if result.slow_score_seconds is None:
        result.slow_score_seconds = compute_slow_score_seconds(result)
    return result


def load_completed_results(
    output_dir: Path, entries: Sequence[RepoEntry], logger: logging.Logger
) -> dict[int, RepoResult]:
    """从已有 ``repo_results.jsonl`` 恢复成功/部分成功的结果；坏行只记录日志并跳过。"""
    results_path = output_dir / "repo_results.jsonl"
    if not results_path.is_file():
        logger.info("恢复模式：未找到已有 repo_results.jsonl，将从头测试当前清单。")
        return {}
    entries_by_index = {entry.index: entry for entry in entries}
    repo_ids_by_index = {entry.index: entry.repo_id for entry in entries}
    completed: dict[int, RepoResult] = {}
    with results_path.open("r", encoding="utf-8") as file:
        for line_number, raw in enumerate(file, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "恢复模式：跳过 repo_results.jsonl 第 %d 行坏 JSON：%s",
                    line_number, exc,
                )
                continue
            if not isinstance(data, dict):
                logger.warning("恢复模式：跳过 repo_results.jsonl 第 %d 行非对象 JSON。", line_number)
                continue
            status = data.get("status")
            if status not in {"success", "partial"}:
                continue
            index = data.get("index")
            repo_id = data.get("repo_id")
            if not isinstance(index, int) or index not in entries_by_index:
                continue
            if repo_id != repo_ids_by_index[index]:
                logger.info(
                    "恢复模式：跳过第 %d 行，index=%s 的 repo_id 不在当前清单对应位置。",
                    line_number, index,
                )
                continue
            completed[index] = repo_result_from_dict(data, entries_by_index[index])
    return completed


def setup_logger(output_dir: Path) -> logging.Logger:
    """创建同时写入 ``run.log`` 和标准输出的日志器并返回。"""
    logger = logging.getLogger("lerobot_video_decode_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def discover_videos(repo_path: Path, exclude_pattern: str) -> list[Path]:
    """递归查找 repo 的 MP4，应用可选排除正则，并按相对路径稳定排序。"""
    videos_dir = repo_path / "videos"
    if not repo_path.is_dir():
        raise FileNotFoundError(f"repo 目录不存在：{repo_path}")
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"repo 中没有 videos 目录：{videos_dir}")
    matcher = re.compile(exclude_pattern) if exclude_pattern else None
    videos = [
        path for path in videos_dir.rglob("*.mp4")
        if path.is_file() and not (matcher and matcher.search(str(path)))
    ]
    videos.sort(key=lambda path: str(path.relative_to(videos_dir)))
    if not videos:
        raise ValueError(f"videos 下没有可用 MP4（也可能全部被排除）：{videos_dir}")
    return videos


def import_runtime_dependencies() -> tuple[Callable[..., Any], Any]:
    """仅正式运行时导入解码函数和 PyAV 模块并返回，使 ``--help`` 无需重依赖。"""
    try:
        import av  # type: ignore
        from lerobot.datasets.video_utils import decode_video_frames
    except Exception as exc:
        raise RuntimeError(
            "无法导入 PyAV/torchvision/LeRobot。请在本仓库已安装依赖的 Python 环境运行。"
        ) from exc
    return decode_video_frames, av


def probe_video(video_path: Path, av_module: Any) -> tuple[float, float, dict[str, Any]]:
    """读取视频时长等元数据，返回时长、探测耗时和元数据；这里不做正式计时解码。"""
    started = time.perf_counter()
    with av_module.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError("文件没有视频流")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        duration = 0.0
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av_module.time_base)
        elif stream.frames and fps > 0:
            duration = float(stream.frames / fps)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("无法获得正数视频时长")
        metadata = {
            "duration_seconds": duration,
            "fps": fps or None,
            "frames": int(stream.frames) if stream.frames else None,
            "width": int(stream.width),
            "height": int(stream.height),
        }
    return duration, time.perf_counter() - started, metadata


def make_timestamps(duration: float, count: int, max_span_seconds: float) -> list[float]:
    """先选 5% 到 95% 区间，再按最大跨度裁剪并均匀生成时间戳，单位为秒。"""
    if count == 1:
        return [duration * 0.5]
    start = duration * TIMESTAMP_START_FRACTION
    end = duration * TIMESTAMP_END_FRACTION
    if end <= start:
        start, end = 0.0, max(0.0, duration * 0.9)
    if max_span_seconds > 0 and end - start > max_span_seconds:
        midpoint = (start + end) * 0.5
        half_span = max_span_seconds * 0.5
        start = max(0.0, midpoint - half_span)
        end = min(duration, midpoint + half_span)
        if end - start < max_span_seconds:
            if start <= 0.0:
                end = min(duration, max_span_seconds)
            elif end >= duration:
                start = max(0.0, duration - max_span_seconds)
    step = (end - start) / (count - 1)
    return [start + step * index for index in range(count)]


def decode_once(
    entry: RepoEntry,
    video_path: Path,
    repeat_index: int,
    timestamps: list[float],
    metadata: dict[str, Any],
    config: Config,
    decode_function: Callable[..., Any],
    event_writer: JsonlWriter,
) -> tuple[bool, float | None]:
    """独立执行一次 LeRobot 解码并记录事件。

    ``entry`` 和 ``video_path`` 标识样本，``timestamps`` 是要读取的秒数位置；
    ``decode_function`` 执行真实解码，``event_writer`` 立即保存结果。返回是否成功，
    以及成功时的解码耗时（秒）；失败或取消时耗时返回 ``None``。
    """
    base_event: dict[str, Any] = {
        "time": datetime.now().astimezone().isoformat(),
        "repo_index": entry.index,
        "repo_id": entry.repo_id,
        "repo_path": str(entry.repo_path),
        "video_path": str(video_path),
        "repeat_index": repeat_index,
        "backend": config.backend,
        "timestamps": timestamps,
        "metadata": metadata,
    }
    if STOP_EVENT.is_set():
        base_event.update(status="cancelled", error="收到停止信号，未开始解码")
        event_writer.append(base_event)
        return False, None
    started = time.perf_counter()
    try:
        # decode_video_frames 每调用一次都会自己打开并关闭视频；重复测试不是复用同一容器，
        # 更接近训练中不同样本各自读取文件的情形。计时只包住这次真实解码调用，前面的
        # PyAV 元数据探测单独累计，不计入 decode_seconds，避免把“看视频多长”误算成解码。
        frames = decode_function(
            video_path=video_path,
            timestamps=timestamps,
            tolerance_s=config.tolerance_s,
            backend=config.backend,
        )
        decode_seconds = time.perf_counter() - started
        shape = list(frames.shape) if hasattr(frames, "shape") else None
        base_event.update(status="success", decode_seconds=decode_seconds, output_shape=shape)
        event_writer.append(base_event)
        del frames
        # 只能帮助 Python 对象尽快回收；不能保证释放所有原生解码器内存。
        gc.collect()
        return True, decode_seconds
    except BaseException as exc:
        decode_seconds = time.perf_counter() - started
        base_event.update(
            status="failed",
            decode_seconds=decode_seconds,
            error=f"{type(exc).__name__}: {exc}",
        )
        event_writer.append(base_event)
        return False, None


def run_repo(
    entry: RepoEntry,
    config: Config,
    decode_function: Callable[..., Any],
    av_module: Any,
    event_writer: JsonlWriter,
    logger: logging.Logger,
) -> RepoResult:
    """测试一个 repo：稳定选择视频并在 repo 内并发解码。

    ``entry`` 指定 repo；``config`` 决定选择量和线程数：选择量为 1 时固定使用
    排序后的首个视频，大于 1 时使用固定 seed 抽样。解码事件会写入
    ``event_writer``，异常会写入 ``logger``。无论完全成功、部分成功还是失败，
    都返回包含次数、耗时和状态的 ``RepoResult``。
    """
    started = time.perf_counter()
    result = RepoResult(
        index=entry.index,
        repo_id=entry.repo_id,
        repo_path=str(entry.repo_path),
        gpu_group=entry.index % config.gpu_count,
    )
    try:
        if STOP_EVENT.is_set():
            raise RuntimeError("收到停止信号，repo 未开始")
        videos = discover_videos(entry.repo_path, config.exclude_video_regex)
        result.discovered_videos = len(videos)
        if config.videos_per_repo == 1:
            # discover_videos 已排序；固定首项可确保选择稳定，且不使用随机抽样。
            sampled = [videos[0]]
        else:
            if len(videos) < config.videos_per_repo:
                raise ValueError(
                    f"视频不足：需要 {config.videos_per_repo} 个，实际只有 {len(videos)} 个"
                )
            # 每个 repo 使用“固定总 seed + 固定输入序号”，所以相同清单和参数会抽到相同视频，
            # 便于不同机器或不同并发配置做公平对比，同时各 repo 又不会共用完全相同的随机序列。
            rng = random.Random(config.seed + entry.index)
            sampled = rng.sample(videos, config.videos_per_repo)
        result.sampled_videos = len(sampled)
        jobs: list[tuple[Path, int, list[float], dict[str, Any]]] = []
        for video_path in sampled:
            try:
                # 先探测元数据来获得视频时长；这段耗时单列为 metadata_seconds，
                # 不会混入后续 decode_seconds，因此 p50/p95 只反映正式解码调用。
                duration, probe_seconds, metadata = probe_video(video_path, av_module)
                result.metadata_seconds += probe_seconds
                # 默认短窗口用于避免 PyAV/torchvision 在单次 decode 中缓存大量中间帧导致内存峰值。
                # 如要复现 BPVA 长跨度慢问题，可设 --max-timestamp-span-seconds 0，但应降低并发。
                timestamps = make_timestamps(
                    duration, config.timestamps_per_call, config.max_timestamp_span_seconds
                )
                for repeat_index in range(config.repeats_per_video):
                    jobs.append((video_path, repeat_index, timestamps, metadata))
            except BaseException as exc:
                event_writer.append({
                    "time": datetime.now().astimezone().isoformat(),
                    "repo_index": entry.index,
                    "repo_id": entry.repo_id,
                    "repo_path": str(entry.repo_path),
                    "video_path": str(video_path),
                    "stage": "metadata_probe",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                result.failed_calls += config.repeats_per_video
        result.planned_calls = config.videos_per_repo * config.repeats_per_video
        if not jobs:
            raise RuntimeError("选中的视频全部损坏或无法读取元数据")
        # 第二层线程池：一个 repo 内最多 workers_per_repo 个解码调用同时运行。
        # 第一层在 main 中并行多个 repo，两层相乘才是潜在总解码线程数。
        with ThreadPoolExecutor(
            max_workers=config.workers_per_repo,
            thread_name_prefix=f"repo-{entry.index}-decode",
        ) as executor:
            futures = [
                executor.submit(
                    decode_once, entry, video_path, repeat_index, timestamps, metadata,
                    config, decode_function, event_writer,
                )
                for video_path, repeat_index, timestamps, metadata in jobs
            ]
            for future in as_completed(futures):
                try:
                    ok, decode_seconds = future.result()
                    if ok and decode_seconds is not None:
                        result.successful_calls += 1
                        result.decode_seconds.append(decode_seconds)
                    else:
                        result.failed_calls += 1
                except BaseException as exc:
                    result.failed_calls += 1
                    event_writer.append({
                        "time": datetime.now().astimezone().isoformat(),
                        "repo_index": entry.index,
                        "repo_id": entry.repo_id,
                        "stage": "worker_future",
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
        if result.successful_calls == 0:
            raise RuntimeError("没有任何成功解码调用；请查看 decode_events.jsonl")
        result.status = "success" if result.failed_calls == 0 else "partial"
    except BaseException as exc:
        result.status = "cancelled" if STOP_EVENT.is_set() else "failed"
        result.error = f"{type(exc).__name__}: {exc}"
        logger.error("repo 失败 | %s | %s", entry.repo_id, result.error)
    finally:
        result.elapsed_seconds = time.perf_counter() - started
        if result.decode_seconds:
            result.p50_seconds = percentile(result.decode_seconds, 0.50)
            result.p95_seconds = percentile(result.decode_seconds, 0.95)
            result.mean_seconds = statistics.fmean(result.decode_seconds)
            result.max_seconds = max(result.decode_seconds)
        result.slow_score_seconds = compute_slow_score_seconds(result)
    return result


def mark_slow_repos(results: list[RepoResult], config: Config) -> int:
    """按最终慢速分数排序，原地标记最慢百分比的 repo；返回可比较 repo 数。

    slow_score_seconds 是用户最终需要看的慢速指标，单位秒，越大越慢。
    p95 只是内部优先采用的耗时统计，可理解为“多数情况下接近最慢的一次耗时”，
    用于避免只看单次偶发最大值。
    """
    comparable = sorted(
        (result for result in results if result.slow_score_seconds is not None),
        key=repo_speed_sort_key,
    )
    comparable_count = len(comparable)
    for result in results:
        result.slow_candidate = False
        if result.slow_score_seconds is None:
            result.slow_reason = "不判慢：没有有效慢速分数（没有成功解码耗时可比较）"
        elif config.slowest_percent_to_filter == 0:
            result.slow_reason = "不判慢：未启用自动淘汰（slowest_percent_to_filter=0）"

    if config.slowest_percent_to_filter == 0 or comparable_count == 0:
        return comparable_count

    slow_count = math.ceil(comparable_count * config.slowest_percent_to_filter / 100.0)
    slow_indices = {result.index for result in comparable[:slow_count]}
    for rank, result in enumerate(comparable, start=1):
        result.slow_candidate = result.index in slow_indices
        if result.slow_candidate:
            result.slow_reason = (
                f"最终慢速分数={result.slow_score_seconds:.2f}秒，"
                f"排在全部{comparable_count}个可比较repo的第{rank}名，"
                f"属于最慢{config.slowest_percent_to_filter:g}%"
            )
        else:
            result.slow_reason = (
                f"最终慢速分数={result.slow_score_seconds:.2f}秒，"
                f"排在全部{comparable_count}个可比较repo的第{rank}名，"
                f"不属于最慢{config.slowest_percent_to_filter:g}%"
            )
    return comparable_count


def repo_speed_sort_key(result: RepoResult) -> tuple[float, float, float, int]:
    """返回速度排名排序键：最终慢速分数、max、mean 从慢到快，最后按原始 index。"""
    return (
        -(result.slow_score_seconds if result.slow_score_seconds is not None else -math.inf),
        -(result.max_seconds if result.max_seconds is not None else -math.inf),
        -(result.mean_seconds if result.mean_seconds is not None else -math.inf),
        result.index,
    )


def ranked_results_with_p95(results: Iterable[RepoResult]) -> list[RepoResult]:
    """返回所有已有最终慢速分数的 repo，最慢在前；函数名保留以兼容内部调用。"""
    return sorted(
        (result for result in results if result.slow_score_seconds is not None),
        key=repo_speed_sort_key,
    )


def ranked_slow_results(results: Iterable[RepoResult]) -> list[RepoResult]:
    """返回已判为慢候选的 repo，按最终慢速分数从慢到快排序。"""
    return sorted(
        (result for result in results if result.slow_candidate),
        key=repo_speed_sort_key,
    )


def slow_rank_row(rank: int, result: RepoResult) -> dict[str, Any]:
    """生成慢候选排名详情行；字段保持简洁，方便直接查看或复制。"""
    return {
        "rank": rank,
        "repo_id": result.repo_id,
        "slow_score_seconds": result.slow_score_seconds,
        "reason": result.slow_reason,
    }


def speed_rank_row(rank: int, result: RepoResult) -> dict[str, Any]:
    """生成小白版全量速度排名行；最终分数越大越慢，is_filtered 表示会被过滤。"""
    return {
        "rank": rank,
        "repo_id": result.repo_id,
        "slow_score_seconds": result.slow_score_seconds,
        "is_filtered": result.slow_candidate,
        "reason": result.slow_reason,
        "measured_calls": result.successful_calls,
    }


def write_final_outputs(
    entries: list[RepoEntry],
    results: list[RepoResult],
    excluded_indices: set[int],
    config: Config,
    started_wall: str,
    elapsed_seconds: float,
    interrupted: bool,
    resumed_from: Path | None = None,
    resumed_results_count: int = 0,
    new_results_count: int | None = None,
) -> dict[str, Path]:
    """根据已有条目和结果完成判慢，并写出表格、排序列表与总览 JSON。

    ``excluded_indices`` 标识主动跳过项，``interrupted`` 标识是否中断；即使结果不完整，
    也会尽量生成可检查的文件。``slow_repos.txt`` 是慢候选列表，按最终慢速分数从慢到快
    排序；``filtered_repo_ids.txt`` 仍保留输入清单的原始顺序，方便继续作为 repo 清单使用。
    返回“文件用途名称到实际路径”的字典。
    """
    output_dir = config.output_dir
    # 最终流程先计算慢候选，再生成原始顺序汇总、慢候选排序、全量速度排名和 summary。
    # 即使运行中断，也尽量用已经完成的 repo 生成一套可检查的汇总文件。
    results_by_index = {result.index: result for result in results}
    comparable_repos = mark_slow_repos(results, config)
    slow_ranked_results = ranked_slow_results(results)
    speed_ranked_results = ranked_results_with_p95(results)
    slow_indices = {result.index for result in slow_ranked_results}
    paths = {
        "summary": output_dir / "summary.json",
        "csv": output_dir / "repo_summary.csv",
        "slow": output_dir / "slow_repos.txt",
        "slow_ranked": output_dir / "slow_repos_ranked.csv",
        "speed_ranking": output_dir / "repo_speed_ranking.csv",
        "filtered": output_dir / "filtered_repo_ids.txt",
        "events": output_dir / "decode_events.jsonl",
        "repo_results": output_dir / "repo_results.jsonl",
        "log": output_dir / "run.log",
    }
    fields = [
        "index", "repo_id", "repo_path", "gpu_group", "status", "error",
        "discovered_videos", "sampled_videos", "planned_calls", "successful_calls",
        "failed_calls", "metadata_seconds", "elapsed_seconds", "p50_seconds",
        "p95_seconds", "mean_seconds", "max_seconds", "slow_score_seconds",
        "slow_candidate", "slow_reason",
    ]
    with paths["csv"].open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda item: item.index):
            writer.writerow(result.public_dict())

    # 慢候选 repo_id 列表按最终慢速分数从慢到快排序；过滤后列表仍按原 repo 清单顺序保留。
    slow_lines = [result.repo_id for result in slow_ranked_results]
    paths["slow"].write_text("".join(line + "\n" for line in slow_lines), encoding="utf-8")

    slow_rank_fields = ["rank", "repo_id", "slow_score_seconds", "reason"]
    slow_rank_rows = [
        slow_rank_row(rank, result)
        for rank, result in enumerate(slow_ranked_results, start=1)
    ]
    with paths["slow_ranked"].open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=slow_rank_fields)
        writer.writeheader()
        writer.writerows(slow_rank_rows)

    speed_rank_fields = [
        "rank", "repo_id", "slow_score_seconds", "is_filtered", "reason", "measured_calls",
    ]
    speed_rank_rows = [
        speed_rank_row(rank, result)
        for rank, result in enumerate(speed_ranked_results, start=1)
    ]
    with paths["speed_ranking"].open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=speed_rank_fields)
        writer.writeheader()
        writer.writerows(speed_rank_rows)

    # 只排除证据充分的慢候选。失败可能来自文件缺失、依赖问题或临时 I/O 错误，并不能
    # 证明 repo 稳定偏慢；视频不足和显式跳过项也没有足够样本，因此全部保留以避免误删。
    filtered_lines = [entry.repo_id for entry in entries if entry.index not in slow_indices]
    paths["filtered"].write_text(
        "".join(line + "\n" for line in filtered_lines), encoding="utf-8"
    )
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    top_slowest_repos = [
        {
            "repo_id": result.repo_id,
            "slow_score_seconds": result.slow_score_seconds,
            "is_filtered": result.slow_candidate,
            "reason": result.slow_reason,
        }
        for result in speed_ranked_results[:20]
    ]
    summary = {
        "started_at": started_wall,
        "finished_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "interrupted": interrupted,
        "resumed_from": str(resumed_from) if resumed_from is not None else None,
        "resumed_results_count": resumed_results_count,
        "new_results_count": (
            len(results) - resumed_results_count if new_results_count is None else new_results_count
        ),
        "repo_list": str(config.repo_list),
        "output_dir": str(output_dir),
        "config": {
            **asdict(config),
            "repo_list": str(config.repo_list),
            "data_root": str(config.data_root),
            "output_dir": str(config.output_dir),
            "resume_output_dir": (
                str(config.resume_output_dir) if config.resume_output_dir is not None else None
            ),
        },
        "concurrency_explanation": {
            "concurrent_repos_formula": "gpu_count * parallel_repos_per_gpu",
            "concurrent_repos": config.concurrent_repos,
            "potential_decode_threads_formula": (
                "gpu_count * parallel_repos_per_gpu * workers_per_repo"
            ),
            "potential_decode_threads": config.potential_decode_threads,
            "note": "GPU 只用于模拟训练并发分组；PyAV/torchvision 在 CPU 解码，不要求 CUDA。",
        },
        "counts": {
            "input_repos": len(entries),
            "explicitly_excluded": len(excluded_indices),
            "completed_results": len(results),
            "status": status_counts,
            "slow_candidates": len(slow_indices),
            "ranked_repos": len(speed_ranked_results),
            "filtered_repo_ids": len(filtered_lines),
        },
        "slow_rule": {
            "slowest_percent_to_filter": config.slowest_percent_to_filter,
            "comparable_repos": comparable_repos,
            "filtered_count": len(slow_indices),
            "logic": (
                f"按最终慢速分数排序，淘汰最慢 {config.slowest_percent_to_filter:g}%；"
                "slow_score_seconds 单位秒，越大越慢。p95 只是内部优先采用的耗时统计，"
                "用户主要看 slow_score_seconds 即可。"
            ),
        },
        "slow_repos": slow_lines,
        "slow_repos_ranked": slow_rank_rows,
        "top_slowest_repos": top_slowest_repos,
        "excluded_repo_ids": [entry.repo_id for entry in entries if entry.index in excluded_indices],
        "unreported_repo_ids": [
            entry.repo_id for entry in entries
            if entry.index not in results_by_index and entry.index not in excluded_indices
        ],
        "files": {name: str(path) for name, path in paths.items()},
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return paths


def install_signal_handler(logger: logging.Logger) -> None:
    """安装 Ctrl+C 处理器：请求温和停止，并让已启动任务和结果文件完成收尾。"""
    def handle_sigint(signum: int, frame: Any) -> None:
        """收到 SIGINT 时设置全局停止标记；再次收到时仅重复提示。"""
        del signum, frame
        if not STOP_EVENT.is_set():
            STOP_EVENT.set()
            logger.warning("收到 Ctrl+C：停止启动新解码；已落盘结果会保留，请稍候收尾。")
        else:
            logger.warning("已在停止中；当前解码调用结束后退出。")
    # 不立刻强杀进程：先设置停止标记，让正在写 JSONL 的线程结束并 flush，随后仍生成汇总。
    signal.signal(signal.SIGINT, handle_sigint)


def print_final_summary(
    config: Config,
    results: list[RepoResult],
    paths: dict[str, Path],
    interrupted: bool,
    resumed_results_count: int = 0,
    new_results_count: int | None = None,
) -> None:
    """在控制台打印小白版最终汇总，包括并发公式、排序后的慢候选和输出路径。"""
    slow = ranked_slow_results(results)
    successful = sum(result.status in {"success", "partial"} for result in results)
    failed = sum(result.status in {"failed", "cancelled"} for result in results)
    with PRINT_LOCK:
        print("\n================ 基准汇总（小白版） ================")
        print(f"运行状态：{'被中断，已有结果已保留' if interrupted else '已完成'}")
        print(f"repo 结果：可用 {successful}，失败/取消 {failed}，慢候选 {len(slow)}")
        if config.resume_output_dir is not None:
            new_count = len(results) - resumed_results_count if new_results_count is None else new_results_count
            print(
                f"恢复来源：{config.resume_output_dir}；"
                f"已恢复 {resumed_results_count}，本次新增 {new_count}"
            )
        print(
            f"本次按最终慢速分数（slow_score_seconds，单位秒，越大越慢）排序，"
            f"自动筛掉最慢 {config.slowest_percent_to_filter:g}%。"
        )
        print(
            "ranking 表只看 slow_score_seconds 和 is_filtered 即可；"
            "p95 是内部优先采用的偏保守耗时统计，用户不用理解也能看结果。"
        )
        print(
            f"并发 repo 数 = GPU 数 {config.gpu_count} × 每 GPU 并行 repo 数 "
            f"{config.parallel_repos_per_gpu} = {config.concurrent_repos}"
        )
        print(
            f"潜在总解码线程 = {config.gpu_count} × {config.parallel_repos_per_gpu} × "
            f"每 repo worker 数 {config.workers_per_repo} = {config.potential_decode_threads}"
        )
        print("注意：GPU 只是模拟训练任务分组；PyAV/torchvision 实际用 CPU 解码，不需要 CUDA。")
        if slow:
            print("慢数据集候选（按最终慢速分数从慢到快排序）：")
            for result in slow:
                print(f"  - {result.repo_id} | {result.slow_reason}")
        else:
            print("慢数据集候选：没有。通常表示 percent=0，或没有有效可比较数据。")
            print(f"可查看全量速度排名了解当前已测 repo 中谁最慢：{paths['speed_ranking']}")
        print("输出文件：")
        labels = {
            "summary": "总览 JSON",
            "csv": "每 repo 表格",
            "slow": "慢候选 repo_id 列表（已按最终慢速分数从慢到快排序）",
            "slow_ranked": "慢候选排序详情",
            "speed_ranking": "全量速度排名",
            "filtered": "保留原顺序的过滤后列表",
            "events": "逐次解码明细",
            "repo_results": "逐 repo 即时结果",
            "log": "运行日志",
        }
        for name, path in paths.items():
            print(f"  - {labels[name]}：{path}")
        print("====================================================", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """运行参数解析、两层并发测试、即时落盘和最终汇总的完整流程。

    ``argv`` 为可选命令行参数序列，省略时读取真实命令行；返回 0 表示完成，
    130 表示收到停止信号，其他致命错误返回 1。
    """
    parser = make_parser()
    try:
        config = config_from_args(parser.parse_args(argv))
    except ValueError as exc:
        parser.error(str(exc))
    config.output_dir.mkdir(parents=True, exist_ok=config.resume_output_dir is not None)
    logger = setup_logger(config.output_dir)
    install_signal_handler(logger)
    started_wall = datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    event_writer = JsonlWriter(config.output_dir / "decode_events.jsonl")
    repo_writer = JsonlWriter(config.output_dir / "repo_results.jsonl")
    results: list[RepoResult] = []
    entries: list[RepoEntry] = []
    excluded_indices: set[int] = set()
    resumed_results_count = 0
    new_results_count = 0
    fatal_error: str | None = None
    try:
        entries = read_repo_entries(config)
        completed_results = (
            load_completed_results(config.output_dir, entries, logger)
            if config.resume_output_dir is not None else {}
        )
        results.extend(completed_results[index] for index in sorted(completed_results))
        resumed_results_count = len(completed_results)
        repo_matcher = re.compile(config.exclude_repo_regex) if config.exclude_repo_regex else None
        runnable = []
        for entry in entries:
            if entry.index in completed_results:
                continue
            if repo_matcher and repo_matcher.search(entry.repo_id):
                excluded_indices.add(entry.index)
                repo_writer.append({
                    "index": entry.index, "repo_id": entry.repo_id,
                    "repo_path": str(entry.repo_path), "status": "excluded",
                    "error": "匹配 --exclude-repo-regex；未测试且会保留在 filtered 列表",
                })
            else:
                runnable.append(entry)
        logger.info("输出目录：%s", config.output_dir)
        if config.resume_output_dir is not None:
            logger.info("恢复模式：从 %s 追加继续运行", config.resume_output_dir)
        logger.info(
            "并发 repo=%d（%d GPU 分组 × 每组 %d repo）；潜在解码线程=%d；GPU 不用于解码",
            config.concurrent_repos, config.gpu_count, config.parallel_repos_per_gpu,
            config.potential_decode_threads,
        )
        logger.info(
            "输入 repo=%d；已恢复=%d；排除=%d；待测试=%d",
            len(entries), resumed_results_count, len(excluded_indices), len(runnable),
        )
        decode_function = av_module = None
        progress = Progress(len(runnable))
        if runnable:
            decode_function, av_module = import_runtime_dependencies()
        # 第一层线程池并行 repo：并发数 = GPU_COUNT × PARALLEL_REPOS_PER_GPU。
        # 这里的 GPU_COUNT 只是把 repo 按训练规模分组和计算并发，并没有选择 CUDA 设备；
        # PyAV/torchvision 仍在 CPU 上解码。每个 repo 内还有 run_repo 创建的第二层线程池。
        with ThreadPoolExecutor(
            max_workers=config.concurrent_repos,
            thread_name_prefix="repo",
        ) as executor:
            futures: dict[Future[RepoResult], RepoEntry] = {}
            for entry in runnable:
                if STOP_EVENT.is_set():
                    break
                futures[executor.submit(
                    run_repo, entry, config, decode_function, av_module, event_writer, logger
                )] = entry
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    result = future.result()
                except BaseException as exc:
                    result = RepoResult(
                        index=entry.index,
                        repo_id=entry.repo_id,
                        repo_path=str(entry.repo_path),
                        gpu_group=entry.index % config.gpu_count,
                        status="failed",
                        error=f"未捕获的 repo 异常：{type(exc).__name__}: {exc}",
                    )
                    logger.error("repo future 异常：%s\n%s", entry.repo_id, traceback.format_exc())
                results.append(result)
                new_results_count += 1
                repo_writer.append(result.public_dict())
                progress.finish(result.status in {"success", "partial"}, entry.repo_id)
    except BaseException as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        STOP_EVENT.set()
        logger.error("运行发生异常，已有 JSONL 结果仍保留：%s", fatal_error)
        logger.debug("异常堆栈：\n%s", traceback.format_exc())
    finally:
        # 无论正常完成、异常还是 Ctrl+C，都先关闭并刷新逐条结果，再基于现有结果写最终汇总。
        event_writer.close()
        repo_writer.close()
        elapsed = time.perf_counter() - started
        try:
            paths = write_final_outputs(
                entries, results, excluded_indices, config, started_wall, elapsed,
                interrupted=STOP_EVENT.is_set(),
                resumed_from=config.resume_output_dir,
                resumed_results_count=resumed_results_count,
                new_results_count=new_results_count,
            )
            if fatal_error:
                summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
                summary["fatal_error"] = fatal_error
                paths["summary"].write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            print_final_summary(
                config, results, paths, STOP_EVENT.is_set(),
                resumed_results_count=resumed_results_count,
                new_results_count=new_results_count,
            )
        except BaseException as final_exc:
            logger.error("生成最终汇总失败，但 JSONL 持续落盘文件仍在：%s", final_exc)
            fatal_error = fatal_error or f"汇总失败：{type(final_exc).__name__}: {final_exc}"
        for handler in logger.handlers:
            handler.flush()
    return 1 if fatal_error else (130 if STOP_EVENT.is_set() else 0)


if __name__ == "__main__":
    raise SystemExit(main())
