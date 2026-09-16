"""Create one LeRobot v3.0 dataset from complete source episodes.

This is a deliberately small adaptation of LeRobot's dataset_tools splitter.  It
keeps only the operation needed by the BP cache generator and uses the API in
this repository (no dependency on the upstream checkout under ``.cache``).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import av
import datasets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    embed_images,
    flatten_dict,
    get_hf_features_from_features,
    load_episodes,
    update_chunk_file_indices,
    write_info,
    write_stats,
)
from lerobot.datasets.video_utils import concatenate_video_files, get_video_info

MIB = 1024 * 1024
ProgressCallback = Callable[["ProgressEvent"], None]


@dataclass(frozen=True)
class ProgressEvent:
    """A stage update suitable for a tqdm adapter or a manifest logger."""

    stage: str
    repo_id: str
    completed: int = 0
    total: int | None = None
    task: str | None = None
    source_episode: int | None = None
    new_episode: int | None = None
    camera: str | None = None
    file: str | None = None


@dataclass(frozen=True)
class SplitWarning:
    level: Literal["warning", "high"]
    message: str
    repo_id: str
    source_episode: int
    camera: str
    size_mb: float


@dataclass(frozen=True)
class VideoFileStat:
    camera: str
    relative_path: str
    source_episodes: tuple[int, ...]
    new_episodes: tuple[int, ...]
    size_mb: float
    oversized_single_episode: bool
    codec: str
    pix_fmt: str


@dataclass
class SplitResult:
    dataset: LeRobotDataset
    root: Path
    episode_mapping: dict[int, int]
    task_mapping: dict[int, int]
    video_files: list[VideoFileStat] = field(default_factory=list)
    warnings: list[SplitWarning] = field(default_factory=list)
    encoder_fallbacks: dict[str, tuple[str, str]] = field(default_factory=dict)


def _emit(callback: ProgressCallback | None, event: ProgressEvent) -> None:
    if callback is not None:
        callback(event)


def validate_episode_indices(episode_indices: Iterable[int], total_episodes: int) -> list[int]:
    """Validate indices while preserving caller order (which defines new indices)."""
    indices = list(episode_indices)
    if not indices:
        raise ValueError("episode_indices must contain at least one episode")
    if total_episodes < 0:
        raise ValueError(f"total_episodes must be non-negative, got {total_episodes}")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise TypeError("episode_indices must contain integers (bool is not accepted)")
    seen: set[int] = set()
    duplicates: set[int] = set()
    for index in indices:
        (duplicates if index in seen else seen).add(index)
    if duplicates:
        raise ValueError(f"episode_indices contains duplicates: {sorted(duplicates)}")
    invalid = [index for index in indices if index < 0 or index >= total_episodes]
    if invalid:
        raise ValueError(f"episode_indices out of range [0, {total_episodes}): {invalid}")
    return indices


def filter_features(features: dict[str, dict], camera_keys: Sequence[str]) -> dict[str, dict]:
    """Keep selected cameras and every non-visual feature."""
    requested = list(camera_keys)
    if len(set(requested)) != len(requested):
        raise ValueError(f"camera_keys contains duplicates: {requested}")
    visual = {key for key, spec in features.items() if spec.get("dtype") in {"image", "video"}}
    unknown = sorted(set(requested) - visual)
    if unknown:
        raise ValueError(f"Unknown camera keys: {unknown}; available: {sorted(visual)}")
    requested_set = set(requested)
    return {
        key: deepcopy(spec)
        for key, spec in features.items()
        if key not in visual or key in requested_set
    }


def _single_episode_warning(
    repo_id: str,
    episode: int,
    camera: str,
    size_mb: float,
    soft_limit_mb: float,
    hard_limit_mb: float,
) -> SplitWarning | None:
    context = f"repo={repo_id} episode={episode} camera={camera} size_mb={size_mb:.3f}"
    if size_mb > hard_limit_mb:
        raise ValueError(f"Single episode video exceeds hard limit: {context}")
    if size_mb > soft_limit_mb:
        return SplitWarning("high", f"Single episode video exceeds soft limit: {context}", repo_id, episode, camera, size_mb)
    return None


def plan_video_packs(
    episode_sizes_mb: Sequence[tuple[int, float]],
    target_mb: float = 5.0,
    soft_limit_mb: float = 10.0,
    hard_limit_mb: float = 20.0,
    *,
    repo_id: str = "unknown",
    camera: str = "unknown",
) -> tuple[list[list[int]], list[SplitWarning]]:
    """Greedily pack measured episode files and enforce 5/10/20-style limits."""
    if not (0 < target_mb <= soft_limit_mb <= hard_limit_mb):
        raise ValueError("limits must satisfy 0 < target_mb <= soft_limit_mb <= hard_limit_mb")
    packs: list[list[int]] = []
    warnings: list[SplitWarning] = []
    current: list[int] = []
    current_size = 0.0
    for episode, size_mb in episode_sizes_mb:
        if size_mb < 0:
            raise ValueError(f"Negative video size for episode {episode}: {size_mb}")
        warning = _single_episode_warning(repo_id, episode, camera, size_mb, soft_limit_mb, hard_limit_mb)
        if warning is not None:
            warnings.append(warning)
        elif size_mb > target_mb:
            context = f"repo={repo_id} episode={episode} camera={camera} size_mb={size_mb:.3f}"
            warnings.append(SplitWarning("warning", f"Single episode video exceeds target: {context}", repo_id, episode, camera, size_mb))
        if current and current_size + size_mb > target_mb:
            packs.append(current)
            current, current_size = [], 0.0
        if size_mb > target_mb:
            packs.append([episode])
        else:
            current.append(episode)
            current_size += size_mb
    if current:
        packs.append(current)
    return packs, warnings


def _assert_output_root(root: Path) -> None:
    if root.exists():
        if not root.is_dir():
            raise FileExistsError(f"output_dir exists and is not a directory: {root}")
        if any(root.iterdir()):
            raise FileExistsError(f"output_dir must not exist or must be empty: {root}")
        root.rmdir()


def _episode_row(dataset: LeRobotDataset, episode_index: int) -> dict[str, Any]:
    episode = dataset.meta.episodes[episode_index]
    path = dataset.root / DEFAULT_EPISODES_PATH.format(
        chunk_index=episode["meta/episodes/chunk_index"],
        file_index=episode["meta/episodes/file_index"],
    )
    frame = pd.read_parquet(path, filters=[("episode_index", "=", episode_index)])
    if len(frame) != 1:
        raise ValueError(f"Expected one metadata row for episode {episode_index}, found {len(frame)}")
    return frame.iloc[0].to_dict()


def _stats_from_row(row: dict[str, Any], kept_features: dict[str, dict]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for key, value in row.items():
        if not key.startswith("stats/"):
            continue
        parts = key.split("/", 2)
        if len(parts) == 3 and parts[1] in kept_features:
            value = _normalize_stat(value)
            spec = kept_features[parts[1]]
            if spec.get("dtype") in {"image", "video"} and isinstance(value, np.ndarray) and value.ndim == 1 and parts[2] != "count":
                value = value.reshape(-1, 1, 1)
            stats.setdefault(parts[1], {})[parts[2]] = value
    return stats


def _normalize_stat(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object:
        def unwrap(item: Any) -> Any:
            while isinstance(item, np.ndarray) and item.size == 1:
                item = item.reshape(-1)[0]
            return item
        return np.asarray([unwrap(item) for item in value])
    return value


def _write_data(
    source: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
    kept_features: dict[str, dict],
    progress: ProgressCallback | None,
) -> tuple[dict[int, dict[str, int]], dict[int, int]]:
    frames: list[pd.DataFrame] = []
    source_task_indices: set[int] = set()
    total = len(episode_mapping)
    data_columns = [key for key, spec in kept_features.items() if spec.get("dtype") != "video"]
    for done, (old, new) in enumerate(episode_mapping.items(), 1):
        path = source.root / source.meta.get_data_file_path(old)
        frame = pd.read_parquet(path, filters=[("episode_index", "=", old)], columns=data_columns)
        if frame.empty:
            raise ValueError(f"Source episode {old} has no frames in {path}")
        source_task_indices.update(int(value) for value in frame["task_index"].unique())
        frame["episode_index"] = new
        frames.append(frame)
        _emit(progress, ProgressEvent("data", source.repo_id, done, total, source_episode=old, new_episode=new, file=str(path)))

    ordered_tasks = sorted(source_task_indices)
    task_mapping = {old: new for new, old in enumerate(ordered_tasks)}
    task_names: list[str] = []
    for old in ordered_tasks:
        matches = source.meta.tasks[source.meta.tasks["task_index"] == old]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one task row for task_index={old}, found {len(matches)}"
            )
        task_names.append(str(matches.index[0]))
    dst_meta.save_episode_tasks(task_names)

    combined = pd.concat(frames, ignore_index=True)
    combined["index"] = np.arange(len(combined), dtype=np.int64)
    combined["task_index"] = combined["task_index"].map(task_mapping).astype("int64")
    output = dst_meta.root / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    hf_features = get_hf_features_from_features(kept_features)
    hf_dataset = datasets.Dataset.from_dict(combined.to_dict(orient="list"), features=hf_features, split="train")
    if dst_meta.image_keys:
        hf_dataset = embed_images(hf_dataset)
    table = hf_dataset.with_format("arrow")[:]
    pq.write_table(table, output, compression="snappy", use_dictionary=True)

    metadata: dict[int, dict[str, int]] = {}
    for new in episode_mapping.values():
        selected = combined[combined["episode_index"] == new]
        metadata[new] = {
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": int(selected["index"].min()),
            "dataset_to_index": int(selected["index"].max()) + 1,
        }
    return metadata, task_mapping


def _encoder_name(codec: str | None) -> str | None:
    aliases = {"av1": "libsvtav1", "dav1d": "libsvtav1", "libdav1d": "libsvtav1", "h264": "libx264", "hevc": "libx265"}
    return aliases.get(codec or "", codec)


def _probe_video(path: Path) -> tuple[str | None, str | None, int, int, float]:
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {path}")
        stream = container.streams.video[0]
        codec = stream.codec_context.name
        pix_fmt = stream.codec_context.pix_fmt
        rate = stream.average_rate or stream.base_rate
        return codec, pix_fmt, stream.width, stream.height, float(rate)


def _encode_episode_segment(
    source_path: Path,
    output_path: Path,
    from_timestamp: float,
    to_timestamp: float,
    frame_count: int,
    fps: int,
    tolerance_s: float,
    target: tuple[str, str] | None = None,
) -> tuple[str, str, bool]:
    """Select by source PTS, then encode exactly one complete episode."""
    source_codec, source_pix_fmt, _, _, _ = _probe_video(source_path)
    source_target = (_encoder_name(source_codec), source_pix_fmt)
    candidates = [target] if target is not None else [source_target, ("libx264", "yuv420p"), ("h264", "yuv420p")]
    candidates = list(dict.fromkeys(pair for pair in candidates if pair[0] and pair[1]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for codec, pix_fmt in candidates:
        output_path.unlink(missing_ok=True)
        try:
            # Reopen for every candidate: decoder state cannot be safely reused after failure.
            with av.open(str(source_path)) as input_container:
                if not input_container.streams.video:
                    raise ValueError(f"No video stream in {source_path}")
                src = input_container.streams.video[0]
                selected: list[av.VideoFrame] = []
                selected_timestamps: list[float] = []
                previous: tuple[av.VideoFrame, float] | None = None
                decoder = iter(input_container.decode(src))
                for frame in decoder:
                    if frame.pts is None or frame.time_base is None:
                        continue
                    timestamp = float(frame.pts * frame.time_base)
                    if timestamp < from_timestamp:
                        previous = (frame, timestamp)
                        continue
                    choices = [(frame, timestamp)]
                    if previous is not None:
                        choices.append(previous)
                    first_frame, first_timestamp = min(
                        choices, key=lambda item: abs(item[1] - from_timestamp)
                    )
                    if abs(first_timestamp - from_timestamp) > tolerance_s:
                        raise ValueError(
                            f"No frame matches from_timestamp={from_timestamp:.9f} in {source_path}; "
                            f"nearest PTS={first_timestamp:.9f}, tolerance={tolerance_s:.9f}"
                        )
                    selected.append(first_frame)
                    selected_timestamps.append(first_timestamp)
                    # If the previous frame won, the current frame is the next episode frame.
                    if first_frame is not frame and len(selected) < frame_count:
                        selected.append(frame)
                        selected_timestamps.append(timestamp)
                    break
                for frame in decoder:
                    if len(selected) >= frame_count:
                        break
                    if frame.pts is None or frame.time_base is None:
                        continue
                    selected.append(frame)
                    selected_timestamps.append(float(frame.pts * frame.time_base))
                if len(selected) != frame_count:
                    raise ValueError(
                        f"Decoded {len(selected)} episode frames from {source_path}, expected {frame_count}"
                    )
                if selected_timestamps[-1] >= to_timestamp + tolerance_s:
                    raise ValueError(
                        f"Episode segment crossed to_timestamp={to_timestamp:.9f}; "
                        f"last PTS={selected_timestamps[-1]:.9f}"
                    )

                with av.open(str(output_path), "w", options={"movflags": "faststart"}) as output_container:
                    dst = output_container.add_stream(codec, rate=Fraction(fps, 1))
                    dst.width, dst.height = src.width, src.height
                    dst.pix_fmt = pix_fmt
                    for index, frame in enumerate(selected):
                        converted = frame.reformat(width=src.width, height=src.height, format=pix_fmt)
                        converted.pts = index
                        converted.time_base = Fraction(1, fps)
                        for packet in dst.encode(converted):
                            output_container.mux(packet)
                    for packet in dst.encode():
                        output_container.mux(packet)
            actual_codec, actual_pix_fmt, _, _, _ = _probe_video(output_path)
            actual = (_encoder_name(actual_codec), actual_pix_fmt)
            requested = (_encoder_name(codec), pix_fmt)
            if actual != requested:
                raise ValueError(f"Encoded stream is {actual}, requested {requested}")
            used_fallback = target is None and requested != source_target
            return requested[0], requested[1], used_fallback
        except (ValueError, av.error.FFmpegError) as exc:
            last_error = exc
            output_path.unlink(missing_ok=True)
    raise RuntimeError(f"Unable to encode {source_path} with candidates {candidates}") from last_error


def _validate_compatible_segments(paths: dict[int, Path], expected: tuple[str, str]) -> None:
    reference: tuple[str | None, str | None, int, int, float] | None = None
    for episode, path in paths.items():
        info = _probe_video(path)
        normalized = (_encoder_name(info[0]), info[1], info[2], info[3], info[4])
        if normalized[:2] != expected:
            raise ValueError(f"Episode {episode} stream {normalized[:2]} differs from target {expected}")
        if reference is None:
            reference = normalized
        elif normalized != reference:
            raise ValueError(f"Incompatible video segments: episode {episode} has {normalized}, expected {reference}")


def _scan_packed_video(
    path: Path,
    pack: list[int],
    episode_lengths: dict[int, int],
    fps: int,
) -> tuple[dict[int, tuple[float, float]], dict[int, dict[str, np.ndarray]]]:
    timestamps: list[float] = []
    images: list[np.ndarray] = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in packed file {path}")
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"Frame without PTS in packed file {path}")
            timestamps.append(float(frame.pts * frame.time_base))
            images.append(frame.to_ndarray(format="rgb24").transpose(2, 0, 1))
    expected_total = sum(episode_lengths[index] for index in pack)
    if len(timestamps) != expected_total:
        raise ValueError(f"Packed file {path} has {len(timestamps)} frames, expected {expected_total}")
    if len(timestamps) > 1 and any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"Packed file {path} has non-monotonic frame PTS")
    interval = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 1.0 / fps
    if abs(interval - 1.0 / fps) > 0.5 / fps:
        raise ValueError(f"Packed file {path} frame interval {interval:.9f} is incompatible with fps={fps}")

    bounds: dict[int, tuple[float, float]] = {}
    visual_stats: dict[int, dict[str, np.ndarray]] = {}
    cursor = 0
    for episode in pack:
        length = episode_lengths[episode]
        segment_ts = timestamps[cursor : cursor + length]
        segment_images = np.stack(images[cursor : cursor + length])
        bounds[episode] = (segment_ts[0], segment_ts[-1] + interval)
        raw_stats = get_feature_stats(segment_images, axis=(0, 2, 3), keepdims=True)
        visual_stats[episode] = {
            key: value if key == "count" else np.squeeze(value / 255.0, axis=0)
            for key, value in raw_stats.items()
        }
        cursor += length
    if cursor != len(timestamps):
        raise AssertionError("Packed frame partition did not consume every frame")
    return bounds, visual_stats


def _split_pack_until_valid(
    pack: list[int],
    temp_paths: dict[int, Path],
    output_dir: Path,
    next_file: list[int],
    target_mb: float,
) -> list[tuple[list[int], Path, float]]:
    """Concat and recursively split multi-episode outputs that exceed the target."""
    file_index = next_file[0]
    next_file[0] += 1
    path = output_dir / f"candidate-{file_index:06d}.mp4"
    if len(pack) == 1:
        shutil.copy2(temp_paths[pack[0]], path)
    else:
        concatenate_video_files([temp_paths[index] for index in pack], path, overwrite=True)
    size_mb = path.stat().st_size / MIB
    if len(pack) > 1 and size_mb > target_mb:
        path.unlink()
        split = max(1, len(pack) // 2)
        return (
            _split_pack_until_valid(pack[:split], temp_paths, output_dir, next_file, target_mb)
            + _split_pack_until_valid(pack[split:], temp_paths, output_dir, next_file, target_mb)
        )
    return [(pack, path, size_mb)]


def _write_videos(
    source: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
    cameras: list[str],
    target_mb: float,
    soft_limit_mb: float,
    hard_limit_mb: float,
    progress: ProgressCallback | None,
) -> tuple[
    dict[int, dict[str, Any]],
    dict[int, dict[str, dict[str, np.ndarray]]],
    list[VideoFileStat],
    list[SplitWarning],
    dict[str, tuple[str, str]],
]:
    metadata = {new: {} for new in episode_mapping.values()}
    recomputed_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {
        new: {} for new in episode_mapping.values()
    }
    stats: list[VideoFileStat] = []
    warnings: list[SplitWarning] = []
    fallbacks: dict[str, tuple[str, str]] = {}
    video_cameras = [camera for camera in cameras if camera in source.meta.video_keys]
    old_by_new = {new: old for old, new in episode_mapping.items()}
    episode_lengths = {
        new: int(source.meta.episodes[old]["length"]) for old, new in episode_mapping.items()
    }
    tolerance = max(float(source.tolerance_s), 0.5 / source.meta.fps + 1e-9)

    with tempfile.TemporaryDirectory(prefix="lerobot-split-") as temp_name:
        temp = Path(temp_name)
        for camera in video_cameras:
            camera_temp = temp / camera.replace("/", "_")
            camera_temp.mkdir(parents=True)
            temp_paths: dict[int, Path] = {}
            target_encoding: tuple[str, str] | None = None
            for done, (old, new) in enumerate(episode_mapping.items(), 1):
                episode = source.meta.episodes[old]
                source_path = source.root / source.meta.get_video_file_path(old, camera)
                from_ts = float(episode[f"videos/{camera}/from_timestamp"])
                to_ts = float(episode[f"videos/{camera}/to_timestamp"])
                temp_path = camera_temp / f"episode-{new:06d}.mp4"
                codec, pix_fmt, fallback = _encode_episode_segment(
                    source_path, temp_path, from_ts, to_ts, episode_lengths[new],
                    source.meta.fps, tolerance, target_encoding,
                )
                if target_encoding is None:
                    target_encoding = (codec, pix_fmt)
                    if fallback:
                        fallbacks[camera] = target_encoding
                temp_paths[new] = temp_path
                _emit(progress, ProgressEvent("video_episode", source.repo_id, done, len(episode_mapping), source_episode=old, new_episode=new, camera=camera, file=str(source_path)))

            if target_encoding is None:
                raise AssertionError(f"No encoded segments for camera {camera}")
            _validate_compatible_segments(temp_paths, target_encoding)
            sizes = [(new, temp_paths[new].stat().st_size / MIB) for new in episode_mapping.values()]
            planned, camera_warnings = plan_video_packs(sizes, target_mb, soft_limit_mb, hard_limit_mb, repo_id=source.repo_id, camera=camera)
            warnings.extend(camera_warnings)
            candidates = camera_temp / "packed"
            candidates.mkdir()
            validated: list[tuple[list[int], Path, float]] = []
            serial = [0]
            for pack in planned:
                validated.extend(_split_pack_until_valid(pack, temp_paths, candidates, serial, target_mb))

            chunk, file_index = 0, 0
            for done, (pack, candidate, size_mb) in enumerate(validated, 1):
                _validate_compatible_segments({index: temp_paths[index] for index in pack}, target_encoding)
                bounds, pack_stats = _scan_packed_video(candidate, pack, episode_lengths, source.meta.fps)
                destination = dst_meta.root / dst_meta.video_path.format(video_key=camera, chunk_index=chunk, file_index=file_index)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(candidate, destination)
                for new in pack:
                    from_ts, to_ts = bounds[new]
                    metadata[new].update({
                        f"videos/{camera}/chunk_index": chunk,
                        f"videos/{camera}/file_index": file_index,
                        f"videos/{camera}/from_timestamp": from_ts,
                        f"videos/{camera}/to_timestamp": to_ts,
                    })
                    recomputed_stats[new][camera] = pack_stats[new]
                if len(pack) > 1 and size_mb > target_mb:
                    raise RuntimeError(f"Packed video exceeds target after verification: {destination} {size_mb:.3f} MB")
                old_pack = tuple(old_by_new[new] for new in pack)
                stats.append(VideoFileStat(camera, str(destination.relative_to(dst_meta.root)), old_pack, tuple(pack), size_mb, len(pack) == 1 and size_mb > target_mb, *target_encoding))
                _emit(progress, ProgressEvent("video_file", source.repo_id, done, len(validated), camera=camera, file=str(destination)))
                chunk, file_index = update_chunk_file_indices(chunk, file_index, dst_meta.chunks_size)

            first = dst_meta.root / dst_meta.video_path.format(video_key=camera, chunk_index=0, file_index=0)
            actual_info = get_video_info(first)
            source_info = deepcopy(source.meta.features[camera].get("info", {}))
            source_info.update(actual_info)
            dst_meta.info["features"][camera]["info"] = source_info
    return metadata, recomputed_stats, stats, warnings, fallbacks

def split_episodes(
    source: LeRobotDataset,
    episode_indices: Iterable[int],
    output_dir: str | Path,
    repo_id: str,
    camera_keys: Sequence[str],
    target_video_mb: float = 5.0,
    single_episode_soft_mb: float = 10.0,
    single_episode_hard_mb: float = 20.0,
    progress_callback: ProgressCallback | None = None,
) -> SplitResult:
    """Build a v3.0 dataset from arbitrary complete source episodes.

    ``episode_indices`` order defines the destination episode order. The caller is
    responsible for atomic publication; this function rejects non-empty outputs.
    """
    if not repo_id or not isinstance(repo_id, str):
        raise ValueError("repo_id must be a non-empty string")
    if source.meta.episodes is None:
        source.meta.episodes = load_episodes(source.root)
    selected = validate_episode_indices(episode_indices, source.meta.total_episodes)
    features = filter_features(source.meta.features, camera_keys)
    root = Path(output_dir)
    _assert_output_root(root)
    mapping = {old: new for new, old in enumerate(selected)}
    _emit(progress_callback, ProgressEvent("start", source.repo_id, 0, len(selected)))

    dst_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=source.meta.fps,
        features=features,
        robot_type=source.meta.robot_type,
        root=root,
        use_videos=any(spec.get("dtype") == "video" for spec in features.values()),
        chunks_size=source.meta.chunks_size,
        data_files_size_in_mb=source.meta.data_files_size_in_mb,
        video_files_size_in_mb=max(1, int(np.ceil(target_video_mb))),
    )
    try:
        data_metadata, task_mapping = _write_data(source, dst_meta, mapping, features, progress_callback)
        video_metadata, video_episode_stats, video_files, warnings, fallbacks = _write_videos(
            source, dst_meta, mapping, list(camera_keys), target_video_mb,
            single_episode_soft_mb, single_episode_hard_mb, progress_callback,
        )
        episode_stats: list[dict[str, dict]] = []
        for done, (old, new) in enumerate(mapping.items(), 1):
            row = _episode_row(source, old)
            stats_for_episode = _stats_from_row(row, features)
            # Never copy source stats for transcoded video. Replace them with stats
            # decoded from the final packed files and their measured PTS boundaries.
            for video_key in source.meta.video_keys:
                stats_for_episode.pop(video_key, None)
            stats_for_episode.update(video_episode_stats[new])
            episode_stats.append(stats_for_episode)
            source_episode = source.meta.episodes[old]
            metadata = dict(data_metadata[new])
            metadata.update(video_metadata[new])
            task_names = [str(task) for task in source_episode.get("tasks", [])]
            dst_meta._save_episode_metadata({
                "episode_index": new,
                "tasks": task_names,
                "length": int(source_episode["length"]),
                **metadata,
                **flatten_dict({"stats": stats_for_episode}),
            })
            _emit(progress_callback, ProgressEvent("metadata", source.repo_id, done, len(mapping), task=task_names[0] if task_names else None, source_episode=old, new_episode=new))
        dst_meta._close_writer()
        dst_meta.info.update({
            "codebase_version": "v3.0",
            "total_episodes": len(mapping),
            "total_frames": sum(int(source.meta.episodes[old]["length"]) for old in selected),
            "total_tasks": len(task_mapping),
            "splits": {"train": f"0:{len(mapping)}"},
        })
        write_info(dst_meta.info, root)
        nonempty_stats = [item for item in episode_stats if item]
        if nonempty_stats:
            write_stats(aggregate_stats(nonempty_stats), root)
        else:
            write_stats({}, root)
        dataset = LeRobotDataset(repo_id=repo_id, root=root, skip_video_file_validation=False)
    except BaseException:
        logging.exception("Episode split failed; leaving output for caller inspection: %s", root)
        raise
    _emit(progress_callback, ProgressEvent("complete", source.repo_id, len(mapping), len(mapping)))
    return SplitResult(dataset, root, mapping, task_mapping, video_files, warnings, fallbacks)
