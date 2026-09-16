from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pandas as pd
import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tools.lerobot_github import filter_features, plan_video_packs, split_episodes, validate_episode_indices


def test_validate_episode_indices_preserves_non_contiguous_order() -> None:
    assert validate_episode_indices([4, 1, 3], 5) == [4, 1, 3]
    with pytest.raises(ValueError, match="duplicates"):
        validate_episode_indices([1, 1], 3)
    with pytest.raises(ValueError, match="out of range"):
        validate_episode_indices([3], 3)
    with pytest.raises(TypeError, match="integers"):
        validate_episode_indices([True], 3)


def test_filter_features_keeps_non_visual_and_requested_cameras() -> None:
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "observation.images.bp": {"dtype": "video", "shape": (4, 4, 3), "names": None},
        "observation.images.other": {"dtype": "image", "shape": (4, 4, 3), "names": None},
    }
    filtered = filter_features(features, ["observation.images.bp"])
    assert set(filtered) == {"observation.state", "observation.images.bp"}
    assert filtered["observation.state"] is not features["observation.state"]
    with pytest.raises(ValueError, match="Unknown camera"):
        filter_features(features, ["missing"])


def test_plan_video_packs_enforces_thresholds() -> None:
    packs, warnings = plan_video_packs(
        [(7, 2.0), (8, 3.0), (9, 5.1), (10, 10.0), (11, 10.1), (12, 20.0)],
        repo_id="org/repo", camera="bp",
    )
    assert packs == [[7, 8], [9], [10], [11], [12]]
    assert [warning.level for warning in warnings] == ["warning", "warning", "high", "high"]
    with pytest.raises(ValueError, match=r"repo=org/repo episode=13 camera=bp size_mb=20\.100"):
        plan_video_packs([(13, 20.1)], repo_id="org/repo", camera="bp")


def _make_no_video_dataset(root: Path) -> LeRobotDataset:
    dataset = LeRobotDataset.create(
        repo_id="local/source", fps=10, root=root, use_videos=False, robot_type="testbot",
        features={"observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]}},
    )
    for episode, task in enumerate(["pick", "place", "pick"]):
        for frame in range(episode + 2):
            dataset.add_frame({"observation.state": np.asarray([episode, frame], dtype=np.float32), "task": task})
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    return LeRobotDataset(repo_id="local/source", root=root)


def test_non_contiguous_no_video_v30_end_to_end(tmp_path: Path) -> None:
    source = _make_no_video_dataset(tmp_path / "source")
    events = []
    result = split_episodes(source, [2, 0], tmp_path / "subset", "local/subset", [], progress_callback=events.append)
    assert result.episode_mapping == {2: 0, 0: 1}
    assert result.task_mapping == {0: 0}
    assert result.dataset.meta.info["codebase_version"] == "v3.0"
    assert result.dataset.meta.total_episodes == 2
    assert result.dataset.meta.total_frames == 6
    assert result.dataset.hf_dataset["episode_index"] == [0, 0, 0, 0, 1, 1]
    assert result.dataset.hf_dataset["index"] == list(range(6))
    assert result.dataset.meta.episodes[0]["length"] == 4
    assert result.dataset.meta.episodes[1]["length"] == 2
    assert result.dataset.meta.episodes[0]["dataset_from_index"] == 0
    assert result.dataset.meta.episodes[1]["dataset_from_index"] == 4
    assert list(result.dataset.meta.tasks.index) == ["pick"]
    assert events[0].stage == "start" and events[-1].stage == "complete"


def test_split_rejects_non_empty_output(tmp_path: Path) -> None:
    source = _make_no_video_dataset(tmp_path / "source")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must not exist or must be empty"):
        split_episodes(source, [0], output, "local/subset", [])
    assert (output / "sentinel").read_text(encoding="utf-8") == "keep"


def test_task_mapping_uses_task_index_column_not_row_position(tmp_path: Path) -> None:
    source = _make_no_video_dataset(tmp_path / "source")
    tasks_path = source.root / "meta/tasks.parquet"
    tasks = pd.DataFrame(
        {"task_index": [9, 0, 4]},
        index=pd.Index(["unused-nine", "pick", "unused-four"], name="task"),
    )
    tasks.to_parquet(tasks_path)
    source.meta.tasks = tasks
    result = split_episodes(source, [2, 0], tmp_path / "subset", "local/subset", [])
    assert result.task_mapping == {0: 0}
    assert list(result.dataset.meta.tasks.index) == ["pick"]


def _rgb_mean(frame: torch.Tensor) -> np.ndarray:
    return frame.float().mean(dim=(1, 2)).cpu().numpy()


def test_video_split_uses_pts_boundaries_concat_metadata_and_recomputed_stats(tmp_path: Path) -> None:
    source_root = tmp_path / "video-source"
    source = LeRobotDataset.create(
        repo_id="local/video-source",
        fps=5,
        root=source_root,
        use_videos=True,
        robot_type="testbot",
        features={
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["x"]},
            "observation.images.bp": {
                "dtype": "video",
                "shape": (64, 64, 3),
                "names": ["height", "width", "channels"],
            },
        },
    )
    colors = [np.asarray([240, 12, 12], dtype=np.uint8), np.asarray([12, 240, 12], dtype=np.uint8)]
    for episode, color in enumerate(colors):
        for _ in range(4):
            source.add_frame(
                {
                    "observation.state": np.asarray([episode], dtype=np.float32),
                    "observation.images.bp": np.broadcast_to(color, (64, 64, 3)).copy(),
                    "task": "color",
                }
            )
        source.save_episode(parallel_encoding=False)
    source.finalize()
    source = LeRobotDataset(repo_id="local/video-source", root=source_root, video_backend="pyav")
    assert source.meta.get_video_file_path(0, "observation.images.bp") == source.meta.get_video_file_path(1, "observation.images.bp")

    # Poison source visual stats: the subset must decode final output instead of copying these values.
    episode_path = source.root / "meta/episodes/chunk-000/file-000.parquet"
    episode_df = pd.read_parquet(episode_path)
    for column in episode_df.columns:
        if column.startswith("stats/observation.images.bp/") and not column.endswith("/count"):
            episode_df[column] = [np.full_like(value, 0.123) for value in episode_df[column]]
    episode_df.to_parquet(episode_path, index=False)
    source.meta.episodes = None

    # Shift the source stream timestamps and matching metadata together. This
    # catches implementations that assume enumerate index == timestamp * fps.
    video_path = source.root / source.meta.get_video_file_path(0, "observation.images.bp")
    shifted_path = video_path.with_name("shifted.mp4")
    with av.open(str(video_path)) as input_container, av.open(str(shifted_path), "w") as output_container:
        input_stream = input_container.streams.video[0]
        output_stream = output_container.add_stream_from_template(input_stream, opaque=True)
        output_stream.time_base = input_stream.time_base
        offset = int(round(1.0 / float(input_stream.time_base)))
        for packet in input_container.demux(input_stream):
            if packet.pts is None or packet.dts is None:
                continue
            packet.pts += offset
            packet.dts += offset
            packet.stream = output_stream
            output_container.mux(packet)
    shifted_path.replace(video_path)
    episode_df = pd.read_parquet(episode_path)
    episode_df["videos/observation.images.bp/from_timestamp"] += 1.0
    episode_df["videos/observation.images.bp/to_timestamp"] += 1.0
    episode_df.to_parquet(episode_path, index=False)
    source.meta.episodes = None

    result = split_episodes(
        source,
        [1, 0],
        tmp_path / "video-subset",
        "local/video-subset",
        ["observation.images.bp"],
        target_video_mb=5,
    )
    assert len(result.video_files) == 1
    assert result.video_files[0].new_episodes == (0, 1)
    assert result.video_files[0].codec in {"libx264", "libsvtav1", "libx265"}

    episodes = result.dataset.meta.episodes
    boundary_indices = [0, episodes[0]["dataset_to_index"] - 1, episodes[1]["dataset_from_index"], episodes[1]["dataset_to_index"] - 1]
    decoded = [result.dataset[index]["observation.images.bp"] for index in boundary_indices]
    expected = [colors[1], colors[1], colors[0], colors[0]]
    for frame, color in zip(decoded, expected, strict=True):
        assert np.allclose(_rgb_mean(frame), color / 255.0, atol=0.08)

    packed_path = result.root / result.video_files[0].relative_path
    with av.open(str(packed_path)) as container:
        stream = container.streams.video[0]
        timestamps = [float(frame.pts * frame.time_base) for frame in container.decode(stream)]
    assert len(timestamps) == 8
    assert episodes[0]["videos/observation.images.bp/from_timestamp"] == pytest.approx(timestamps[0])
    assert episodes[1]["videos/observation.images.bp/from_timestamp"] == pytest.approx(timestamps[4])
    assert episodes[1]["videos/observation.images.bp/to_timestamp"] == pytest.approx(timestamps[-1] + 0.2)

    written = pd.read_parquet(result.root / "meta/episodes/chunk-000/file-000.parquet")
    visual_mean = np.asarray(written.iloc[0]["stats/observation.images.bp/mean"], dtype=float).reshape(-1)
    assert not np.allclose(visual_mean, 0.123)
    assert np.allclose(visual_mean, colors[1] / 255.0, atol=0.08)
