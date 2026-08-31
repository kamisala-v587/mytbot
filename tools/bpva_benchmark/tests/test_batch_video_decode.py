from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from lerobot.datasets.behavior_prompt_dataset import BehaviorPromptConfig, BehaviorPromptLeRobotDataset
from lerobot.datasets.lerobot_dataset import BatchedVideoDecodeError, LeRobotDataset
from lerobot.policies.BPVA.configuration_bpva import BPVADatasetConfig


class FakeHFDataset:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, index):
        if isinstance(index, list):
            return {key: [self.rows[i][key] for i in index] for key in self.rows[0]}
        return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in self.rows[index].items()}

    def __len__(self):
        return len(self.rows)


def make_dataset(rows, episodes, *, image_transforms=None):
    dataset = object.__new__(LeRobotDataset)
    dataset.hf_dataset = FakeHFDataset(rows)
    dataset._lazy_loading = False
    dataset.writer = None
    dataset.delta_indices = None
    dataset._absolute_to_relative_idx = None
    dataset.root = Path('/dataset')
    dataset.tolerance_s = 0.002
    dataset.video_backend = 'fake'
    dataset.image_transforms = image_transforms
    dataset.meta = SimpleNamespace(
        video_keys=['camera'],
        camera_keys=['camera'],
        episodes=episodes,
        tasks=pd.DataFrame(index=['task-zero']),
        robot_type='fake_robot',
        get_video_file_path=lambda episode_index, video_key: Path('shared.mp4'),
    )
    return dataset


def row(index, episode, timestamp):
    return {
        'index': torch.tensor(index),
        'episode_index': torch.tensor(episode),
        'timestamp': torch.tensor(timestamp),
        'task_index': torch.tensor(0),
    }


def test_get_items_default_is_strictly_item_wise():
    calls = []
    recording_type = type(
        'RecordingDataset',
        (LeRobotDataset,),
        {'__getitem__': lambda self, index: calls.append(index) or {'index': index}},
    )
    dataset = object.__new__(recording_type)

    result = dataset.get_items([2, 1, 2])

    assert calls == [2, 1, 2]
    assert result == [{'index': 2}, {'index': 1}, {'index': 2}]


def test_batch_decode_groups_path_applies_episode_offsets_and_restores_order(monkeypatch):
    episodes = [
        {'videos/camera/from_timestamp': 10.0},
        {'videos/camera/from_timestamp': 20.0},
    ]
    dataset = make_dataset([row(100, 0, 0.25), row(200, 1, 0.5)], episodes)
    calls = []

    def fake_decode(path, timestamps, tolerance, backend):
        calls.append((path, list(timestamps), tolerance, backend))
        return torch.tensor(timestamps).reshape(-1, 1, 1, 1)

    monkeypatch.setattr('lerobot.datasets.lerobot_dataset.decode_video_frames', fake_decode)

    samples = dataset.get_items([1, 0, 1], batch_video_decode=True)

    assert len(calls) == 1
    assert calls[0][0] == Path('/dataset/shared.mp4')
    assert calls[0][1] == pytest.approx([20.5, 10.25, 20.5])
    assert [sample['index'].item() for sample in samples] == [200, 100, 200]
    assert [sample['camera'].item() for sample in samples] == pytest.approx([20.5, 10.25, 20.5])
    assert all(sample['task'] == 'task-zero' for sample in samples)
    assert all(sample['robot_type'] == 'fake_robot' for sample in samples)


def test_batch_decode_splits_single_and_multi_frame_requests(monkeypatch):
    dataset = make_dataset(
        [row(0, 0, 0.0), row(1, 0, 1.0)],
        [{'videos/camera/from_timestamp': 5.0}],
    )
    dataset.delta_indices = {'camera': [0]}
    dataset._get_query_indices = lambda absolute, episode: ({'camera': [absolute]}, {})
    dataset._query_hf_dataset = lambda query: {}
    dataset._get_query_timestamps = lambda timestamp, query: {
        'camera': [timestamp] if timestamp == 0 else [timestamp, timestamp + 0.5]
    }
    monkeypatch.setattr(
        'lerobot.datasets.lerobot_dataset.decode_video_frames',
        lambda path, timestamps, tolerance, backend: torch.arange(len(timestamps) * 2).reshape(-1, 2, 1, 1),
    )

    first, second = dataset.get_items([0, 1], batch_video_decode=True)

    assert first['camera'].shape == (2, 1, 1)
    assert second['camera'].shape == (2, 2, 1, 1)
    assert torch.equal(first['camera'], torch.tensor([[[0]], [[1]]]))
    assert torch.equal(second['camera'], torch.tensor([[[[2]], [[3]]], [[[4]], [[5]]]]))


def test_batch_decode_failure_is_specialized_and_precedes_transforms(monkeypatch):
    transform_calls = []
    dataset = make_dataset(
        [row(0, 0, 0.0)],
        [{'videos/camera/from_timestamp': 0.0}],
        image_transforms=lambda image: transform_calls.append(image) or image,
    )
    monkeypatch.setattr(
        'lerobot.datasets.lerobot_dataset.decode_video_frames',
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError('decode failed')),
    )

    with pytest.raises(BatchedVideoDecodeError, match='prepare or decode'):
        dataset.get_items([0], batch_video_decode=True)
    assert transform_calls == []


class PromptDataset:
    def __init__(self, *, fail_batch=False):
        self.fail_batch = fail_batch
        self.batch_calls = []
        self.item_calls = []
        self.meta = SimpleNamespace(
            camera_keys=['camera'], robot_type='egodex_v', features={}
        )
        self.hf_dataset = FakeHFDataset([row(10, 3, 0.0), row(11, 3, 1.0)])

    def get_items(self, indices, *, batch_video_decode=False):
        self.batch_calls.append((list(indices), batch_video_decode))
        if self.fail_batch:
            raise BatchedVideoDecodeError('retryable')
        return [self[index] for index in indices]

    def __getitem__(self, index):
        self.item_calls.append(index)
        return {'camera': torch.tensor([float(index)]), 'index': torch.tensor(10 + index)}


def make_behavior(prompt_dataset, *, enabled):
    dataset = object.__new__(BehaviorPromptLeRobotDataset)
    dataset.prompt_ds = prompt_dataset
    dataset.current_ds = SimpleNamespace(_missing_state_action_placeholder_dim=1)
    dataset.prompt_cfg = BehaviorPromptConfig(
        prompt_action_chunk_size=1, num_chunks=2, batch_prompt_video_decode=enabled
    )
    dataset._episode_ranges = {3: (0, 2, 10)}
    dataset._batch_decode_warned_workers = set()
    dataset._sample_prompt_episode = lambda current_episode, current_task: 3
    return dataset


def test_behavior_prompt_uses_batch_api_and_falls_back_on_specialized_error(caplog):
    successful = PromptDataset()
    make_behavior(successful, enabled=True)._build_prompt(
        {'episode_index': torch.tensor(0), 'task_index': torch.tensor(0)}
    )
    assert successful.batch_calls == [([0, 1], True)]

    failing = PromptDataset(fail_batch=True)
    behavior = make_behavior(failing, enabled=True)
    current = {'episode_index': torch.tensor(0), 'task_index': torch.tensor(0)}
    behavior._build_prompt(current)
    behavior._build_prompt(current)
    assert failing.item_calls == [0, 1, 0, 1]
    assert caplog.text.count('falling back to item-wise decode') == 1


def test_batch_decode_config_defaults_off():
    assert BehaviorPromptConfig().batch_prompt_video_decode is False
    assert BPVADatasetConfig(repo_id='fake/repo').batch_prompt_video_decode is False
