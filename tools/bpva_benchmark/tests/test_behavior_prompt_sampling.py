import random
from types import SimpleNamespace

from lerobot.datasets.behavior_prompt_dataset import BehaviorPromptLeRobotDataset


def make_sampler(task_to_episodes, *, all_episodes=None, seed=0):
    dataset = BehaviorPromptLeRobotDataset.__new__(BehaviorPromptLeRobotDataset)
    dataset._task_to_episodes = task_to_episodes
    episode_ids = all_episodes if all_episodes is not None else {
        episode for episodes in task_to_episodes.values() for episode in episodes
    }
    dataset._episode_ranges = {episode: (0, 1, 0) for episode in episode_ids}
    dataset.rng = random.Random(seed)
    return dataset


def test_samples_uniformly_within_the_same_task_including_current_episode():
    dataset = make_sampler({"task seven": [10, 20, 30]})

    samples = {dataset._sample_prompt_episode(20, "task seven") for _ in range(100)}

    assert samples == {10, 20, 30}


def test_single_episode_task_returns_that_episode():
    dataset = make_sampler({"task seven": [10]})

    assert dataset._sample_prompt_episode(10, "task seven") == 10


def test_missing_task_falls_back_to_all_episodes_in_the_repo():
    dataset = make_sampler({"task seven": [10, 20]}, all_episodes={10, 20, 99})

    samples = {dataset._sample_prompt_episode(99, "missing task") for _ in range(100)}

    assert samples == {10, 20, 99}


def test_cross_dataset_task_indices_match_by_task_name():
    import pandas as pd

    current = SimpleNamespace(repo_id="current", meta=SimpleNamespace(tasks=pd.DataFrame(
        {"task_index": [41]}, index=["pick cube"]
    )))
    prompt = SimpleNamespace(repo_id="prompt", meta=SimpleNamespace(tasks=pd.DataFrame(
        {"task_index": [3]}, index=["pick cube"]
    )))
    dataset = BehaviorPromptLeRobotDataset.__new__(BehaviorPromptLeRobotDataset)
    dataset.current_ds = current
    dataset.prompt_ds = prompt
    dataset._current_task_names = dataset._build_task_index_to_name(current)
    dataset._prompt_task_names = dataset._build_task_index_to_name(prompt)
    assert dataset._current_task_identity(41) == "pick cube"
    assert dataset._prompt_task_names[3] == "pick cube"
