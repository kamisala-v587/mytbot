import random
from types import SimpleNamespace

from lerobot.datasets.behavior_prompt_dataset import BehaviorPromptLeRobotDataset


def make_sampler(task_to_episodes, *, all_episodes=None, seed=0, mode="random"):
    dataset = BehaviorPromptLeRobotDataset.__new__(BehaviorPromptLeRobotDataset)
    dataset._task_to_episodes = task_to_episodes
    episode_ids = all_episodes if all_episodes is not None else {
        episode for episodes in task_to_episodes.values() for episode in episodes
    }
    dataset._episode_ranges = {episode: (0, 1, 0) for episode in episode_ids}
    dataset.prompt_cfg = SimpleNamespace(bp_sampling_mode=mode)
    dataset.rng = random.Random(seed)
    return dataset


def test_random_samples_uniformly_from_all_episodes_in_current_repo():
    dataset = make_sampler({"task seven": [10, 20, 30]})

    samples = {dataset._sample_prompt_episode(20, "task seven") for _ in range(100)}

    assert samples == {10, 20, 30}


def test_single_episode_task_returns_that_episode():
    dataset = make_sampler({"task seven": [10]})

    assert dataset._sample_prompt_episode(10, "task seven") == 10


def test_random_uses_all_repo_episodes_even_when_task_is_missing():
    dataset = make_sampler({"task seven": [10, 20]}, all_episodes={10, 20, 99})

    samples = {dataset._sample_prompt_episode(99, "missing task") for _ in range(100)}

    assert samples == {10, 20, 99}




def test_random_samples_other_tasks_in_loaded_repo():
    dataset = make_sampler(
        {"task seven": [10, 20], "task eight": [30]},
        all_episodes={10, 20, 30, 99},
        mode="random",
    )

    samples = {dataset._sample_prompt_episode(20, "task seven") for _ in range(100)}

    assert samples == {10, 20, 30, 99}


def test_avoid_excludes_current_episode_but_not_other_tasks():
    dataset = make_sampler(
        {"task seven": [10, 20], "task eight": [30]},
        all_episodes={10, 20, 30, 99},
        mode="avoid",
    )

    samples = {dataset._sample_prompt_episode(20, "task seven") for _ in range(100)}

    assert samples == {10, 30, 99}


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


def test_random_in_task_does_not_sample_other_task():
    dataset = make_sampler({"task": [10, 20], "other": [30]}, mode="random_in_task")
    assert {dataset._sample_prompt_episode(10, "task") for _ in range(100)} == {10, 20}


def test_avoid_in_task_masks_singleton_instead_of_repo_fallback():
    dataset = make_sampler({"task": [10], "other": [20]}, mode="avoid_in_task")
    assert dataset._sample_prompt_episode(10, "task") is None


def test_avoid_in_task_samples_other_matching_episode():
    dataset = make_sampler({"task": [10, 20], "other": [30]}, mode="avoid_in_task")
    assert dataset._sample_prompt_episode(10, "task") == 20


def test_repo_aliases_match_original_modes():
    for mode, alias in [("random", "random_in_repo"), ("avoid", "avoid_in_repo")]:
        a = make_sampler({"task": [10], "other": [20, 30]}, mode=mode)
        b = make_sampler({"task": [10], "other": [20, 30]}, mode=alias)
        assert [a._sample_prompt_episode(10, "task") for _ in range(100)] == [b._sample_prompt_episode(10, "task") for _ in range(100)]
