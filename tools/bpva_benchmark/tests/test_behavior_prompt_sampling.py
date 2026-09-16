import logging
import random
from types import SimpleNamespace

import pytest

from lerobot.datasets.behavior_prompt_dataset import BehaviorPromptConfig, BehaviorPromptLeRobotDataset
from lerobot.policies.BPVA.configuration_bpva import BPVADatasetConfig
from lerobot.policies.BPVAv2.configuration_bpva import BPVAv2DatasetConfig


def make_sampler(task_to_episodes, *, all_episodes=None, policy="neighbor", seed=0, namespace_shared=True):
    dataset = BehaviorPromptLeRobotDataset.__new__(BehaviorPromptLeRobotDataset)
    dataset._task_to_episodes = task_to_episodes
    episode_ids = all_episodes if all_episodes is not None else {
        episode for episodes in task_to_episodes.values() for episode in episodes
    }
    dataset._episode_ranges = {episode: (0, 1, 0) for episode in episode_ids}
    dataset.prompt_cfg = SimpleNamespace(
        same_episode_policy=policy, prompt_episode_namespace_shared=namespace_shared
    )
    dataset.rng = random.Random(seed)
    return dataset


def test_neighbor_middle_episode_only_samples_immediate_neighbors():
    dataset = make_sampler({"task seven": [10, 20, 30]})

    samples = {dataset._sample_prompt_episode(20, "task seven") for _ in range(100)}

    assert samples == {10, 30}


@pytest.mark.parametrize(("current", "expected"), [(10, 20), (30, 20)])
def test_neighbor_boundary_samples_only_existing_side(current, expected):
    dataset = make_sampler({"task seven": [10, 20, 30]})

    assert dataset._sample_prompt_episode(current, "task seven") == expected


def test_neighbor_missing_current_falls_back_to_same_task_random_sampling():
    dataset = make_sampler({"task seven": [10, 20, 30]}, all_episodes={10, 20, 30, 99})

    samples = {dataset._sample_prompt_episode(99, "task seven") for _ in range(100)}

    assert samples == {10, 20, 30}


def test_neighbor_without_neighbor_preserves_avoid_soft_fallback():
    dataset = make_sampler({"task seven": [10]})

    assert dataset._sample_prompt_episode(10, "task seven") == 10


@pytest.mark.parametrize(
    ("policy", "current", "expected"),
    [("allow", 20, {10, 20, 30}), ("avoid", 20, {10, 30}), ("forbid", 20, {10, 30})],
)
def test_existing_policies_keep_their_candidate_behavior(policy, current, expected):
    dataset = make_sampler({"task seven": [10, 20, 30]}, policy=policy)

    samples = {dataset._sample_prompt_episode(current, "task seven") for _ in range(100)}

    assert samples == expected


def test_forbid_without_other_candidate_raises():
    dataset = make_sampler({7: [10]}, policy="forbid")

    with pytest.raises(RuntimeError, match="No different prompt episode"):
        dataset._sample_prompt_episode(10, "task seven")


def test_neibor_alias_is_normalized_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        config = BehaviorPromptConfig(same_episode_policy="  NeiBor  ")

    assert config.same_episode_policy == "neighbor"
    assert "deprecated; use 'neighbor'" in caplog.text


def test_policy_is_stripped_and_lowercased():
    assert BehaviorPromptConfig(same_episode_policy="  ALLOW ").same_episode_policy == "allow"


def test_invalid_policy_raises_clear_error():
    with pytest.raises(ValueError, match=r"Invalid BP same-episode policy 'nearest'.*neighbor"):
        BehaviorPromptConfig(same_episode_policy="nearest")


@pytest.mark.parametrize("config_type", [BPVADatasetConfig, BPVAv2DatasetConfig])
def test_dataset_config_normalizes_neibor_alias(config_type, caplog):
    with caplog.at_level(logging.WARNING):
        config = config_type(repo_id="fake/repo", bp_same_episode_policy=" NeIbOr ")

    assert config.bp_same_episode_policy == "neighbor"
    assert "deprecated; use 'neighbor'" in caplog.text


@pytest.mark.parametrize("config_type", [BPVADatasetConfig, BPVAv2DatasetConfig])
def test_dataset_config_rejects_invalid_policy_early(config_type):
    with pytest.raises(ValueError, match="Invalid BP same-episode policy"):
        config_type(repo_id="fake/repo", bp_same_episode_policy="nearest")


def test_cache_neighbor_degrades_to_same_task_random(caplog):
    with caplog.at_level(logging.WARNING):
        cfg = BehaviorPromptConfig(same_episode_policy="neighbor", prompt_episode_namespace_shared=False)
    assert "degrades to same-task random" in caplog.text
    dataset = make_sampler(
        {"same task": [101, 102, 103]}, policy=cfg.same_episode_policy, namespace_shared=False
    )
    samples = {dataset._sample_prompt_episode(999, "same task") for _ in range(100)}
    assert samples == {101, 102, 103}


def test_cross_dataset_task_indices_match_by_task_name():
    import pandas as pd

    current = SimpleNamespace(repo_id="current", meta=SimpleNamespace(tasks=pd.DataFrame(
        {"task_index": [41]}, index=["pick cube"]
    )))
    prompt = SimpleNamespace(repo_id="cache", meta=SimpleNamespace(tasks=pd.DataFrame(
        {"task_index": [3]}, index=["pick cube"]
    )))
    dataset = BehaviorPromptLeRobotDataset.__new__(BehaviorPromptLeRobotDataset)
    dataset.current_ds = current
    dataset.prompt_ds = prompt
    dataset._current_task_names = dataset._build_task_index_to_name(current)
    dataset._prompt_task_names = dataset._build_task_index_to_name(prompt)
    assert dataset._current_task_identity(41) == "pick cube"
    assert dataset._prompt_task_names[3] == "pick cube"


def test_cache_forbid_rejected_and_avoid_warns(caplog):
    with pytest.raises(ValueError, match="requires current and prompt datasets"):
        BehaviorPromptConfig(same_episode_policy="forbid", prompt_episode_namespace_shared=False)
    with caplog.at_level(logging.WARNING):
        BehaviorPromptConfig(same_episode_policy="avoid", prompt_episode_namespace_shared=False)
    assert "behaves as 'allow'" in caplog.text
