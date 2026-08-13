"""单仓 LeRobot 数据读取与统计。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transforms.constants import get_feature_mapping, get_mask_mapping, infer_embodiment_variant
from lerobot.utils.constants import ACTION, OBS_STATE

from .core import RunningStats


def _dataset(repo_id: str, root: str | None) -> LeRobotDataset:
    path = Path(repo_id)
    return LeRobotDataset(str(path)) if path.is_absolute() else LeRobotDataset(repo_id, root=root)


def _stack(dataset: LeRobotDataset, key: str, indices: np.ndarray) -> torch.Tensor:
    """读取一列并统一为 Tensor，兼容 hf transform 返回 Tensor 或 Python/NumPy 值。"""
    if indices.size == 0:
        raise ValueError(f"不能读取空索引：feature={key}")
    selected = dataset.hf_dataset.select(indices.astype(np.int64, copy=False).tolist())
    values = selected[key][:]
    if isinstance(values, torch.Tensor):
        return values
    if isinstance(values, np.ndarray):
        return torch.as_tensor(values)
    if not values:
        raise ValueError(f"hf_dataset 返回空列：feature={key}")
    return torch.stack([value if isinstance(value, torch.Tensor) else torch.as_tensor(value) for value in values])


def _matrix(value: torch.Tensor) -> torch.Tensor:
    """把逐帧 scalar/vector 规范为 [frames, dim]。"""
    if value.ndim == 0:
        return value.reshape(1, 1)
    return value if value.ndim > 1 else value[:, None]


def _feature_shape(feature: dict[str, Any]) -> list[int]:
    shape = feature.get("shape", ())
    if isinstance(shape, int):
        return [int(shape)]
    return [int(dim) for dim in shape]


def _feature_dim(feature: dict[str, Any]) -> int:
    shape = _feature_shape(feature)
    return int(np.prod(shape)) if shape else 1


def _delta_values(action: torch.Tensor, state: torch.Tensor, starts: np.ndarray,
                  chunk_size: int, mask: torch.Tensor) -> torch.Tensor:
    """构造 [window, chunk, dim] delta，并显式控制 mask 广播。"""
    gather = torch.as_tensor(starts[:, None] + np.arange(chunk_size)[None, :], dtype=torch.long)
    state_base = torch.where(mask.reshape(1, -1), state, torch.zeros_like(state))
    return action[gather] - state_base[:, None, :]


def _validate_delta_layout(repo_id: str, features: dict[str, Any], mapping: dict[str, Any], mask: torch.Tensor) -> None:
    state_dim = sum(_feature_dim(features[key]) for key in mapping[OBS_STATE])
    action_dim = sum(_feature_dim(features[key]) for key in mapping[ACTION])
    if state_dim != action_dim or mask.numel() != action_dim:
        raise ValueError(
            f"{repo_id} delta mapping 维度不兼容：state={state_dim}, action={action_dim}, mask={mask.numel()}"
        )


def _normalize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _sample_starts(repo_id: str, from_ids: np.ndarray, to_ids: np.ndarray, chunk_size: int,
                   per_episode: int | None, per_repo: int | None, seed: int) -> dict[int, np.ndarray]:
    digest = hashlib.sha256(f"{seed}|{repo_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    selected: list[tuple[int, int]] = []
    for episode, (start, end) in enumerate(zip(from_ids, to_ids)):
        count = max(0, int(end - start) - chunk_size + 1)
        starts = np.arange(count, dtype=np.int64)
        if per_episode is not None and count > per_episode:
            starts = np.sort(rng.choice(starts, per_episode, replace=False))
        selected.extend((episode, int(index)) for index in starts)
    if per_repo is not None and len(selected) > per_repo:
        chosen = np.sort(rng.choice(len(selected), per_repo, replace=False))
        selected = [selected[int(i)] for i in chosen]
    output: dict[int, list[int]] = {}
    for episode, start in selected:
        output.setdefault(episode, []).append(start)
    return {episode: np.asarray(starts, dtype=np.int64) for episode, starts in output.items()}


def enrich_cached_result(result: dict[str, Any], root: str | None) -> dict[str, Any]:
    """为旧缓存补齐 mapping/mask/compute，并移除历史视觉统计。"""
    if all(key in result for key in ("mapping", "mask", "compute")):
        cleaned = dict(result)
        cleaned.pop("visual_stats", None)
        return cleaned
    dataset = _dataset(result["repo_id"], root)
    features = dataset.meta.features
    mapping = get_feature_mapping(dataset.meta.robot_type, features)
    mask = torch.as_tensor(get_mask_mapping(dataset.meta.robot_type, features), dtype=torch.bool)
    result = dict(result)
    result.pop("visual_stats", None)
    result["mapping"] = {OBS_STATE: list(mapping[OBS_STATE]), ACTION: list(mapping[ACTION])}
    result["mask"] = _normalize(mask)
    result.setdefault("compute", {"legacy_cache": True, "skip_action": bool(result.get("skip_action_stats", False))})
    if result["compute"].get("skip_action"):
        payload = dict(result["payload"])
        for key in mapping[ACTION]:
            payload[key] = None
        result["payload"] = payload
    return result


def compute_one(job: tuple[str, str, int, str | None, int | None, int | None, int, tuple[str, ...]]) -> dict[str, Any]:
    repo_id, action_mode, chunk_size, root, per_episode, per_repo, seed, skip_types = job
    dataset = _dataset(repo_id, root)
    features = dataset.meta.features
    robot_type = dataset.meta.robot_type
    resolved = infer_embodiment_variant(robot_type, features)
    mapping = get_feature_mapping(robot_type, features)
    mask_tensor = get_mask_mapping(robot_type, features)
    mask = torch.as_tensor(mask_tensor, dtype=torch.bool)
    visual_keys = list(dataset.meta.video_keys) + list(dataset.meta.image_keys)
    keys = [key for key in features if key not in visual_keys]
    action_keys = list(mapping[ACTION])
    missing = [key for key in mapping[OBS_STATE] + action_keys if key not in features]
    if missing:
        raise ValueError(f"{repo_id} mapping 引用了缺失 feature：{missing}")
    if action_mode == "delta":
        _validate_delta_layout(repo_id, features, mapping, mask)
    stats = {key: RunningStats() for key in keys}
    skip_action = robot_type in skip_types or resolved in skip_types
    from_ids = np.asarray(dataset.meta.episodes["dataset_from_index"], dtype=np.int64)
    to_ids = np.asarray(dataset.meta.episodes["dataset_to_index"], dtype=np.int64)
    sampled = per_episode is not None or per_repo is not None
    starts_map = _sample_starts(repo_id, from_ids, to_ids, chunk_size, per_episode, per_repo, seed) if sampled else None
    valid_windows = 0

    for episode, (start, end) in enumerate(zip(from_ids, to_ids)):
        frame_indices = np.arange(start, end, dtype=np.int64)
        episode_len = len(frame_indices)
        # 非 action 永远扫描全帧；abs action 也永远扫描全帧，均不受 chunk/sampling 影响。
        for key in keys:
            if key not in action_keys or (action_mode == "abs" and not skip_action):
                stats[key].update(_stack(dataset, key, frame_indices).cpu().numpy())
        if action_mode != "delta" or skip_action or episode_len < chunk_size:
            continue
        starts = starts_map.get(episode, np.empty(0, dtype=np.int64)) if sampled else np.arange(episode_len - chunk_size + 1)
        if not len(starts):
            continue
        valid_windows += len(starts)
        action = torch.cat([_matrix(_stack(dataset, key, frame_indices)).reshape(episode_len, -1)
                            for key in action_keys], dim=-1)
        anchor_indices = int(start) + starts
        state = torch.cat([_matrix(_stack(dataset, key, anchor_indices)).reshape(len(starts), -1)
                           for key in mapping[OBS_STATE]], dim=-1)
        delta = _delta_values(action, state, starts, chunk_size, mask)
        offset = 0
        for key in action_keys:
            dim = _feature_dim(features[key])
            stats[key].update(delta[..., offset:offset + dim].cpu().numpy())
            offset += dim

    payload = {key: (None if skip_action and key in action_keys else value.to_payload()) for key, value in stats.items()}
    return {
        "repo_id": repo_id, "repo_path": str(Path(dataset.root).resolve()), "robot_type": robot_type,
        "resolved_robot_type": resolved, "keys": keys,
        "shapes": {key: _feature_shape(features[key]) for key in keys},
        "mapping": {OBS_STATE: list(mapping[OBS_STATE]), ACTION: action_keys}, "mask": _normalize(mask),
        "payload": payload,
        "compute": {"total_frames": int(sum(to_ids - from_ids)), "total_episodes": len(from_ids),
                    "valid_delta_windows": int(valid_windows), "sampled": sampled,
                    "skip_action": skip_action, "action_mode": action_mode,
                    "chunk_size": chunk_size if action_mode == "delta" else None},
    }
