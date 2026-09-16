from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class BPConfig:
    repo_id_file: Path
    bp_cache_root: Path
    mapping_output: Path
    sample_ratio: float = 0.1
    sampling_scope: Literal["dataset", "task"] = "dataset"
    sampling_mode: Literal["random", "first"] = "random"
    seed: int = 42
    bp_camera_keys: tuple[str, ...] = ("observation.images.image0",)
    target_video_mb: float = 5.0
    single_episode_soft_mb: float = 10.0
    single_episode_hard_mb: float = 20.0
    resume: bool = False
    overwrite: bool = False
    dry_run: bool = False

    def validate(self) -> None:
        if not 0 < self.sample_ratio <= 1:
            raise ValueError("sample-ratio 必须在 (0, 1] 内")
        if self.sampling_scope not in {"dataset", "task"}:
            raise ValueError("sampling-scope 必须为 dataset 或 task")
        if self.sampling_mode not in {"random", "first"}:
            raise ValueError("sampling-mode 必须为 random 或 first")
        if not self.bp_camera_keys or len(set(self.bp_camera_keys)) != len(self.bp_camera_keys):
            raise ValueError("bp-camera-keys 必须非空且不能重复")
        if not (0 < self.target_video_mb <= self.single_episode_soft_mb <= self.single_episode_hard_mb):
            raise ValueError("视频阈值必须满足 0 < target <= soft <= hard")
        if self.resume and self.overwrite:
            raise ValueError("--resume 与 --overwrite 不能同时使用")

    def effective_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("repo_id_file", "bp_cache_root", "mapping_output"):
            value[key] = str(value[key])
        value["bp_camera_keys"] = list(self.bp_camera_keys)
        return value
