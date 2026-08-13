"""Norm stats 的纯统计与 schema 工具。"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


class RunningStats:
    """以 float64 保存可合并的向量统计量。"""

    def __init__(self) -> None:
        self.count = 0
        self.mean: np.ndarray | None = None
        self.mean_sq: np.ndarray | None = None
        self.min: np.ndarray | None = None
        self.max: np.ndarray | None = None

    def update(self, batch: Any) -> None:
        values = np.asarray(batch, dtype=np.float64)
        if values.ndim == 0:
            values = values.reshape(1, 1)
        elif values.ndim == 1:
            values = values[:, None]
        else:
            values = values.reshape(-1, values.shape[-1])
        if values.shape[0] == 0:
            return
        other = RunningStats.from_payload({
            "count": int(values.shape[0]),
            "mean": values.mean(axis=0).tolist(),
            "mean_sq": np.square(values).mean(axis=0).tolist(),
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
        })
        self.merge(other)

    def merge(self, other: "RunningStats") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean.copy()
            self.mean_sq = other.mean_sq.copy()
            self.min = other.min.copy()
            self.max = other.max.copy()
            return
        if self.mean.shape != other.mean.shape:
            raise ValueError(f"统计维度不一致：{self.mean.shape} != {other.mean.shape}")
        total = self.count + other.count
        self.mean = (self.mean * self.count + other.mean * other.count) / total
        self.mean_sq = (self.mean_sq * self.count + other.mean_sq * other.count) / total
        self.min = np.minimum(self.min, other.min)
        self.max = np.maximum(self.max, other.max)
        self.count = total

    def to_payload(self) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0, "mean": None, "mean_sq": None, "min": None, "max": None}
        return {
            "count": self.count, "mean": self.mean.tolist(), "mean_sq": self.mean_sq.tolist(),
            "min": self.min.tolist(), "max": self.max.tolist(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RunningStats":
        stats = cls()
        count = payload.get("count", 0)
        if isinstance(count, list):
            count = count[0] if count else 0
        stats.count = int(count)
        if stats.count:
            for name in ("mean", "mean_sq", "min", "max"):
                value = payload.get(name)
                if value is None:
                    raise ValueError(f"非空统计缺少 {name}")
                setattr(stats, name, np.asarray(value, dtype=np.float64))
        return stats

    def statistics(self) -> dict[str, Any]:
        if self.count == 0:
            raise ValueError("统计量为空，不能生成最终结果")
        std = np.sqrt(np.maximum(self.mean_sq - np.square(self.mean), 0.0))
        return {"min": self.min.tolist(), "max": self.max.tolist(), "mean": self.mean.tolist(),
                "std": std.tolist(), "count": [self.count]}


def schema_signature(result: dict[str, Any]) -> dict[str, Any]:
    """default 聚合要求所有非视觉 schema 和变换定义完全一致。"""
    return {key: result.get(key) for key in ("keys", "shapes", "mapping", "mask")}


def validate_schema(results: list[dict[str, Any]]) -> None:
    if not results:
        raise ValueError("没有可聚合的数据集")
    expected = schema_signature(results[0])
    for result in results[1:]:
        if schema_signature(result) != expected:
            raise ValueError(f"非视觉 feature/mapping/mask 不一致：{results[0]['repo_id']} vs {result['repo_id']}")


def merge_payloads(results: list[dict[str, Any]]) -> dict[str, Any]:
    validate_schema(results)
    output: dict[str, Any] = {}
    for key in results[0]["keys"]:
        merged = RunningStats()
        for result in results:
            payload = result["payload"].get(key)
            if payload is not None:  # skip-action 不制造样本数，也不参与该 feature 聚合
                merged.merge(RunningStats.from_payload(payload))
        if merged.count:
            output[key] = merged.statistics()
        elif all(result["payload"].get(key) is None for result in results):
            # 全部 repo 主动跳过 action 时保留可消费的零占位，但 count 明确为 0。
            shape = results[0]["shapes"][key]
            dim = int(shape[0]) if shape else 1
            zeros = [0.0] * dim
            output[key] = {"min": zeros, "max": zeros, "mean": zeros, "std": zeros, "count": [0]}
    return output



