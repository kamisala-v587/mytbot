"""Norm stats 内部实现；公开 CLI 仅为 tools/run_norm_stats.py。"""

from .core import RunningStats

__all__ = ["RunningStats"]
