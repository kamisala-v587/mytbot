"""Data-related utility functions used by the trainer main loop."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def move_batch_to_device(
    batch: Any,
    device: torch.device,
    *,
    non_blocking: bool = True,
) -> Any:
    """Move every tensor in ``batch`` to ``device``, recursing into dict/list.

    Returns a new structure with the same shape; the input ``batch`` is not
    mutated. dict / list containers are rebuilt; tensors are moved via
    ``.to(device, non_blocking=...)``; everything else is passed through
    by reference.

    Parameters
    ----------
    batch : Any
        Typically a dict produced by the dataloader, but recursion accepts
        dict / list / tensor / arbitrary leaves.
    device : torch.device
        Target device, typically ``self.device`` on the trainer.
    non_blocking : bool
        Whether to use pinned-memory async copy. Default True because the
        trainer uses pinned loaders; pass False if the dataloader hasn't
        pinned memory.
    """
    # Any mapping, not only ``dict``: a processor hands back a ``BatchFeature``,
    # which is a ``UserDict`` and fails an ``isinstance(x, dict)`` test. Such a
    # batch used to be passed through untouched, and training only worked
    # because one distributed wrapper moved the inputs itself -- on another
    # backend the first embedding lookup met an index still on the host.
    if isinstance(batch, Mapping):
        return {
            k: move_batch_to_device(v, device, non_blocking=non_blocking)
            for k, v in batch.items()
        }
    if isinstance(batch, list):
        return [
            move_batch_to_device(v, device, non_blocking=non_blocking) for v in batch
        ]
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    return batch
