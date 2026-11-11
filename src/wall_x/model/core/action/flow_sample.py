"""A finished flow sample, and the means to run the decoder over it again.

A second pass over a sampled action chunk -- the residual refiner, whether it
is being served or trained -- has to see the same observation the chunk was
sampled from. That observation lives in the sampler's own prefix K/V and in the
closure that evaluates the action postfix against it, so the sampler hands both
back rather than letting a second caller rebuild them.

Kept here, away from the model package, so that the sampler, the refiner and
their tests all name the same record without pulling the decoder in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch


@dataclass(frozen=True)
class CachedFlowSample:
    """One candidate chunk, with the decoder pass that produced it.

    ``run_lm(action_embeds, adarms_conds)`` evaluates the action postfix
    against the prefix K/V the candidate was sampled with, exactly as the
    integration steps did. ``num_steps`` is the integration budget the
    candidate came from: the refiner learns the residual of that distribution,
    so it travels with the sample rather than being assumed.
    """

    candidate: torch.Tensor
    run_lm: Callable[[torch.Tensor, Any], Any]
    embed_dtype: torch.dtype
    dof_mask: Optional[torch.Tensor]
    num_steps: int


__all__ = ["CachedFlowSample"]
