"""One training step of the residual refiner, inside the model's forward.

The refiner learns the one mapping demonstrations never show: from the base
model's own output to the demonstration. A step therefore has to *sample* from
the base model before it can score anything, and the sample has to come from
the same computation serving uses -- same integration budget, same cached
observation -- or the residual is fitted to a distribution that is never served.

So this runs inside ``forward``. Two reasons, both load-bearing:

- the sampler reads the model's weights, and under FSDP those are only
  materialized while the module's own forward is running;
- the refinement pass must reuse the closure the sampler built over the cached
  prefix, which is what makes it the same computation.

The candidate is integrated without gradients and detached: the refiner learns
to correct the base model, not to change what it produces. Gradients are
enabled only for the refinement pass, which is sound -- the cached prefix is a
constant as far as the twin weights are concerned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
from transformers.utils import ModelOutput

from wall_x.model.core.refiner.refine import RefinerConfig, refine
from wall_x.model.core.refiner.train import (
    Margins,
    margins_for,
    measure,
    refinement_loss,
    within_each,
)

# The first ``ARMS * ARM_WIDTH`` dimensions are the arms; the rest is padding
# this embodiment does not provide.
ARMS = 2
ARM_WIDTH = 10


@dataclass
class RefinerStepOutput(ModelOutput):
    """What a refiner step reports. ``loss`` is what the trainer reads.

    A ``ModelOutput`` rather than a plain dataclass, because this is the return
    value of the model's ``forward``: FSDP walks that value to find the tensors
    it has to re-gather the parameters for in backward, and it can only walk
    containers it knows. An opaque object means the backward runs against
    parameters that were already freed.
    """

    loss: Optional[torch.Tensor] = None
    refiner_loss: Optional[torch.Tensor] = None
    kept: Optional[int] = None
    dropped: Optional[int] = None
    correction: Optional[float] = None


class RefinerTrainingMixin:
    """``refiner_train_forward``: sample from the base model, correct, score."""

    def refiner_train_forward(
        self,
        *,
        action_chunk: torch.Tensor,
        dof_mask: Optional[torch.Tensor] = None,
        dataset_names: Optional[Sequence[str]] = None,
        refiner_margins: Optional[Mapping[str, Margins]] = None,
        refiner_default_margins: Optional[Margins] = None,
        task_names: Optional[Sequence[str]] = None,
        **observation: Any,
    ) -> RefinerStepOutput:
        config = RefinerConfig.from_vla_config(self.config.vla_config)
        if not config.enabled:
            raise ValueError(
                "a refiner training step was requested but model.refiner.enabled "
                "is false, so there are no twin weights to train")
        if action_chunk is None:
            raise ValueError(
                "a refiner step scores a correction against a demonstration; this "
                "batch carries none. Selection samples cannot train the refiner -- "
                "set data.robodojo.selection to false for stage 2")

        batch, horizon, width = action_chunk.shape
        for key in ("sample_time", "moe_token_types", "frame_index", "labels"):
            observation.pop(key, None)

        # The sampler reuses the prefix K/V across its integration steps, and
        # gradient checkpointing refuses to produce a cache while the module is
        # in training mode -- the postfix would then attend to nothing. Nothing
        # else here depends on training mode: this model has no dropout, and
        # the refinement pass is a few dozen tokens, so recomputation would buy
        # nothing anyway. Gradients still flow in eval mode.
        was_training = self.training
        self.eval()
        # A step runs the decoder many times -- once to prefill, once per
        # integration step, once per refinement pass -- and then backward
        # through the last of them. FSDP frees the gathered parameters after
        # each forward by default, and the backward then finds nothing to read.
        # Hold them for the step instead.
        resharding = getattr(self, "set_reshard_after_forward", None)
        if resharding is not None:
            resharding(False)
        try:
            with torch.no_grad():
                sample = self.sample_flow_with_cache(
                    action_horizon=horizon,
                    action_dim=width,
                    num_inference_timesteps=config.num_steps,
                    dof_mask=dof_mask,
                    dataset_names=dataset_names,
                    **observation,
                )
                candidate = sample.candidate.detach()
                if dof_mask is not None:
                    # Dimensions this embodiment does not provide are written to
                    # one value, so the refiner is not shown whatever the
                    # sampler happened to leave in them.
                    candidate = torch.where(
                        dof_mask.to(torch.bool), candidate,
                        torch.zeros_like(candidate))

            valid = None
            if dof_mask is not None:
                # A step whose dimensions are all padding is a step the episode
                # did not have; it is not a correction anyone can learn.
                valid = dof_mask[..., :ARMS * ARM_WIDTH].amax(dim=-1) > 0

            keep = None
            if refiner_margins is not None:
                keep = self._refiner_keep(
                    candidate, action_chunk, dataset_names, valid,
                    refiner_margins, refiner_default_margins, task_names, batch)

            with torch.enable_grad():
                refined = refine(
                    self.model,
                    candidate,
                    dof_mask=dof_mask,
                    embed_dtype=sample.embed_dtype,
                    run_lm=sample.run_lm,
                    config=config,
                )
                loss = refinement_loss(
                    refined, action_chunk, dof_mask=dof_mask, valid=valid, keep=keep)
        finally:
            if resharding is not None:
                resharding(True)
            self.train(was_training)

        kept = int(keep.sum()) if keep is not None else batch
        return RefinerStepOutput(
            loss=loss,
            refiner_loss=loss.detach(),
            kept=kept,
            dropped=batch - kept,
            correction=float((refined - candidate).abs().mean().detach()),
        )

    def _refiner_keep(
        self,
        candidate: torch.Tensor,
        demonstration: torch.Tensor,
        dataset_names,
        valid,
        table: Mapping[str, Margins],
        default: Optional[Margins],
        tasks: Optional[Sequence[str]],
        batch: int,
    ) -> torch.Tensor:
        """Which candidates are close enough to the demonstration to be scored.

        The margins are metres, radians and gripper steps, so the comparison
        happens in those units rather than in the normalized space the loss
        uses.
        """
        if tasks is None:
            raise ValueError(
                "margins are per task, so a batch has to say which task each of "
                "its samples came from")
        if len(tasks) != batch:
            raise ValueError(
                f"{len(tasks)} task names for {batch} samples")
        normalizer = self.action_preprocessor.normalizer_action
        arms = ARMS * ARM_WIDTH
        physical = [normalizer.unnormalize_data(x, dataset_names)[..., :arms].float()
                    for x in (candidate, demonstration)]
        deviation = measure(physical[0], physical[1], arms=ARMS, valid=valid)
        return within_each(deviation, margins_for(tasks, table, default))


__all__ = ["RefinerStepOutput", "RefinerTrainingMixin"]
