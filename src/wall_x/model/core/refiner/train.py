"""Training the refiner: what it learns from, and what it is not asked to learn.

The refiner learns one mapping the base model cannot learn from demonstrations
alone -- "my own output" to "the demonstration" -- so its training needs the
base model's own samples, not noised demonstrations. A step therefore samples a
candidate from the frozen base, refines it, and scores the refinement against
the demonstration.

Not every candidate deserves to be scored. When the base model picks a
different reasonable way to do the task, or misses badly, pulling the refiner
towards this particular demonstration teaches it to undo a decision rather than
to correct an error. So a candidate too far from the demonstration is dropped
from the loss, by a per-task margin measured in metres, radians and steps.
Those margins are a property of the base model being refined, so they are
calibrated against it rather than carried around as constants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

# The layout of one arm inside an action: position, 6D rotation, gripper.
ARM_WIDTH = 10
POSITION, ROTATION, GRIPPER = slice(0, 3), slice(3, 9), slice(9, 10)


@dataclass(frozen=True)
class Margins:
    """How far a candidate may be from the demonstration and still be scored.

    ``position`` is metres, ``rotation`` radians, and ``gripper_steps`` counts
    the steps whose gripper disagrees by more than ``gripper`` of its range.
    A margin of ``None`` is not checked.
    """

    position: Optional[float] = None
    rotation: Optional[float] = None
    gripper: float = 0.5
    gripper_steps: Optional[int] = None


@dataclass(frozen=True)
class Deviation:
    """How far each candidate in a batch is from its demonstration.

    One number per candidate, taken over every step and both arms, which is
    what the margins are compared against.
    """

    position: torch.Tensor
    rotation: torch.Tensor
    gripper_steps: torch.Tensor


def _rotation_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The angle between two 6D-encoded orientations, in radians.

    The 6D encoding holds two rows of a rotation matrix; the third is their
    cross product. The angle comes from the trace of the relative rotation.

    A 6D block of zeros is not an orientation. Normalizing it would quietly
    produce an angle of 120 degrees -- enough to drop the candidate -- so it is
    refused instead: it means the layout is wrong or the dimensions are padding
    that should not have been measured.
    """
    def matrix(six: torch.Tensor) -> torch.Tensor:
        first, second = six[..., :3], six[..., 3:]
        lengths = torch.stack([first.norm(dim=-1), second.norm(dim=-1)], dim=-1)
        if bool((lengths < 1e-6).any()):
            raise ValueError(
                "a 6D orientation of zero length cannot be compared; the action "
                "layout does not match, or padding dimensions are being measured")
        first = first / first.norm(dim=-1, keepdim=True)
        second = second - first * (first * second).sum(-1, keepdim=True)
        second = second / second.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.stack([first, second, torch.cross(first, second, dim=-1)], dim=-2)

    relative = matrix(a) @ matrix(b).transpose(-1, -2)
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.arccos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))


def measure(
    candidate: torch.Tensor,
    demonstration: torch.Tensor,
    *,
    arms: int = 2,
    valid: Optional[torch.Tensor] = None,
    gripper_tolerance: float = 0.5,
) -> Deviation:
    """How far ``candidate`` is from ``demonstration``, per candidate.

    Both are ``[batch, steps, dims]`` of unnormalized actions, laid out as
    ``arms`` blocks of position, 6D rotation and gripper. ``valid`` masks the
    steps that are real, so padding at the end of an episode is not measured.
    """
    if candidate.shape != demonstration.shape:
        raise ValueError(
            f"candidate is {tuple(candidate.shape)} but the demonstration is "
            f"{tuple(demonstration.shape)}")
    batch, steps, _ = candidate.shape
    if valid is None:
        valid = torch.ones((batch, steps), dtype=torch.bool, device=candidate.device)
    position = torch.zeros(batch, device=candidate.device)
    rotation = torch.zeros(batch, device=candidate.device)
    disagreeing = torch.zeros((batch, steps), dtype=torch.bool, device=candidate.device)
    for arm in range(arms):
        base = arm * ARM_WIDTH
        a, b = candidate[..., base:base + ARM_WIDTH], demonstration[..., base:base + ARM_WIDTH]
        distance = (a[..., POSITION] - b[..., POSITION]).norm(dim=-1)
        # Padding carries no orientation, and a block of zeros is refused as an
        # orientation -- rightly, but not over steps that do not count. The
        # masked steps are given one that compares to itself; the angles they
        # produce are masked out below anyway.
        upright = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                               device=candidate.device, dtype=candidate.dtype)
        counts = valid.unsqueeze(-1)
        angle = _rotation_angle(
            torch.where(counts, a[..., ROTATION], upright.expand_as(a[..., ROTATION])),
            torch.where(counts, b[..., ROTATION], upright.expand_as(b[..., ROTATION])))
        position = torch.maximum(position, distance.masked_fill(~valid, 0.0).amax(dim=-1))
        rotation = torch.maximum(rotation, angle.masked_fill(~valid, 0.0).amax(dim=-1))
        gap = (a[..., GRIPPER] - b[..., GRIPPER]).abs().squeeze(-1)
        disagreeing |= (gap > gripper_tolerance) & valid
    return Deviation(position=position, rotation=rotation,
                     gripper_steps=disagreeing.sum(dim=-1))


def within(deviation: Deviation, margins: Margins) -> torch.Tensor:
    """Which candidates are close enough to be scored."""
    keep = torch.ones_like(deviation.position, dtype=torch.bool)
    if margins.position is not None:
        keep &= deviation.position <= margins.position
    if margins.rotation is not None:
        keep &= deviation.rotation <= margins.rotation
    if margins.gripper_steps is not None:
        keep &= deviation.gripper_steps <= margins.gripper_steps
    return keep


def within_each(deviation: Deviation, margins: Sequence[Margins]) -> torch.Tensor:
    """Which candidates are close enough, with one margin per candidate.

    A batch mixes tasks and the margins are per task, so the comparison is a
    vector one. A margin of ``None`` leaves its candidate unconstrained, which
    is what an infinite limit expresses.
    """
    count = int(deviation.position.shape[0])
    if len(margins) != count:
        raise ValueError(f"{len(margins)} margins for {count} candidates")

    def limit(attribute: str) -> torch.Tensor:
        values = [getattr(margin, attribute) for margin in margins]
        return torch.tensor(
            [float("inf") if value is None else float(value) for value in values],
            device=deviation.position.device, dtype=torch.float32)

    return (
        (deviation.position.float() <= limit("position"))
        & (deviation.rotation.float() <= limit("rotation"))
        & (deviation.gripper_steps.float() <= limit("gripper_steps"))
    )


def margins_for(tasks: Sequence[str], table: Mapping[str, Margins],
                default: Optional[Margins] = None) -> Sequence[Margins]:
    """The margins of each candidate's task.

    A task with no entry falls back to ``default``; without one, that is an
    error rather than an unfiltered task, because an unfiltered task trains on
    candidates the margins were meant to drop.
    """
    out = []
    for task in tasks:
        found = table.get(task, default)
        if found is None:
            raise KeyError(
                f"no margins calibrated for task {task!r}; calibrate it or pass a "
                "default, rather than training it unfiltered")
        out.append(found)
    return out


def refinement_loss(
    refined: torch.Tensor,
    demonstration: torch.Tensor,
    *,
    dof_mask: Optional[torch.Tensor] = None,
    valid: Optional[torch.Tensor] = None,
    keep: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """How far the refinement still is from the demonstration.

    The mean absolute error over what is being trained: the dimensions this
    embodiment provides (``dof_mask``), the steps that are real (``valid``) and
    the candidates worth scoring (``keep``). Returns a zero that still carries
    a gradient path when nothing is left to score, so a step whose whole batch
    was dropped does not break the backward pass.
    """
    if refined.shape != demonstration.shape:
        raise ValueError(
            f"refinement is {tuple(refined.shape)} but the demonstration is "
            f"{tuple(demonstration.shape)}")
    error = (refined - demonstration).abs()
    weight = torch.ones_like(error)
    if dof_mask is not None:
        weight = weight * dof_mask.to(weight.dtype)
    if valid is not None:
        weight = weight * valid.to(weight.dtype).unsqueeze(-1)
    if keep is not None:
        weight = weight * keep.to(weight.dtype).reshape(-1, 1, 1)
    total = weight.sum()
    if total == 0:
        return (error * weight).sum()
    return (error * weight).sum() / total


__all__ = [
    "ARM_WIDTH",
    "Deviation",
    "Margins",
    "margins_for",
    "measure",
    "refinement_loss",
    "load_margins",
    "within",
    "within_each",
]


def load_margins(path: str) -> Tuple[Dict[str, Margins], Optional[Margins]]:
    """Read a margins file written by ``scripts/calibrate_refiner_margins.py``.

    Returns the per-task table and the default for a task the file does not
    mention. There is no default unless the file names one: a task nobody
    measured would otherwise train unfiltered, which is the thing the margins
    exist to prevent.

    Tasks are keyed by their instruction, which is what identifies a task in
    this dataset.
    """
    with open(path, encoding="utf-8") as stream:
        record = json.load(stream)
    schema = record.get("schema")
    if schema != "refiner_margins_v1":
        raise ValueError(
            f"{path} declares schema {schema!r}; this reads refiner_margins_v1")
    tasks = record.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError(f"{path} carries no measured tasks")

    def parse(entry: Mapping[str, Any]) -> Margins:
        return Margins(
            position=_optional_float(entry.get("position")),
            rotation=_optional_float(entry.get("rotation")),
            gripper=float(entry.get("gripper", 0.5)),
            gripper_steps=(None if entry.get("gripper_steps") is None
                           else int(entry["gripper_steps"])),
        )

    table = {str(task): parse(entry) for task, entry in tasks.items()}
    default = table.pop("default", None)
    return table, default


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)
