"""Scoring a selection answer when more than one answer is right.

A keyframe is a moment, not a frame: writing it one candidate early or late
records the same progress. The annotation says so, giving each written slot an
interval of candidates that count. Training a plain cross-entropy against the
single annotated slot therefore punishes answers that are correct, and teaches
the model a precision the annotation does not have.

This scores the accepted set instead: the loss is the negative log of the
probability the model puts anywhere in it. With one accepted slot that is
exactly cross-entropy, so an exact annotation is unaffected.

Three functions, used in two places. The data side finds the answer's slot
tokens and turns each one's accepted interval into token ids; the model side
replaces the per-token cross-entropy at those positions.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

# Padding for a row of accepted ids: rows are as wide as the widest interval.
NO_TOKEN = -1


def answer_slots(
    input_ids: Sequence[int], write_id: int, slot_ids: Sequence[int]
) -> Tuple[int, ...]:
    """Where the answer writes a slot, in answer order.

    A selection prompt names every candidate as ``<|R_i|>`` before it asks the
    question, so only the slot tokens *after* the decision token belong to the
    answer.
    """
    ids = list(input_ids)
    try:
        start = ids.index(write_id)
    except ValueError:
        return ()                      # the answer skips, and writes nothing
    wanted = set(slot_ids)
    return tuple(i for i in range(start + 1, len(ids)) if ids[i] in wanted)


def accepted_token_ids(
    slot_ids: Sequence[int], ranges: Sequence[Tuple[int, int]]
) -> torch.Tensor:
    """``[len(ranges), width]`` token ids that count, padded with ``NO_TOKEN``.

    Each range is the inclusive span of candidate slots that would be a correct
    answer for one written keyframe.
    """
    rows = []
    for low, high in ranges:
        if not 0 <= low <= high < len(slot_ids):
            raise ValueError(
                f"slot range ({low}, {high}) is outside the {len(slot_ids)} candidates")
        rows.append([slot_ids[slot] for slot in range(low, high + 1)])
    width = max((len(row) for row in rows), default=1)
    return torch.tensor(
        [row + [NO_TOKEN] * (width - len(row)) for row in rows], dtype=torch.long)


def interval_cross_entropy(
    logits: torch.Tensor, accepted: torch.Tensor
) -> torch.Tensor:
    """``-log`` of the probability on the accepted tokens, one value per row.

    ``logits`` is ``[N, vocab]`` and ``accepted`` an ``[N, width]`` tensor of
    token ids padded with ``NO_TOKEN``. Equals cross-entropy when a row accepts
    exactly one token.
    """
    if logits.shape[0] != accepted.shape[0]:
        raise ValueError(
            f"{logits.shape[0]} rows of logits but {accepted.shape[0]} of accepted ids")
    real = accepted != NO_TOKEN
    if not bool(real.any(dim=-1).all()):
        raise ValueError(
            "an answer that accepts no token at all cannot be scored")
    logits = logits.to(torch.float32)
    everything = torch.logsumexp(logits, dim=-1)
    gathered = logits.gather(1, accepted.clamp(min=0).to(logits.device))
    # -inf on the padding leaves it out of the numerator while the denominator
    # still carries the whole distribution.
    accepted_only = torch.logsumexp(
        gathered.masked_fill(~real.to(gathered.device), float("-inf")), dim=-1)
    return everything - accepted_only


__all__ = ["NO_TOKEN", "accepted_token_ids", "answer_slots",
           "interval_cross_entropy"]
