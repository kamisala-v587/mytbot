"""Which frames a selection pass looks at.

Pure index arithmetic over one episode, shared by dataset construction and
inference. At inference the client may not hold every frame of the window, so
it maps each position onto the nearest frame it has; the positions themselves
are these functions.
"""

from __future__ import annotations

from typing import List


def candidate_positions(now: int, *, window: int, count: int) -> List[int]:
    """The ``count`` candidate frames of a selection pass at frame ``now``.

    The last ``window`` frames before ``now`` are divided into ``count`` equal
    spans and one frame is taken from the middle of each, so the candidates are
    phase-free: they do not depend on where in the episode the pass happens to
    fall. Near the episode start the window is shorter and the candidates
    bunch up, repeating frames rather than reaching before frame 0.
    """
    start = max(0, now - window)
    span = max(1, now - start)
    return [start + int((i + 0.5) * span / count) for i in range(count)]


def short_term_positions(now: int, *, count: int, stride: int, oldest: int = 0) -> List[int]:
    """Recent-view frames at ``now - stride * k`` for k = ``count``..1, oldest first.

    A fixed ``count`` of frames: anchors closer to ``oldest`` than
    ``count * stride`` repeat the oldest frame instead of dropping taps, so
    every sample carries the same number of images.
    """
    if count <= 0:
        return []
    return [max(oldest, now - k * stride) for k in range(count, 0, -1)]


__all__ = ["candidate_positions", "short_term_positions"]
