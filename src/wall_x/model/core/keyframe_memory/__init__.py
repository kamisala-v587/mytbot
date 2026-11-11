"""Keyframe memory: what training and serving must agree on, byte for byte.

A checkpoint trained with keyframe memory expects its recalled frames in one
exact form -- a prompt layout, a box coordinate convention and a frame-index
arithmetic. Dataset construction and inference both render that form, so the
pieces live here rather than in either of them, and neither has to import the
other:

- :mod:`prompts` -- the two prompt layouts (selection pass, action pass).
- :mod:`boxes` -- the 0..1000 box convention and the memory crop.
- :mod:`windows` -- which frames a selection pass looks at.

Nothing here touches tensors or reads a dataset, so it is cheap to import and
testable on its own.
"""

from wall_x.model.core.keyframe_memory.boxes import (
    BOX_SCALE,
    Box,
    crop_rect,
    crop_to_box,
    normalize_box,
)
from wall_x.model.core.keyframe_memory.prompts import (
    IMAGE,
    SELECTION_QUESTION,
    SELECTION_SYSTEM_PROMPT,
    inject_memory,
    selection_prompt,
)
from wall_x.model.core.keyframe_memory.windows import (
    candidate_positions,
    short_term_positions,
)

__all__ = [
    "BOX_SCALE",
    "Box",
    "IMAGE",
    "SELECTION_QUESTION",
    "SELECTION_SYSTEM_PROMPT",
    "candidate_positions",
    "crop_rect",
    "crop_to_box",
    "inject_memory",
    "normalize_box",
    "selection_prompt",
    "short_term_positions",
]
