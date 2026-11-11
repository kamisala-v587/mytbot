"""The object box of a memory frame: its coordinates, and the crop it drives.

A box marks the object a keyframe is about. It travels through the model as
four integers in ``0..BOX_SCALE`` of the frame, independent of resolution --
the selector answers in those coordinates, and the memory frame is cropped to
them before the action pass sees it, so the memory keeps the object and drops
the rest of the scene.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from PIL import Image

# Box coordinates are integers in 0..BOX_SCALE of the frame: (x1, y1, x2, y2).
BOX_SCALE = 1000
Box = Tuple[int, int, int, int]

EXPAND = 0.20  # each edge moves outward by this fraction of the box side
MIN_SIDE = 56  # px at the processing resolution; smaller crops grow around the centre
# Training varies the framing instead of always using EXPAND, so the model does
# not come to rely on one: each edge moves outward by a fraction drawn from this
# range, then the whole rectangle shifts by up to SHIFT of the box side.
EXPAND_RANGE = (0.10, 0.30)
SHIFT = 0.10


def normalize_box(box: Sequence[int], width: int, height: int) -> Box:
    """A raw-pixel box on a ``width`` x ``height`` frame, in 0..``BOX_SCALE``.

    The inverse of the scaling in :func:`crop_rect`. Annotations record pixels
    of the source frame; this is the form the model reads and writes.
    """
    x1, y1, x2, y2 = (int(v) for v in box)

    def scale(value: int, dim: int) -> int:
        return min(BOX_SCALE, max(0, int(round(value * float(BOX_SCALE) / dim))))

    return scale(x1, width), scale(y1, height), scale(x2, width), scale(y2, height)


def crop_rect(box: Box, width: int, height: int, rng=None) -> Tuple[int, int, int, int]:
    """Integer crop rectangle for a 0..1000 ``box`` on a ``width`` x ``height`` image.

    Without ``rng`` the framing is fixed, which is the convention inference
    follows. With one it is varied the way training varies it, so a model is
    not trained on a single framing of each object.
    """
    sx, sy = width / BOX_SCALE, height / BOX_SCALE
    x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
    bw, bh = x2 - x1, y2 - y1
    if rng is None:
        x1, x2 = x1 - EXPAND * bw, x2 + EXPAND * bw
        y1, y2 = y1 - EXPAND * bh, y2 + EXPAND * bh
    else:
        lo, hi = EXPAND_RANGE
        x1, x2 = x1 - rng.uniform(lo, hi) * bw, x2 + rng.uniform(lo, hi) * bw
        y1, y2 = y1 - rng.uniform(lo, hi) * bh, y2 + rng.uniform(lo, hi) * bh
        shift_x, shift_y = rng.uniform(-SHIFT, SHIFT) * bw, rng.uniform(-SHIFT, SHIFT) * bh
        x1, x2 = x1 + shift_x, x2 + shift_x
        y1, y2 = y1 + shift_y, y2 + shift_y

    def clamp(lo: float, hi: float, limit: int) -> Tuple[float, float]:
        lo, hi = max(0.0, lo), min(float(limit), hi)
        side = min(float(MIN_SIDE), float(limit))
        if hi - lo < side:
            centre = (lo + hi) / 2.0
            lo, hi = centre - side / 2.0, centre + side / 2.0
            if lo < 0.0:
                hi, lo = hi - lo, 0.0
            if hi > limit:
                lo, hi = lo - (hi - limit), float(limit)
            lo = max(0.0, lo)
        return lo, hi

    x1, x2 = clamp(x1, x2, width)
    y1, y2 = clamp(y1, y2, height)
    ix2, iy2 = min(int(round(x2)), width), min(int(round(y2)), height)
    ix1 = max(0, min(int(round(x1)), ix2 - 1))
    iy1 = max(0, min(int(round(y1)), iy2 - 1))
    return ix1, iy1, ix2, iy2


def crop_to_box(image, box: Box, rng=None):
    """Crop an already-resized memory image to its box and resize it back.

    Keeping the size unchanged keeps the image-token count of the prompt
    unchanged. Accepts and returns a PIL image or an HWC uint8 array. ``rng``
    varies the framing as training does; without it the framing is the fixed
    one inference uses.
    """
    pil = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    width, height = pil.size
    out = pil.crop(crop_rect(box, width, height, rng)).resize((width, height), Image.BICUBIC)
    return out if isinstance(image, Image.Image) else np.asarray(out)
