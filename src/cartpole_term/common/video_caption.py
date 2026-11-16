"""Caption bar for rollout videos: says which policy is on screen
(DEVIATIONS.md 2026-11-15, user: "in the videos specify which recovery
policy I'm watching"). Pure numpy + Pillow, no mjlab.
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MIN_WIDTH = 640  # frames narrower than this are upscaled so the text stays legible
BAR_H = 30


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _fit(d: ImageDraw.ImageDraw, text: str, size: int, max_width: float):
    for s in range(size, 7, -1):
        font = _font(s)
        if d.textlength(text, font=font) <= max_width:
            return font
    return _font(8)


def caption_frame(frame: np.ndarray, title: str, right: str = "") -> np.ndarray:
    """`frame` with a black bar across the top: `title` on the left (shrunk
    to fit), `right` (e.g. "fall 3/8") on the right."""
    img = Image.fromarray(frame).convert("RGB")
    if img.width < MIN_WIDTH:
        img = img.resize((MIN_WIDTH, round(img.height * MIN_WIDTH / img.width)), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    w = img.width
    d.rectangle([0, 0, w, BAR_H], fill=(0, 0, 0))
    right_w = 0.0
    if right:
        f = _font(18)
        right_w = d.textlength(right, font=f)
        d.text((w - 8 - right_w, 6), right, fill=(255, 255, 255), font=f)
    d.text((8, 4), title, fill=(255, 255, 255), font=_fit(d, title, 22, w - 24 - right_w))
    return np.asarray(img)


def file_tag(tag: str) -> str:
    """'_<tag>' for a file name ('' when no tag), letters/digits/-/_ only."""
    if not tag:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError(f"policy tag must be letters, digits, - or _: {tag!r}")
    return f"_{tag}"
