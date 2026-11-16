"""Caption bar naming the policy on screen (common/video_caption.py)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_p = Path(__file__).resolve().parent.parent / "src" / "cartpole_term" / "common" / "video_caption.py"
_spec = importlib.util.spec_from_file_location("_video_caption", _p)
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)


def test_caption_draws_a_bar_and_keeps_small_frames_legible():
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)   # mjlab's default render size
    out = vc.caption_frame(frame, "SmallFall: Hull growth reward | settle lock on", "fall 3/8")
    assert out.shape == (480, 640, 3)                    # upscaled to 640 wide
    bar = out[: vc.BAR_H]
    assert (bar > 200).any() and (bar == 0).mean() > 0.5  # white text on black
    assert (out[vc.BAR_H + 5:] == 128).all()             # the picture below is untouched


def test_long_title_still_fits_and_big_frames_keep_their_size():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = vc.caption_frame(frame, "x" * 300)
    assert out.shape == frame.shape


def test_file_tag():
    assert vc.file_tag("") == "" and vc.file_tag("hullgrow") == "_hullgrow"
    with pytest.raises(ValueError):
        vc.file_tag("hull growth")
