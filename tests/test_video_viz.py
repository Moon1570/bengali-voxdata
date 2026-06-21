"""Tests for utils.video.build_frame_index (pure logic, no cv2 needed)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.utils.video import build_frame_index  # noqa: E402


def test_interpolates_between_samples():
    # one track sampled at t=0s and t=1s (fps=10 -> frames 0 and 10)
    tracks = {"t1": [{"t": 0.0, "bbox": [0, 0, 10, 10]},
                     {"t": 1.0, "bbox": [10, 10, 20, 20]}]}
    # max_hold must cover the 10-frame gap for interpolation to fill it
    idx = build_frame_index(tracks, fps=10.0, max_hold_s=1.0)
    # midpoint frame 5 should be halfway interpolated
    mid = dict((tid, b) for tid, b in idx[5])
    assert mid["t1"] == [5.0, 5.0, 15.0, 15.0]


def test_holds_last_sample():
    tracks = {"t1": [{"t": 0.0, "bbox": [0, 0, 10, 10]}]}
    idx = build_frame_index(tracks, fps=10.0, max_hold_s=0.5)  # hold 5 frames
    assert all("t1" == idx[f][0][0] for f in range(0, 5))
    assert 5 not in idx  # held only for max_hold frames


def test_large_gap_not_stretched():
    # samples 2s apart at fps=10 -> frames 0 and 20; max_hold 0.5s (5 frames)
    tracks = {"t1": [{"t": 0.0, "bbox": [0, 0, 10, 10]},
                     {"t": 2.0, "bbox": [0, 0, 10, 10]}]}
    idx = build_frame_index(tracks, fps=10.0, max_hold_s=0.5)
    assert 4 in idx and 5 not in idx  # only held 5 frames after the first sample
