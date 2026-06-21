"""utils.video — frame iteration and overlay rendering.

Includes ``render_bbox_video`` which draws face-track bounding boxes (from s04)
onto a video so results can be inspected visually.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict

logger = logging.getLogger("utils.video")

# Distinct BGR colours cycled per track id for readability.
_COLORS = [
    (66, 135, 245), (46, 204, 113), (231, 76, 60), (241, 196, 15),
    (155, 89, 182), (26, 188, 156), (230, 126, 34), (52, 152, 219),
]


def _color_for(track_id: str) -> tuple[int, int, int]:
    """Pick a stable colour for a track id."""
    return _COLORS[hash(track_id) % len(_COLORS)]


def build_frame_index(tracks: dict[str, list[dict]], fps: float,
                      max_hold_s: float = 0.5) -> dict[int, list[tuple]]:
    """Map ``frame_idx -> [(track_id, bbox), ...]`` from sampled track points.

    Boxes are linearly interpolated between consecutive samples for smoothness;
    across a gap larger than ``max_hold_s`` (e.g. a re-detection after the face
    left frame) the earlier box is only held for ``max_hold_s`` rather than
    stretched. The final sample of each track is held for ``max_hold_s`` too.
    """
    max_hold = max(1, int(round(fps * max_hold_s)))
    index: dict[int, list[tuple]] = defaultdict(list)
    for tid, seq in tracks.items():
        pts = [(int(round(p["t"] * fps)), p["bbox"])
               for p in sorted(seq, key=lambda p: p["t"])]
        for (f0, b0), (f1, b1) in zip(pts, pts[1:]):
            gap = f1 - f0
            if gap <= 0:
                continue
            if gap > max_hold:
                for f in range(f0, f0 + max_hold):
                    index[f].append((tid, b0))
                continue
            for f in range(f0, f1):
                a = (f - f0) / gap
                box = [b0[i] + (b1[i] - b0[i]) * a for i in range(4)]
                index[f].append((tid, box))
        if pts:
            fl, bl = pts[-1]
            for f in range(fl, fl + max_hold):
                index[f].append((tid, bl))
    return index


def render_bbox_video(video_path: str, tracks: dict[str, list[dict]],
                      out_path: str, max_hold_s: float = 0.5) -> str:
    """Render ``video_path`` with track bounding boxes drawn, to ``out_path``.

    ``tracks`` maps ``track_id -> [{"t": s, "bbox": [x1,y1,x2,y2]}, ...]``
    (as written by s04). Returns ``out_path``.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    index = build_frame_index(tracks, fps, max_hold_s)

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for tid, bbox in index.get(frame_idx, ()):
                x1, y1, x2, y2 = (int(round(v)) for v in bbox)
                color = _color_for(tid)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = tid.split("_")[-1]  # short track tag, e.g. t003
                cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
    logger.info("wrote %d frames -> %s", frame_idx, out_path)
    return out_path
