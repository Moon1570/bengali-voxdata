"""Stage 4 — facetrack: shots -> face tracks (InsightFace SCRFD + ByteTrack).

For each shot with ``status='done'`` (from s03), detect faces per sampled frame
with InsightFace's SCRFD detector and link them across frames with ByteTrack.
Each resulting track's bbox sequence is written to
``data/interim/tracks/<video_id>/<track_id>.json`` and one ``tracks`` row is
upserted (``track_id, shot_id, video_id, bbox_path``).

Heavy CV imports (cv2, insightface, supervision) are loaded lazily so the rest
of the pipeline and the unit tests do not require them.

Idempotent and resumable: shots that already have tracks are skipped.
"""
from __future__ import annotations

import json
import logging
import os

from ..utils import gpu

logger = logging.getLogger("stage.s04_facetrack")


class FaceDetector:
    """Thin wrapper around InsightFace's SCRFD detector (lazy-loaded)."""

    def __init__(self, model: str = "buffalo_l", det_threshold: float = 0.5,
                 det_size: int = 640) -> None:
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name=model, allowed_modules=["detection"],
                                providers=gpu.onnx_providers())
        self.app.prepare(ctx_id=gpu.ctx_id(), det_thresh=det_threshold,
                         det_size=(det_size, det_size))

    def detect(self, frame) -> tuple[list[list[float]], list[float]]:
        """Return (boxes_xyxy, scores) for all faces in a BGR frame."""
        faces = self.app.get(frame)
        boxes = [f.bbox.tolist() for f in faces]
        scores = [float(f.det_score) for f in faces]
        return boxes, scores


def _track_id(shot_id: str, tid: int) -> str:
    """Deterministic track id within a shot."""
    return f"{shot_id}_t{tid:03d}"


def process_shot(detector: FaceDetector, video_path: str, start_t: float,
                 end_t: float, sample_fps: float = 5.0) -> dict[int, list[dict]]:
    """Detect + ByteTrack faces over a shot's frames.

    Samples the shot at ``sample_fps`` and returns a mapping of
    ``tracker_id -> [{"t": seconds, "bbox": [x1,y1,x2,y2], "score": s}, ...]``.
    A fresh tracker is used per shot so ids do not leak across shots.
    """
    import cv2
    import numpy as np
    import supervision as sv

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(fps / max(sample_fps, 0.1))))
    tracker = sv.ByteTrack(frame_rate=int(round(sample_fps)))

    tracks: dict[int, list[dict]] = {}
    frame_idx = int(start_t * fps)
    end_idx = int(end_t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    try:
        while frame_idx <= end_idx:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame_idx - int(start_t * fps)) % stride == 0:
                boxes, scores = detector.detect(frame)
                if boxes:
                    dets = sv.Detections(
                        xyxy=np.array(boxes, dtype=float),
                        confidence=np.array(scores, dtype=float),
                        class_id=np.zeros(len(boxes), dtype=int),
                    )
                else:
                    dets = sv.Detections.empty()
                tracked = tracker.update_with_detections(dets)
                t = frame_idx / fps
                for xyxy, tid in zip(tracked.xyxy, tracked.tracker_id):
                    tracks.setdefault(int(tid), []).append(
                        {"t": round(float(t), 3),
                         "bbox": [round(float(x), 1) for x in xyxy]})
            frame_idx += 1
    finally:
        cap.release()
    return tracks


def _filter_tracks(tracks: dict[int, list[dict]],
                   min_frames: int) -> dict[int, list[dict]]:
    """Drop tracks shorter than ``min_frames`` (detection noise)."""
    return {tid: seq for tid, seq in tracks.items() if len(seq) >= min_frames}


def _write_track(path: str, seq: list[dict]) -> None:
    """Persist one track's bbox sequence as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(seq, fh)


def run(cfg, mf, limit: int | None = None) -> None:
    """Detect + track faces for each done shot and upsert ``tracks`` rows.

    ``limit`` caps the number of shots processed this run. Shots that already
    have tracks are skipped (resumable); failures are soft per shot.
    """
    ft = cfg.facetrack
    model = getattr(ft, "det_model", "buffalo_l")
    # buffalo_l bundles the SCRFD-10G detector used by the config's det_model.
    if str(model).startswith("scrfd"):
        model = "buffalo_l"
    det_threshold = float(getattr(ft, "det_threshold", 0.5))
    sample_fps = float(getattr(ft, "sample_fps", 5.0))
    min_frames = int(getattr(ft, "min_track_frames", 3))
    tracks_root = os.path.join(cfg.paths.data, "interim", "tracks")

    shots = mf.query(
        "SELECT s.shot_id, s.video_id, s.start_t, s.end_t, v.local_path "
        "FROM shots s JOIN videos v ON v.video_id = s.video_id "
        "WHERE s.status='done'")
    if not shots:
        logger.warning("no shots with status='done'; run s03 first")
        return

    detector = None
    processed = 0
    for sh in shots:
        if limit is not None and processed >= limit:
            break
        sid, vid = sh["shot_id"], sh["video_id"]

        if mf.count("tracks", "shot_id=?", (sid,)) > 0:
            logger.info("%s already has tracks; skipping", sid)
            continue
        if not sh["local_path"] or not os.path.exists(sh["local_path"]):
            logger.error("%s: video file missing; skipping", sid)
            continue

        try:
            if detector is None:
                logger.info("loading detector %s ...", model)
                detector = FaceDetector(model, det_threshold)
            raw = process_shot(detector, sh["local_path"], sh["start_t"],
                               sh["end_t"], sample_fps)
            kept = _filter_tracks(raw, min_frames)
        except Exception as exc:  # noqa: BLE001 — fail soft per shot
            logger.error("%s: face tracking failed: %s", sid, exc)
            continue

        for tid, seq in kept.items():
            track_id = _track_id(sid, tid)
            bbox_path = os.path.join(tracks_root, vid, f"{track_id}.json")
            _write_track(bbox_path, seq)
            mf.upsert("tracks", {
                "track_id": track_id, "shot_id": sid, "video_id": vid,
                "bbox_path": bbox_path, "status": "done",
            })
        processed += 1
        logger.info("%s: %d track(s) kept (of %d raw)", sid, len(kept), len(raw))

    logger.info("s04 complete: %d track(s) across %d shot(s)",
                mf.count("tracks"), processed)
