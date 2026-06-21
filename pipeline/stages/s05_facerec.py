"""Stage 5 — facerec: tracks -> ``speaker_id``, ``facerec_score`` (ArcFace).

For each face track from s04, compute an ArcFace embedding (InsightFace
``buffalo_l``) and compare it by cosine similarity to the seed embedding of the
video's target speaker. Tracks at/above ``facerec.cosine_threshold`` are
assigned that ``speaker_id`` (status stays ``done``); tracks below are rejected
(``status='skipped'``, ``speaker_id`` left null). ``facerec_score`` is always
recorded.

Heavy CV imports stay lazy. Idempotent: tracks already scored are skipped.
"""
from __future__ import annotations

import json
import logging
import os

from ..utils import gpu

logger = logging.getLogger("stage.s05_facerec")


def cosine(a, b) -> float:
    """Cosine similarity between two 1-D vectors (numpy arrays)."""
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _iou(a, b) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


class FaceRecognizer:
    """InsightFace detection + ArcFace recognition wrapper (lazy-loaded)."""

    def __init__(self, model: str = "buffalo_l", det_size: int = 640) -> None:
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name=model,
                                allowed_modules=["detection", "recognition"],
                                providers=gpu.onnx_providers())
        self.app.prepare(ctx_id=gpu.ctx_id(), det_size=(det_size, det_size))

    def embed_largest(self, frame):
        """Return the normed embedding of the largest detected face, or None."""
        faces = self.app.get(frame)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding

    def embed_at_bbox(self, frame, bbox):
        """Return the embedding of the detected face best matching ``bbox``."""
        faces = self.app.get(frame)
        if not faces:
            return None
        best = max(faces, key=lambda f: _iou(f.bbox, bbox))
        if _iou(best.bbox, bbox) <= 0:
            return None
        return best.normed_embedding


def seed_embedding(rec: FaceRecognizer, seed_dir: str):
    """Average ArcFace embedding over the seed images in ``seed_dir`` (or None)."""
    import cv2
    import numpy as np

    if not seed_dir or not os.path.isdir(seed_dir):
        return None
    embs = []
    for fn in sorted(os.listdir(seed_dir)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        img = cv2.imread(os.path.join(seed_dir, fn))
        if img is None:
            continue
        emb = rec.embed_largest(img)
        if emb is not None:
            embs.append(emb)
    if not embs:
        return None
    return np.mean(np.stack(embs), axis=0)


def track_embedding(rec: FaceRecognizer, video_path: str, bbox_seq: list[dict],
                    n_samples: int = 3):
    """Average embedding over a few evenly-spaced frames of a track (or None)."""
    import cv2
    import numpy as np

    if not bbox_seq:
        return None
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, len(bbox_seq) // n_samples)
    samples = bbox_seq[::step][:n_samples]
    embs = []
    try:
        for pt in samples:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(pt["t"] * fps)))
            ok, frame = cap.read()
            if not ok:
                continue
            emb = rec.embed_at_bbox(frame, pt["bbox"])
            if emb is not None:
                embs.append(emb)
    finally:
        cap.release()
    if not embs:
        return None
    return np.mean(np.stack(embs), axis=0)


def _load_bbox(path: str) -> list[dict]:
    """Load a track's bbox sequence JSON."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def run(cfg, mf, limit: int | None = None) -> None:
    """Assign speaker_id + facerec_score to each track; reject below threshold.

    ``limit`` caps the number of tracks processed. Tracks that already have a
    ``facerec_score`` are skipped (resumable); failures are soft per track.
    """
    threshold = float(getattr(cfg.facerec, "cosine_threshold", 0.45))
    model = getattr(cfg.facerec, "model", "buffalo_l")
    n_samples = int(getattr(cfg.facerec, "track_samples", 3))

    tracks = mf.query(
        "SELECT t.track_id, t.bbox_path, t.video_id, v.speaker_id, v.local_path "
        "FROM tracks t JOIN videos v ON v.video_id = t.video_id "
        "WHERE t.status='done' AND t.facerec_score IS NULL")
    if not tracks:
        logger.warning("no unscored tracks; run s04 first (or already scored)")
        return

    rec = None
    seed_cache: dict[str, object] = {}
    assigned = 0
    processed = 0
    for tr in tracks:
        if limit is not None and processed >= limit:
            break
        tid, spk = tr["track_id"], tr["speaker_id"]
        try:
            if rec is None:
                logger.info("loading recognizer %s ...", model)
                rec = FaceRecognizer(model)

            if spk not in seed_cache:
                row = mf.get_speaker(spk)
                seed_cache[spk] = (seed_embedding(rec, row["seed_dir"])
                                   if row else None)
            seed = seed_cache[spk]
            if seed is None:
                logger.warning("%s: speaker %s has no usable seed; skipping",
                               tid, spk)
                continue

            emb = track_embedding(rec, tr["local_path"], _load_bbox(tr["bbox_path"]),
                                  n_samples)
            if emb is None:
                logger.warning("%s: no face embedding; rejecting", tid)
                mf.upsert("tracks", {"track_id": tid, "facerec_score": 0.0,
                                     "status": "skipped"})
                processed += 1
                continue

            score = cosine(emb, seed)
            if score >= threshold:
                mf.upsert("tracks", {"track_id": tid, "speaker_id": spk,
                                     "facerec_score": round(score, 4), "status": "done"})
                assigned += 1
            else:
                mf.upsert("tracks", {"track_id": tid, "facerec_score": round(score, 4),
                                     "status": "skipped"})
            processed += 1
            logger.info("%s: score=%.3f -> %s", tid, score,
                        spk if score >= threshold else "rejected")
        except Exception as exc:  # noqa: BLE001 — fail soft per track
            logger.error("%s: facerec failed: %s", tid, exc)

    logger.info("s05 complete: %d/%d track(s) assigned (threshold=%.2f)",
                assigned, processed, threshold)
