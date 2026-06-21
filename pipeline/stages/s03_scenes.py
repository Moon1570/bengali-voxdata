"""Stage 3 — scenes: video -> ``shots`` (PySceneDetect ContentDetector).

For each video with ``status='done'`` (from s02), detect shot boundaries and
write one ``shots`` row per shot (``shot_id, video_id, start_t, end_t``). Shots
cover the whole video with no gaps; a video with no detected cuts yields a
single shot spanning its full duration.

Idempotent and resumable: a video that already has shots is skipped.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("stage.s03_scenes")


def detect_shots(video_path: str, threshold: float = 27.0,
                 detector: str = "content",
                 fallback_duration: float | None = None) -> list[tuple[float, float]]:
    """Return ``[(start_s, end_s), ...]`` shot spans for a video.

    Uses PySceneDetect's ContentDetector. If no cuts are found, returns a
    single span covering the whole video (using the detected end, or
    ``fallback_duration`` when detection yields nothing).
    """
    from scenedetect import ContentDetector, detect

    if detector != "content":
        raise ValueError(f"unsupported detector {detector!r}; only 'content'")

    scenes = detect(video_path, ContentDetector(threshold=threshold))
    spans = [(start.get_seconds(), end.get_seconds()) for start, end in scenes]
    if spans:
        return spans
    if fallback_duration and fallback_duration > 0:
        return [(0.0, float(fallback_duration))]
    return []


def _shot_id(video_id: str, idx: int) -> str:
    """Deterministic shot id so re-runs do not duplicate rows."""
    return f"{video_id}_s{idx:04d}"


def run(cfg, mf, limit: int | None = None) -> None:
    """Detect shots for each done video and upsert ``shots`` rows.

    ``limit`` caps the number of videos processed this run. Videos that already
    have shots are skipped (resumable).
    """
    threshold = float(getattr(cfg.scenes, "threshold", 27.0))
    detector = getattr(cfg.scenes, "detector", "content")

    videos = mf.query(
        "SELECT video_id, local_path, duration_s FROM videos WHERE status='done'")
    if not videos:
        logger.warning("no videos with status='done'; run s02 first")
        return

    processed = 0
    for v in videos:
        if limit is not None and processed >= limit:
            break
        vid, path, dur = v["video_id"], v["local_path"], v["duration_s"]

        if mf.count("shots", "video_id=?", (vid,)) > 0:
            logger.info("%s already has shots; skipping", vid)
            continue
        if not path or not os.path.exists(path):
            logger.error("%s: local file missing (%s); skipping", vid, path)
            continue

        try:
            spans = detect_shots(path, threshold, detector, fallback_duration=dur)
        except Exception as exc:  # noqa: BLE001 — fail soft per video
            logger.error("%s: scene detection failed: %s", vid, exc)
            continue

        for idx, (start_t, end_t) in enumerate(spans):
            mf.upsert("shots", {
                "shot_id": _shot_id(vid, idx), "video_id": vid,
                "start_t": round(start_t, 3), "end_t": round(end_t, 3),
                "status": "done",
            })
        processed += 1
        logger.info("%s: %d shot(s) detected", vid, len(spans))

    logger.info("s03 complete: %d shot(s) across %d video(s)",
                mf.count("shots"), processed)
