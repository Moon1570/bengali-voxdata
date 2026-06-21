"""Render face-track bounding boxes (from s04) onto videos for visual review.

Reads the ``tracks`` table + bbox-sequence artifacts from the manifest and
writes an annotated MP4 per video to ``data/interim/viz/<video_id>.bbox.mp4``.

Usage:
    python -m scripts.render_tracks --config config/config.local.yaml
    python -m scripts.render_tracks --config config/config.local.yaml --video-id VID
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import load_config  # noqa: E402
from pipeline.manifest import Manifest  # noqa: E402
from pipeline.utils.video import render_bbox_video  # noqa: E402

logger = logging.getLogger("render_tracks")


def load_tracks_for_video(mf: Manifest, video_id: str) -> tuple[dict, dict, set]:
    """Return (tracks, labels, assigned) for one video from the manifest.

    ``tracks`` maps track_id -> bbox sequence; ``labels`` maps assigned tracks
    to a "speaker score" caption; ``assigned`` is the set of track ids s05
    matched to the target speaker. All track rows are included (assigned or
    not) so the QA video shows both kept and rejected faces.
    """
    tracks: dict[str, list[dict]] = {}
    labels: dict[str, str] = {}
    assigned: set[str] = set()
    rows = mf.query(
        "SELECT track_id, bbox_path, speaker_id, facerec_score FROM tracks "
        "WHERE video_id=? AND bbox_path IS NOT NULL", (video_id,))
    for r in rows:
        if not (r["bbox_path"] and os.path.exists(r["bbox_path"])):
            continue
        with open(r["bbox_path"], encoding="utf-8") as fh:
            tracks[r["track_id"]] = json.load(fh)
        if r["speaker_id"]:
            assigned.add(r["track_id"])
            score = r["facerec_score"]
            labels[r["track_id"]] = (f"{r['speaker_id']} {score:.2f}"
                                     if score is not None else r["speaker_id"])
    return tracks, labels, assigned


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="render_tracks")
    p.add_argument("--config", default="config/config.local.yaml")
    p.add_argument("--video-id", default=None, help="render one video (default: all)")
    p.add_argument("--limit", type=int, default=None, help="max videos to render")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    mf = Manifest(cfg.paths.manifest)
    mf.connect()

    if args.video_id:
        video_ids = [args.video_id]
    else:
        video_ids = [r["video_id"] for r in mf.query(
            "SELECT DISTINCT video_id FROM tracks WHERE status='done'")]
    if args.limit is not None:
        video_ids = video_ids[:args.limit]
    if not video_ids:
        logger.warning("no videos with tracks; run s04 first")
        return 0

    viz_root = os.path.join(cfg.paths.data, "interim", "viz")
    for vid in video_ids:
        row = mf.query("SELECT local_path FROM videos WHERE video_id=?", (vid,))
        if not row or not row[0]["local_path"] or not os.path.exists(row[0]["local_path"]):
            logger.error("%s: source video missing; skipping", vid)
            continue
        tracks, labels, assigned = load_tracks_for_video(mf, vid)
        if not tracks:
            logger.warning("%s: no track artifacts found; skipping", vid)
            continue
        out_path = os.path.join(viz_root, f"{vid}.bbox.mp4")
        render_bbox_video(row[0]["local_path"], tracks, out_path,
                          labels=labels, assigned=assigned)
        logger.info("%s: rendered %d track(s), %d assigned -> %s",
                    vid, len(tracks), len(assigned), out_path)

    mf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
