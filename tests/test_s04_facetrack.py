"""Stage s04 (facetrack) tests — helpers + mocked run() (no models needed)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import Config  # noqa: E402
from pipeline.manifest import Manifest  # noqa: E402
from pipeline.stages import s04_facetrack as s04  # noqa: E402


def test_track_id_deterministic():
    assert s04._track_id("v1_s0003", 7) == "v1_s0003_t007"


def test_filter_tracks_drops_short():
    raw = {1: [{"t": 0}, {"t": 1}, {"t": 2}], 2: [{"t": 0}]}
    kept = s04._filter_tracks(raw, min_frames=3)
    assert set(kept) == {1}


def test_write_track_roundtrip(tmp_path):
    p = str(tmp_path / "sub" / "t.json")
    seq = [{"t": 0.0, "bbox": [1.0, 2.0, 3.0, 4.0]}]
    s04._write_track(p, seq)
    assert json.load(open(p)) == seq


def _cfg(tmp_path):
    return Config(raw={
        "paths": {"data": str(tmp_path), "manifest": str(tmp_path / "m.db")},
        "speakers": {"seed_csv": "", "webui_db": ""},
        "facetrack": {"det_model": "buffalo_l", "det_threshold": 0.5,
                      "sample_fps": 5.0, "min_track_frames": 2},
    })


def _shot(mf, tmp_path, shot_id="v1_s0000", vid="v1"):
    p = tmp_path / f"{vid}.mp4"
    p.write_bytes(b"\x00")
    mf.upsert("videos", {"video_id": vid, "speaker_id": "sp",
                         "local_path": str(p), "status": "done"})
    mf.upsert("shots", {"shot_id": shot_id, "video_id": vid,
                        "start_t": 0.0, "end_t": 5.0, "status": "done"})


def _fake_tracks(*a, **k):
    return {
        1: [{"t": 0.0, "bbox": [0, 0, 10, 10]}, {"t": 0.2, "bbox": [1, 1, 11, 11]}],
        2: [{"t": 0.0, "bbox": [5, 5, 9, 9]}],  # too short -> filtered (min 2)
    }


def test_run_writes_tracks_and_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(s04, "FaceDetector", lambda *a, **k: object())
    monkeypatch.setattr(s04, "process_shot", _fake_tracks)
    mf = Manifest(str(tmp_path / "m.db")); mf.connect()
    _shot(mf, tmp_path)
    s04.run(_cfg(tmp_path), mf)
    rows = mf.query("SELECT track_id, bbox_path FROM tracks")
    assert len(rows) == 1 and rows[0]["track_id"] == "v1_s0000_t001"
    assert os.path.exists(rows[0]["bbox_path"])  # artifact written


def test_run_idempotent_and_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(s04, "FaceDetector", lambda *a, **k: object())
    monkeypatch.setattr(s04, "process_shot", _fake_tracks)
    mf = Manifest(str(tmp_path / "m.db")); mf.connect()
    _shot(mf, tmp_path, shot_id="v1_s0000")
    _shot(mf, tmp_path, shot_id="v1_s0001")
    cfg = _cfg(tmp_path)
    s04.run(cfg, mf, limit=1)
    assert len({r["shot_id"] for r in mf.query("SELECT shot_id FROM tracks")}) == 1
    s04.run(cfg, mf)              # finish the rest
    s04.run(cfg, mf)              # rerun -> no duplicates
    assert mf.count("tracks") == 2


def test_run_no_shots_is_noop(tmp_path):
    mf = Manifest(str(tmp_path / "m.db")); mf.connect()
    s04.run(_cfg(tmp_path), mf)
    assert mf.count("tracks") == 0
