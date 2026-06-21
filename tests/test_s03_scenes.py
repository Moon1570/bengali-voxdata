"""Stage s03 (scenes) tests — mocked detection, idempotency, fallback, limit."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import Config  # noqa: E402
from pipeline.manifest import Manifest  # noqa: E402
from pipeline.stages import s03_scenes as s03  # noqa: E402


def _cfg(tmp_path):
    return Config(raw={
        "paths": {"data": str(tmp_path), "manifest": str(tmp_path / "m.db")},
        "speakers": {"seed_csv": "", "webui_db": ""},
        "scenes": {"detector": "content", "threshold": 27.0},
    })


def _video(mf, tmp_path, vid="v1", dur=120.0):
    # create a real file so the existence check passes
    p = tmp_path / f"{vid}.mp4"
    p.write_bytes(b"\x00")
    mf.upsert("videos", {"video_id": vid, "speaker_id": "sp",
                         "local_path": str(p), "duration_s": dur, "status": "done"})
    return str(p)


def test_run_writes_contiguous_shots(tmp_path, monkeypatch):
    monkeypatch.setattr(s03, "detect_shots",
                        lambda *a, **k: [(0.0, 10.0), (10.0, 25.0), (25.0, 40.0)])
    mf = Manifest(str(tmp_path / "m.db")); mf.connect()
    _video(mf, tmp_path)
    s03.run(_cfg(tmp_path), mf)
    rows = mf.query("SELECT shot_id, start_t, end_t FROM shots ORDER BY shot_id")
    assert [r["shot_id"] for r in rows] == ["v1_s0000", "v1_s0001", "v1_s0002"]
    # contiguous coverage, no gaps
    for a, b in zip(rows, rows[1:]):
        assert a["end_t"] == b["start_t"]


def test_run_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(s03, "detect_shots", lambda *a, **k: [(0.0, 10.0), (10.0, 20.0)])
    mf = Manifest(str(tmp_path / "m.db")); mf.connect()
    _video(mf, tmp_path)
    cfg = _cfg(tmp_path)
    s03.run(cfg, mf)
    s03.run(cfg, mf)  # second run skips (shots already exist)
    assert mf.count("shots") == 2


def test_run_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(s03, "detect_shots", lambda *a, **k: [(0.0, 5.0)])
    mf = Manifest(str(tmp_path / "m.db")); mf.connect()
    _video(mf, tmp_path, vid="v1")
    _video(mf, tmp_path, vid="v2")
    s03.run(_cfg(tmp_path), mf, limit=1)
    vids = {r["video_id"] for r in mf.query("SELECT DISTINCT video_id FROM shots")}
    assert len(vids) == 1


def test_run_missing_file_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(s03, "detect_shots", lambda *a, **k: [(0.0, 5.0)])
    mf = Manifest(str(tmp_path / "m.db")); mf.connect()
    mf.upsert("videos", {"video_id": "gone", "speaker_id": "sp",
                         "local_path": str(tmp_path / "nope.mp4"),
                         "duration_s": 30.0, "status": "done"})
    s03.run(_cfg(tmp_path), mf)
    assert mf.count("shots") == 0


def test_detect_shots_fallback_no_cuts(monkeypatch):
    # simulate detection returning no scenes -> single full-duration shot
    import pipeline.stages.s03_scenes as mod
    monkeypatch.setattr("scenedetect.detect", lambda *a, **k: [])
    spans = mod.detect_shots("any.mp4", fallback_duration=42.0)
    assert spans == [(0.0, 42.0)]
