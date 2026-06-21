"""Stage s05 (facerec) tests — cosine/iou + mocked assignment (no models)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import Config  # noqa: E402
from pipeline.manifest import Manifest  # noqa: E402
from pipeline.stages import s05_facerec as s05  # noqa: E402


def test_cosine():
    assert abs(s05.cosine([1, 0, 0], [1, 0, 0]) - 1.0) < 1e-9
    assert abs(s05.cosine([1, 0], [0, 1])) < 1e-9
    assert s05.cosine([0, 0], [1, 1]) == 0.0


def test_iou():
    assert s05._iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert s05._iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert abs(s05._iou([0, 0, 10, 10], [5, 0, 15, 10]) - (50 / 150)) < 1e-9


def _cfg(tmp_path, threshold=0.5):
    return Config(raw={
        "paths": {"data": str(tmp_path), "manifest": str(tmp_path / "m.db")},
        "speakers": {"seed_csv": "", "webui_db": ""},
        "facerec": {"model": "buffalo_l", "cosine_threshold": threshold},
    })


def _setup(mf, tmp_path):
    seed = tmp_path / "seeds"; seed.mkdir()
    mf.upsert("speakers", {"speaker_id": "sp", "name": "S",
                           "seed_dir": str(seed), "status": "done"})
    vp = tmp_path / "v.mp4"; vp.write_bytes(b"\x00")
    mf.upsert("videos", {"video_id": "v1", "speaker_id": "sp",
                         "local_path": str(vp), "status": "done"})
    for tid in ("v1_s0_t001", "v1_s0_t002"):
        bp = tmp_path / f"{tid}.json"
        bp.write_text(json.dumps([{"t": 0.0, "bbox": [0, 0, 10, 10]}]))
        mf.upsert("tracks", {"track_id": tid, "shot_id": "v1_s0", "video_id": "v1",
                             "bbox_path": str(bp), "status": "done"})


def test_run_assigns_and_rejects(tmp_path, monkeypatch):
    import numpy as np
    monkeypatch.setattr(s05, "FaceRecognizer", lambda *a, **k: object())
    monkeypatch.setattr(s05, "seed_embedding", lambda rec, d: np.array([1.0, 0.0, 0.0]))

    # First track processed matches the seed (cosine 1.0); second is orthogonal.
    def fake_track_emb(rec, video_path, seq, n):
        return fake_track_emb.lookup.pop(0)
    fake_track_emb.lookup = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    monkeypatch.setattr(s05, "track_embedding", fake_track_emb)

    mf = Manifest(str(tmp_path / "m.db")); mf.connect()
    _setup(mf, tmp_path)
    s05.run(_cfg(tmp_path, threshold=0.5), mf)
    rows = {r["track_id"]: r for r in mf.query(
        "SELECT track_id, speaker_id, facerec_score, status FROM tracks ORDER BY track_id")}
    assert rows["v1_s0_t001"]["speaker_id"] == "sp"
    assert rows["v1_s0_t001"]["status"] == "done"
    assert rows["v1_s0_t002"]["speaker_id"] is None
    assert rows["v1_s0_t002"]["status"] == "skipped"


def test_run_idempotent(tmp_path, monkeypatch):
    import numpy as np
    monkeypatch.setattr(s05, "FaceRecognizer", lambda *a, **k: object())
    monkeypatch.setattr(s05, "seed_embedding", lambda rec, d: np.array([1.0, 0.0]))
    monkeypatch.setattr(s05, "track_embedding", lambda *a, **k: np.array([1.0, 0.0]))
    mf = Manifest(str(tmp_path / "m.db")); mf.connect()
    _setup(mf, tmp_path)
    s05.run(_cfg(tmp_path), mf)
    # all scored now; a second run finds nothing to do
    before = mf.query("SELECT facerec_score FROM tracks")
    s05.run(_cfg(tmp_path), mf)
    after = mf.query("SELECT facerec_score FROM tracks")
    assert [r["facerec_score"] for r in before] == [r["facerec_score"] for r in after]
