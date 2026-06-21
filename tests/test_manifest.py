"""Manifest tests (M0 acceptance)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.manifest import Manifest  # noqa: E402


def test_schema_created(tmp_path):
    mf = Manifest(str(tmp_path / "m.db"))
    with mf:
        names = {r["name"] for r in mf.query(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"speakers", "videos", "shots", "tracks", "utterances"} <= names


def test_upsert_and_get(tmp_path):
    mf = Manifest(str(tmp_path / "m.db"))
    with mf:
        mf.upsert("speakers", {"speaker_id": "sp1", "name": "A", "status": "pending"})
        assert mf.get_speaker("sp1")["name"] == "A"
        # upsert again updates, does not duplicate
        mf.upsert("speakers", {"speaker_id": "sp1", "name": "B"})
        assert mf.get_speaker("sp1")["name"] == "B"
        assert mf.count("speakers") == 1


def test_mark_status(tmp_path):
    mf = Manifest(str(tmp_path / "m.db"))
    with mf:
        mf.upsert("speakers", {"speaker_id": "sp1", "name": "A"})
        mf.mark_status("speakers", "sp1", "done")
        assert mf.get_speaker("sp1")["status"] == "done"
        assert len(mf.speakers_with_status("done")) == 1


def test_invalid_status_and_table(tmp_path):
    mf = Manifest(str(tmp_path / "m.db"))
    with mf:
        mf.upsert("speakers", {"speaker_id": "sp1", "name": "A"})
        for bad in ("bogus", ""):
            try:
                mf.mark_status("speakers", "sp1", bad)
                assert False, "expected ValueError"
            except ValueError:
                pass
        try:
            mf.upsert("nope", {"x": 1})
            assert False, "expected ValueError"
        except ValueError:
            pass
