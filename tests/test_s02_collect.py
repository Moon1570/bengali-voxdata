"""Stage s02 (collect) tests — duration parse, filtering, mocked run()."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import Config  # noqa: E402
from pipeline.manifest import Manifest  # noqa: E402
from pipeline.stages import s02_collect as s02  # noqa: E402


def test_parse_iso8601_duration():
    assert s02.parse_iso8601_duration("PT1H2M3S") == 3723.0
    assert s02.parse_iso8601_duration("PT45S") == 45.0
    assert s02.parse_iso8601_duration("PT5M") == 300.0
    assert s02.parse_iso8601_duration("") == 0.0


def test_filter_candidates_duration_and_lang():
    cands = [
        {"video_id": "a", "duration_s": 30, "lang": "bn"},     # too short
        {"video_id": "b", "duration_s": 120, "lang": "bn-BD"}, # keep
        {"video_id": "c", "duration_s": 120, "lang": "en"},    # wrong lang
        {"video_id": "d", "duration_s": 120, "lang": None},    # unknown -> keep
        {"video_id": "e", "duration_s": 5000, "lang": "bn"},   # too long
    ]
    kept = {c["video_id"] for c in s02.filter_candidates(cands, 60, 1800, "bn")}
    assert kept == {"b", "d"}


def _cfg(tmp_path):
    return Config(raw={
        "paths": {"data": str(tmp_path), "manifest": str(tmp_path / "m.db")},
        "speakers": {"seed_csv": "", "webui_db": ""},
        "collect": {"per_speaker_videos": 2, "min_duration_s": 60,
                    "max_duration_s": 1800, "lang_filter": "bn"},
    })


def _seed_speaker(mf):
    mf.upsert("speakers", {"speaker_id": "sp_a", "name": "Speaker A",
                           "region": "Bangladesh", "profession": "actor",
                           "status": "done"})


def test_run_downloads_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_API_KEY", "fake-key")
    calls = {"download": 0}

    def fake_search(query, api_key, max_results=10, relevance_language="bn"):
        return [
            {"video_id": "v1", "url": "u1", "title": "T1", "duration_s": 120, "lang": "bn"},
            {"video_id": "v2", "url": "u2", "title": "T2", "duration_s": 200, "lang": None},
        ]

    def fake_download(url, dest_dir):
        calls["download"] += 1
        return os.path.join(dest_dir, "vid.mp4"), {"id": "x"}

    monkeypatch.setattr(s02, "search_videos", fake_search)
    monkeypatch.setattr(s02, "download_video", fake_download)

    mf = Manifest(str(tmp_path / "m.db")); mf.connect()
    _seed_speaker(mf)
    s02.run(_cfg(tmp_path), mf)
    assert mf.count("videos", "status='done'") == 2
    assert calls["download"] == 2

    # rerun: already-done videos are skipped (no new downloads)
    s02.run(_cfg(tmp_path), mf)
    assert calls["download"] == 2
    assert mf.count("videos") == 2


def test_run_respects_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_API_KEY", "fake-key")
    monkeypatch.setattr(s02, "search_videos", lambda *a, **k: [
        {"video_id": f"v{i}", "url": f"u{i}", "title": f"T{i}",
         "duration_s": 120, "lang": "bn"} for i in range(5)])
    monkeypatch.setattr(s02, "download_video",
                        lambda url, d: (os.path.join(d, "v.mp4"), {}))
    mf = Manifest(str(tmp_path / "m.db")); mf.connect()
    _seed_speaker(mf)
    s02.run(_cfg(tmp_path), mf, limit=1)
    assert mf.count("videos", "status='done'") == 1


def test_run_no_done_speakers_is_noop(tmp_path):
    mf = Manifest(str(tmp_path / "m.db")); mf.connect()
    s02.run(_cfg(tmp_path), mf)
    assert mf.count("videos") == 0
