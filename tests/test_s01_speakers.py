"""Stage s01 (speakers) tests — sources, completeness, idempotency."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import Config  # noqa: E402
from pipeline.manifest import Manifest  # noqa: E402
from pipeline.stages import s01_speakers as s01  # noqa: E402


def _cfg(tmp_path, csv_path="", webui_db=""):
    return Config(raw={
        "paths": {"data": str(tmp_path), "manifest": str(tmp_path / "m.db")},
        "speakers": {"seed_csv": csv_path, "webui_db": webui_db,
                     "use_wikidata": False, "min_speakers": 1},
    })


def _seed(tmp_path):
    p = tmp_path / "seed.csv"
    p.write_text(
        "speaker_id,name,wikidata_id,region,profession,gender,seed_dir\n"
        "sp_a,Full Person,Q1,Bangladesh,actor,male,\n"
        ",Partial Person,,India,,female,\n",
        encoding="utf-8",
    )
    return str(p)


def test_csv_completeness_and_status(tmp_path):
    mf = Manifest(str(tmp_path / "m.db")); mf.connect()
    s01.run(_cfg(tmp_path, csv_path=_seed(tmp_path)), mf)
    assert mf.get_speaker("sp_a")["status"] == "done"
    # deterministic id derived from name; missing profession -> pending
    partial = mf.get_speaker("sp_partial_person")
    assert partial is not None and partial["status"] == "pending"
    assert mf.count("speakers", "status='done'") == 1


def test_idempotent_rerun(tmp_path):
    cfg = _cfg(tmp_path, csv_path=_seed(tmp_path))
    mf = Manifest(str(tmp_path / "m.db")); mf.connect()
    s01.run(cfg, mf)
    first = mf.count("speakers")
    s01.run(cfg, mf)
    assert mf.count("speakers") == first == 2


def test_webui_fills_gaps(tmp_path):
    # web-UI DB provides a fully-populated POI not present in the CSV.
    import webui.db as wdb
    db = str(tmp_path / "poi.db")
    wdb.init_db(db)
    wdb.create_speaker({"speaker_id": "sp_w", "name": "Web POI",
                        "region": "Bangladesh", "profession": "creator"}, db)
    mf = Manifest(str(tmp_path / "m.db")); mf.connect()
    s01.run(_cfg(tmp_path, webui_db=db), mf)
    assert mf.get_speaker("sp_w")["status"] == "done"


def test_no_sources_is_noop(tmp_path):
    mf = Manifest(str(tmp_path / "m.db")); mf.connect()
    s01.run(_cfg(tmp_path), mf)
    assert mf.count("speakers") == 0
