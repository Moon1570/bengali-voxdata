"""Stage 1 — speakers: seed CSV + web-UI POIs + Wikidata -> ``speakers`` rows.

Sources, merged by ``speaker_id`` (later sources fill missing fields only):
  1. ``speakers.seed_csv``  — optional manual seed list.
  2. ``speakers.webui_db``  — POIs entered through the researcher web UI.
  3. Wikidata (``speakers.use_wikidata``) — enrich region/profession/gender.

A speaker is marked ``status='done'`` once both ``region`` and ``profession``
are populated (the stage-1 acceptance bar); otherwise it stays ``pending``.
Idempotent: re-running upserts the same rows.
"""
from __future__ import annotations

import csv
import logging
import os
import re
import sqlite3

logger = logging.getLogger("stage.s01_speakers")

# Speaker fields carried into the manifest ``speakers`` table.
FIELDS = ("speaker_id", "name", "wikidata_id", "region", "profession",
          "gender", "seed_dir")


def _slug(name: str) -> str:
    """Return a filesystem/id-safe slug derived from a name."""
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return slug or "poi"


def _ensure_id(row: dict) -> str:
    """Return the row's speaker_id, or a deterministic one derived from the
    wikidata_id / name so re-runs stay idempotent (no random suffixes)."""
    sid = (row.get("speaker_id") or "").strip()
    if sid:
        return sid
    wd = (row.get("wikidata_id") or "").strip()
    if wd:
        return f"wd_{wd.lower()}"
    return f"sp_{_slug(row.get('name', ''))}"


def read_seed_csv(path: str) -> list[dict]:
    """Read seed speakers from a CSV file. Returns [] if the file is absent."""
    if not path or not os.path.exists(path):
        logger.info("no seed CSV at %s; skipping", path)
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("name") or "").strip()]
    logger.info("read %d speaker(s) from seed CSV", len(rows))
    return rows


def read_webui_pois(db_path: str) -> list[dict]:
    """Read POIs seeded through the web UI. Returns [] if the DB is absent."""
    if not db_path or not os.path.exists(db_path):
        logger.info("no web-UI DB at %s; skipping", db_path)
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT speaker_id, name, wikidata_id, region, profession, gender, "
            "seed_dir FROM speakers")]
    except sqlite3.OperationalError as exc:
        logger.warning("web-UI DB unreadable (%s); skipping", exc)
        return []
    finally:
        conn.close()
    logger.info("read %d POI(s) from web-UI DB", len(rows))
    return rows


def wikidata_enrich(name: str, wikidata_id: str | None) -> dict:
    """Best-effort Wikidata lookup of region/profession/gender for a speaker.

    Returns a (possibly empty) dict of newly found fields. Network or parse
    failures are swallowed so one lookup never crashes the stage.
    """
    try:
        from SPARQLWrapper import JSON, SPARQLWrapper
    except ImportError:
        return {}

    if wikidata_id:
        where = f"BIND(wd:{wikidata_id} AS ?p)"
    elif name:
        safe = name.replace('"', '\\"')
        where = (
            f'?p rdfs:label "{safe}"@bn. '
            f'?p wdt:P31 wd:Q5.'  # instance of human
        )
    else:
        return {}

    query = f"""
    SELECT ?countryLabel ?occLabel ?genderLabel WHERE {{
      {where}
      OPTIONAL {{ ?p wdt:P27 ?country. }}
      OPTIONAL {{ ?p wdt:P106 ?occ. }}
      OPTIONAL {{ ?p wdt:P21 ?gender. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,bn". }}
    }} LIMIT 1
    """
    try:
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql",
                               agent="bengali-voxdata/0.1 (s01_speakers)")
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        sparql.setTimeout(20)
        bindings = sparql.query().convert()["results"]["bindings"]
    except Exception as exc:  # noqa: BLE001 — fail soft on any network/parse error
        logger.warning("Wikidata lookup failed for %r: %s", name or wikidata_id, exc)
        return {}

    if not bindings:
        return {}
    b = bindings[0]
    out = {}
    if "countryLabel" in b:
        out["region"] = b["countryLabel"]["value"]
    if "occLabel" in b:
        out["profession"] = b["occLabel"]["value"]
    if "genderLabel" in b:
        out["gender"] = b["genderLabel"]["value"]
    return out


def _merge(base: dict, extra: dict) -> dict:
    """Fill only empty/missing fields of ``base`` from ``extra``."""
    out = dict(base)
    for k, v in extra.items():
        if v and not (out.get(k) or ""):
            out[k] = v
    return out


def run(cfg, mf, limit: int | None = None) -> None:
    """Collect seed + web-UI speakers, enrich via Wikidata, upsert to manifest.

    Idempotent and resumable: existing speakers are re-upserted; a speaker with
    both region and profession is marked ``done``, otherwise ``pending``.
    """
    sp_cfg = cfg.speakers
    use_wikidata = bool(getattr(sp_cfg, "use_wikidata", False))

    # Merge sources by speaker_id (CSV first, then web-UI fills gaps).
    merged: dict[str, dict] = {}
    for src in (read_seed_csv(getattr(sp_cfg, "seed_csv", "")),
                read_webui_pois(getattr(sp_cfg, "webui_db", ""))):
        for raw in src:
            sid = _ensure_id(raw)
            present = {k: raw[k] for k in FIELDS if (raw.get(k) or "")}
            present["speaker_id"] = sid
            merged[sid] = _merge(merged.get(sid, {}), present)

    if not merged:
        logger.warning("no speakers found from any source; nothing to do")
        return

    items = list(merged.values())
    if limit is not None:
        items = items[:limit]

    done = 0
    for row in items:
        sid = row["speaker_id"]
        try:
            if use_wikidata and not (row.get("region") and row.get("profession")):
                row = _merge(row, wikidata_enrich(row.get("name", ""),
                                                  row.get("wikidata_id")))
            record = {k: row.get(k) for k in FIELDS}
            complete = bool(record.get("region") and record.get("profession"))
            record["status"] = "done" if complete else "pending"
            mf.upsert("speakers", record)
            done += int(complete)
            logger.info("speaker %s (%s) -> %s", sid, record.get("name"),
                        record["status"])
        except Exception as exc:  # noqa: BLE001 — fail soft per item
            logger.error("speaker %s failed: %s", sid, exc)
            mf.upsert("speakers", {"speaker_id": sid, "name": row.get("name"),
                                   "status": "failed"})

    min_speakers = int(getattr(sp_cfg, "min_speakers", 0) or 0)
    logger.info("s01 complete: %d/%d speakers fully populated (min_speakers=%d)",
                done, len(items), min_speakers)
    if done < min_speakers:
        logger.warning("only %d speakers have region+profession (< min_speakers=%d)",
                       done, min_speakers)
