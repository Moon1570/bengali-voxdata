"""SQLite data layer for the POI (speaker) web UI.

The ``speakers`` table mirrors the manifest schema in PLAN.md §4 so stage
``s01_speakers`` can consume these rows directly. Extra columns
(``image_path``, ``notes``, timestamps) and the ``speaker_tags`` table support
the researcher-facing form, labelling, and tagging.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

# Default DB path; overridable via env so the UI can point at the real manifest.
DB_PATH = os.environ.get("POI_DB", os.path.join(os.path.dirname(__file__), "poi.db"))

# Allowed pipeline statuses (PLAN.md §4).
STATUSES = ("pending", "done", "failed", "skipped")


def _now() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str | None = None) -> sqlite3.Connection:
    """Open a connection with row access by column name and WAL mode."""
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str | None = None) -> None:
    """Create the ``speakers`` and ``speaker_tags`` tables if absent."""
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS speakers (
                speaker_id  TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                wikidata_id TEXT,
                region      TEXT,
                profession  TEXT,
                gender      TEXT,
                seed_dir    TEXT,
                image_path  TEXT,
                notes       TEXT,
                status      TEXT NOT NULL DEFAULT 'pending',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS speaker_tags (
                speaker_id TEXT NOT NULL,
                tag        TEXT NOT NULL,
                PRIMARY KEY (speaker_id, tag),
                FOREIGN KEY (speaker_id) REFERENCES speakers(speaker_id)
                    ON DELETE CASCADE
            );
            """
        )
        conn.commit()


def create_speaker(data: dict[str, Any], db_path: str | None = None) -> str:
    """Insert a new speaker row and return its ``speaker_id``."""
    ts = _now()
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO speakers
                (speaker_id, name, wikidata_id, region, profession, gender,
                 seed_dir, image_path, notes, status, created_at, updated_at)
            VALUES
                (:speaker_id, :name, :wikidata_id, :region, :profession, :gender,
                 :seed_dir, :image_path, :notes, :status, :created_at, :updated_at)
            """,
            {
                "speaker_id": data["speaker_id"],
                "name": data["name"],
                "wikidata_id": data.get("wikidata_id"),
                "region": data.get("region"),
                "profession": data.get("profession"),
                "gender": data.get("gender"),
                "seed_dir": data.get("seed_dir"),
                "image_path": data.get("image_path"),
                "notes": data.get("notes"),
                "status": data.get("status", "pending"),
                "created_at": ts,
                "updated_at": ts,
            },
        )
        conn.commit()
    return data["speaker_id"]


def list_speakers(db_path: str | None = None) -> list[dict[str, Any]]:
    """Return all speakers (newest first) with their tags attached."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM speakers ORDER BY created_at DESC"
        ).fetchall()
        speakers = [dict(r) for r in rows]
        for sp in speakers:
            sp["tags"] = _tags_for(conn, sp["speaker_id"])
    return speakers


def get_speaker(speaker_id: str, db_path: str | None = None) -> dict[str, Any] | None:
    """Return one speaker (with tags), or ``None`` if it does not exist."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM speakers WHERE speaker_id = ?", (speaker_id,)
        ).fetchone()
        if row is None:
            return None
        sp = dict(row)
        sp["tags"] = _tags_for(conn, speaker_id)
        return sp


def update_speaker(speaker_id: str, fields: dict[str, Any],
                   db_path: str | None = None) -> None:
    """Update the given editable columns for a speaker."""
    allowed = {
        "name", "wikidata_id", "region", "profession", "gender",
        "seed_dir", "image_path", "notes", "status",
    }
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return
    sets["updated_at"] = _now()
    assignments = ", ".join(f"{k} = :{k}" for k in sets)
    sets["speaker_id"] = speaker_id
    with connect(db_path) as conn:
        conn.execute(
            f"UPDATE speakers SET {assignments} WHERE speaker_id = :speaker_id",
            sets,
        )
        conn.commit()


def set_status(speaker_id: str, status: str, db_path: str | None = None) -> None:
    """Set a speaker's pipeline status (validated against ``STATUSES``)."""
    if status not in STATUSES:
        raise ValueError(f"invalid status {status!r}; expected one of {STATUSES}")
    update_speaker(speaker_id, {"status": status}, db_path)


def add_tag(speaker_id: str, tag: str, db_path: str | None = None) -> None:
    """Attach a normalised (lower, stripped) tag to a speaker; idempotent."""
    tag = tag.strip().lower()
    if not tag:
        return
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO speaker_tags (speaker_id, tag) VALUES (?, ?)",
            (speaker_id, tag),
        )
        conn.commit()


def remove_tag(speaker_id: str, tag: str, db_path: str | None = None) -> None:
    """Detach a tag from a speaker."""
    with connect(db_path) as conn:
        conn.execute(
            "DELETE FROM speaker_tags WHERE speaker_id = ? AND tag = ?",
            (speaker_id, tag.strip().lower()),
        )
        conn.commit()


def _tags_for(conn: sqlite3.Connection, speaker_id: str) -> list[str]:
    """Return the sorted tag list for a speaker on an open connection."""
    rows = conn.execute(
        "SELECT tag FROM speaker_tags WHERE speaker_id = ? ORDER BY tag",
        (speaker_id,),
    ).fetchall()
    return [r["tag"] for r in rows]
