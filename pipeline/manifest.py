"""SQLite manifest wrapper (PLAN.md §4) — the pipeline's source of truth.

All stages communicate only through this manifest and the ``data/`` tree. The
``Manifest`` class owns the schema and provides thin generic helpers
(``upsert``, ``query``, ``mark_status``) plus per-table convenience methods.
WAL mode is enabled for concurrent reads.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Iterable

# Primary-key column for each manifest table (used by upsert / mark_status).
TABLE_PK = {
    "speakers": "speaker_id",
    "videos": "video_id",
    "shots": "shot_id",
    "tracks": "track_id",
    "utterances": "utt_id",
}

VALID_STATUS = ("pending", "done", "failed", "skipped")

SCHEMA = """
CREATE TABLE IF NOT EXISTS speakers (
    speaker_id  TEXT PRIMARY KEY,
    name        TEXT,
    wikidata_id TEXT,
    region      TEXT,
    profession  TEXT,
    gender      TEXT,
    seed_dir    TEXT,
    status      TEXT DEFAULT 'pending'
);
CREATE TABLE IF NOT EXISTS videos (
    video_id   TEXT PRIMARY KEY,
    speaker_id TEXT,
    url        TEXT,
    title      TEXT,
    duration_s REAL,
    lang       TEXT,
    local_path TEXT,
    status     TEXT DEFAULT 'pending'
);
CREATE TABLE IF NOT EXISTS shots (
    shot_id  TEXT PRIMARY KEY,
    video_id TEXT,
    start_t  REAL,
    end_t    REAL,
    status   TEXT DEFAULT 'pending'
);
CREATE TABLE IF NOT EXISTS tracks (
    track_id         TEXT PRIMARY KEY,
    shot_id          TEXT,
    video_id         TEXT,
    bbox_path        TEXT,
    speaker_id       TEXT,
    facerec_score    REAL,
    asd_score        REAL,
    is_active_speaker INTEGER,
    status           TEXT DEFAULT 'pending'
);
CREATE TABLE IF NOT EXISTS utterances (
    utt_id         TEXT PRIMARY KEY,
    video_id       TEXT,
    speaker_id     TEXT,
    track_id       TEXT,
    start_t        REAL,
    end_t          REAL,
    audio_16k      TEXT,
    audio_24k      TEXT,
    transcript     TEXT,
    transcript_conf REAL,
    sector         TEXT,
    dialect        TEXT,
    dialect_conf   REAL,
    snr_db         REAL,
    overlap_flag   INTEGER,
    length_s       REAL,
    tier           TEXT,
    verified       INTEGER,
    source_url     TEXT,
    status         TEXT DEFAULT 'pending'
);
"""


class Manifest:
    """Thin SQLite wrapper around the pipeline manifest database."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        """Open (once) and return the connection, ensuring schema + WAL mode."""
        if self._conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(SCHEMA)
            conn.commit()
            self._conn = conn
        return self._conn

    def close(self) -> None:
        """Close the underlying connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Manifest":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def upsert(self, table: str, row: dict[str, Any]) -> None:
        """Insert or replace a row by the table's primary key."""
        if table not in TABLE_PK:
            raise ValueError(f"unknown table {table!r}")
        cols = list(row.keys())
        placeholders = ", ".join(f":{c}" for c in cols)
        col_list = ", ".join(cols)
        updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != TABLE_PK[table])
        sql = (
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT({TABLE_PK[table]}) DO UPDATE SET {updates}"
            if updates else
            f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES ({placeholders})"
        )
        conn = self.connect()
        conn.execute(sql, row)
        conn.commit()

    def query(self, sql: str, params: Iterable[Any] | dict[str, Any] = ()) -> list[sqlite3.Row]:
        """Run a read query and return all rows."""
        return self.connect().execute(sql, params).fetchall()

    def mark_status(self, table: str, row_id: str, status: str) -> None:
        """Set the ``status`` of one row identified by its primary key."""
        if table not in TABLE_PK:
            raise ValueError(f"unknown table {table!r}")
        if status not in VALID_STATUS:
            raise ValueError(f"invalid status {status!r}; expected {VALID_STATUS}")
        conn = self.connect()
        conn.execute(
            f"UPDATE {table} SET status=? WHERE {TABLE_PK[table]}=?", (status, row_id)
        )
        conn.commit()

    def count(self, table: str, where: str | None = None,
              params: Iterable[Any] = ()) -> int:
        """Return the row count for a table, optionally filtered by a WHERE."""
        sql = f"SELECT COUNT(*) AS n FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return int(self.connect().execute(sql, params).fetchone()["n"])

    # -- per-table helpers -------------------------------------------------

    def get_speaker(self, speaker_id: str) -> sqlite3.Row | None:
        """Return one speaker row, or ``None``."""
        rows = self.query("SELECT * FROM speakers WHERE speaker_id=?", (speaker_id,))
        return rows[0] if rows else None

    def speakers_with_status(self, status: str = "done") -> list[sqlite3.Row]:
        """Return speakers filtered by status (default ``done``)."""
        return self.query("SELECT * FROM speakers WHERE status=?", (status,))
