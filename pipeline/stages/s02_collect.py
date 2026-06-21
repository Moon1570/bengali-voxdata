"""Stage 2 — collect: speaker names -> downloaded video + videos/info.json (yt-dlp)."""
from __future__ import annotations


def run(cfg, mf, limit: int | None = None) -> None:
    """Read upstream rows with status='done', process, write artifacts to
    data/, upsert results, set status. Idempotent and resumable.

    TODO: implement in this module's dedicated branch.
    """
    raise NotImplementedError("s02_collect not yet implemented.")
