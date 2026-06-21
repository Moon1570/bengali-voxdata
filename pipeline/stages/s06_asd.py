"""Stage 6 — asd: active-speaker tracks -> is_active_speaker, asd_score (TalkNet-ASD)."""
from __future__ import annotations


def run(cfg, mf, limit: int | None = None) -> None:
    """Read upstream rows with status='done', process, write artifacts to
    data/, upsert results, set status. Idempotent and resumable.

    TODO: implement in this module's dedicated branch.
    """
    raise NotImplementedError("s06_asd not yet implemented.")
