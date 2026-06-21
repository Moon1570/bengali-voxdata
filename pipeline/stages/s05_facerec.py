"""Stage 5 — facerec: tracks -> speaker_id, facerec_score (ArcFace buffalo_l)."""
from __future__ import annotations


def run(cfg, mf, limit: int | None = None) -> None:
    """Read upstream rows with status='done', process, write artifacts to
    data/, upsert results, set status. Idempotent and resumable.

    TODO: implement in this module's dedicated branch.
    """
    raise NotImplementedError("s05_facerec not yet implemented.")
