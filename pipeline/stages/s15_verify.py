"""Stage 15 — verify: random subset -> review bundle; ingest results."""
from __future__ import annotations


def run(cfg, mf, limit: int | None = None) -> None:
    """Read upstream rows with status='done', process, write artifacts to
    data/, upsert results, set status. Idempotent and resumable.

    TODO: implement in this module's dedicated branch.
    """
    raise NotImplementedError("s15_verify not yet implemented.")
