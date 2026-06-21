"""CLI orchestrator for the pipeline.

Usage:
    python -m pipeline.run --stage s01 --config config/config.yaml [--limit N]
    python -m pipeline.run --stage all --config config/config.yaml

Each stage module under ``pipeline/stages`` exposes ``run(cfg, mf, limit)``.
The orchestrator loads config, opens the manifest, and dispatches by name.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys

from .config import load_config
from .manifest import Manifest

# Ordered stage list — also the order used by ``--stage all``.
STAGES = [
    "s01_speakers", "s02_collect", "s03_scenes", "s04_facetrack",
    "s05_facerec", "s06_asd", "s07_diarize", "s08_audio", "s09_asr",
    "s10_sector", "s11_dialect", "s12_group", "s13_quality", "s14_metadata",
    "s15_verify", "s16_release",
]

# Map both the short alias (s01) and full name (s01_speakers) to the module.
STAGE_ALIASES = {name.split("_")[0]: name for name in STAGES}


def _resolve_stage(token: str) -> str:
    """Map a CLI stage token (``s01`` or ``s01_speakers``) to a module name."""
    if token in STAGES:
        return token
    if token in STAGE_ALIASES:
        return STAGE_ALIASES[token]
    raise SystemExit(f"unknown stage {token!r}; choose from: all, "
                     + ", ".join(STAGE_ALIASES))


def _run_stage(name: str, cfg, mf, limit: int | None) -> None:
    """Import ``pipeline.stages.<name>`` and invoke its ``run``."""
    module = importlib.import_module(f"pipeline.stages.{name}")
    logging.getLogger("pipeline.run").info("=== running %s ===", name)
    module.run(cfg, mf, limit=limit)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the orchestrator."""
    p = argparse.ArgumentParser(prog="pipeline.run",
                                description="Bengali multimodal dataset pipeline.")
    p.add_argument("--stage", required=True,
                   help="stage to run: 'all', a short id (s01), or full name (s01_speakers)")
    p.add_argument("--config", default="config/config.yaml",
                   help="path to config.yaml (default: config/config.yaml)")
    p.add_argument("--limit", type=int, default=None,
                   help="process at most N items (for fast local testing)")
    p.add_argument("--resume", action="store_true",
                   help="resume: stages skip items already marked done (default behaviour)")
    p.add_argument("--log-level", default="INFO", help="logging level (default INFO)")
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse args, load config + manifest, dispatch stage(s)."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    mf = Manifest(cfg.paths.manifest)
    mf.connect()
    try:
        targets = STAGES if args.stage == "all" else [_resolve_stage(args.stage)]
        for name in targets:
            _run_stage(name, cfg, mf, args.limit)
    finally:
        mf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
