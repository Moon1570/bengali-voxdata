"""Fetch reference face images for seed speakers from Wikimedia Commons.

For each speaker with a ``wikidata_id``, downloads the canonical portrait
(Wikidata ``P18``) plus a few images from the speaker's Commons category
(``P373``) into ``data/raw/seeds/<speaker_id>/`` and records ``seed_dir`` in the
manifest. These images seed the ArcFace embeddings used by stage s05 (facerec).

Usage:
    python -m scripts.fetch_seed_images --config config/config.local.yaml
    python -m scripts.fetch_seed_images --config config/config.local.yaml --max-images 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import load_config  # noqa: E402
from pipeline.manifest import Manifest  # noqa: E402

logger = logging.getLogger("fetch_seed_images")

UA = "bengali-voxdata/0.1 (research dataset; contact ch@ht15.de)"
_WD_API = "https://www.wikidata.org/w/api.php"
_COMMONS_API = "https://commons.wikimedia.org/w/api.php"
_FILEPATH = "https://commons.wikimedia.org/wiki/Special:FilePath/"
_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def _get_json(url: str) -> dict:
    """GET a URL with the required Wikimedia User-Agent and parse JSON."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _claim_values(qid: str, prop: str) -> list[str]:
    """Return string-valued claims for a Wikidata property (e.g. P18, P373)."""
    url = (f"{_WD_API}?action=wbgetclaims&entity={qid}&property={prop}"
           f"&format=json")
    claims = _get_json(url).get("claims", {}).get(prop, [])
    out = []
    for c in claims:
        val = c.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(val, str):
            out.append(val)
    return out


def _category_images(category: str, limit: int) -> list[str]:
    """Return up to ``limit`` image filenames from a Commons category."""
    cat = category if category.startswith("Category:") else f"Category:{category}"
    url = (f"{_COMMONS_API}?action=query&list=categorymembers"
           f"&cmtitle={urllib.parse.quote(cat)}&cmtype=file&cmlimit=50&format=json")
    members = _get_json(url).get("query", {}).get("categorymembers", [])
    files = [m["title"].split("File:", 1)[-1] for m in members
             if m["title"].lower().endswith(_IMG_EXT)]
    return files[:limit]


def seed_filenames(qid: str, max_images: int) -> list[str]:
    """Collect candidate Commons filenames for a speaker: P18 first, then
    a few from the Commons category (P373), de-duplicated."""
    names: list[str] = []
    for fn in _claim_values(qid, "P18"):
        if fn not in names:
            names.append(fn)
    if len(names) < max_images:
        for cat in _claim_values(qid, "P373"):
            for fn in _category_images(cat, max_images):
                if fn not in names:
                    names.append(fn)
                if len(names) >= max_images:
                    break
            if len(names) >= max_images:
                break
    return names[:max_images]


def download_image(filename: str, dest_dir: str) -> str | None:
    """Download one Commons file into ``dest_dir``; return the local path."""
    os.makedirs(dest_dir, exist_ok=True)
    safe = filename.replace(" ", "_")
    url = _FILEPATH + urllib.parse.quote(safe)
    ext = os.path.splitext(safe)[1].lower() or ".jpg"
    local = os.path.join(dest_dir, f"{abs(hash(filename)) % 10**8}{ext}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as resp, open(local, "wb") as fh:
            fh.write(resp.read())
    except Exception as exc:  # noqa: BLE001 — skip a bad file, keep going
        logger.warning("failed to download %s: %s", filename, exc)
        return None
    return local


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fetch_seed_images")
    p.add_argument("--config", default="config/config.local.yaml")
    p.add_argument("--max-images", type=int, default=4)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    mf = Manifest(cfg.paths.manifest)
    mf.connect()
    seeds_root = os.path.join(cfg.paths.data, "raw", "seeds")

    speakers = mf.query(
        "SELECT speaker_id, name, wikidata_id FROM speakers WHERE wikidata_id IS NOT NULL")
    if not speakers:
        logger.warning("no speakers with a wikidata_id; run s01 first")
        return 0

    for sp in speakers:
        sid, qid = sp["speaker_id"], sp["wikidata_id"]
        dest = os.path.join(seeds_root, sid)
        try:
            names = seed_filenames(qid, args.max_images)
        except Exception as exc:  # noqa: BLE001 — fail soft per speaker
            logger.error("%s (%s): metadata fetch failed: %s", sid, qid, exc)
            continue
        if not names:
            logger.warning("%s (%s): no images found on Wikidata/Commons", sid, qid)
            continue
        saved = [pth for fn in names if (pth := download_image(fn, dest))]
        if saved:
            mf.upsert("speakers", {"speaker_id": sid, "seed_dir": dest})
            logger.info("%s: saved %d image(s) -> %s", sid, len(saved), dest)

    mf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
