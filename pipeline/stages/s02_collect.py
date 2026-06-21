"""Stage 2 — collect: speaker names -> downloaded video + ``videos`` rows.

For each speaker with ``status='done'`` (from s01), search YouTube for
candidate videos, filter by duration and language, download with ``yt-dlp``
into ``data/raw/<speaker_id>/`` (with ``info.json``), and upsert ``videos``
rows storing ``source_url``.

Search uses the YouTube Data API when ``YT_API_KEY`` is set (richer metadata),
falling back to ``yt-dlp``'s ``ytsearch`` otherwise. Fail-soft per item and
idempotent: videos already downloaded (``status='done'``) are skipped.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
import urllib.request

logger = logging.getLogger("stage.s02_collect")

_API = "https://www.googleapis.com/youtube/v3"
_ISO_DUR = re.compile(
    r"P(?:(?P<d>\d+)D)?T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?")


def parse_iso8601_duration(value: str) -> float:
    """Convert an ISO-8601 duration (e.g. ``PT1H2M3S``) to seconds."""
    m = _ISO_DUR.fullmatch(value or "")
    if not m:
        return 0.0
    d, h, mn, s = (int(m.group(k) or 0) for k in ("d", "h", "m", "s"))
    return float(d * 86400 + h * 3600 + mn * 60 + s)


def _api_get(endpoint: str, params: dict) -> dict:
    """GET a YouTube Data API endpoint and return the parsed JSON."""
    url = f"{_API}/{endpoint}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search_videos(query: str, api_key: str, max_results: int = 10,
                  relevance_language: str = "bn") -> list[dict]:
    """Search YouTube via the Data API; return candidate dicts with metadata.

    Each candidate: ``video_id, url, title, duration_s, lang``. Raises on a
    failed request — callers handle that per speaker.
    """
    search = _api_get("search", {
        "key": api_key, "q": query, "part": "id", "type": "video",
        "maxResults": min(max_results, 50), "relevanceLanguage": relevance_language,
    })
    ids = [it["id"]["videoId"] for it in search.get("items", [])
           if it.get("id", {}).get("videoId")]
    if not ids:
        return []
    details = _api_get("videos", {
        "key": api_key, "id": ",".join(ids),
        "part": "contentDetails,snippet",
    })
    out = []
    for it in details.get("items", []):
        snip = it.get("snippet", {})
        out.append({
            "video_id": it["id"],
            "url": f"https://www.youtube.com/watch?v={it['id']}",
            "title": snip.get("title"),
            "duration_s": parse_iso8601_duration(
                it.get("contentDetails", {}).get("duration", "")),
            "lang": snip.get("defaultAudioLanguage") or snip.get("defaultLanguage"),
        })
    return out


def ytsearch_fallback(query: str, n: int = 10) -> list[dict]:
    """Search via yt-dlp (no API key). Language is unknown from search alone."""
    from yt_dlp import YoutubeDL

    opts = {"quiet": True, "noprogress": True, "extract_flat": True,
            "skip_download": True}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    out = []
    for e in (info or {}).get("entries", []):
        if not e:
            continue
        out.append({
            "video_id": e.get("id"),
            "url": e.get("url") or f"https://www.youtube.com/watch?v={e.get('id')}",
            "title": e.get("title"),
            "duration_s": float(e.get("duration") or 0.0),
            "lang": None,
        })
    return out


def filter_candidates(candidates: list[dict], min_s: float, max_s: float,
                      lang_filter: str | None) -> list[dict]:
    """Keep candidates within the duration bounds and matching the language.

    Unknown language (``None``) is kept — many Bengali uploads omit the field;
    downstream diarization/ASR confirm the language. A known, non-matching
    language is dropped.
    """
    kept = []
    for c in candidates:
        dur = c.get("duration_s") or 0.0
        if dur and not (min_s <= dur <= max_s):
            continue
        lang = (c.get("lang") or "").lower()
        if lang_filter and lang and not lang.startswith(lang_filter.lower()):
            continue
        kept.append(c)
    return kept


def download_video(url: str, dest_dir: str) -> tuple[str, dict]:
    """Download one video + its info.json with yt-dlp; return (path, info)."""
    from yt_dlp import YoutubeDL

    os.makedirs(dest_dir, exist_ok=True)
    opts = {
        "outtmpl": os.path.join(dest_dir, "%(id)s.%(ext)s"),
        "writeinfojson": True,
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "quiet": True,
        "noprogress": True,
        "ignoreerrors": False,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info), info


def run(cfg, mf, limit: int | None = None) -> None:
    """Search, filter, and download videos for each done speaker.

    ``limit`` caps the total number of videos downloaded this run (handy for
    ``--limit 3`` smoke tests). Idempotent: already-downloaded videos skip.
    """
    cc = cfg.collect
    per_speaker = int(getattr(cc, "per_speaker_videos", 5))
    min_s = float(getattr(cc, "min_duration_s", 0))
    max_s = float(getattr(cc, "max_duration_s", 1e9))
    lang_filter = getattr(cc, "lang_filter", None)
    raw_root = os.path.join(cfg.paths.data, "raw")
    api_key = os.environ.get("YT_API_KEY", "").strip()

    speakers = mf.speakers_with_status("done")
    if not speakers:
        logger.warning("no speakers with status='done'; run s01 first")
        return
    if not api_key:
        logger.warning("YT_API_KEY not set; using yt-dlp ytsearch fallback")

    remaining = limit  # global download budget (None = unlimited)
    for sp in speakers:
        if remaining is not None and remaining <= 0:
            break
        name, sid = sp["name"], sp["speaker_id"]
        try:
            raw = (search_videos(name, api_key, max_results=per_speaker * 3)
                   if api_key else ytsearch_fallback(name, n=per_speaker * 3))
        except Exception as exc:  # noqa: BLE001 — fail soft per speaker
            logger.error("search failed for %s (%s): %s", sid, name, exc)
            continue

        wanted = filter_candidates(raw, min_s, max_s, lang_filter)[:per_speaker]
        logger.info("%s (%s): %d candidate(s) after filter", sid, name, len(wanted))

        for cand in wanted:
            if remaining is not None and remaining <= 0:
                break
            vid = cand["video_id"]
            existing = mf.query("SELECT status FROM videos WHERE video_id=?", (vid,))
            if existing and existing[0]["status"] == "done":
                logger.info("  %s already downloaded; skipping", vid)
                continue
            try:
                local_path, _info = download_video(cand["url"],
                                                   os.path.join(raw_root, sid))
                mf.upsert("videos", {
                    "video_id": vid, "speaker_id": sid, "url": cand["url"],
                    "title": cand.get("title"), "duration_s": cand.get("duration_s"),
                    "lang": cand.get("lang") or lang_filter,
                    "local_path": local_path, "status": "done",
                })
                if remaining is not None:
                    remaining -= 1
                logger.info("  downloaded %s -> %s", vid, local_path)
            except Exception as exc:  # noqa: BLE001 — fail soft per item
                logger.error("  download failed for %s: %s", vid, exc)
                mf.upsert("videos", {
                    "video_id": vid, "speaker_id": sid, "url": cand["url"],
                    "title": cand.get("title"), "status": "failed",
                })

    logger.info("s02 complete: %d video(s) in manifest (%d done)",
                mf.count("videos"), mf.count("videos", "status='done'"))
