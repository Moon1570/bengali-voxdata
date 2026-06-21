# POI Manager (web UI)

A small Flask app for **stage s01 (speakers)**: researchers seed Persons of
Interest (POIs) — name, region, profession, gender, Wikidata ID, a reference
image, free-form tags, and notes — then browse, **label** (pipeline status),
and **tag** them.

Data is written through `webui/db.py` into a SQLite database whose `speakers`
table matches the pipeline manifest schema (PLAN.md §4), so stage
`s01_speakers` can consume these rows directly.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r webui/requirements.txt
python -m webui.app                 # http://127.0.0.1:5000
```

## Configuration (env vars)

| Var          | Default            | Purpose                                  |
|--------------|--------------------|------------------------------------------|
| `POI_DB`     | `webui/poi.db`     | SQLite path (point at the manifest to share data) |
| `POI_SECRET` | `dev-poi-secret`   | Flask session secret (set in production) |
| `PORT`       | `5000`             | HTTP port                                |

## Data model

- **speakers** — `speaker_id, name, wikidata_id, region, profession, gender,
  seed_dir, image_path, notes, status, created_at, updated_at`
- **speaker_tags** — `(speaker_id, tag)`, normalised lowercase, deduped

Reference images are stored under `webui/uploads/<speaker_id>/` (gitignored);
`seed_dir` points at that folder for the face-recognition seed in later stages.
