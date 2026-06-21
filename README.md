# bengali-voxdata

A pipeline that turns in-the-wild Bengali video into a verified multimodal dataset of
**aligned video + audio + transcript**, usable for speaker verification, ASR, TTS, and
diarization. The release contains **manifest + annotations + reconstruction scripts**
(URLs and timestamps) — **not** redistributed media.

See [plan.md](plan.md) for the full specification.

## Status

Scaffold only. The 16 stages (`pipeline/stages/s01..s16`) and the M0 core
(`pipeline/config.py`, `pipeline/manifest.py`, `pipeline/run.py`) are stubbed and land
milestone-by-milestone, each module on its own branch with a commit per function.

## Quickstart (Docker)

```bash
cp .env.example .env            # add HF_TOKEN for pyannote (stage 7)
cd docker
docker compose build
docker compose run --rm pipeline --stage all --config /app/config/config.yaml
# single stage / test:
docker compose run --rm pipeline --stage s02 --limit 3
```

## Local dev (without Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Config keys to edit first

`config/config.yaml`:

- `speakers.min_speakers`
- `facerec.cosine_threshold` (raise if pilot precision < 95%)
- `asr.mode` — `stub` (default) or `provider`
- `asr.language_code` / `asr.region_language_map`

## How ASR works

ASR runs in **stub mode by default** so the pipeline is testable without the
transcription script. To use the real provider, set `asr.mode: provider` and drop your
free-Google-API transcription script into `pipeline/asr/transcribe.py` (the single
integration point — see plan.md §5a). Nothing else in the repo imports that script.

## Pilot gate

Do **not** run full-scale collection until both gates in `scripts/pilot_eval.py` pass:
face-recognition precision ≥ 0.95 and ASR WER ≤ 0.20.

## Legal note

The release distributes **URLs + timestamps + annotations, never the media**
(`release.redistribute_media: false` — this MUST stay false).
