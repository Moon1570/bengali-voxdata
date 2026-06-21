# PLAN.md — Bengali Multimodal Speech Dataset Pipeline

> **For the coding agent (Copilot):** Build this repository exactly as specified. Work milestone by milestone (§9). Each stage is an independent, resumable module with a fixed input/output contract (§5). Do not skip the manifest design (§4) — every stage reads and writes it. Everything runs in Docker (§7). When a milestone's acceptance check passes, move to the next.

Must: Create different branch for each module, different commits for each functions. 

---

## 1. Goal

A pipeline that turns in-the-wild Bengali video into a verified multimodal dataset of **aligned video + audio + transcript**, usable for speaker verification, ASR, TTS, and diarization. Output is **manifest + annotations + reconstruction scripts** (URLs and timestamps), **not** redistributed media.

## 2. Principles (apply to all code)

- **Resumable & idempotent.** Every stage skips items already completed (check the manifest). Re-running a stage must not duplicate work or corrupt outputs.
- **Config-driven.** No hardcoded paths, model names, or thresholds. Everything comes from `config/config.yaml`.
- **Manifest is the source of truth.** Stages communicate only through the SQLite manifest (§4) and the `data/` tree (§3). No stage imports another stage.
- **`--limit N` everywhere** for fast local testing on a handful of items.
- **GPU optional for dev.** Detect CUDA; fall back to CPU with a warning so the scaffold runs on a laptop.
- **Structured logging** (`logging`, one logger per stage), type hints, and docstrings on every public function.
- **Fail soft per item.** One bad video must not crash the stage; log the error, mark the item `status=failed`, continue.

## 3. Repository layout

```
bengali-voxdata/
├── PLAN.md                  # this file
├── README.md                # quickstart (generate from §10)
├── .env.example             # HF_TOKEN=...  (plus any env the user's ASR script needs)
├── .gitignore               # data/, *.pyc, .env, __pycache__
├── requirements.txt
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── config/
│   ├── config.yaml
│   └── speakers_seed.csv    # optional manual seed list
├── pipeline/
│   ├── __init__.py
│   ├── run.py               # CLI orchestrator
│   ├── config.py            # load + validate config.yaml -> dataclass
│   ├── manifest.py          # SQLite wrapper (§4)
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── s01_speakers.py
│   │   ├── s02_collect.py
│   │   ├── s03_scenes.py
│   │   ├── s04_facetrack.py
│   │   ├── s05_facerec.py
│   │   ├── s06_asd.py
│   │   ├── s07_diarize.py
│   │   ├── s08_audio.py
│   │   ├── s09_asr.py
│   │   ├── s10_sector.py
│   │   ├── s11_dialect.py
│   │   ├── s12_group.py
│   │   ├── s13_quality.py
│   │   ├── s14_metadata.py
│   │   ├── s15_verify.py
│   │   └── s16_release.py
│   ├── asr/
│   │   ├── __init__.py
│   │   └── transcribe.py    # adapter + stub (§5a); user's script slots in here
│   └── utils/
│       ├── __init__.py
│       ├── io.py            # path helpers, json/parquet io
│       ├── video.py         # frame iteration, ffmpeg wrappers
│       ├── audio.py         # load/resample, SNR
│       └── gpu.py           # device selection
├── scripts/
│   ├── pilot_eval.py        # the go/no-go gate (§8)
│   └── reconstruct.py       # rebuild dataset from URLs + timestamps
├── data/                    # gitignored, mounted as a volume
│   ├── raw/                 # downloaded video + info.json
│   ├── interim/             # shots, tracks, embeddings, clips
│   ├── processed/           # final per-utterance artifacts
│   └── release/             # manifest.jsonl, datasheet.md, scripts
├── tests/
│   └── test_manifest.py
└── manifest.db              # created at runtime (gitignored)
```

## 4. Manifest (SQLite) — `pipeline/manifest.py`

Single file `manifest.db`. Provide a thin wrapper class `Manifest` with `connect()`, `upsert(table, row)`, `query(sql, params)`, `mark_status(table, id, status)`, and per-table helpers. Use `WAL` mode for concurrent reads.

**Tables and key columns:**

```
speakers(speaker_id PK, name, wikidata_id, region, profession, gender, seed_dir, status)
videos(video_id PK, speaker_id FK, url, title, duration_s, lang, local_path, status)
shots(shot_id PK, video_id FK, start_t, end_t, status)
tracks(track_id PK, shot_id FK, video_id, bbox_path, speaker_id,
       facerec_score, asd_score, is_active_speaker, status)
utterances(utt_id PK, video_id, speaker_id, track_id,
           start_t, end_t, audio_16k, audio_24k, transcript, transcript_conf,
           sector, dialect, dialect_conf, snr_db, overlap_flag, length_s,
           tier, verified, source_url, status)
```

`status` ∈ `pending | done | failed | skipped`. Each stage filters on the previous stage's `status='done'`.

## 5. Stage contract

Every `sNN_*.py` exposes:

```python
def run(cfg: Config, mf: Manifest, limit: int | None = None) -> None:
    """Read upstream rows with status='done', process, write artifacts to
    data/, upsert results, set status. Idempotent and resumable."""
```

### Stage specs (real tools — these names go in the code)

| # | Module | Input → Output | Library / Model | Acceptance |
|---|--------|----------------|-----------------|------------|
| 1 | speakers | seed/Wikidata → `speakers` rows | `SPARQLWrapper` (Wikidata), optional LLM NER; Bengali + transliterated names | ≥N speakers with region+profession populated |
| 2 | collect | speaker names → downloaded video + `videos`,`info.json` | `yt-dlp` (python API), filter by `duration_s`, `lang=bn`; store `source_url` | videos downloaded, metadata captured, lang filtered |
| 3 | scenes | video → `shots` | `scenedetect` `ContentDetector` (threshold from cfg) | shots cover full video, no gaps |
| 4 | facetrack | shots → face tracks (bbox sequences) | `insightface` SCRFD/RetinaFace detect + **ByteTrack** | each track has continuous bboxes; one row per track |
| 5 | facerec | tracks → `speaker_id`, `facerec_score` | `insightface` ArcFace (`buffalo_l`); compare to speaker seed embedding (cosine) | tracks assigned or rejected at cfg threshold |
| 6 | asd | active-speaker tracks → `is_active_speaker`, `asd_score` | **TalkNet-ASD** (or Light-ASD) via git submodule | only target+speaking tracks kept |
| 7 | diarize | video audio → speaker segments + overlap | `pyannote.audio` `speaker-diarization-3.1` (needs `HF_TOKEN`) | RTTM produced; overlap regions flagged |
| 8 | audio | active+single-speaker segments → wav clips → `utterances` | `ffmpeg` → 16 kHz mono PCM (`audio_16k`) **and** 24 kHz copy (`audio_24k`) | clips exist, correct sample rate, length in cfg bounds |
| 9 | asr | clips → `transcript`, `transcript_conf`, words | **User-provided transcription script** (free Google API) called through the adapter `pipeline/asr/transcribe.py` (§5a). Ships with a **stub mode** so the pipeline runs before the script is added | non-empty transcript per clip (real or stub); WER measured in pilot |
| 10 | sector | speaker text → `sector` | zero-shot `transformers` pipeline, `xlm-roberta-large` / BanglaBERT; labels in cfg | every speaker labelled |
| 11 | dialect | clip audio + region → `dialect`, `dialect_conf` | `wav2vec2` features + **weak label from `speakers.region`**; probabilistic | dialect stored as probability, not hard class |
| 12 | group | utterances → consistent `speaker_id` clusters | ArcFace embedding clustering (`sklearn` Agglomerative) | utterances grouped per identity |
| 13 | quality | clip → `snr_db`, `overlap_flag`, `tier` | WADA-SNR (`utils/audio.py`) + overlap from §7 + length | `tier ∈ {tts_grade, asr_grade}` per cfg thresholds |
| 14 | metadata | utterance row → JSON-LD file | hand-rolled JSON-LD writer | valid JSON-LD per utterance with all fields |
| 15 | verify | random subset → review bundle (CSV + clips) | export for **Label Studio / CVAT**; ingest results back | accuracy computed on reviewed subset |
| 16 | release | manifest → `manifest.jsonl`, `datasheet.md`, `reconstruct.py` | pandas; **URLs+timestamps only, no media** | release bundle reconstructs dataset from URLs |

### 5a. ASR adapter contract (stage 9)

The user supplies the transcription script (a free Google API wrapper that takes audio and returns text). **Copilot does not implement the Google call.** Instead, build everything around a single adapter and a stub, so the pipeline is testable today and the real script drops in later with no other changes.

Create `pipeline/asr/transcribe.py`:

```python
from dataclasses import dataclass

@dataclass
class TranscriptResult:
    text: str
    confidence: float | None = None
    words: list | None = None          # optional [{word, start, end}, ...]

def transcribe(audio_path: str, language_code: str) -> TranscriptResult:
    """Transcribe one clip. INTEGRATION POINT — the user will drop their
    free-Google-API script in here (or have this import/call it).
    Until then, raise NotImplementedError; `asr.mode: stub` bypasses this."""
    raise NotImplementedError("Add the user-provided transcription script here.")
```

Rules for stage `s09_asr.py`:
- Read `asr.mode` from config. **`stub`** → write a fixed placeholder transcript (e.g. `"<asr-stub>"`, `confidence=None`) so milestones M3–M5 run without the script. **`provider`** → call `transcribe()`.
- Choose `language_code` per speaker region via `asr.region_language_map`, falling back to `asr.language_code`.
- Per-clip: on success set `transcript`, `transcript_conf`, optional words, `status=done`; on exception log and set `status=failed`, continue.
- Respect `asr.requests_per_minute` (client-side rate limit) and `asr.max_retries` (exponential backoff) — the user's script may hit an external quota.
- Keep the adapter the **only** file that imports the user's script, so nothing else in the repo depends on its internals.

## 6. Config — `config/config.yaml` (Copilot: create with these keys)

```yaml
paths:
  data: /app/data
  manifest: /app/manifest.db
speakers:
  seed_csv: /app/config/speakers_seed.csv
  use_wikidata: true
  min_speakers: 50
  languages: [bn]
collect:
  per_speaker_videos: 5
  min_duration_s: 60
  max_duration_s: 1800
  lang_filter: bn
scenes:
  detector: content
  threshold: 27.0
facetrack:
  det_model: scrfd_10g_bnkps
  det_threshold: 0.5
facerec:
  model: buffalo_l
  cosine_threshold: 0.45   # raise if pilot precision < 95%
asd:
  backend: talknet
  score_threshold: 0.0
audio:
  sr_asr: 16000
  sr_tts: 24000
  min_len_s: 1.0
  max_len_s: 15.0
asr:
  mode: stub                    # stub | provider  (stub runs without the script)
  language_code: bn-IN          # default; auto-switch by speaker region
  region_language_map:
    Bangladesh: bn-BD
    India: bn-IN
  max_retries: 5
  requests_per_minute: 300      # client-side rate limit for the user's script
sector:
  model: xlm-roberta-large
  labels: [actor, politician, news_anchor, athlete, creator, academic, other]
dialect:
  weak_label_from_region: true
quality:
  tts_min_snr_db: 20
  tts_max_overlap_s: 0.0
  asr_max_overlap_s: 0.5
verify:
  sample_size: 200
release:
  redistribute_media: false   # MUST stay false
```

## 7. Docker

### `docker/Dockerfile`
```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3-pip python3.10-dev git ffmpeg wget ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
ENTRYPOINT ["python", "-m", "pipeline.run"]
CMD ["--help"]
```

### `docker/docker-compose.yml`
```yaml
services:
  pipeline:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    image: bengali-voxdata:latest
    env_file: ../.env
    working_dir: /app
    volumes:
      - ../data:/app/data
      - ../config:/app/config
      - ../manifest.db:/app/manifest.db
      - ../pipeline/asr:/app/pipeline/asr           # drop the user's transcription script here
      - hf-cache:/root/.cache/huggingface
      - torch-cache:/root/.cache/torch
    shm_size: "8gb"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
volumes:
  hf-cache:
  torch-cache:
```

### `requirements.txt` (pin exact versions during build)
```
torch
torchaudio
yt-dlp
scenedetect[opencv]
opencv-python-headless
insightface
onnxruntime-gpu
pyannote.audio>=3.1
transformers
huggingface_hub
pandas
pyarrow
pyyaml
tqdm
soundfile
librosa
numpy
scikit-learn
ffmpeg-python
SPARQLWrapper
# TalkNet-ASD: add as git submodule under third_party/
# ASR: the user's transcription script and its deps drop into pipeline/asr/ (add to this file when supplied)
```

### Run commands (put in README)
```bash
cp .env.example .env            # add HF_TOKEN for pyannote
cd docker
docker compose build
docker compose run --rm pipeline --stage all --config /app/config/config.yaml
# single stage / test:
docker compose run --rm pipeline --stage s02 --limit 3
```

## 8. Pilot gate — `scripts/pilot_eval.py` (run before scaling)

Runs the full pipeline on a small set (≈50 videos / 10 speakers), then computes and prints **two gates**:

1. **Face-recognition precision** against a hand-labelled track→identity file → **PASS if ≥ 0.95**.
2. **ASR WER** against human reference transcripts for ~30 min of clips (use `jiwer`) → **PASS if ≤ 0.20**.

Also report: % segments lost to overlap filtering, avg usable seconds per video, ASD false-positive rate. Exit non-zero if either gate fails. **Do not run full-scale collection until both gates pass.**

## 9. Build milestones (do in order; each must pass its check)

- **M0 — Scaffold.** Repo layout, `config.py`, `manifest.py`, `run.py` CLI (`--stage`, `--config`, `--limit`, `--resume`), Docker builds, `tests/test_manifest.py` green. *Check:* `docker compose run --rm pipeline --help` works; manifest tests pass.
- **M1 — Acquisition (s01–s02).** Seed/Wikidata speakers + yt-dlp collection with filters. *Check:* `--stage s02 --limit 3` downloads 3 videos and populates `videos`.
- **M2 — Visual (s03–s06).** Scenes, face track, face rec, ASD on the M1 sample. *Check:* at least one active-speaker track correctly tied to a target speaker.
- **M3 — Audio (s07–s09).** Diarization+overlap, clip extraction at both sample rates, transcripts via the ASR adapter. *Check:* clips play; in `asr.mode: stub` every clip gets a placeholder transcript and the stage completes (swap to `provider` once the user's script is in place).
- **M4 — Annotation & quality (s10–s14).** Sector, dialect (weak labels), grouping, SNR/overlap tiering, JSON-LD. *Check:* every utterance has `tier` and a valid JSON-LD file.
- **M5 — Verify & release + gate (s15–s16, pilot_eval).** Review bundle, release manifest + datasheet + `reconstruct.py`, pilot gate script. *Check:* `pilot_eval.py` runs and prints both gate results; `reconstruct.py` rebuilds clips from URLs.

## 10. README (generate this)

Quickstart, the four Docker commands from §7, the config keys to edit first (`speakers.min_speakers`, `facerec.cosine_threshold`, `asr.mode`, `asr.language_code` / `asr.region_language_map`), how ASR works (**stub mode by default**; set `asr.mode: provider` and drop your transcription script into `pipeline/asr/transcribe.py` per §5a), the pilot-gate warning, and the legal note: **release distributes URLs + timestamps + annotations, never the media** (`release.redistribute_media: false`).

## 11. Definition of done

- `docker compose build` succeeds; `--stage all` runs end-to-end on a small `--limit` set without crashing.
- Manifest fully populated; every `utterances` row has audio paths, transcript, sector, dialect, tier, JSON-LD.
- `pilot_eval.py` reports both gates.
- `release/` contains `manifest.jsonl`, `datasheet.md`, and a working `reconstruct.py`.
- No media files in the release bundle.