"""ASR adapter (stage 9 integration point).

This is the ONLY file that imports the user's free-Google-API transcription
script, so nothing else in the repo depends on its internals.
"""
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
