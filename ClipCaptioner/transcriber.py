"""Word-level transcription via faster-whisper."""

from __future__ import annotations

from pathlib import Path

import config
from models import Word


class TranscriptionError(Exception):
    """Raised when transcription produces no usable words."""


_MODEL = None


def _load_model():
    """Load the Whisper model once and reuse it across clips.

    The first call downloads the weights into the HuggingFace cache, which is
    why this is lazy rather than done at import time.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    from faster_whisper import WhisperModel

    print(
        f"Loading Whisper '{config.WHISPER_MODEL}' "
        f"({config.WHISPER_DEVICE}/{config.WHISPER_COMPUTE_TYPE})... "
        "first run downloads the model."
    )
    _MODEL = WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )
    return _MODEL


def _clean(text: str) -> str:
    return text.strip()


def _initial_prompt(title: str | None) -> str | None:
    """Prime the decoder with words it should expect.

    Whisper conditions on this text, which pulls proper nouns towards the
    right spelling. Costs nothing at runtime.
    """
    parts = [part for part in (config.WHISPER_VOCABULARY or ()) if part.strip()]
    if title:
        parts.insert(0, title)

    if not parts:
        return None
    return ", ".join(parts) + "."


def transcribe_words(wav_path: Path, title: str | None = None) -> list[Word]:
    """Transcribe audio into individually timed words."""
    model = _load_model()

    segments, _info = model.transcribe(
        str(wav_path),
        language=config.WHISPER_LANGUAGE,
        word_timestamps=True,
        # Skips long silences, which keeps timestamps from drifting.
        vad_filter=True,
        initial_prompt=_initial_prompt(title),
    )

    words: list[Word] = []
    for segment in segments:
        for word in segment.words or ():
            text = _clean(word.word)
            if not text:
                continue
            start = float(word.start)
            end = float(word.end)
            if end <= start:
                end = start + 0.05
            words.append(Word(text=text, start_s=start, end_s=end))

    if not words:
        raise TranscriptionError(f"No speech detected in {wav_path.name}")

    return words
