"""Word-level transcription via faster-whisper."""

from __future__ import annotations

from pathlib import Path

import config
from models import Word


class TranscriptionError(Exception):
    """Raised when transcription produces no usable words."""


_MODEL = None

# Longest a single word may occupy. Real speech tops out around 2s for a drawn
# out word; anything beyond that is a bad timestamp, not slow talking.
MAX_WORD_DURATION_S = 2.5


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


def _initial_prompt() -> str | None:
    """Prime the decoder with names it should expect.

    Only the curated vocabulary goes in here. Feeding it the filename was
    tried and reverted: a clip named after its video ("... - Alex Hormozi
    (4K)") primed the model to *say* that name, and it opened by hallucinating
    "Alex Hormozi is the founder..." while discarding the first 19 seconds of
    real speech. A prompt steers the decoder, so anything in it must be words
    that are genuinely spoken.
    """
    parts = [part.strip() for part in (config.WHISPER_VOCABULARY or ()) if part.strip()]
    if not parts:
        return None
    return ", ".join(parts) + "."


def detect_language(wav_path: Path) -> str | None:
    """Work out what is being spoken, once, so every part agrees.

    A configured language always wins - if the user says the stream is Urdu,
    detection cannot overrule them.
    """
    if config.WHISPER_LANGUAGE:
        return config.WHISPER_LANGUAGE

    model = _load_model()
    try:
        _segments, info = model.transcribe(str(wav_path), vad_filter=True)
        language = getattr(info, "language", None)
        probability = getattr(info, "language_probability", 0.0) or 0.0
    except Exception:
        return None

    if not language:
        return None

    print(f"    language: {language} ({probability:.0%} confident)")
    return str(language)


def transcribe_words(
    wav_path: Path,
    title: str | None = None,
    language: str | None = None,
) -> list[Word]:
    """Transcribe audio into individually timed words.

    `title` is accepted and ignored; see _initial_prompt for why.
    """
    model = _load_model()

    segments, _info = model.transcribe(
        str(wav_path),
        language=language or config.WHISPER_LANGUAGE,
        word_timestamps=True,
        # Skips long silences, which keeps timestamps from drifting.
        vad_filter=True,
        initial_prompt=_initial_prompt(),
        # Stops a hallucinated phrase becoming context for the next window and
        # running away with the rest of the clip.
        condition_on_previous_text=False,
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
            # A single word never lasts this long. When Whisper misfires it
            # emits one spanning many seconds, which would hold a caption on
            # screen over speech that has already moved on.
            end = min(end, start + MAX_WORD_DURATION_S)
            words.append(Word(text=text, start_s=start, end_s=end))

    if not words:
        raise TranscriptionError(f"No speech detected in {wav_path.name}")

    return words
