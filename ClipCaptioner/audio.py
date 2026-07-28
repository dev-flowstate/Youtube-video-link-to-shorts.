"""Extract Whisper-ready audio from a video file."""

from __future__ import annotations

from pathlib import Path

import ffmpeg_tools

# Whisper expects 16 kHz mono PCM. Feeding it anything else just makes
# faster-whisper resample internally.
SAMPLE_RATE = 16_000
CHANNELS = 1


def extract_audio(video_path: Path, wav_path: Path) -> Path:
    """Write a 16 kHz mono WAV alongside the given video."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    if wav_path.exists():
        wav_path.unlink()

    args = [
        ffmpeg_tools.tool_path("ffmpeg"),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        str(CHANNELS),
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    ffmpeg_tools.run(args)

    if not wav_path.exists() or wav_path.stat().st_size == 0:
        raise ffmpeg_tools.FFmpegError(f"Audio extraction produced nothing for {video_path.name}")

    return wav_path
