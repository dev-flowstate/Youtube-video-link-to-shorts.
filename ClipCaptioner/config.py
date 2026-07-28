"""Tunable settings for the captioning pipeline.

Everything a user is likely to change lives here, so the other modules stay
free of magic numbers.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent

# Folder holding the cut clips produced by YouTubeReplayDownloader. Points at
# that project's default output folder, so the two steps line up with no setup.
# Override per run with: python main.py --input "D:\some\folder"
INPUT_DIR = _PROJECT_ROOT.parent / "YouTubeReplayDownloader" / "output"

# Folder the captioned vertical clips are written to.
OUTPUT_DIR = _PROJECT_ROOT / "output"

# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

# faster-whisper model size. "small" balances accuracy against CPU time.
# "base" is ~3x faster and noticeably less accurate; "medium" is far slower.
WHISPER_MODEL = "small"

# int8 is the right choice without an NVIDIA GPU.
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# Force a language to skip auto-detection. None means detect.
WHISPER_LANGUAGE: str | None = "en"

# ---------------------------------------------------------------------------
# Caption grouping
# ---------------------------------------------------------------------------

# Words visible on screen at once. Clipper captions stay tiny and fast.
MAX_WORDS_PER_GROUP = 3

# Characters on screen at once, so long words do not overflow the frame.
MAX_CHARS_PER_GROUP = 20

# A silence longer than this starts a new caption group.
GROUP_SPLIT_SILENCE_S = 0.45

# Groups shorter than this get padded so they do not flicker.
MIN_GROUP_DURATION_S = 0.30

# ---------------------------------------------------------------------------
# Caption styling
# ---------------------------------------------------------------------------

# Both are installed on Windows by default. "Impact" is narrower and taller.
FONT_NAME = "Arial Black"
FONT_SIZE = 110

# ASS colours are &HBBGGRR - blue and red are swapped relative to hex web
# colours. These constants are the raw BGR values.
COLOR_BASE = "FFFFFF"       # white - inactive words
COLOR_ACTIVE = "00E5FF"     # amber - the word being spoken
COLOR_OUTLINE = "000000"    # black outline

OUTLINE_WIDTH = 7
SHADOW_DEPTH = 4

# Scale bump applied to the active word, as a percentage.
ACTIVE_SCALE_PERCENT = 112

# Distance from the bottom of the frame to the caption baseline, in pixels.
# 1920-tall output, so ~560 puts captions just below centre.
CAPTION_MARGIN_V = 560

# Uppercase reads louder and is the clipper convention.
UPPERCASE_CAPTIONS = True

# ---------------------------------------------------------------------------
# Output video
# ---------------------------------------------------------------------------

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920

VIDEO_CRF = 20
VIDEO_PRESET = "medium"
AUDIO_BITRATE = "160k"

# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

# Clips longer than this are split into parts. YouTube Shorts caps at 3
# minutes; 179s leaves a safety margin. Set SPLIT_LONG_CLIPS to False to
# render one long vertical video instead.
SPLIT_LONG_CLIPS = True
MAX_PART_DURATION_S = 179.0

# Never emit a trailing fragment shorter than this; merge it into the
# previous part instead.
MIN_PART_DURATION_S = 15.0
