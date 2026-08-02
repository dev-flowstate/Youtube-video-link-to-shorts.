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

# Names and jargon Whisper should expect. The model biases towards this text
# when decoding, which is the difference between "WILLEY" and "Willy".
# Proper nouns are exactly what a podcast clip is about and exactly what a
# small model gets wrong, so list the people, places and brands in your videos.
WHISPER_VOCABULARY: list[str] = [
    # "MrBeast", "Chandler", "Karl", "Chris",
]

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

# Intel QuickSync encodes ~3x faster than libx264 on this class of laptop and
# keeps the work off the CPU cores. Set to False to force software encoding,
# e.g. if you dislike the quality or your machine has no Intel iGPU. The
# renderer falls back to libx264 automatically when QSV is unavailable.
PREFER_HARDWARE_ENCODER = True
HARDWARE_ENCODER = "h264_qsv"

# QSV uses -global_quality, libx264 uses -crf. Lower is better quality for both.
HARDWARE_QUALITY = 22
VIDEO_CRF = 20
VIDEO_PRESET = "medium"
AUDIO_BITRATE = "160k"

# ---------------------------------------------------------------------------
# Face tracking
# ---------------------------------------------------------------------------

# Follow faces when choosing the 9:16 crop window instead of always cropping
# the centre. Falls back to centre framing whenever no face is found.
TRACK_FACES = True

# How often to look for a face, in samples per second. Higher tracks fast
# movement better and costs roughly linearly in detection time. Below about
# 3 the crop visibly lags the subject.
TRACKING_SAMPLE_FPS = 4.0

# Frames are downscaled to this width before detection. Bigger finds smaller
# faces but costs proportionally more.
TRACKING_FRAME_WIDTH = 480

# Detections below this confidence are ignored.
TRACKING_MIN_SCORE = 0.6

# Rate the finished crop path is written out at. The crop moves in steps, so
# this needs to be high enough that the steps are invisible - below about 10
# the motion reads as stuttery rather than smooth.
TRACKING_OUTPUT_FPS = 15.0

# Width of the smoothing window in seconds. This is the main dial: larger is
# smoother and slower to react, smaller is snappier and more jittery.
# Applied forwards and backwards, so it adds no lag.
TRACKING_SMOOTHING_WINDOW_S = 1.2

# Ceiling on how fast the crop may pan within a shot, in source pixels per
# second. Stops a mis-detection whipping the frame across the scene.
TRACKING_MAX_PAN_PX_PER_S = 320.0

# Ignore a face this far from the current framing - almost always a
# background person rather than the subject. Fraction of source width.
TRACKING_MAX_JUMP_FRACTION = 0.45

# Mean pixel difference between consecutive samples that counts as a scene
# cut. At a cut the crop repositions instantly instead of sliding across the
# frame, because gliding through a cut looks like a mistake. Set too low,
# handheld camera movement registers as a cut and the crop snaps constantly.
TRACKING_CUT_THRESHOLD = 40.0

# Shots shorter than this are merged into the previous one. Real cuts are
# rarely this close together; back-to-back detections are camera shake.
TRACKING_MIN_SHOT_S = 0.7

# Movements smaller than this are ignored, so the crop sits still during
# normal talking-head micro-motion. Applied to the raw signal *before*
# smoothing - afterwards it would put the steps back in.
TRACKING_DEADZONE_PX = 12.0


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------

# Write a suggested title beside each rendered clip, as a .txt file. The title
# is the clip's own strongest line, chosen by scoring the transcript - no model
# is involved, so it can never invent something that was not said.
WRITE_TITLES = True

# Titles longer than this get cut off on mobile. YouTube shows roughly this
# much before truncating.
TITLE_MAX_CHARS = 70

# ---------------------------------------------------------------------------
# Ending on a complete thought
# ---------------------------------------------------------------------------

# End clips where a sentence finishes rather than merely where speech pauses.
# A pause happens inside sentences too, so without this a clip can stop
# halfway through an idea, which reads worse than an extra few seconds.
END_ON_COMPLETE_THOUGHT = True

# Bounds on where that sentence ending may be, in seconds from the clip's
# start. Finishing the thought is worth going over the nominal 90s for, but
# not by much - a clip that runs on defeats the format.
THOUGHT_MIN_SECONDS = 35.0
THOUGHT_MAX_SECONDS = 110.0

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
