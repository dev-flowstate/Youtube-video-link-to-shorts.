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

# faster-whisper model size. Measured on a real podcast clip, both int8 on
# CPU: small transcribed it in 19s with 4.7% of words below 0.5 confidence,
# medium in 47s with 2.8%. Medium
# roughly halves the words the model was unsure about, and the ones it fixed
# were real - "I asked them" for "I ask them", "noticed" for "notice".
#
# 2.5x slower, but that is 47s against a render that takes minutes, and a wrong
# word is burned into the video where nobody can fix it. Drop back to "small"
# if a batch is running overnight and accuracy matters less than throughput.
WHISPER_MODEL = "medium"

# int8 is the right choice without an NVIDIA GPU.
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# Language of the speech. None auto-detects, which is the right default: not
# every stream is in English, and *forcing* a language makes Whisper decode
# foreign speech into that language, producing transliterated nonsense rather
# than a translation or an honest failure. Set a code like "en" or "ur" only
# when you already know what the source is.
WHISPER_LANGUAGE: str | None = None

# Detection is run once per clip and pinned for all of its parts. Per-part
# detection was rejected: a part dominated by gunfire or music can detect
# differently from its neighbour, so one clip would caption in two languages.
DETECT_LANGUAGE_ONCE = True

# Burn the captions into the video. Turned into a prompt at startup, or set
# from the command line with --captions / --no-captions. With this off the
# clips are still cropped to vertical, face-tracked, titled and named - only
# the subtitle burn-in is skipped.
BURN_CAPTIONS = True

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

# Characters on screen at once. Arial Black runs about 0.62 x the font size
# per character, so at FONT_SIZE 96 in a 1080-wide frame with CAPTION_MARGIN_H
# either side, 15 characters is roughly the widest line that still fits.
# Raising this without lowering FONT_SIZE pushes text off the frame.
MAX_CHARS_PER_GROUP = 15

# A silence longer than this starts a new caption group.
GROUP_SPLIT_SILENCE_S = 0.45

# Groups shorter than this get padded so they do not flicker.
MIN_GROUP_DURATION_S = 0.30

# ---------------------------------------------------------------------------
# Caption styling
# ---------------------------------------------------------------------------

# Both are installed on Windows by default. "Impact" is narrower and taller.
# Used for Latin scripts only - see SCRIPT_FONTS.
FONT_NAME = "Arial Black"
FONT_SIZE = 88

# Arial Black carries no Arabic, Devanagari or CJK glyphs, so captioning a
# non-English stream with it renders every word as an empty box. The font is
# therefore chosen from the detected language. Every entry below was confirmed
# installed on this machine; a language with no entry falls back to FONT_NAME.
#
# Right-to-left scripts need no special handling here: the bundled FFmpeg
# reports libfribidi and libharfbuzz, so libass reorders and shapes them once
# the font actually has the glyphs.
SCRIPT_FONTS: dict[str, str] = {
    # Arabic script
    "ur": "Segoe UI",   # Urdu - PMWC ships a [UR] feed
    "ar": "Segoe UI",
    "fa": "Segoe UI",
    "ps": "Segoe UI",
    # Devanagari and neighbours
    "hi": "Nirmala UI",
    "mr": "Nirmala UI",
    "ne": "Nirmala UI",
    "bn": "Nirmala UI",
    "pa": "Nirmala UI",
    "ta": "Nirmala UI",
    "te": "Nirmala UI",
    # CJK
    "zh": "Microsoft YaHei",
    "ja": "Yu Gothic",
    "ko": "Malgun Gothic",
    # Cyrillic, Greek and the rest of Latin are covered by the default.
}


def font_for_language(language: str | None) -> str:
    """Pick a font whose glyphs cover the language's script."""
    if not language:
        return FONT_NAME
    return SCRIPT_FONTS.get(language.lower().split("-")[0], FONT_NAME)

# Clear space either side of the caption, in pixels. Text wider than
# TARGET_WIDTH minus twice this wraps to a second line rather than running off
# the frame.
CAPTION_MARGIN_H = 60

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

# How the 16:9 source becomes a 9:16 clip.
#
#   "crop"    - cut a tall slice out of the frame. Right for a talking head:
#               the subject fills the screen and the empty sides are no loss.
#   "fit"     - shrink the whole frame to fit the width and fill the rest with
#               a blurred copy. Right for gameplay, where cropping throws away
#               the minimap, the kill feed and often the fight itself.
#   "stacked" - the streamer's camera above, the rest of the frame below.
#               Right for a stream, where the two things worth watching sit in
#               different parts of the screen.
#
# Cropping is not a mild choice. A 9:16 slice of a 16:9 frame keeps 31.6% of
# the width, so two thirds of the picture is discarded. On a talking head that
# is empty room; on a stream it is the game the reaction is about.
#
# EsportsClipper marks its output as gameplay, which switches this to "fit"
# automatically - see apply_content_marker in pipeline.py.
CROP_MODE = "crop"

# Blur strength of the backdrop in "fit" mode. Blurred by shrinking and
# regrowing rather than with gblur, which is far too slow over a whole clip.
FIT_BACKDROP_BLUR = 12

# ---------------------------------------------------------------------------
# Stacked layout
# ---------------------------------------------------------------------------

# Share of the height given to the streamer's camera. The rest holds the full
# frame, so whatever the reaction is about stays on screen.
STACKED_CAMERA_SHARE = 0.45

# The camera box is found from where faces actually appear, not from a fixed
# corner - it sits in a different place for every streamer, and a wrong guess
# would enlarge a patch of wall. This is how much room to leave around the
# face, as a multiple of its width, so hair and shoulders are included rather
# than a tight head crop.
STACKED_CAMERA_PADDING = 1.9

# Shape of the enlarged camera panel. Roughly 4:3, which suits a webcam better
# than a square and leaves the lower panel most of the height.
STACKED_CAMERA_ASPECT = 4.0 / 3.0

# A face must be found in at least this share of sampled frames before the
# stacked layout is used. Below it there is no reliable camera to enlarge, so
# the clip falls back to "fit" - which loses nothing - rather than guessing.
STACKED_MIN_FACE_RATE = 0.25

# Follow faces when choosing the 9:16 crop window instead of always cropping
# the centre. Falls back to centre framing whenever no face is found. Has no
# effect in "fit" mode, where there is no crop window to move.
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

# Write a .txt beside each clip holding the description and hashtags to paste
# when uploading. Off by default so the output folder holds videos only.
#
# This does not affect the title itself: clips are still named after their
# strongest line and still carry it in their MP4 metadata.
WRITE_UPLOAD_NOTES = False

# Titles longer than this get cut off on mobile. YouTube shows roughly this
# much before truncating.
TITLE_MAX_CHARS = 70

# Write the title with Gemini instead of picking the best existing sentence.
# The heuristic can only choose a line the clip already contains, which is
# honest but blunt - it cannot say what a moment is *about*, only quote it.
#
# Needs GEMINI_API_KEY in the environment and `py -m pip install google-genai`.
# With either missing, or if the call fails, the heuristic runs instead, so
# this is safe to leave on.
USE_GEMINI_TITLES = True
GEMINI_MODEL = "gemini-3-flash-preview"

# A generated title is rejected unless this share of its meaningful words also
# appear in the transcript. The heuristic physically cannot invent anything; a
# model can, so it has to earn the same guarantee rather than be trusted with
# it.
GEMINI_MIN_GROUNDING = 0.6

# ---------------------------------------------------------------------------
# Ending on a complete thought
# ---------------------------------------------------------------------------

# End clips where a sentence finishes rather than merely where speech pauses.
# A pause happens inside sentences too, so without this a clip can stop
# halfway through an idea, which reads worse than an extra few seconds.
END_ON_COMPLETE_THOUGHT = True

# Bounds on where that sentence ending may be, in seconds from the clip's
# start. Finishing the thought is worth running a little past the target for,
# but not by much - a clip that runs on defeats the format.
#
# The floor can only ever make a clip longer, so it is well below the target:
# at 35s it was dragging short clips out to meet it, which is the opposite of
# what it is for.
THOUGHT_MIN_SECONDS = 20.0
THOUGHT_MAX_SECONDS = 55.0

# How much of each source clip is tail padding rather than clip.
#
# The downloader adds footage past its own cut so a sentence can be finished
# beyond it, and content.json says how much. It matters because the end is
# chosen relative to where the clip was *meant* to end: aim at the padded
# length instead and the padding is kept, which turned every 45s clip into a
# 65s one. Clips with no padding, like EsportsClipper's, leave this at zero.
SOURCE_TAIL_PADDING_SECONDS = 0.0

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
