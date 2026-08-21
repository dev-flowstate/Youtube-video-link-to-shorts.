"""Tunable settings for the captioning pipeline.

Everything a user is likely to change lives here, so the other modules stay
free of magic numbers.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent

# Model downloads land beside the project rather than on the system drive.
#
# Set here rather than left to the environment on purpose. HF_HOME is also set
# as a user environment variable, but a terminal inherits its environment from
# when its parent window opened, so a VS Code window opened before that was set
# cannot see it - which is exactly how a missing GEMINI_API_KEY once sent this
# pipeline silently down a path nobody wanted. An existing value always wins,
# so anyone who has pointed it somewhere deliberately keeps their setting.
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT.parent / ".cache" / "huggingface"))

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
# Stock footage
# ---------------------------------------------------------------------------

# Where fetched clips are kept. Cached rather than re-downloaded, so a second
# clip mentioning the sea costs nothing and the folder grows into a library
# that works offline.
STOCK_DIR = _PROJECT_ROOT / "stock"

# Pexels is used rather than anything scraped from YouTube. Its licence permits
# commercial and monetised use with no attribution, which is the difference
# between a feature and a channel strike.
#
# The key lives in the PEXELS_API_KEY environment variable, never in the repo.
# Without it the pipeline still runs and simply produces no cutaways.

# Portrait suits a 1080x1920 frame. Landscape stock has to be cropped or
# letterboxed to fit, which wastes most of what was downloaded.
STOCK_ORIENTATION = "portrait"

# Measured on the *short* side, not the height. Portrait stock is tall by
# definition, so a height test passes a 720x1280 clip that is only 720 wide and
# then upscales it half again to fill a 1080-wide frame. The short side is the
# one that has to cover the frame whichever way the footage is oriented.
#
# Above this the file is discarded in favour of the smallest that still clears
# it: downloading 4K to show for one second is bytes and decode time for detail
# nobody sees.
STOCK_MIN_SHORT_SIDE = 1080

# Source clips longer than this are whole scenes. A cutaway needs a second or
# two, so a long download is spent on footage that is cut away from at once.
STOCK_MAX_SOURCE_SECONDS = 30

STOCK_TIMEOUT_S = 30

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
# Off. The key available here allows about five requests a day, and three
# features share it - titles, hook selection and b-roll search terms - so a
# single clip can exhaust it. The failure is the quiet kind: every one of them
# falls back to a heuristic and the run still finishes, so the only symptom is
# that results silently get worse partway through a batch, differently on every
# run.
#
# A pipeline cannot depend on that. The heuristics are now the workflow rather
# than the safety net, which also makes a run deterministic and free. Set this
# back to True only with a key that has real quota behind it.
USE_GEMINI_TITLES = False
# Was gemini-3-flash-preview, whose free tier allows twenty requests a day.
# Three features share this setting - titles, hook selection and b-roll search
# terms - so one clip can spend three of that twenty, and a batch exhausts it
# before the first handful of clips are done. The failure is quiet: every one
# of them falls back to a heuristic and the run still finishes, so the only
# symptom is that the results get worse.
#
# Checked directly: on this key gemini-3-flash-preview and gemini-flash-latest
# both answer 429, and gemini-3.1-flash-lite answers normally. It is also the
# model EsportsClipper settled on after measuring 17 of 18 frames correct.
GEMINI_MODEL = "gemini-3.1-flash-lite"

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
# Opening on a hook
# ---------------------------------------------------------------------------

# Start clips on the sentence most likely to stop a scroll, rather than
# wherever the downloader's run-up happened to land.
#
# The end is already chosen this way; the start was not, and it matters more.
# There is no thumbnail in a Shorts feed, so the first seconds are the whole
# audition, and what they most often contained was the back half of a sentence
# the viewer never heard the front of.
OPEN_ON_HOOK = True

# How far into a clip the start may move, in seconds.
#
# The downloader keeps HOOK_LEAD_SECONDS of run-up in front of the moment
# people replayed, and that run-up is exactly what there is to choose from.
# Past it the start would be eating into the moment the clip exists for.
#
# Nothing else bounds this: THOUGHT_MIN_SECONDS stops a clip being trimmed
# down to nothing, but on a long clip that alone would allow the opening to
# skip most of the way to the payoff.
# Raised from 10.0 after measuring it on eight real clips. The run-up is the
# clip's setup by construction, so the strongest sentence tends to sit near the
# far end of it, and 10.0 kept stopping just short: one clip's first sentence
# boundary fell at 10.65s and so never moved at all, and another opened on
# "I think, you know, one of my," where a second and a half later it could have
# opened on "Two people told me they never wanna come back."
#
# 10.0 was also conservative for a reason that does not hold: the downloader
# snaps the cut outward to a pause, so the moment the clip exists for sits at
# HOOK_LEAD_SECONDS or later, not earlier.
# Measured from the clip's first frame, so it has to move with
# HOOK_LEAD_SECONDS. That is now 4.0, meaning the replayed moment sits about
# four seconds in; leaving this at 15 would let the opening drift eleven
# seconds *past* the very moment the clip exists for.
#
# Six allows a couple of seconds past the peak, which is still on the moment,
# while covering the whole run-up. If no sentence starts inside it the opening
# stays where it is - and unlike before, a mid-sentence open is survivable now
# that the hook card tells the viewer what is coming.
MAX_START_SHIFT_SECONDS = 6.0

# Let Gemini pick the opening instead of scoring sentences by their shape.
# Shape can see a question mark, a number or a name; it cannot see whether a
# line is worth hearing. Needs GEMINI_API_KEY and google-genai exactly as
# USE_GEMINI_TITLES does, costs one request per clip, and falls back to the
# heuristic without either - so it is safe to leave on.
# Off. The key available here allows about five requests a day, and three
# features share it - titles, hook selection and b-roll search terms - so a
# single clip can exhaust it. The failure is the quiet kind: every one of them
# falls back to a heuristic and the run still finishes, so the only symptom is
# that results silently get worse partway through a batch, differently on every
# run.
#
# A pipeline cannot depend on that. The heuristics are now the workflow rather
# than the safety net, which also makes a run deterministic and free. Set this
# back to True only with a key that has real quota behind it.
USE_GEMINI_HOOKS = False

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

# ---------------------------------------------------------------------------
# Filler footage
# ---------------------------------------------------------------------------

# Ambient motion under a podcast clip, to hold attention the way parkour or
# subway-surfer gameplay does on other channels. Off by default and turned
# into a prompt at startup - see _ask_about_filler in main.py - or set from
# the command line with --parkour / --no-parkour. Filler is an opinion about
# the clip, not a correction to it, so a piped or scheduled run leaves it off
# rather than guessing.
# Cut away to stock footage when the speaker names something concrete - a
# charging phone as he says the battery is draining, children as he says "if
# you're a kid". On by default because it is the point of fetching stock at
# all; the planner is conservative and most clips get none.
# Drop a clip whose transcript reads as a sponsor segment.
#
# The most-replayed graph cannot tell an advert from an insight: listeners
# scrub across sponsor reads, and scrubbing registers as replay activity just
# as rewatching does. On a real 1h50m episode two of the four clips produced
# were adverts - half the output, and the half nobody watches.
#
# The signals are reliable because a read has to carry them: it spells out a
# domain for listeners who cannot click, offers a code, quotes a discount, and
# names who is paying. Ordinary conversation does none of that. Measured on
# real text, sponsor reads scored 8.5 and 18.0 against 0.0 to 2.5 for talk that
# merely mentions a website.
SKIP_SPONSOR_READS = True

USE_BROLL = True

PARKOUR_FILLER = False

# Where your own filler footage lives. Drop Minecraft parkour, Subway Surfers
# or anything else in here and it is used ahead of anything fetched.
#
# This folder exists because Pexels cannot supply game footage and never will.
# It is a real-life stock library, and a game recording belongs to whoever
# recorded it. Asked for "minecraft parkour" it ignores the word minecraft and
# returns people leaping between rooftops; asked for "subway surfers gameplay"
# it returns a photograph of a subway train. The filler in the first renders
# was a Pexels clip called "yoga and freerunning" - a nearly static shot of a
# tree, occupying thirty percent of every frame with nothing moving.
FILLER_DIR = _PROJECT_ROOT / "filler"

# Fallback topics, used only when FILLER_DIR is empty. Real-life footage, since
# that is all a stock site has. Kept because some filler beats a black strip,
# but supplying your own is the point.
FILLER_TOPICS: list[str] = ["parkour", "satisfying"]

# Share of the height given to the podcast. The rest holds the filler strip.
# The podcast is the content and has to keep dominating the frame; the filler
# is peripheral motion meant to catch the eye, not compete with it, so the
# split leans hard towards the podcast - the opposite emphasis from
# STACKED_CAMERA_SHARE, where both halves of a stream reaction are actually
# worth watching and the split sits close to even.
FILLER_PODCAST_SHARE = 0.70

# ---------------------------------------------------------------------------
# Opening hook card
# ---------------------------------------------------------------------------

# A line of the clip's own words across the top of its first seconds, saying
# what a cold viewer is about to hear.
#
# On by default, and the only one of these extras that is. A Shorts feed gives
# a clip about a second to be understood, with no thumbnail and no title on
# screen, and OPEN_ON_HOOK can only choose *which* sentence the viewer lands
# mid-way through - it cannot make a spoken sentence arrive faster than speech.
# The card can: reading eight words costs well under a second, saying them
# costs three. Set from the command line with --hook-card / --no-hook-card.
#
# Unlike filler and cutaways this needs no prompt, because it competes with
# nothing: it occupies the upper third, which every layout here leaves empty,
# and only for the opening seconds.
HOOK_CARD = True

# How long the card stays up, in seconds.
#
# Tied to HOOK_MAX_WORDS by one number: a short on-screen line is scanned at
# roughly four words a second, so ten words is about two and a half. Under
# that the last words are still being read when the card fades, which is worse
# than never showing them.
#
# It cannot go much past this either. The seed-audience test this exists for is
# settled in the first one to three seconds, so a card still up at four is no
# longer context for the clip - it is the clip, with the speaker talking
# underneath it. 2.5 fits the reading and still clears the frame inside the
# window being judged.
HOOK_CARD_SECONDS = 2.5

# Longest a card may be, in words. Two independent limits land near the same
# number, which is why it is this one.
#
# Reading: a line on screen is scanned rather than read, at roughly four words
# a second, so ten words is about two and a half - which is what
# HOOK_CARD_SECONDS is set to. The two numbers are the same measurement.
#
# Fitting: Arial Black runs about 0.62 x the font size per character, so at
# HOOK_FONT_SIZE in a 1080-wide frame with CAPTION_MARGIN_H either side a line
# holds roughly 21 characters. Ten words of average length is about 55
# characters, which wraps to three lines and still ends in the top quarter of
# the frame, well clear of a face. Raising this without lowering
# HOOK_FONT_SIZE walks the card down towards the speaker.
#
# The card is always a whole sentence, never part of one, so this is a filter
# rather than a budget: a longer sentence is passed over, not cut down. Six,
# seven and eight were all measured on real transcripts and were too tight -
# at eight the best line in one clip, "They want to put $465,000 up to a coin
# flip", was thrown away for "Best man's gonna win".
HOOK_MAX_WORDS = 10

# Smaller than the captions on purpose. The card carries a whole sentence where
# a caption carries three words, so at FONT_SIZE it would wrap to four lines and
# reach the speaker's face; and it should read as a different thing from the
# captions rather than as one that has drifted up the frame.
HOOK_FONT_SIZE = 72

# Distance from the top of the frame to the top of the card, in pixels.
#
# The upper third is the one part of a 1080x1920 Short that every layout here
# leaves empty: a cropped or fitted talking head puts the face around the
# middle, the stacked layout's camera panel starts at the top but a face inside
# it still sits low, and captions sit at CAPTION_MARGIN_V from the *bottom*,
# which is around y=1150 - so the two cannot collide.
#
# 150 rather than 0 because a phone's status bar and notch overlay the very top
# of a full-bleed video, and because text hard against the frame edge reads as a
# mistake.
HOOK_MARGIN_V = 150

# Transparency of the card's backing box, as ASS alpha: 00 is opaque, FF is
# invisible. The box is what makes the card read as a card rather than as a
# mistimed caption, and it is what guarantees the text stays legible over a
# bright or busy frame, where an outline alone is a gamble. 40 is solid enough
# to carry white text over anything and still shows the footage moving behind
# it.
HOOK_BOX_ALPHA = "40"
