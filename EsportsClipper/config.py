"""Tunable settings for esports fight clipping.

Everything a user is likely to change lives here, so the other modules stay
free of magic numbers.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

STREAM_URL = "https://www.youtube.com/live/BFOzTy2ML88?si=smO-PB85IA3vb91c"

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# A five hour broadcast has to be downloaded whole before clips can be cut,
# because partial downloads stall against YouTube. Capping the height keeps
# that at a manageable size, and the output is cropped to 1080x1920 for Shorts
# anyway, so a taller source buys very little.
SOURCE_MAX_HEIGHT = 1080

# ---------------------------------------------------------------------------
# Clip shape
# ---------------------------------------------------------------------------

# Clips shorter than the minimum are incidental pot-shots, not fights.
MIN_CLIP_SECONDS = 15.0
MAX_CLIP_SECONDS = 45.0

# How far back the clip may reach for the start of the engagement. The actual
# start is found by following the gunfire, so a short sharp trade produces a
# short sharp clip instead of one padded with someone looting.
MIN_PRE_ROLL_SECONDS = 8.0
MAX_PRE_ROLL_SECONDS = 32.0

# Kept after the peak for the caster's reaction, which is the payoff.
POST_ROLL_SECONDS = 13.0

# Gunfire density, as a fraction of this fight's own peak, that still counts as
# "shooting". Walking backwards stops below this.
ACTION_FLOOR = 0.20

# ---------------------------------------------------------------------------
# How many clips
# ---------------------------------------------------------------------------

CLIPS_PER_HOUR = 9
MIN_CLIPS = 10
MAX_CLIPS = 60

# Join every clip into one long video for a full upload. The dead air between
# fights is never included, so this is the day's action back to back.
MAKE_COMPILATION = True

# ---------------------------------------------------------------------------
# Analysis resolution
# ---------------------------------------------------------------------------

# Fights are short, so the curve is far finer than the podcast pipeline's 5s.
BUCKET_SECONDS = 2.0

# Smoothed over seconds, not minutes - a firefight lasts under a minute.
SMOOTH_SECONDS = 8.0

# Opening ceremony and outro. The gunfire gate is the real defence against
# these; this only stops a long pre-show skewing the scale.
EDGE_TRIM_SECONDS = 900.0

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# How far ahead a caster reaction still counts towards the moment that caused
# it. The scream lands on the kill, which is the *end* of a fight, so without
# this the run-up scores nothing.
CASTER_LOOKAHEAD_SECONDS = 30.0

# score = gunfire * (HYPE_BASE + HYPE_GAIN * hype)
# Multiplicative on purpose: gunfire gates, hype only amplifies. A ceremony is
# deafening and has no gunfire, so it must score zero however loud it gets.
HYPE_BASE = 0.35
HYPE_GAIN = 1.0

# Below this many gunfire onsets in a bucket, the score is forced to zero. Stops
# a stray shot before a chicken dinner being multiplied into a "fight".
MIN_GUNFIRE_ONSETS = 4.0

# Curves are scaled by this percentile rather than their maximum, so the ~6
# match-winning screams saturate instead of defining the range and flattening
# every ordinary fight into a tie.
NORMALISE_PERCENTILE = 95.0

# ---------------------------------------------------------------------------
# Vision detection
# ---------------------------------------------------------------------------

# Look at the video rather than listen to it. Audio detection was measured
# against known fights on a real broadcast and does not work: the game is mixed
# so far under the casters that studio segments scored 2.4x *higher* on gunfire
# onsets than the fights did, and five separate features came out flat between
# the two. What is obvious on screen is simply not in the waveform.
#
# Needs GEMINI_API_KEY and google-genai. Falls back to audio when unavailable.
USE_VISION = True

# Measured against frames from a real broadcast, labelled by hand: 17 of 18
# correct, and 12 of 12 studio frames correctly kept out - which is the whole
# complaint about the audio detector. Its one error was calling a quiet
# gameplay frame a fight, which costs a dull clip rather than an interview.
#
# Pinned rather than an alias like "gemini-flash-lite-latest", so the behaviour
# does not change under you. gemini-3-flash-preview scored the same but its
# free tier allows twenty requests a day, and a broadcast needs a hundred-odd.
VISION_MODEL = "gemini-3.1-flash-lite"

# Roughly one sampled frame per this many seconds. Frames come from keyframes,
# which cost almost nothing to decode and land on the broadcast's own scene
# cuts - exactly where it switches between the game and the desk.
VISION_SAMPLE_SECONDS = 10.0

# Frames are tiled into a grid so one call covers many moments. A 3x3 grid cuts
# the number of calls by nine.
VISION_FRAME_WIDTH = 320
VISION_GRID_COLUMNS = 3
VISION_GRID_ROWS = 3

# Fight frames closer together than this belong to one engagement. Wide enough
# that a single mislabelled frame mid-fight does not split it in two.
VISION_BRIDGE_SECONDS = 25.0

# A free key allows only a handful of requests a minute. Staying under the
# ceiling is faster than sprinting into it and backing off, and much quieter.
# Raise it if the key is on a paid tier - it is the main thing setting how long
# a broadcast takes to watch.
VISION_REQUESTS_PER_MINUTE = 10.0

# The quota comes back on its own, so waiting is the right response. Giving up
# would leave a stretch of the broadcast unlabelled, and unlabelled reads as
# "no fight here".
VISION_MAX_RETRIES = 4
VISION_RETRY_SECONDS = 20.0

# Refuse to report on a broadcast this badly labelled. Without it, a run where
# most calls failed would quietly claim there were no fights.
VISION_MAX_UNLABELLED = 0.25

# ---------------------------------------------------------------------------
# Gunfire detection (audio fallback only - see USE_VISION)
# ---------------------------------------------------------------------------

# Energy above 4 kHz as a fraction of the total. Gunshots are broadband; voice
# sits between 300 and 3400 Hz. This is what separates a shot from a hard
# consonant, and is the first dial to touch if detection is noisy.
HF_RATIO_MIN = 0.55

# An onset is a flux peak this many times above the rolling median. Relative,
# so broadcast loudness does not matter.
ONSET_FLUX_FACTOR = 1.8

# Music onsets are evenly spaced; gunfire is ragged. Buckets whose onset
# intervals vary less than this read as a drum track and are penalised.
MUSIC_REGULARITY_MIN = 0.35
