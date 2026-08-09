"""Where the broadcast overlay sits on screen.

Every rectangle is in **fractions of the frame**, not pixels, so the same
numbers work whether a frame arrives at 1920x1080, 1280x720 or the 960x540 the
feature pass actually runs at.

Measured by hand off a verified firefight frame from the EN EWC broadcast at
1280x720, by cropping each region and looking at it rather than guessing. The
regions marked provisional were not fully visible in that frame and want
confirming against more footage before anything leans on them.

The point of these is *not* to read the overlay like a template. A stored
bitmap of one tournament's art stops working the moment the art changes. These
only say "look here"; what the features actually measure is structure that
survives a re-skin - does anything persist here while the world moves, how many
tick marks are lit, did a banner appear.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    """A rectangle as fractions of frame width and height."""

    x0: float
    y0: float
    x1: float
    y1: float

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """This region as (x0, y0, x1, y1) pixel bounds, clamped to the frame."""
        px0 = max(0, min(width - 1, int(round(self.x0 * width))))
        py0 = max(0, min(height - 1, int(round(self.y0 * height))))
        px1 = max(px0 + 1, min(width, int(round(self.x1 * width))))
        py1 = max(py0 + 1, min(height, int(round(self.y1 * height))))
        return px0, py0, px1, py1


# ---------------------------------------------------------------------------
# Verified against a real gameplay frame
# ---------------------------------------------------------------------------

# "Remaining 8 | Team 2 | Time 29:27" strip along the very top left.
TOP_STATS = Region(0.014, 0.008, 0.210, 0.036)

# The players-remaining digits inside that strip. Deliberately loose - the
# number is one or two digits and shifts as it shrinks.
REMAINING_DIGITS = Region(0.056, 0.006, 0.082, 0.038)

# "Elims 4" panel directly below.
ELIMS_PANEL = Region(0.020, 0.053, 0.134, 0.082)

# The observed squad's four player rows. A knocked player shows a red bar under
# their name, which is the cheapest knock signal available - no OCR, no icons.
TEAM_ROSTER = Region(0.021, 0.093, 0.145, 0.239)

# The four-team bar across the top. Each slot carries a rank badge, a logo, a
# name and one tick per living player. Tick colour matters: green for the teams
# currently in frame, red for a knocked player, grey for teams elsewhere.
#
# The right edge was measured at 0.844 off a single frame, which put the slot
# pitch about 3 px wide at 960 and let the error accumulate - by the fourth
# slot the computed tick zone had walked clean past the actual ticks. Pulled in
# after checking the pitch across all four slots.
TEAM_BAR = Region(0.203, 0.000, 0.830, 0.042)
TEAM_BAR_SLOTS = 4

# Within one slot, the ticks sit towards the right-hand end. Widened from an
# initial (0.72, 1.00) for the same reason: measured against a real bar the
# ticks occupy roughly the middle-right of a slot, not its last quarter.
TEAM_SLOT_TICKS = (0.55, 0.95)

# This production does not always show the bar. It alternates between the bar
# and a right-hand standings table carrying the same alive information, so the
# team-bar region is empty for roughly half of a match. That is why gameplay
# presence is decided across three regions rather than this one: any single
# region can be absent for minutes at a time without meaning the game is off.
#
# Ticks are slanted slashes, so at 960 wide their x-projections overlap and
# connected components merge and fragment. Measured against a hand-read bar the
# green/grey split came out wrong while the total was close (17 against 16), so
# the total is worth using and the colour breakdown is not.
TEAM_BAR_INTERMITTENT = True

# Minimap, top right.
MINIMAP = Region(0.877, 0.004, 0.979, 0.139)

# The blue-zone countdown bar and the "02:12  Stage 9" line beneath the minimap.
ZONE_BAR = Region(0.877, 0.178, 0.979, 0.197)
ZONE_STAGE = Region(0.877, 0.201, 0.979, 0.226)

# Just the stage number, for digit matching.
STAGE_DIGITS = Region(0.963, 0.201, 0.981, 0.226)

# The observed player's card, loadout and webcam along the bottom.
BOTTOM_CARD = Region(0.305, 0.792, 0.789, 1.000)

# ---------------------------------------------------------------------------
# Provisional - confirm against more footage
# ---------------------------------------------------------------------------

# Kill and knock notices stack down the left-middle as dark banners with bright
# text. This broadcast has no persistent kill feed in the usual top-right spot;
# the one notice visible in the reference frame sat here.
#
# Raw pixel change is the wrong measure for this strip, because it is mostly
# live footage and the camera moves. What identifies a notice is the banner
# itself - a box holding bright text - which camera motion does not produce.
#
# Two corrections after testing against real notices. The plate is translucent
# rather than black, so over a bright ground it reads around 150 and a
# dark-pixel test misses it entirely; what holds is "mostly dark rows carrying
# a minority of bright glyphs". And the bottom edge was clipping the top of the
# picture-in-picture monitor in some frames, which added ink that was not a
# notice, so it is raised.
KILLFEED_STRIP = Region(0.000, 0.278, 0.195, 0.520)

# Floating damage numbers appear around the centre, near whoever the camera is
# following. Direct evidence of a hit landing.
#
# This region is broad, and a plain bright-blob count inside it also catches
# player nametags and zone warnings - it scored 21 blobs on a frame holding no
# damage numbers at all. What it measures is honestly "small bright words on a
# dark plate near the middle of the screen", which correlates with a fight
# without being only damage numbers.
DAMAGE_ZONE = Region(0.300, 0.150, 0.750, 0.550)

# The standings table that covers the right of the screen during analysis. Its
# signal is *negative*: a table on screen means the broadcast is talking about
# the game, not showing it.
STANDINGS_OVERLAY = Region(0.620, 0.150, 1.000, 0.850)

# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

# Regions whose persistence decides whether live gameplay is on screen at all.
# Spread across three corners on purpose: a caster cam or a lower-third can
# cover one of them, and two out of three still answers the question.
PRESENCE_REGIONS = {
    "minimap": MINIMAP,
    "topleft": TOP_STATS,
    "teambar": TEAM_BAR,
}

# Everything that belongs to the overlay rather than the world. Masked out of
# whole-frame motion features, so a ticking clock does not read as action.
OVERLAY_REGIONS = (
    TOP_STATS,
    ELIMS_PANEL,
    TEAM_ROSTER,
    TEAM_BAR,
    MINIMAP,
    ZONE_BAR,
    ZONE_STAGE,
    BOTTOM_CARD,
)
