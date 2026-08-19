"""Begin and end a clip on a whole thought, not merely on a pause.

Snapping to a pause stops clips cutting mid-word, but a pause happens between
sentences *and* inside them, so a clip can still stop halfway through an idea.
Punctuation is what marks the difference, and the transcript has it.

Length gives a little either way: finishing the sentence matters more than
hitting an exact runtime, but a clip that runs on and on defeats the point of
the format, so the search is bounded at both ends.

The start is the same problem read backwards, and it matters more. The
downloader leaves a run-up before the moment people replayed, so the clip opens
wherever that run-up happened to land - most often mid-sentence, which is the
one thing an opening cannot afford to be. In a Shorts feed there is no
thumbnail; the first seconds are the whole audition. So the same run-up is
searched for a sentence worth opening on, and the clip starts there instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import config
import gemini_hook
import titler
from models import CaptionGroup, join_word_texts

# A thought ends at one of these. A comma or dash does not.
_TERMINATORS = (".", "!", "?")

# Breathing room after the final word so its tail is not clipped.
_TAIL_S = 0.35


def sentence_end_times(groups: list[CaptionGroup]) -> list[float]:
    """Times at which a complete sentence finishes."""
    ends: list[float] = []

    for group in groups:
        for word in group.words:
            if word.text.rstrip().endswith(_TERMINATORS):
                ends.append(word.end_s + _TAIL_S)

    return ends


def choose_end(
    groups: list[CaptionGroup],
    natural_end_s: float,
) -> float:
    """Pick the clip's end: the sentence ending nearest the natural cut.

    Nearest rather than latest. The natural end came from where audience
    activity died down, so it marks the moment the clip is about; drifting to
    the furthest sentence inside the budget would pad every clip out to the
    ceiling and bury the point.
    """
    if not config.END_ON_COMPLETE_THOUGHT:
        return natural_end_s

    candidates = [
        end
        for end in sentence_end_times(groups)
        if config.THOUGHT_MIN_SECONDS <= end <= config.THOUGHT_MAX_SECONDS
    ]
    if not candidates:
        return natural_end_s

    return min(candidates, key=lambda end: abs(end - natural_end_s))


def trim_groups(groups: list[CaptionGroup], end_s: float) -> list[CaptionGroup]:
    """Drop caption groups that fall past the chosen end."""
    return [group for group in groups if group.end_s <= end_s]


# ---------------------------------------------------------------------------
# Opening on a hook
# ---------------------------------------------------------------------------

# The cut sits a little before the opening word, the mirror of _TAIL_S, so its
# first consonant is not shaved off. Never more than half the silence in front
# of it, though, or on a tight sentence break the cut reaches back and opens on
# the tail of the previous word - worse than the problem it solves.
_LEAD_S = 0.25

# How much of what follows an opening is shown when ranking it. A hook is
# judged partly on what it promises, and the sentence alone does not show that.
_CONTEXT_WORDS = 40

# Openers that promise nothing: a clip beginning on one has spent its first
# second announcing that it is about to say something. Deliberately not
# titler's list, which is single words only - "you know" and "I mean" read as
# filler whole and as ordinary words apart.
_FILLER_WORDS = (
    "you know",
    "i mean",
    "kind of",
    "sort of",
    "i guess",
    "so",
    "and",
    "but",
    "or",
    "well",
    "okay",
    "ok",
    "yeah",
    "yes",
    "no",
    "um",
    "uh",
    "oh",
    "now",
    "then",
    "just",
    "like",
    "right",
    "anyway",
    "actually",
    "basically",
)

# Anchored, and stopping on a word boundary. Matching a bare prefix instead
# scored "Nobody knows why this happens" as filler because it begins "no", and
# that sentence is the best hook on the list.
_FILLER_OPENER = re.compile(r"^(?:" + "|".join(_FILLER_WORDS) + r")\b", re.I)

# A concrete quantity, which reads as specific and specific is persuasive.
# Digits and currency rather than titler's word list, because digits survive
# translation and "million" does not - the spelled-out magnitudes are scored
# in the English branch instead.
_NUMBER = re.compile(r"[\d$]")

# English magnitudes, for the numbers Whisper spells out rather than digitises.
_MAGNITUDES = frozenset("million billion thousand hundred percent dollars".split())

# A capital letter that is not the sentence's first: usually a name, and a name
# is a stake. Scripts with no capitals simply never match, which is the right
# answer for them rather than a wrong one.
_INNER_CAPITAL = re.compile(r"(?<!^)\b[A-Z][a-z]{2,}")

# A hook has to be a sentence, not a grunt. Below this it is "Exactly." or
# "Right, yeah." - real sentences, but nothing to open a clip on.
_MIN_HOOK_WORDS = 4

# Past this a sentence is a paragraph, and whatever the hook in it was, the
# viewer has scrolled before it arrives.
_MAX_HOOK_WORDS = 35


@dataclass(frozen=True)
class Opening:
    """A place the clip could begin, and the sentence it would begin on."""

    index: int      # which caption group starts it
    start_s: float  # where to cut, in seconds into the clip as it stands
    text: str       # the sentence the clip would open on
    context: str    # that sentence plus a little of what follows it


def _starts_sentence(previous: CaptionGroup) -> bool:
    return previous.words[-1].text.rstrip().endswith(_TERMINATORS)


def _openings(groups: list[CaptionGroup]) -> list[Opening]:
    """Every place the clip could begin, in order.

    Sentence starts only, because opening mid-sentence is the exact failure
    being fixed. caption_builder always breaks a group after a sentence-ending
    word, so a sentence never begins inside a group: the group boundaries hold
    all of them, and trimming to one drops whole groups rather than splitting
    any.
    """
    words = [word for group in groups for word in group.words]

    # Where each group's words begin in that flat list, so the text following
    # a candidate can be read without walking the groups again for each one.
    positions: list[int] = []
    seen = 0
    for group in groups:
        positions.append(seen)
        seen += len(group.words)

    openings: list[Opening] = []

    for index, group in enumerate(groups):
        if index and not _starts_sentence(groups[index - 1]):
            continue

        if index == 0:
            # Where the clip already opens. Pinned to zero rather than to the
            # first word, so that choosing it means leaving the clip alone.
            start_s = 0.0
        else:
            gap = max(0.0, group.start_s - groups[index - 1].end_s)
            start_s = max(0.0, group.start_s - min(_LEAD_S, gap / 2.0))

        window = words[positions[index] : positions[index] + _CONTEXT_WORDS]

        sentence: list[str] = []
        for word in window:
            sentence.append(word.text)
            if word.text.rstrip().endswith(_TERMINATORS):
                break

        openings.append(
            Opening(
                index=index,
                start_s=start_s,
                text=join_word_texts(sentence).strip(),
                context=join_word_texts([word.text for word in window]).strip(),
            )
        )

    return openings


def score_opening(opening: Opening, language: str | None = None) -> float:
    """Rate how well a sentence would work as the first thing a viewer hears.

    The signals that survive translation - a question, a number, a name, a
    sentence with some substance to it - are scored for every clip. The word
    lists are English, so they are scored only when the transcript is.
    """
    text = opening.text.strip()
    if not text:
        return -10.0

    tokens = text.lower().split()
    score = 0.0

    # A question is the strongest opening there is: it is a gap the viewer
    # wants closed. The Arabic mark is included for the reason titler includes
    # it - Whisper emits it for Urdu and Arabic.
    if text.rstrip().endswith(("?", "؟")):
        score += 3.0

    if _NUMBER.search(text):
        score += 2.0

    if _INNER_CAPITAL.search(text):
        score += 1.0

    # Whisper capitalises the beginning of a sentence, so a lower-case first
    # letter is the plainest evidence available that the clip currently opens
    # mid-thought - the failure this whole search exists to fix.
    if text[:1].islower():
        score -= 2.5

    if len(tokens) < _MIN_HOOK_WORDS:
        score -= 2.5
    elif len(tokens) > _MAX_HOOK_WORDS:
        score -= 1.5

    if titler.is_english(language):
        if _FILLER_OPENER.match(text):
            score -= 3.0
        # Stripped of punctuation, or a sentence ending "...a million." would
        # never match the word it plainly contains.
        if _MAGNITUDES & {token.strip(".,!?;:'\"") for token in tokens}:
            score += 1.0

    return score


def choose_start(
    groups: list[CaptionGroup],
    usable_end_s: float,
    language: str | None = None,
) -> float:
    """Pick where the clip opens, in seconds into the clip as it stands.

    Zero means leave it alone, and is the right answer whenever nothing in the
    run-up beats the opening the clip already has.
    """
    if not config.OPEN_ON_HOOK or not groups:
        return 0.0

    openings = [
        opening
        for opening in _openings(groups)
        # Past the run-up, the start would be eating the moment the clip
        # exists for; below the floor, the clip is no longer worth posting.
        if opening.start_s <= config.MAX_START_SHIFT_SECONDS
        and usable_end_s - opening.start_s >= config.THOUGHT_MIN_SECONDS
    ]

    # One candidate is the opening it already has: nothing to choose between.
    if len(openings) < 2:
        return 0.0

    index = gemini_hook.choose_index([opening.context for opening in openings])
    if index is not None:
        return openings[index].start_s

    # Ties go to the earliest, so a later sentence has to actually be better
    # to justify throwing away everything in front of it.
    chosen = max(
        openings,
        key=lambda opening: (score_opening(opening, language), -opening.start_s),
    )
    return chosen.start_s


def trim_start(groups: list[CaptionGroup], start_s: float) -> list[CaptionGroup]:
    """Drop caption groups before the chosen start and rebase the rest.

    Every caption time is measured from the clip's first frame, so moving the
    first frame moves all of them. Whole groups only: a chosen start is always
    a sentence start, and caption_builder always breaks a group there, so no
    group straddles the cut.
    """
    return [group.shifted(-start_s) for group in groups if group.start_s >= start_s]
