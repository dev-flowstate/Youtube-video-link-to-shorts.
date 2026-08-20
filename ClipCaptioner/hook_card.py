"""Say what the clip is about, across the top of its first seconds.

A Shorts viewer decides in about a second, with no thumbnail, no title in view
and often no sound - and what they land in is the middle of somebody else's
conversation. The clip already opens on the strongest sentence in its run-up,
which is thought.choose_start's job, but a sentence takes two or three seconds
to *say* and the decision is made before it finishes.

So the claim goes on screen whole, readable at a glance, while the speaker is
still getting to it. A ten-word line is scanned in about two and a half
seconds; saying it aloud takes four.

Nothing here invents anything, and nothing here cuts a sentence in half. The
card is one whole sentence the speaker actually said, with the filler taken off
its ends, so it can never promise something the clip does not say and can never
say something the speaker did not mean - the same guarantee titler.py gives a
title, and for the same reason.

A clip whose lines are all too long, too short or too empty gets no card at
all. That is a designed outcome rather than a failure: an empty upper third
costs nothing, where "You had an umbrella?" across the opening frame spends the
one second the clip gets on nothing. See _says_something, which is the rule
that made the difference on real footage.
"""

from __future__ import annotations

import re

import config
import titler
from models import Word

# A card is a claim, not a grunt. Below three words it is "Exactly." or "No
# way." - true sentences that tell a cold viewer nothing.
_MIN_CARD_WORDS = 3

# How long a question has to be before it counts as saying something on its
# own - see _says_something. Six was tried and let "Sir, can you come over
# here?" and "You're good, buddy, you're good, okay?" through on real footage.
_MIN_QUESTION_WORDS = 7

# Words and phrases that fill time. Stripped off both ends of a line rather
# than merely scored down, because they sit around otherwise good ones: "So it
# cost four hundred grand" is a card and "So" spent on nothing is not, and a
# cut at a comma leaves "I think, you know" trailing where the claim should be.
#
# Its own list rather than titler's or thought's, for the reason thought.py
# gives about its own: this one strips, from both ends, repeatedly, and needs
# the multi-word phrases whole - "you know" is filler and "know" is not.
# Longest first, so the alternation cannot match "so" inside "sort of".
_HEAD_FILLER = (
    "oh my gosh", "oh my god", "you know", "i mean", "i guess", "kind of",
    "sort of", "anyway", "actually", "basically", "okay", "yeah", "right",
    "well", "then", "just", "like", "now", "and", "but", "yes", "so", "or",
    "ok", "um", "uh", "oh",
)

# A shorter list at the end of a line, because most of the words above can be
# what a sentence is *about* when they land last: stripping the tail of "That
# is what I like" leaves "That is what I", and stripping "You have the right"
# leaves nonsense. These cannot be anything but filler wherever they appear.
_TAIL_FILLER = (
    "you know what i mean", "you know", "i mean", "i guess", "or something",
    "kind of", "sort of", "anyway", "um", "uh",
)

_FILLER_HEAD = re.compile(r"^(?:\W*\b(?:" + "|".join(_HEAD_FILLER) + r")\b\W*)+", re.I)
_FILLER_TAIL = re.compile(r"(?:\W*\b(?:" + "|".join(_TAIL_FILLER) + r")\b\W*)+$", re.I)

# Words that mark a line as a claim rather than an observation. Its own short
# list rather than titler's, because titler's are private to that module and a
# card is scored on extremity alone, where a title is also scored on how much
# of the clip's subject it carries.
_STAKES = frozenset(
    """
    never always nobody everyone everything nothing worst best biggest only
    first last impossible insane crazy wrong terrified shocked hate love
    """.split()
)

# A concrete quantity. Digits and currency only, exactly as thought.py scores
# them: digits survive translation and the spelled-out magnitudes do not.
_NUMBER = re.compile(r"[\d$\u00a3\u20ac]")

# A capital that is not the sentence's first, which in practice is a name, and
# a name gives the claim someone to belong to.
_INNER_CAPITAL = re.compile(r"(?<!^)\b[A-Z][a-z]{2,}")


def _tokens(text: str) -> list[str]:
    return text.split()


def _strip_filler(text: str, english: bool) -> str:
    """Drop filler off both ends of a line, as much of it as is there.

    English only. The list is English, so running it over a Urdu or Chinese
    transcript could only take a real word off a real sentence.
    """
    if not english:
        return text

    return _FILLER_TAIL.sub("", _FILLER_HEAD.sub("", text)).strip()


def _tidy(text: str) -> str:
    """Turn the chosen words back into a line worth reading.

    A full stop is dropped because the card is a caption, not prose; a
    question mark is kept because it *is* the hook.
    """
    text = re.sub(r"\s+", " ", text).strip(" ,;:-\u2014")

    if text.endswith("."):
        text = text[:-1]

    if not text:
        return ""

    return text[0].upper() + text[1:]


def _content_words(text: str) -> set[str]:
    return {token.strip(".,!?;:'\"").lower() for token in _tokens(text)}


def _is_question(text: str) -> bool:
    # The Arabic mark is included because Whisper emits it for Urdu and Arabic.
    return text.rstrip().endswith(("?", "\u061f"))


def _says_something(text: str, english: bool) -> bool:
    """Whether the line tells a stranger anything at all.

    This is the floor, and it is the single most important rule here. Without
    it the best-scoring line on four real clips was "You had an umbrella?" - a
    real sentence, a real question, and completely empty to somebody who has
    just arrived. Scoring alone cannot fix that, because on a clip where every
    line is empty something still has to come top; a floor lets the answer be
    "no card", which is the correct one.

    A figure, a name, or a word carrying stakes is what separated the lines
    worth showing from the ones that merely parsed. A long question passes on
    its own, because a question is the one shape that states what the clip
    is *for* without needing any of those - but only a long one. Short
    conversational questions are the exact failure above.

    Its limits are worth stating: this cannot tell "How do you know when
    you're actually in love?" from "You have to make it sound like that, huh?"
    Both are long questions and only one is worth showing. Nothing available
    here without a model can, so the occasional dull card gets through. It is
    always a real line from the clip, which bounds how wrong it can be.
    """
    if _NUMBER.search(text) or _INNER_CAPITAL.search(text):
        return True

    if _is_question(text) and len(_tokens(text)) >= _MIN_QUESTION_WORDS:
        return True

    return english and bool(_content_words(text) & _STAKES)


def _score(text: str, progress: float, english: bool) -> float:
    """Rate how well a line would work as the card.

    Broadly the signals thought.score_opening ranks an opening by, because the
    question they answer is the same one: does a stranger who hears nothing
    else understand what this is about.

    A figure outscores a question mark here, where thought.py has it the other
    way round. thought.py is choosing which sentence the viewer *hears* first,
    and a question asked aloud pulls them through the next line. The card is
    read at a glance and gone, so the specific thing beats the open one:
    "They want to put $465,000 up to a coin flip" against "You have to make it
    sound like that, huh?" - both real lines from the same clip.
    """
    score = 0.0

    if _is_question(text):
        score += 2.0

    if _NUMBER.search(text):
        score += 2.5

    if _INNER_CAPITAL.search(text):
        score += 1.5

    if english:
        score += 1.8 * min(2, len(_content_words(text) & _STAKES))

    # A line the speaker reaches forty seconds in is a promise the viewer has
    # to wait forty seconds for, and for those forty seconds the card disagrees
    # with the audio under it. Mild rather than absolute: a genuinely stronger
    # claim later still wins, which is the text-only version of re-cutting so
    # the payoff comes first.
    score += 1.5 * (1.0 - progress)

    return score


def choose(words: list[Word], language: str | None = None) -> str | None:
    """The line to put on the card, or None if the clip has no good one."""
    if not words:
        return None

    limit = config.HOOK_MAX_WORDS
    english = titler.is_english(language)

    span_s = max(1e-6, words[-1].end_s - words[0].start_s)
    first_s = words[0].start_s

    ranked: list[tuple[float, str]] = []

    for sentence in titler.build_sentences(words):
        text = _tidy(_strip_filler(sentence.text, english))
        length = len(_tokens(text))

        # Whole sentences only. Cutting a long one back to its first clause was
        # built and then taken out again: measured over four real transcripts
        # it supplied one candidate out of eleven, and every way it could go
        # wrong went wrong loudly. "So it costs four hundred and fifty six
        # thousand dollars" cut at "and" reads "It costs four hundred", which
        # is not a fragment but a wrong figure; "It wasn't the money, it was
        # the betrayal" cut at the comma says the opposite of the sentence it
        # came from. Filler stripping already recovers most long-looking lines,
        # since a good deal of their length is "Oh my gosh" on the front.
        if not _MIN_CARD_WORDS <= length <= limit:
            continue

        # No card at all beats an empty one, so this is a floor rather than a
        # penalty: a clip where nothing clears it gets nothing.
        if not _says_something(text, english):
            continue

        progress = min(1.0, max(0.0, (sentence.start_s - first_s) / span_s))
        ranked.append((_score(text, progress, english), text))

    if not ranked:
        return None

    return max(ranked, key=lambda entry: entry[0])[1]
