"""Pick a title for a clip out of its own transcript.

No model is involved. Sentences are rebuilt from the word stream and scored
against a handful of properties that reliably separate a hook from filler:
questions, superlatives, numbers and money, emotional weight, a title-shaped
length, and closeness to the moment the clip was cut around.

The winning sentence is the clip's own words, so the title always matches what
is actually said.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import config
from models import Word

# Words that mark a claim worth clicking on. Curated rather than learned:
# these are the terms that carry stakes, extremity or novelty.
_INTENSIFIERS = frozenset(
    """
    never ever always most best worst biggest smallest hardest easiest first
    last only greatest craziest insane crazy wild unbelievable impossible
    ridiculous incredible literally actually seriously honestly nobody
    everyone everything nothing anyone somebody
    """.split()
)

_EMOTION = frozenset(
    """
    love hate scared terrified afraid shocked stunned amazed surprised angry
    furious excited nervous embarrassed proud regret painful terrible awful
    amazing awesome perfect brutal savage
    """.split()
)

_QUESTION_OPENERS = frozenset("why how what who when where which".split())

# Openers that carry no information and read badly as the first word.
_LEADING_FILLER = frozenset(
    """
    so and but or like well okay ok yeah yes no um uh oh now then just
    basically anyway right actually i'm
    """.split()
)

_FILLER = frozenset(
    """
    um uh er ah like yeah yep nope okay ok right well just really very
    """.split()
)

_MONEY_OR_NUMBER = re.compile(r"[\$£€]|\d|\b(million|billion|thousand|hundred|percent)\b", re.I)

_SENTENCE_END = (".", "!", "?")

# Titles that run long get cut off by YouTube on mobile.
IDEAL_MIN_WORDS = 4
IDEAL_MAX_WORDS = 12


@dataclass(frozen=True)
class Sentence:
    """A rebuilt sentence with the time range it was spoken over."""

    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words).strip()

    @property
    def start_s(self) -> float:
        return self.words[0].start_s

    @property
    def end_s(self) -> float:
        return self.words[-1].end_s


def build_sentences(words: list[Word], gap_s: float = 0.6) -> list[Sentence]:
    """Rebuild sentences from the word stream.

    Whisper punctuates, but not always, so a long pause also ends a sentence.
    """
    sentences: list[Sentence] = []
    current: list[Word] = []

    for index, word in enumerate(words):
        current.append(word)

        ends_here = word.text.endswith(_SENTENCE_END)
        if not ends_here and index + 1 < len(words):
            ends_here = words[index + 1].start_s - word.end_s >= gap_s

        if ends_here:
            sentences.append(Sentence(words=current))
            current = []

    if current:
        sentences.append(Sentence(words=current))

    return sentences


def _normalise(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def score_sentence(sentence: Sentence, midpoint_s: float, span_s: float) -> float:
    """Rate how well a sentence would work as a title."""
    text = sentence.text
    tokens = _normalise(text)
    if not tokens:
        return 0.0

    score = 0.0

    # A question is the strongest hook there is.
    if text.rstrip().endswith("?"):
        score += 3.0
    if tokens[0] in _QUESTION_OPENERS:
        score += 1.5

    # Stakes and extremity.
    score += 1.8 * min(3, sum(1 for token in tokens if token in _INTENSIFIERS))
    score += 1.4 * min(2, sum(1 for token in tokens if token in _EMOTION))

    # Concrete numbers read as specific, which is persuasive.
    if _MONEY_OR_NUMBER.search(text):
        score += 2.0

    # Title-shaped length.
    count = len(tokens)
    if IDEAL_MIN_WORDS <= count <= IDEAL_MAX_WORDS:
        score += 2.0
    elif count < IDEAL_MIN_WORDS:
        score -= 2.5 * (IDEAL_MIN_WORDS - count)
    else:
        score -= 0.35 * (count - IDEAL_MAX_WORDS)

    # Filler drags a line down in proportion to how much of it there is.
    filler = sum(1 for token in tokens if token in _FILLER)
    score -= 2.2 * (filler / count)

    # The clip is cut around its peak, so the middle is the moment itself.
    if span_s > 0:
        centre = (sentence.start_s + sentence.end_s) / 2.0
        closeness = 1.0 - min(1.0, abs(centre - midpoint_s) / (span_s / 2.0))
        score += 1.5 * closeness

    return score


def _tidy(text: str) -> str:
    """Clean a spoken sentence into something that reads as a title."""
    text = re.sub(r"\s+", " ", text).strip()

    # Drop leading filler words, which are common in speech and weak in print.
    while True:
        match = re.match(r"^([A-Za-z']+)[,\s]+(.*)$", text)
        if not match or match.group(1).lower() not in _LEADING_FILLER:
            break
        remainder = match.group(2).strip()
        if not remainder:
            break
        text = remainder

    text = text.strip(" ,;:-—")

    # Keep a question mark, drop a trailing full stop.
    if text.endswith("."):
        text = text[:-1]

    if not text:
        return text

    if len(text) > config.TITLE_MAX_CHARS:
        cut = text[: config.TITLE_MAX_CHARS].rsplit(" ", 1)[0]
        text = cut.rstrip(" ,;:-—") + "..."

    return text[0].upper() + text[1:]


def make_title(words: list[Word], fallback: str) -> str:
    """Choose the most title-worthy line the clip actually contains."""
    if not words:
        return fallback

    sentences = [s for s in build_sentences(words) if s.text.strip()]
    if not sentences:
        return fallback

    span_s = words[-1].end_s - words[0].start_s
    midpoint_s = words[0].start_s + span_s / 2.0

    ranked = sorted(
        sentences,
        key=lambda s: -score_sentence(s, midpoint_s, span_s),
    )

    for sentence in ranked:
        title = _tidy(sentence.text)
        if len(_normalise(title)) >= IDEAL_MIN_WORDS:
            return title

    return fallback
