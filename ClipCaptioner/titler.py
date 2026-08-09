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
import gemini_titler
from models import Word, join_word_texts

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
        return join_word_texts([word.text for word in self.words]).strip()

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


def is_english(language: str | None) -> bool:
    """Whether the English term lists apply to this transcript.

    Every word list here is English. Scoring a Urdu or Chinese transcript
    against them matches nothing, so those clips would all tie at zero and
    silently fall back to the filename. Detecting that up front lets the
    language-neutral signals decide instead.
    """
    return language is None or language.lower().split("-")[0] == "en"


def score_sentence(
    sentence: Sentence,
    midpoint_s: float,
    span_s: float,
    language: str | None = None,
) -> float:
    """Rate how well a sentence would work as a title."""
    text = sentence.text
    tokens = _normalise(text)
    if not tokens:
        # Scripts with no Latin letters normalise to nothing, so fall back to
        # counting whitespace-separated words.
        tokens = text.split()
    if not tokens:
        return 0.0

    score = 0.0
    english = is_english(language)

    # A question is the strongest hook there is, and question marks are used
    # across scripts - Urdu's "؟" included, once normalised by Whisper.
    if text.rstrip().endswith(("?", "؟")):
        score += 3.0

    if english:
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
    if english:
        filler = sum(1 for token in tokens if token in _FILLER)
        score -= 2.2 * (filler / count)

    # The clip is cut around its peak, so the middle is the moment itself.
    if span_s > 0:
        centre = (sentence.start_s + sentence.end_s) / 2.0
        closeness = 1.0 - min(1.0, abs(centre - midpoint_s) / (span_s / 2.0))
        score += 1.5 * closeness

    return score


def _tidy(text: str, english: bool = True) -> str:
    """Clean a spoken sentence into something that reads as a title."""
    text = re.sub(r"\s+", " ", text).strip()

    # Drop leading filler words, which are common in speech and weak in print.
    # The list is English, so this is skipped for anything else rather than
    # stripping a word that only looks like filler.
    while english:
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


# Words carrying no topical meaning. Removing them is what turns raw word
# counts into something resembling keywords.
_STOPWORDS = frozenset(
    """
    a about after all also am an and any are as at be because been before
    being but by can cant come could did didnt do does doing dont down each
    even for from get gets getting go going got had has have having he her
    here hers him his how i if im in into is isnt it its just know let like
    ll me more most much my no not now of off on once one only or other our
    out over own re said same say says see she should so some such take than
    that thats the their them then there these they thing things think this
    those through to too up us very want was wasnt we well went were what
    when where which while who why will with would yeah yes you your youre
    """.split()
)


def keywords(words: list[Word], limit: int = 6) -> list[str]:
    """Most-repeated meaningful words, as topic hints.

    Frequency with stopwords removed. Crude, but repetition genuinely tracks
    what a clip is about, and it cannot hallucinate a topic the clip lacks.
    """
    counts: dict[str, int] = {}

    for word in words:
        for token in _normalise(word.text):
            # Contractions are grammar, never a topic - "we're" is not a tag.
            if "'" in token:
                continue
            if len(token) < 4 or token in _STOPWORDS:
                continue
            counts[token] = counts.get(token, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, count in ranked[:limit] if count > 1]


def build_hashtags(words: list[Word]) -> list[str]:
    """Hashtags for the description.

    Three is the practical ceiling - YouTube ignores them all past about
    fifteen, and a short, relevant set reads better than a wall of them.
    #Shorts is always first because it is what routes a vertical video into
    the Shorts feed.
    """
    tags = ["#Shorts"]
    for keyword in keywords(words, limit=4):
        if len(tags) >= 3:
            break
        tags.append("#" + keyword.capitalize())
    return tags


def build_description(title: str, words: list[Word], source_name: str) -> str:
    """Assemble the text to paste into YouTube's description box.

    The first line repeats the title because that is the part search actually
    weighs; the rest is context and tags.
    """
    lines = [title, ""]

    topics = keywords(words, limit=6)
    if topics:
        lines.append("Topics: " + ", ".join(topics))
        lines.append("")

    lines.append(f"Clipped from: {source_name}")
    lines.append("")
    lines.append(" ".join(build_hashtags(words)))
    return "\n".join(lines)


def make_title(words: list[Word], fallback: str, language: str | None = None) -> str:
    """Title the clip, preferring Gemini and falling back to its own words.

    Gemini can say what a moment is about; the heuristic below can only quote
    it. But the heuristic can never invent, so it stays as the fallback for
    when Gemini is unavailable or returns something the clip does not support.
    """
    if not words:
        return fallback

    generated = gemini_titler.make_title(join_word_texts([w.text for w in words]))
    if generated:
        return generated

    sentences = [s for s in build_sentences(words) if s.text.strip()]
    if not sentences:
        return fallback

    span_s = words[-1].end_s - words[0].start_s
    midpoint_s = words[0].start_s + span_s / 2.0

    ranked = sorted(
        sentences,
        key=lambda s: -score_sentence(s, midpoint_s, span_s, language),
    )

    english = is_english(language)
    for sentence in ranked:
        title = _tidy(sentence.text, english)
        # Non-Latin scripts have no letters for _normalise to find, so length
        # is counted in whitespace-separated words instead.
        length = len(_normalise(title)) if english else len(title.split())
        if length >= IDEAL_MIN_WORDS:
            return title

    return fallback
