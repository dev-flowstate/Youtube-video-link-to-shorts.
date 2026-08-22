"""Plan short cutaways to stock footage when the speaker names something real.

The idea is the one every good editor uses: when someone says "the sea", the
sea is on screen for a moment. What makes it work is restraint. Footage that
merely illustrates the word - a stressed actor over the word "stress", a brain
animation over "brain" - is the thing that makes a clip look automated, so this
module is deliberately hard to please. It cuts away only for subjects a camera
could actually be pointed at, only a couple of times a minute, and never over
the hook.

Nothing here renders. `plan()` returns a list of `Cutaway`s and the caller
decides what to do with them. An empty plan is the normal outcome for an
abstract clip, not a failure, so every path that cannot produce a cutaway
returns fewer of them rather than raising.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import config
import stock
import titler
from models import CaptionGroup, Word, join_word_texts

# ---------------------------------------------------------------------------
# Tuning. These belong in config.py; they live here only until the workers
# editing that file are clear, at which point they should move across intact.
# ---------------------------------------------------------------------------

# How long a cutaway holds. Long enough to read as a deliberate cut and not a
# glitch, short enough that the speaker is back before the viewer wonders where
# they went. Anything past about two seconds stops being a cutaway and starts
# being a scene the viewer did not ask for.
# Retuned together after watching a finished batch: "there wasnt enough stock
# video for people to visualise and the book stock video was for too long".
#
# Both complaints were the same cap. A 40s clip allowed
# min(MAX_CUTAWAYS, 40/60 * MAX_CUTAWAYS_PER_MINUTE) = 2 cutaways, each 2.5s -
# too few to register as a pattern, and each lingering long enough to notice.
#
# The duration has now moved twice, and the first move was my mistake. It began
# at 1.5s, and when cutaways seemed invisible I raised it to 2.5 - treating a
# frequency problem as a length problem. Fixing the frequency is what lets the
# length come back down, so it is near where it started.
#
# A 40s clip now allows 5 cutaways of 1.6s: 8 seconds of footage, a fifth of
# the runtime. Short enough to read as a punctuation mark rather than a cutaway
# the clip has to recover from, frequent enough to be a pattern the viewer
# notices. Not below about 1.5s, where a shot reads as a glitch rather than an
# illustration.
CUTAWAY_DURATION_S = 1.6

# A cutaway shortened by the tail guard below this is dropped instead. Under a
# second reads as a flicker.
MIN_CUTAWAY_DURATION_S = 1.0

# Nothing cuts away inside the opening seconds. The first frames are where a
# viewer decides whether to keep watching, and that decision is made on the
# speaker's face and their first sentence. Replacing either with stock footage
# reads as an advert, and on a Short - which loops - it is also the frame that
# follows the last one.
HOOK_GUARD_S = 5.0

# Nor in the closing seconds, so the clip ends on the person talking.
TAIL_GUARD_S = 1.5

# Minimum gap between two cutaways. At six seconds and 1.6s shots the worst
# case is footage on screen a quarter of the time, with each cut still having
# several seconds to be forgotten before the next.
#
# The worst case does not happen. Measured across twelve real clips the highest
# share was 10.1%, because qualifying moments are far rarer than this spacing
# allows - see MAX_CUTAWAYS_PER_MINUTE below. This is a guard against a
# slideshow, not the thing deciding how often a cutaway appears.
MIN_SPACING_S = 6.0

# Not the brake, despite being written as one. A 45s clip may now take six
# cutaways and a 55s clip seven, but measured across twelve real clips the
# counts came out 0 to 4, and raising this from 4.0 produced exactly one extra
# cutaway in the whole sample.
#
# What actually decides how many appear is how few spoken words are in
# _CONCRETE_NOUNS: one clip of 173 words matched twice. So this is a ceiling
# that no longer binds, kept only to stop a dense clip becoming a montage. The
# next change that wants more cutaways has to widen what counts as a subject,
# not raise this number again.
MAX_CUTAWAYS_PER_MINUTE = 8.0

# Regardless of length. Guards a long input from turning into a montage.
MAX_CUTAWAYS = 8

# How many candidates get as far as being considered. Sized above the cap so a
# moment that resolves to no footage can be backfilled by the next one down.
#
# This was six because six was what fitted in a single Gemini request. That
# request no longer happens - USE_GEMINI_SEARCH_TERMS is off - so six had
# stopped being a batch size and become a silent ceiling on cutaways. Measured
# on a Steve Jobs clip naming six distinct subjects, it bound exactly.
SHORTLIST_SIZE = 10

# Ask Gemini to turn each moment into a search phrase. Costs one request per
# clip. Without a key, without the package, or on any failure, the concrete
# noun itself is used instead, so this is safe to leave on.
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
USE_GEMINI_SEARCH_TERMS = False

# How often a noun must be said before the *fallback* will cut away for it.
# Only applies when the model did not answer for that moment.
#
# Measured on the stress podcast: the model rejects "at the top of a hill" and
# "Calm Coast" as a metaphor and a proper noun, and the fallback cannot - it
# sees two concrete nouns and fetches a hillside and a coastline. Repetition is
# the one signal left that a thing is genuinely the subject rather than a turn
# of phrase, so without a model to ask, a noun said once is not enough. It
# costs some real cutaways and removes most of the embarrassing ones.
FALLBACK_MIN_MENTIONS = 2

_ENV_KEY = "GEMINI_API_KEY"

# ---------------------------------------------------------------------------
# What earns a cutaway
# ---------------------------------------------------------------------------

# The rule: cut away only for a noun naming something physical that a camera
# could be pointed at, and that a viewer would recognise in one second without
# a caption.
#
# That rules out abstractions by construction, which is the point. Stock
# footage of "stress" is a paid actor holding their head, and footage of
# "energy" is a stranger jogging - both illustrate the word while saying
# nothing about the sentence, and both are why automated b-roll is easy to
# spot. It also rules out subjects that are technically physical but arrive as
# medical-stock cliche: brain, mind, heart, breathing and sleep are all left
# out on purpose, which matters here because a podcast about stress says
# "brain" constantly and would otherwise cut away to the same rotating scan
# every twelve seconds.
#
# An allowlist rather than a part-of-speech tagger because the costs are not
# symmetric. A missed cutaway is invisible; a bad one is the first thing a
# viewer notices. A list can only ever fail by staying quiet.
#
# Words that are usually figurative in conversation are left out for the same
# reason - "space", "field", "watch", "running", "driving", "climbing". Gemini
# rejects those readings when it is asked, but the fallback below cannot, and
# "create the space and the time" fetching a starfield is exactly the kind of
# cutaway this module exists to avoid. Anything ambiguous enough to need the
# model to save it does not belong in the list.
_CONCRETE_NOUNS = frozenset(
    """
    ocean sea beach wave shore coast island tide sand reef coral
    forest tree woods jungle leaf branch grass meadow garden flower palm pine
    mountain hill cliff valley canyon desert dune cave glacier iceberg rock stone
    river lake stream waterfall rain snow storm lightning thunder ice puddle
    sky cloud sun sunset sunrise moon star rainbow fog
    fire flame smoke volcano ash
    dog cat puppy kitten horse bird fish shark whale dolphin lion tiger bear wolf
    elephant snake butterfly bee spider cow sheep goat pig chicken duck eagle owl
    monkey rabbit deer fox mouse frog turtle octopus penguin ant
    city street road highway traffic bridge building skyscraper tower castle
    statue fountain bench fence roof wall alley rooftop
    office kitchen bedroom bathroom hospital school classroom gym restaurant
    cafe airport station library factory farm stadium church supermarket
    home house apartment hotel museum park store shop market bar prison
    car truck bus train plane airplane boat ship bike bicycle motorcycle
    rocket helicopter subway tram taxi ambulance tractor skateboard scooter
    kayak canoe sail
    phone laptop computer screen keyboard camera television clock robot drone
    satellite printer headphones microphone
    battery charger cable
    book newspaper letter map guitar piano drum candle mirror window door
    stairs desk chair table bed pillow blanket ladder rope hammer knife
    scissors brush pen pencil paper envelope box basket bottle glass cup
    plate spoon fork lock suitcase backpack shoes jacket glasses umbrella
    coffee tea water bread pizza food fruit apple cake rice pasta burger
    sandwich salad soup egg cheese milk chocolate banana orange lemon honey
    money cash coin wallet gold diamond
    ball football basketball tennis soccer baseball golf boxing yoga skiing
    surfboard
    baby kid child children crowd family friend team audience doctor nurse
    teacher student athlete runner chef farmer
    fireworks balloon flag crown trophy medal gift wedding party concert
    festival parade
    walking swimming hiking dancing cooking surfing sleeping reading writing
    heart brain blood bone muscle smile tear
    surgery medicine pill syringe wheelchair bandage mask stethoscope
    college university graduation diploma campus lecture blackboard notebook
    exam globe microscope telescope laboratory
    work worker meeting whiteboard presentation handshake interview
    briefcase elevator warehouse construction crane
    garage workshop toolbox wrench screwdriver drill
    typewriter calendar diary journal stamp postcard poster billboard
    sculpture painting canvas gallery theatre stage curtain spotlight
    violin trumpet saxophone flute vinyl radio speaker
    compass anchor lighthouse harbour pier dock net
    tent campfire trail tunnel railway platform runway
    watch hourglass bucket broom lamp lantern torch
    seed soil harvest wheat corn vineyard orchard greenhouse barn
    bakery barber mechanic
    frost wind hurricane tornado flood earthquake
    """.split()
)

# Words from the list above that arrive figuratively as often as literally.
# These alone still need saying twice, because a single mention is more likely
# to be a turn of phrase than a subject: "weather the storm", "opens doors",
# "rock bottom", "a wave of change", "the key to success". Everything else
# fires on one mention, which is what makes the feature visible at all - at two
# mentions for every word it fired once across nine clips.
#
# The second group was found by testing rather than predicted: "baby" is an
# exclamation at least as often as a noun, and "Let's get it, baby" pulled
# footage of an infant. The same goes for man, boy, girl, dog and beast in
# excited speech, and for gold and bomb as praise.
_FIGURATIVE_NOUNS = frozenset(
    """
    storm fire wave rock stone door window key ladder bridge chain
    light star cloud water road path
    baby kid man boy girl dog beast bomb gold ball
    work seed harvest anchor compass wind blood brain net mask stage
    """.split()
)

# Where the bare word searches badly. "space" returns storage units, "battery"
# returns AA cells, "watch" returns people watching. Everything not listed here
# is searched for as it was said.
_SEARCH_OVERRIDES = {
    "heart": "heartbeat pulse monitor",
    "work": "person working at desk",
    "net": "fishing net",
    "stage": "empty theatre stage",
    "mask": "face mask",
    "speaker": "concert loudspeaker",
    "vinyl": "vinyl record player",
    "globe": "spinning globe",
    "trail": "hiking trail",
    "platform": "train platform",
    "watch": "wristwatch on wrist",
    "battery": "phone battery charging",
    "charger": "phone charging cable",
    "kid": "children playing",
    "kids": "children playing",
    "child": "children playing",
    "children": "children playing",
    "phone": "smartphone in hand",
    "screen": "glowing computer screen",
    "sea": "ocean waves",
    "star": "night sky stars",
    "wave": "ocean waves",
    "water": "flowing water",
    "food": "cooking food",
    "money": "cash money",
    "tennis": "playing tennis",
}

# A phrase longer than this is a sentence, not a search query.
_MAX_TERM_WORDS = 4

_PROMPT = """You choose stock footage for a short vertical video.

Below is the transcript of one clip, then a numbered list of moments in it
where the speaker named something that might be worth cutting away to for
about a second.

For each moment, reply with a stock-footage search phrase of two to four plain
words describing what should be on screen. Reply "skip" instead whenever a
cutaway there would add nothing.

Skip a moment when:
- the word is a dead figure of speech - "the top of a hill", "power through" -
  where no picture is meant at all
- the footage would only illustrate the word rather than the point being made
- what appears would be a person acting out a feeling
- the thing is mentioned only in passing and the talk moves straight on

Keep a moment when the speaker is deliberately painting a picture, even a
figurative one. If they say a brain is like a phone battery and keep returning
to it, a draining phone battery is exactly what should be on screen.

Rules:
- One line per moment, written as "<number>: <phrase or skip>".
- Plain English a stock library would match. No quotes, no punctuation, no
  explanation, nothing but the phrase.
- Describe a scene a camera could film, never a concept.

Transcript:
{transcript}

Moments:
{moments}"""


@dataclass(frozen=True)
class Cutaway:
    """One planned cut to stock footage."""

    start_s: float
    duration_s: float
    topic: str
    clip: Path | None  # resolved footage, or None if none could be had

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


@dataclass(frozen=True)
class _Candidate:
    """A moment that might deserve a cutaway, before anything is decided."""

    word: Word
    noun: str
    mentions: int
    context: str


def _normalise(text: str) -> str:
    """A word reduced to comparable letters."""
    return re.sub(r"[^a-z]", "", text.lower())


def _stems(token: str) -> list[str]:
    """The token and the singulars it might be a plural of.

    Crude on purpose. Full lemmatisation would need a dependency for a gain of
    a handful of words, and a wrong stem here costs a missed cutaway, not a bad
    one.
    """
    forms = [token]
    if token.endswith("ies") and len(token) > 4:
        forms.append(token[:-3] + "y")
    if token.endswith("es") and len(token) > 4:
        forms.append(token[:-2])
    if token.endswith("s") and len(token) > 3:
        forms.append(token[:-1])
    return forms


def _match(word: Word) -> str | None:
    """The concrete noun this word names, if it names one."""
    token = _normalise(word.text)
    if len(token) < 3:
        return None
    for form in _stems(token):
        if form in _CONCRETE_NOUNS:
            return form
    return None


def _context(words: list[Word], index: int, span: int = 8) -> str:
    """The words either side of a match, for the model to judge it in."""
    start = max(0, index - span)
    end = min(len(words), index + span + 1)
    return join_word_texts([w.text for w in words[start:end]])


def _candidates(groups: list[CaptionGroup]) -> list[_Candidate]:
    """Every moment a concrete noun is spoken, with how often the clip says it."""
    words = [word for group in groups for word in group.words]

    hits: list[tuple[int, Word, str]] = []
    for index, word in enumerate(words):
        noun = _match(word)
        if noun:
            hits.append((index, word, noun))

    counts: dict[str, int] = {}
    for _, _, noun in hits:
        counts[noun] = counts.get(noun, 0) + 1

    return [
        _Candidate(
            word=word,
            noun=noun,
            mentions=counts[noun],
            context=_context(words, index),
        )
        for index, word, noun in hits
    ]


def _cap(clip_length_s: float) -> int:
    return min(MAX_CUTAWAYS, int(clip_length_s / 60.0 * MAX_CUTAWAYS_PER_MINUTE))


def _window(start_s: float, clip_length_s: float) -> float | None:
    """How long a cutaway starting here may run, or None if it may not."""
    if start_s < HOOK_GUARD_S:
        return None
    available = (clip_length_s - TAIL_GUARD_S) - start_s
    duration = min(CUTAWAY_DURATION_S, available)
    if duration < MIN_CUTAWAY_DURATION_S:
        return None
    return duration


def _gemini_available() -> bool:
    if not USE_GEMINI_SEARCH_TERMS or not os.environ.get(_ENV_KEY):
        return False
    try:
        from google import genai  # noqa: F401
    except ImportError:
        return False
    return True


def _clean_term(text: str) -> str | None:
    """A model's line reduced to a usable search phrase, or None to skip it.

    No grounding check of the kind gemini_titler.py runs, because the whole
    point of asking is that the phrase may reinterpret: "my kids' battery
    levels" should become "children playing", which shares no words with what
    was said. Nothing the model returns is ever shown to a viewer - it is only
    ever a search query - so the worst a bad phrase can do is fetch footage
    that does not match, and the shape checks below are enough to catch a reply
    that is prose rather than a query.
    """
    text = re.sub(r"\s+", " ", text).strip().strip("\"'").strip()
    text = re.sub(r"[^A-Za-z \-]", "", text).strip().lower()
    if not text or text in {"skip", "none", "no", "nothing"}:
        return None
    if len(text.split()) > _MAX_TERM_WORDS:
        return None
    return text


def _gemini_terms(transcript: str, shortlist: list[_Candidate]) -> dict[int, str | None]:
    """One request for the whole clip: a search phrase per shortlisted moment.

    Returns only the moments the model actually answered for. A moment it
    skipped maps to None, which drops it; a moment it did not mention at all is
    absent, and the caller falls back to the noun.
    """
    if not _gemini_available() or not shortlist:
        return {}

    moments = "\n".join(
        f'{number}. "{candidate.noun}" - {candidate.context}'
        for number, candidate in enumerate(shortlist, start=1)
    )

    try:
        from google import genai

        client = genai.Client(api_key=os.environ[_ENV_KEY])
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=_PROMPT.format(transcript=transcript[:6000], moments=moments),
        )
        reply = getattr(response, "text", "") or ""
    except Exception as exc:
        # Never lose a clip over b-roll. The nouns still search.
        print(f"    Gemini b-roll terms unavailable ({type(exc).__name__}) - using the nouns")
        return {}

    terms: dict[int, str | None] = {}
    for line in reply.splitlines():
        match = re.match(r"\s*(\d+)\s*[:.\)-]\s*(.+)", line)
        if not match:
            continue
        number = int(match.group(1))
        if 1 <= number <= len(shortlist):
            terms[number - 1] = _clean_term(match.group(2))

    return terms


def plan(
    groups: list[CaptionGroup],
    clip_length_s: float,
    language: str | None = None,
) -> list[Cutaway]:
    """Plan the cutaways for one clip, in the order they appear.

    `language` is optional so existing two-argument calls keep working; it
    exists because every noun above is English, and scoring a Urdu transcript
    against them would match nothing at best and the wrong thing at worst.

    Returns only cutaways that resolved to footage on disk, so a caller can
    render the list as it stands. An empty list is the ordinary result for a
    clip that never names anything worth seeing.
    """
    if not groups or clip_length_s <= 0:
        return []

    if not titler.is_english(language):
        return []

    candidates = _candidates(groups)
    candidates = [c for c in candidates if _window(c.word.start_s, clip_length_s) is not None]
    if not candidates:
        return []

    cap = _cap(clip_length_s)
    if cap <= 0:
        return []

    # A noun the clip keeps returning to is what the clip is about; a noun said
    # once in passing usually is not. Earlier wins ties, so a cutaway lands
    # while the subject is still live rather than after the speaker has moved
    # on to something else.
    ranked = sorted(candidates, key=lambda c: (-c.mentions, c.word.start_s))

    # One noun, one cutaway. Two clips of the sea in the same 45 seconds is
    # worse than one, however often the sea comes up.
    shortlist: list[_Candidate] = []
    seen: set[str] = set()
    for candidate in ranked:
        if candidate.noun in seen:
            continue
        seen.add(candidate.noun)
        shortlist.append(candidate)
        if len(shortlist) >= SHORTLIST_SIZE:
            break

    transcript = join_word_texts([w.text for g in groups for w in g.words])
    terms = _gemini_terms(transcript, shortlist)

    chosen: list[Cutaway] = []
    for index, candidate in enumerate(shortlist):
        if len(chosen) >= cap:
            break

        if index in terms:
            topic = terms[index]
            if topic is None:  # the model judged a cutaway here pointless
                continue
        else:
            # Nothing judged this moment, so it has to clear the bar alone.
            needed = (
                FALLBACK_MIN_MENTIONS
                if candidate.noun in _FIGURATIVE_NOUNS
                else 1
            )
            if candidate.mentions < needed:
                continue
            topic = _SEARCH_OVERRIDES.get(candidate.noun, candidate.noun)

        start_s = candidate.word.start_s
        if any(abs(start_s - c.start_s) < MIN_SPACING_S for c in chosen):
            continue

        duration_s = _window(start_s, clip_length_s)
        if duration_s is None:
            continue

        footage = stock.fetch(topic, want=1)
        if not footage:
            continue

        chosen.append(
            Cutaway(
                start_s=start_s,
                duration_s=duration_s,
                topic=topic,
                clip=footage[0],
            )
        )

    return sorted(chosen, key=lambda c: c.start_s)
