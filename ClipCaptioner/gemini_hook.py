"""Choose which sentence a clip opens on, with Gemini.

The heuristic in thought.py can only measure a sentence's *shape*: is it a
question, does it carry a number, does it open on filler. That is enough to
throw out the worst openings and not enough to find the best one. Whether a
line stops someone scrolling is a fact about what it means, and shape is a
poor proxy for meaning.

So the model is asked instead, and given the same choices the heuristic had -
it picks one of them rather than writing anything, which is why nothing here
needs the grounding check gemini_titler.py performs. An answer that is not one
of the offered numbers is simply not an answer, and the caller falls back.

Entirely optional. Without a key, without the package, or on any API failure,
the caller falls back to the heuristic.
"""

from __future__ import annotations

import os
import re

import config

_ENV_KEY = "GEMINI_API_KEY"

_PROMPT = """You decide where a short vertical video clip should start.

Below are the places it could start, numbered. Each shows the sentence the clip
would open on, followed by a little of what comes after it.

Pick the one whose opening sentence would best stop someone scrolling past: a
claim, a question, a stake, a surprising fact, a number, a name. Reject an
opening that begins mid-thought, that begins on filler ("yeah", "so", "and",
"you know"), or that answers a question the viewer never heard.

Option 1 is where the clip already starts. Every later option throws away
everything before it, so choose one only if its opening is clearly stronger.

Reply with the number alone and nothing else.

{options}"""


def available() -> bool:
    """Whether a Gemini-chosen opening can even be attempted."""
    if not config.USE_GEMINI_HOOKS or not os.environ.get(_ENV_KEY):
        return False
    try:
        from google import genai  # noqa: F401
    except ImportError:
        return False
    return True


def _format_options(options: list[str]) -> str:
    return "\n\n".join(f"{number}. {text}" for number, text in enumerate(options, start=1))


def _parse_choice(reply: str, count: int) -> int | None:
    """The offered number the model picked, as a 0-based index.

    The first number in the reply, because a model told to answer with a
    number alone still sometimes explains itself first. Anything outside the
    offered range is treated as no answer rather than clamped - a model that
    invents an option has not chosen one of ours.
    """
    match = re.search(r"\d+", reply)
    if not match:
        return None

    choice = int(match.group(0))
    if not 1 <= choice <= count:
        return None

    return choice - 1


def choose_index(options: list[str]) -> int | None:
    """Ask Gemini which opening to use, or None to leave it to the heuristic.

    Takes and returns plain positions rather than the caller's own type, so
    this module stays a leaf and thought.py can import it without a cycle.
    """
    if not available() or len(options) < 2:
        return None

    try:
        from google import genai

        client = genai.Client(api_key=os.environ[_ENV_KEY])
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=_PROMPT.format(options=_format_options(options)),
        )
        reply = (getattr(response, "text", "") or "").strip()
    except Exception as exc:
        # Never lose a clip over an opening. The heuristic still works.
        print(f"    Gemini opening unavailable ({type(exc).__name__}) - scoring them here")
        return None

    return _parse_choice(reply, len(options))
