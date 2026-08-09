"""Write a clip title with Gemini, grounded in what was actually said.

The heuristic in titler.py can only pick a sentence the clip already contains.
That is honest but blunt: it can quote the moment, never describe it. A model
can say what a clip is *about*, which is usually the better title.

The tradeoff is that a model can also invent. So anything it returns is checked
back against the transcript and thrown away if it drifts - the guarantee that a
title reflects the clip is kept by verification rather than by trust.

Entirely optional. Without a key, without the package, or on any API failure,
the caller falls back to the heuristic.
"""

from __future__ import annotations

import os
import re

import config

_ENV_KEY = "GEMINI_API_KEY"

_PROMPT = """You write titles for short vertical video clips cut from a longer video.

Below is the transcript of one clip. Write a single YouTube Shorts title for it.

Rules:
- Use only what the transcript actually says. Never add a fact, name, number or
  claim that is not in it.
- Say what the moment is about. Do not simply copy a sentence back.
- Keep it under {max_chars} characters.
- No quotation marks, no hashtags, no emoji, no trailing full stop.
- Write it in the same language as the transcript.
- Reply with the title alone and nothing else.

Transcript:
{transcript}"""


def available() -> bool:
    """Whether a Gemini title can even be attempted."""
    if not config.USE_GEMINI_TITLES or not os.environ.get(_ENV_KEY):
        return False
    try:
        from google import genai  # noqa: F401
    except ImportError:
        return False
    return True


def _words(text: str) -> set[str]:
    """Meaningful words, lowercased. Short ones carry no grounding signal."""
    return {w for w in re.findall(r"[^\W\d_]+", text.lower()) if len(w) > 3}


def grounding(title: str, transcript: str) -> float:
    """Share of the title's meaningful words that appear in the transcript.

    A cheap check, and enough to catch the failure that matters: a title about
    something the clip never mentions. Short words are ignored because they
    match by coincidence and would flatter a bad title.
    """
    title_words = _words(title)
    if not title_words:
        return 0.0
    return len(title_words & _words(transcript)) / len(title_words)


def _clean(text: str) -> str:
    """Strip the wrapping a model adds despite being told not to.

    Order matters: the lead-in comes off first, because a quote usually sits
    *inside* it (`Title: "..."`), and stripping quotes first would leave the
    opening one stranded once the lead-in went.
    """
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(title|here['’]?s (a |the )?title)\s*[:\-]\s*", "", text, flags=re.I)
    text = text.strip().strip('"').strip("'").strip()
    return text.rstrip(".").strip()


def make_title(transcript: str) -> str | None:
    """Ask Gemini for a title, or None if it cannot be trusted or obtained."""
    if not available() or not transcript.strip():
        return None

    try:
        from google import genai

        client = genai.Client(api_key=os.environ[_ENV_KEY])
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=_PROMPT.format(
                max_chars=config.TITLE_MAX_CHARS,
                transcript=transcript[:6000],
            ),
        )
        title = _clean(getattr(response, "text", "") or "")
    except Exception as exc:
        # Never lose a clip over a title. The heuristic still works.
        print(f"    Gemini title unavailable ({type(exc).__name__}) - using the transcript")
        return None

    if not title:
        return None

    if len(title) > config.TITLE_MAX_CHARS:
        title = title[: config.TITLE_MAX_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:-")

    score = grounding(title, transcript)
    if score < config.GEMINI_MIN_GROUNDING:
        print(f"    Gemini title discarded - only {score:.0%} grounded in the clip")
        return None

    return title
