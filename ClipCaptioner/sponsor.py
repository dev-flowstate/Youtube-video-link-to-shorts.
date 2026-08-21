"""Spot a sponsor read, so a clip slot is not spent on an advert.

A podcast episode yields only a handful of clips, and the most-replayed graph
does not know an advert from an insight - listeners scrub across sponsor
segments, which registers as replay activity in exactly the same way. On a
recent episode two of four clips were sponsor reads: half the output, and the
half nobody watches.

The signals are unusually reliable because a sponsor read has to carry the
things that make it a sponsor read. It names a domain, spells it out for
listeners ("eightsleep dot com slash huberman"), offers a code or a discount,
and says who is paying. Ordinary conversation does none of that.

Judged over the whole clip rather than a single hit: one mention of a percentage
is nothing, while a domain plus a code plus a discount inside forty seconds is
not a coincidence.
"""

from __future__ import annotations

import re

from models import CaptionGroup

# A single hit means little; the score is what decides. Weights are rough, and
# deliberately favour the things only an advert says.
_PATTERNS: list[tuple[str, float, str]] = [
    (r"\bpromo\s?code\b", 3.0, "promo code"),
    (r"\bdiscount\s?code\b", 3.0, "discount code"),
    (r"\buse\s+code\b", 3.0, "use code"),
    (r"\bcoupon\b", 2.5, "coupon"),
    (r"\bsponsor(ed|ship|s)?\b", 2.5, "sponsor"),
    (r"\bthis\s+episode\s+is\s+brought\s+to\s+you\b", 3.0, "brought to you by"),
    (r"\bbrought\s+to\s+you\s+by\b", 3.0, "brought to you by"),
    (r"\bfree\s+shipping\b", 2.0, "free shipping"),
    (r"\bmoney[- ]back\s+guarantee\b", 2.0, "money-back guarantee"),
    (r"\bfree\s+trial\b", 2.0, "free trial"),
    # Spelled-out URLs are the giveaway. A host says "dot com slash huberman"
    # only when reading a link aloud for listeners who cannot click one.
    (r"\bdot\s?com\b", 2.5, "spoken URL"),
    (r"\.com\b", 2.0, "written URL"),
    (r"\bdot\s?com\s+slash\b", 3.0, "spoken URL with path"),
    (r"\bslash\s+\w+\b", 1.5, "spoken path"),
    (r"\b\d{1,2}\s?(percent|%)\s+off\b", 3.0, "percentage off"),
    (r"\bsave\s+up\s+to\b", 2.0, "save up to"),
    (r"\bsave\s+\$?\d", 2.0, "save an amount"),
    (r"\b\$\d+\s+off\b", 3.0, "amount off"),
    (r"\bgo\s+to\s+\w+\s?dot\s?com\b", 3.0, "go to a URL"),
    (r"\bcheck\s+(them|it)\s+out\s+at\b", 2.0, "check it out at"),
    (r"\blink\s+in\s+the\s+description\b", 2.5, "link in description"),
    (r"\bterms\s+and\s+conditions\b", 2.5, "terms and conditions"),
]

# Above this a clip is treated as an advert. Set from real reads: a genuine
# sponsor segment trips several patterns and lands well past it, while a
# conversation that happens to mention a website scores one hit and stays under.
SCORE_THRESHOLD = 5.0


def score(text: str) -> tuple[float, list[str]]:
    """How much this reads like an advert, and what gave it away."""
    lowered = text.lower()
    total = 0.0
    hits: list[str] = []

    for pattern, weight, label in _PATTERNS:
        if re.search(pattern, lowered):
            total += weight
            hits.append(label)

    return total, hits


def is_sponsor(groups: list[CaptionGroup]) -> tuple[bool, float, list[str]]:
    """Whether these captions are a sponsor read, with the evidence."""
    text = " ".join(word.text for group in groups for word in group.words)
    total, hits = score(text)
    return total >= SCORE_THRESHOLD, total, hits
