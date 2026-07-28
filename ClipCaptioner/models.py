"""Data structures shared across the captioning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Word:
    """A single transcribed word with its spoken time range."""

    text: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def shifted(self, offset_s: float) -> Word:
        """Return a copy moved along the timeline by offset_s."""
        return Word(
            text=self.text,
            start_s=self.start_s + offset_s,
            end_s=self.end_s + offset_s,
        )


@dataclass
class CaptionGroup:
    """A short burst of words shown on screen together."""

    words: list[Word] = field(default_factory=list)

    @property
    def start_s(self) -> float:
        return self.words[0].start_s

    @property
    def end_s(self) -> float:
        return self.words[-1].end_s

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)

    def shifted(self, offset_s: float) -> CaptionGroup:
        return CaptionGroup(words=[word.shifted(offset_s) for word in self.words])


@dataclass(frozen=True)
class ClipPart:
    """One rendered slice of a source clip."""

    index: int
    total: int
    start_s: float
    end_s: float
    groups: list[CaptionGroup]

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass(frozen=True)
class VideoInfo:
    """Probed properties of a source video."""

    width: int
    height: int
    duration_s: float
