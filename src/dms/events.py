"""Timestamped events: the common output format of every module (INITIALPLAN §2.2, principle 5)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Event:
    type: str
    start: float  # seconds from the start of the recording
    end: float
    confidence: float = 1.0
    info: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)
