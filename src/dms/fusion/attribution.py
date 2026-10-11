"""Context-aware event attribution (#1, rule-based v1; INITIALPLAN §7.2).

All modules emit timestamped `Event`s. This merges them on one timeline and attaches a *likely
context* to each risky event from co-occurring signals. It is attribution from co-occurrence,
not causal inference; the rules are the baseline for a learned model (internship).

| Pattern                                                      | Context                       | Status           |
|--------------------------------------------------------------|-------------------------------|------------------|
| HARSH_BRAKE + EYES_OFF_ROAD / PHONE / TEXTING in previous 2 s | likely_distraction            | testable         |
| HARSH_BRAKE + HAZARD_AHEAD around it + attentive driver       | likely_hazard_response        | testable         |
| LANE_CHANGE without MIRROR_CHECK in previous 3 s              | no_mirror_check               | testable         |
| EYES_OFF_ROAD while vehicle stopped                           | suppressed (stationary)       | testable         |
| DROWSY / MICROSLEEP + LANE_DEPARTURE / ROAD_EDGE_DRIFT        | critical_fatigue              | deployment rule  |
| PHONE_USE / TEXTING while moving                              | phone_while_moving            | deployment rule  |
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dms.events import Event

DISTRACTION = {"EYES_OFF_ROAD", "PHONE_USE", "PHONE_CALL", "TEXTING", "EATING", "DRINKING"}
FATIGUE = {"DROWSY", "MICROSLEEP"}
LANE_DRIFT = {"LANE_DEPARTURE", "ROAD_EDGE_DRIFT"}


@dataclass
class Attributed:
    event: Event
    context: str
    severity: str  # info | warning | critical
    evidence: list[Event] = field(default_factory=list)
    status: str = "testable"  # testable | deployment_rule

    def to_dict(self) -> dict:
        return {"type": self.event.type, "start": self.event.start, "end": self.event.end,
                "context": self.context, "severity": self.severity, "status": self.status,
                "evidence": [e.type for e in self.evidence], **self.event.info}


@dataclass
class SpeedTrack:
    t: np.ndarray
    kmh: np.ndarray

    def at(self, t: float) -> float:
        if len(self.t) == 0:
            return float("nan")
        return float(np.interp(t, self.t, self.kmh))


def _overlapping(events: list[Event], types: set[str], start: float, end: float) -> list[Event]:
    return [e for e in events if e.type in types and e.start <= end and e.end >= start]


def attribute(events: list[Event], speed: SpeedTrack | None = None, mirror_check_available: bool = True,
              stopped_kmh: float = 3.0) -> list[Attributed]:
    """Return one Attributed record per risky event (sorted by time)."""
    events = sorted(events, key=lambda e: e.start)
    out: list[Attributed] = []
    for ev in events:
        if ev.type == "HARSH_BRAKE":
            distract = _overlapping(events, DISTRACTION, ev.start - 2.0, ev.start)
            hazard = _overlapping(events, {"HAZARD_AHEAD"}, ev.start - 2.0, ev.end + 0.5)
            if distract:
                out.append(Attributed(ev, "likely_distraction", "critical", distract))
            elif hazard:
                out.append(Attributed(ev, "likely_hazard_response", "info", hazard))
            else:
                out.append(Attributed(ev, "unexplained_harsh_brake", "warning"))
        elif ev.type in ("HARSH_ACCEL", "SHARP_TURN", "SPEEDING"):
            out.append(Attributed(ev, ev.type.lower(), "warning"))
        elif ev.type == "LANE_CHANGE":
            if not mirror_check_available:
                out.append(Attributed(ev, "mirror_check_unknown", "info"))
            else:
                mirror = _overlapping(events, {"MIRROR_CHECK"}, ev.start - 3.0, ev.start)
                if mirror:
                    out.append(Attributed(ev, "lane_change_with_mirror_check", "info", mirror))
                else:
                    out.append(Attributed(ev, "no_mirror_check", "warning"))
        elif ev.type in DISTRACTION:
            moving = speed is None or not (speed.at(ev.start) < stopped_kmh and speed.at(ev.end) < stopped_kmh)
            if not moving:
                out.append(Attributed(ev, "suppressed_stationary", "info"))
            elif ev.type in ("PHONE_USE", "PHONE_CALL", "TEXTING"):
                out.append(Attributed(ev, "phone_while_moving", "critical", status="deployment_rule"))
            else:
                out.append(Attributed(ev, ev.type.lower(), "warning"))
        elif ev.type in FATIGUE:
            drift = _overlapping(events, LANE_DRIFT, ev.start - 1.0, ev.end + 3.0)
            if drift:
                out.append(Attributed(ev, "critical_fatigue", "critical", drift, status="deployment_rule"))
            else:
                out.append(Attributed(ev, ev.type.lower(), "warning" if ev.type == "DROWSY" else "critical"))
    return out


def summary(attributed: list[Attributed]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for a in attributed:
        if a.context != "suppressed_stationary":
            counts[a.context] = counts.get(a.context, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
