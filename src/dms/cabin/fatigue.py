"""Rule-based cabin monitor (v1 baseline, INITIALPLAN §4.2 and §4.4).

Turns per-frame features into timestamped events using driver-relative thresholds:
BLINK, SLOW_BLINK, MICROSLEEP, YAWN, NOD, EYES_OFF_ROAD (head-pose proxy), DROWSY (PERCLOS rule).

These rules are the starting point; the learned fatigue classifier (XGBoost/GRU on window
features) and L2CS-Net gaze replace the corresponding rules in later phases.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from dms.baseline.calibration import Baseline
from dms.events import Event


@dataclass
class _Run:
    """Tracks one continuous condition (e.g. eyes closed) over time."""

    start: float | None = None

    def update(self, active: bool, t: float) -> tuple[bool, float | None]:
        """Returns (just_ended, start_of_ended_run)."""
        if active and self.start is None:
            self.start = t
            return False, None
        if not active and self.start is not None:
            s, self.start = self.start, None
            return True, s
        return False, None

    def duration(self, t: float) -> float:
        return 0.0 if self.start is None else t - self.start


@dataclass
class CabinState:
    """Live values for display / logging."""

    eyes_closed: bool = False
    mouth_open: bool = False
    off_road: bool = False
    perclos: float = math.nan
    drowsy: bool = False
    closure_s: float = 0.0
    face_visible: bool = False
    counts: dict = field(default_factory=lambda: {"BLINK": 0, "SLOW_BLINK": 0, "MICROSLEEP": 0, "YAWN": 0, "NOD": 0})


class CabinMonitor:
    def __init__(self, cfg: dict, baseline: Baseline):
        self.cfg = cfg
        self.base = baseline
        self.state = CabinState()
        self._eyes = _Run()
        self._mouth = _Run()
        self._covered = _Run()
        self._last_cover_t = -math.inf
        self._last_yawn_end = -math.inf
        self._off_road = _Run()
        self._nod_start: float | None = None
        self._perclos: deque[tuple[float, bool]] = deque()
        self._off_road_emitted = False
        self._drowsy_since: float | None = None

    # ---- per-frame classification -------------------------------------------------
    def eyes_closed(self, f: dict) -> bool:
        e = self.cfg["eyes"]
        by_ear = f["ear"] < e["closed_ear_ratio"] * self.base.ear_open
        by_score = not math.isnan(f["blink_score"]) and f["blink_score"] > e["closed_blink_score"]
        return by_ear or by_score

    def mouth_open(self, f: dict) -> bool:
        m = self.cfg["mouth"]
        by_jaw = not math.isnan(f["jaw_open"]) and f["jaw_open"] > m["yawn_jaw_score"]
        return f["mar"] > m["yawn_mar"] or by_jaw

    def head_off_road(self, f: dict) -> bool:
        h = self.cfg["head"]
        if math.isnan(f["yaw"]) or math.isnan(self.base.yaw):
            return False
        return (
            abs(f["yaw"] - self.base.yaw) > h["off_road_yaw_deg"]
            or abs(f["pitch"] - self.base.pitch) > h["off_road_pitch_deg"]
        )

    # ---- streaming update ---------------------------------------------------------
    def update(self, t: float, f: dict | None) -> list[Event]:
        """Feed one frame (f=None when no face is visible). Returns events that ended now."""
        events: list[Event] = []
        st = self.state
        st.face_visible = f is not None
        e_cfg, m_cfg, h_cfg, p_cfg = (self.cfg[k] for k in ("eyes", "mouth", "head", "perclos"))

        closed = f is not None and self.eyes_closed(f)
        mouth = f is not None and self.mouth_open(f)
        off_road = f is not None and self.head_off_road(f)

        # Eye closures -> BLINK / SLOW_BLINK / MICROSLEEP
        ended, s = self._eyes.update(closed, t)
        if ended:
            d = t - s
            if d >= e_cfg["microsleep_min_s"]:
                kind = "MICROSLEEP"
            elif d > e_cfg["blink_max_s"]:
                kind = "SLOW_BLINK"
            elif d >= e_cfg["blink_min_s"]:
                kind = "BLINK"
            else:
                kind = None
            if kind:
                events.append(Event(kind, s, t))
                st.counts[kind] += 1

        # Wide-open mouth held long enough -> YAWN
        ended, s = self._mouth.update(mouth, t)
        if ended and t - s >= m_cfg["yawn_min_s"]:
            self._add_yawn(events, s, t, covered=False)

        # Hand held over the mouth (mouth not visible) -> YAWN with hand.
        # In deployment this is ambiguous with eating / drinking; the distraction module
        # (objects in hand) decides which one it is.
        hm = math.nan if f is None else f.get("hand_mouth", math.nan)
        if not math.isnan(hm) and hm < m_cfg.get("hand_mouth_dist", 0.3):
            self._last_cover_t = t
        covered = t - self._last_cover_t <= m_cfg.get("hand_gap_s", 0.5)
        ended, s = self._covered.update(covered, t)
        if ended and self._last_cover_t - s >= m_cfg.get("yawn_hand_min_s", 2.0):
            self._add_yawn(events, s, self._last_cover_t, covered=True)

        # Head turned away from neutral for long enough -> EYES_OFF_ROAD (head-pose proxy)
        ended, s = self._off_road.update(off_road, t)
        if ended and t - s >= h_cfg["off_road_min_s"]:
            events.append(Event("EYES_OFF_ROAD", s, t, info={"source": "head_pose"}))

        # Nod: pitch drops below neutral and recovers quickly
        if f is not None and not math.isnan(f["pitch"]) and not math.isnan(self.base.pitch):
            dropped = abs(f["pitch"] - self.base.pitch) > h_cfg["nod_pitch_deg"]
            if dropped and self._nod_start is None:
                self._nod_start = t
            elif not dropped and self._nod_start is not None:
                if h_cfg.get("nod_min_s", 0.0) <= t - self._nod_start <= h_cfg["nod_max_s"]:
                    events.append(Event("NOD", self._nod_start, t))
                    st.counts["NOD"] += 1
                self._nod_start = None

        # PERCLOS over a sliding window -> rule-based DROWSY state
        self._perclos.append((t, closed))
        while self._perclos and t - self._perclos[0][0] > p_cfg["window_s"]:
            self._perclos.popleft()
        span = t - self._perclos[0][0]
        if span >= p_cfg["min_fill"] * p_cfg["window_s"]:
            st.perclos = sum(c for _, c in self._perclos) / len(self._perclos)
            drowsy = st.perclos >= p_cfg["drowsy_threshold"]
        else:
            st.perclos, drowsy = math.nan, False
        if drowsy and self._drowsy_since is None:
            self._drowsy_since = t
        elif not drowsy and self._drowsy_since is not None:
            events.append(Event("DROWSY", self._drowsy_since, t, info={"rule": "perclos"}))
            self._drowsy_since = None

        st.eyes_closed, st.mouth_open, st.off_road, st.drowsy = closed, mouth, off_road, drowsy
        st.closure_s = self._eyes.duration(t)
        return events

    def _add_yawn(self, events: list[Event], start: float, end: float, covered: bool) -> None:
        if start <= self._last_yawn_end + 1.0:  # same yawn seen by both cues
            self._last_yawn_end = max(self._last_yawn_end, end)
            return
        events.append(Event("YAWN", start, end, info={"hand_over_mouth": covered}))
        self.state.counts["YAWN"] += 1
        self._last_yawn_end = end

    def live_alerts(self, t: float) -> list[str]:
        """Conditions that are happening right now (for on-screen warnings)."""
        a = []
        if self._eyes.duration(t) >= self.cfg["eyes"]["microsleep_min_s"]:
            a.append("EYES CLOSED - MICROSLEEP")
        if self._mouth.duration(t) >= self.cfg["mouth"]["yawn_min_s"]:
            a.append("YAWNING")
        if self._off_road.duration(t) >= self.cfg["head"]["off_road_min_s"]:
            a.append("EYES OFF ROAD")
        if self.state.drowsy:
            a.append(f"DROWSY (PERCLOS {self.state.perclos:.0%})")
        return a
