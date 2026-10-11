"""Candidate harsh-driving events from vehicle-frame acceleration (INITIALPLAN §5).

Rule-based candidates (threshold + minimum duration on smoothed acceleration) with per-event
features. A supervised classifier trained on Pune VIIT labels refines these in the next step.
"""

from __future__ import annotations

import numpy as np

from dms.events import Event


def smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    x = np.nan_to_num(x)
    return np.convolve(x, np.ones(n) / n, mode="same")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def detect_events(
    t: np.ndarray,
    longitudinal: np.ndarray,
    lateral: np.ndarray,
    speed_at: np.ndarray,
    cfg: dict,
    rate_hz: float = 25.0,
) -> list[Event]:
    """t, longitudinal, lateral at rate_hz (g); speed_at = vehicle speed (km/h) at each sample."""
    n = max(1, int(cfg["smooth_s"] * rate_hz))
    lon, lat = smooth(longitudinal, n), np.abs(smooth(lateral, n))
    moving = speed_at >= cfg["min_speed_kmh"]
    e = cfg["events"]
    min_len = int(e["min_duration_s"] * rate_hz)
    out: list[Event] = []
    for kind, mask, sig in (
        ("HARSH_BRAKE", (lon <= -e["harsh_brake_g"]) & moving, -lon),
        ("HARSH_ACCEL", (lon >= e["harsh_accel_g"]) & moving, lon),
        ("SHARP_TURN", (lat >= e["sharp_turn_g"]) & moving, lat),
    ):
        for s, end in _runs(mask):
            if end - s + 1 < min_len:
                continue
            seg = sig[s : end + 1]
            jerk = np.abs(np.diff(seg)) * rate_hz if len(seg) > 1 else np.array([0.0])
            out.append(
                Event(
                    kind,
                    float(t[s]),
                    float(t[end]),
                    info={
                        "peak_g": float(seg.max()),
                        "mean_g": float(seg.mean()),
                        "jerk_max_gps": float(jerk.max()),
                        "speed_start_kmh": float(speed_at[s]),
                        "speed_end_kmh": float(speed_at[end]),
                    },
                )
            )
    return sorted(out, key=lambda ev: ev.start)
