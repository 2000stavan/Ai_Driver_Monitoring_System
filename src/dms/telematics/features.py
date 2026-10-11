"""Window features for the driving-behaviour classifier (INITIALPLAN §5.3).

Built only from signals that every source has, so a model trained on UAH-DriveSet (Spain) can be
applied unchanged to LEVIN (Pune) and to our own phone drives:

  speed tier   1 Hz speed (GPS or OBD): speed profile, longitudinal acceleration / braking and jerk
               derived from the speed itself (no phone orientation needed)
  motion tier  phone accelerometer, gravity removed, smoothed over 1 s: overall dynamic acceleration
               magnitude (orientation-free, so a badly mounted phone does not matter)

UAH-only extras (lane offset, car-following, speed limit) are added by the caller for the ablation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

G = 9.81


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or len(x) < n:
        return x
    k = np.ones(n) / n
    pad = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


class SpeedSeries:
    """1 Hz speed with longitudinal acceleration and jerk derived from it."""

    def __init__(self, t: np.ndarray, speed_kmh: np.ndarray, smooth_s: int = 3):
        ok = np.isfinite(t) & np.isfinite(speed_kmh)
        t, v = t[ok], np.clip(speed_kmh[ok], 0, 200)
        self.t = np.arange(np.ceil(t[0]), np.floor(t[-1]) + 1)
        self.v = np.interp(self.t, t, v)
        vs = _smooth(self.v, smooth_s) / 3.6
        self.a_lon = np.gradient(vs) / G          # g
        self.jerk = np.gradient(self.a_lon)       # g / s


class MotionSeries:
    """Dynamic acceleration magnitude (g) at 10 Hz, smoothed over 1 s."""

    def __init__(self, t: np.ndarray, dyn_acc: np.ndarray, rate_hz: float = 10.0):
        mag = np.linalg.norm(dyn_acc, axis=1) if dyn_acc.ndim == 2 else np.abs(dyn_acc)
        self.t = np.arange(t[0], t[-1], 1.0 / rate_hz)
        self.m = _smooth(np.interp(self.t, t, mag), int(rate_hz))


def speed_features(s: SpeedSeries, t0: float, t1: float) -> dict:
    m = (s.t >= t0) & (s.t < t1)
    v, a, j = s.v[m], s.a_lon[m], s.jerk[m]
    if len(v) < 10:
        return {}
    moving = v > 5
    crossings = np.sum(np.diff(np.sign(a[np.abs(a) > 0.03])) != 0) if (np.abs(a) > 0.03).sum() > 1 else 0
    return {
        "speed_mean": v.mean(), "speed_std": v.std(), "speed_max": v.max(),
        "speed_cv": v.std() / max(v.mean(), 1.0), "moving_frac": moving.mean(),
        "acc_p95": np.percentile(a, 95), "brake_p05": np.percentile(a, 5),
        "acc_max": a.max(), "brake_max": -a.min(), "a_lon_std": a.std(),
        "frac_acc_gt_0.1": (a > 0.1).mean(), "frac_brake_gt_0.1": (a < -0.1).mean(),
        "frac_brake_gt_0.2": (a < -0.2).mean(),
        "jerk_std": j.std(), "jerk_abs_p95": np.percentile(np.abs(j), 95),
        "acc_sign_changes_per_min": crossings * 60.0 / len(v),
    }


def motion_features(ms: MotionSeries, t0: float, t1: float) -> dict:
    m = (ms.t >= t0) & (ms.t < t1)
    x = ms.m[m]
    if len(x) < 20:
        return {}
    return {
        "dyn_mean": x.mean(), "dyn_std": x.std(), "dyn_p95": np.percentile(x, 95), "dyn_max": x.max(),
        "dyn_frac_gt_0.1": (x > 0.1).mean(), "dyn_frac_gt_0.2": (x > 0.2).mean(),
    }


def trip_windows(speed: SpeedSeries, motion: MotionSeries | None = None, extras=None,
                 window_s: float = 60.0, step_s: float = 10.0) -> pd.DataFrame:
    """One row per window. extras(t0, t1) -> dict adds source-specific features (UAH ablation)."""
    rows = []
    t0 = speed.t[0]
    while t0 + window_s <= speed.t[-1] + 1e-9:
        t1 = t0 + window_s
        f = speed_features(speed, t0, t1)
        if f:
            f = {"t_start": t0, "t_end": t1, **f}
            if motion is not None:
                f.update(motion_features(motion, t0, t1))
            if extras is not None:
                f.update(extras(t0, t1))
            rows.append(f)
        t0 += step_s
    return pd.DataFrame(rows)


SPEED_COLS = ["speed_mean", "speed_std", "speed_max", "speed_cv", "moving_frac", "acc_p95", "brake_p05",
              "acc_max", "brake_max", "a_lon_std", "frac_acc_gt_0.1", "frac_brake_gt_0.1", "frac_brake_gt_0.2",
              "jerk_std", "jerk_abs_p95", "acc_sign_changes_per_min"]
MOTION_COLS = ["dyn_mean", "dyn_std", "dyn_p95", "dyn_max", "dyn_frac_gt_0.1", "dyn_frac_gt_0.2"]
