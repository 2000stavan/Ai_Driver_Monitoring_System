"""Window-level fatigue features for the learned classifier (INITIALPLAN §4.2).

Per-frame features (from extract_cabin_features) -> personal calibration -> CabinMonitor events
-> one row per sliding window with PERCLOS, blink statistics, yawns, nods and head-pose variability.
These rows are what XGBoost / GRU models are trained on (UTA-RLDD -> INVEDRIFAC).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from dms.baseline.calibration import Baseline, Calibrator
from dms.cabin.fatigue import CabinMonitor

FRAME_COLS = ["ear", "mar", "blink_score", "jaw_open", "yaw", "pitch", "roll"]


def _frame_dict(row) -> dict | None:
    if not row.face or math.isnan(row.ear):
        return None
    d = {c: getattr(row, c) for c in FRAME_COLS}
    for c in ("n_hands", "hand_mouth", "hand_ear"):
        d[c] = getattr(row, c, math.nan)
    return d


def calibrate(df: pd.DataFrame, cfg: dict, calib_s: float | None = None) -> Baseline:
    """Personal baseline from the first seconds of a recording with a visible face."""
    cal = Calibrator.from_config(cfg, calib_s)
    for row in df.itertuples():
        cal.add(row.t, _frame_dict(row))
        if cal.done(row.t):
            break
    return cal.result()


def run_monitor(df: pd.DataFrame, cfg: dict, calib_s: float | None = None, baseline: Baseline | None = None):
    """Calibrate on the first seconds with a visible face (or use a given baseline, e.g. from the same
    driver's alert drive), then run the monitor over all frames.
    Returns (per-frame DataFrame with eyes_closed / off_road flags, list of events, monitor)."""
    mon = CabinMonitor(cfg, baseline or calibrate(df, cfg, calib_s))
    closed = np.zeros(len(df), dtype=bool)
    off = np.zeros(len(df), dtype=bool)
    events = []
    for i, row in enumerate(df.itertuples()):
        events += mon.update(row.t, _frame_dict(row))
        closed[i], off[i] = mon.state.eyes_closed, mon.state.off_road
    out = df.assign(eyes_closed=closed, off_road=off)
    return out, events, mon


def window_features(df: pd.DataFrame, cfg: dict, window_s: float = 60.0, step_s: float = 10.0,
                    baseline: Baseline | None = None) -> pd.DataFrame:
    """One row of features per sliding window over a single recording."""
    frames, events, mon = run_monitor(df, cfg, baseline=baseline)
    base = mon.base
    ev = pd.DataFrame([e.to_dict() for e in events]) if events else pd.DataFrame(columns=["type", "start", "end"])
    ev["dur"] = ev["end"] - ev["start"]
    t_end = frames["t"].iloc[-1]
    rows = []
    start = 0.0
    while start + window_s <= t_end + 1e-9:
        stop = start + window_s
        f = frames[(frames["t"] >= start) & (frames["t"] < stop)]
        vis = f[f["face"].astype(bool)]
        w = ev[(ev["start"] >= start) & (ev["start"] < stop)]
        closures = w[w["type"].isin(["BLINK", "SLOW_BLINK", "MICROSLEEP"])]
        minutes = window_s / 60.0
        rows.append(
            {
                "t_start": start,
                "t_end": stop,
                "face_ratio": len(vis) / max(len(f), 1),
                "perclos": vis["eyes_closed"].mean() if len(vis) else math.nan,
                "blink_rate": (w["type"] == "BLINK").sum() / minutes,
                "slow_blink_rate": (w["type"] == "SLOW_BLINK").sum() / minutes,
                "microsleeps": int((w["type"] == "MICROSLEEP").sum()),
                "closure_mean_s": closures["dur"].mean() if len(closures) else 0.0,
                "closure_max_s": closures["dur"].max() if len(closures) else 0.0,
                "yawns": int((w["type"] == "YAWN").sum()),
                "nods": int((w["type"] == "NOD").sum()),
                "ear_rel_mean": (vis["ear"] / base.ear_open).mean() if len(vis) else math.nan,
                "ear_rel_std": (vis["ear"] / base.ear_open).std() if len(vis) else math.nan,
                "mar_mean": vis["mar"].mean() if len(vis) else math.nan,
                "pitch_rel_mean": (vis["pitch"] - base.pitch).mean() if len(vis) else math.nan,
                "pitch_std": vis["pitch"].std() if len(vis) else math.nan,
                "yaw_std": vis["yaw"].std() if len(vis) else math.nan,
                "off_road_ratio": vis["off_road"].mean() if len(vis) else math.nan,
            }
        )
        start += step_s
    return pd.DataFrame(rows)
