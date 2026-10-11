"""Personal baseline (#4, INITIALPLAN §7.1).

Collects the driver's normal open-eye EAR, mouth ratio and neutral head pose during a
calibration window, so thresholds become driver-relative instead of fixed.

Robust version (v1.1): a driver who is still settling in, looking at the mirror or turned to talk
during calibration inflates the open-eye EAR (INVEDRIFAC Sub 26: head turned ~37° in the first
10 s -> EAR baseline 0.52 instead of ~0.34 -> hundreds of false microsleeps). So:
  1. the neutral head pose is the median of the frames near the dominant pose (iterated),
  2. only near-frontal frames (within `frontal_deg` of it) are used for the eye / mouth levels,
  3. EAR outliers (tracking glitches) are removed with a median-absolute-deviation rule,
  4. the open-eye level is the 75th percentile of what is left (blinks pull the median down).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Baseline:
    ear_open: float
    mar_rest: float
    yaw: float
    pitch: float
    n_frames: int
    n_used: int = 0  # frames that passed the frontal-pose and outlier filters


def _dominant_pose(yaw: np.ndarray, pitch: np.ndarray, radius: float, iters: int = 3) -> tuple[float, float]:
    """Median pose, re-estimated from the frames within `radius` of it (a cheap mode-seeking step)."""
    ry, rp = float(np.median(yaw)), float(np.median(pitch))
    for _ in range(iters):
        near = np.hypot(yaw - ry, pitch - rp) <= radius
        if near.sum() < 10:
            break
        ry, rp = float(np.median(yaw[near])), float(np.median(pitch[near]))
    return ry, rp


class Calibrator:
    def __init__(self, seconds: float, frontal_deg: float = 15.0, mad_k: float = 3.0, min_frames: int = 10):
        self.seconds = seconds
        self.frontal_deg = frontal_deg
        self.mad_k = mad_k
        self.min_frames = min_frames
        self._start: float | None = None
        self._rows: list[dict] = []

    @classmethod
    def from_config(cls, cfg: dict, seconds: float | None = None) -> "Calibrator":
        c = cfg["calibration"]
        return cls(seconds or c["seconds"], c.get("frontal_deg", 15.0), c.get("mad_k", 3.0))

    def add(self, t: float, feats: dict | None) -> None:
        if self._start is None:
            self._start = t
        if feats is not None:
            self._rows.append(feats)

    def progress(self, t: float) -> float:
        if self._start is None:
            return 0.0
        return min(1.0, (t - self._start) / self.seconds)

    def done(self, t: float) -> bool:
        return self.progress(t) >= 1.0 and len(self._rows) >= self.min_frames

    def result(self) -> Baseline:
        if len(self._rows) < self.min_frames:
            raise RuntimeError("Not enough frames with a visible face to calibrate.")

        def col(name: str) -> np.ndarray:
            return np.array([r[name] for r in self._rows], dtype=float)

        ear, mar, yaw, pitch = col("ear"), col("mar"), col("yaw"), col("pitch")
        keep = ~np.isnan(ear)
        pose_ok = keep & ~np.isnan(yaw) & ~np.isnan(pitch)
        ref_yaw = ref_pitch = math.nan
        if pose_ok.sum() >= self.min_frames:
            ref_yaw, ref_pitch = _dominant_pose(yaw[pose_ok], pitch[pose_ok], self.frontal_deg)
            frontal = pose_ok & (np.hypot(yaw - ref_yaw, pitch - ref_pitch) <= self.frontal_deg)
            if frontal.sum() >= self.min_frames:
                keep = frontal
        med = np.median(ear[keep])
        mad = 1.4826 * np.median(np.abs(ear[keep] - med))
        inlier = keep & (np.abs(ear - med) <= self.mad_k * max(mad, 1e-3))
        if inlier.sum() >= self.min_frames:
            keep = inlier
        return Baseline(
            ear_open=float(np.percentile(ear[keep], 75)),
            mar_rest=float(np.nanmedian(mar[keep])),
            yaw=ref_yaw,
            pitch=ref_pitch,
            n_frames=len(self._rows),
            n_used=int(keep.sum()),
        )
