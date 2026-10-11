"""Phone / device axes -> vehicle axes (INITIALPLAN §5).

1. Gravity: the median acceleration vector points "up" (an accelerometer at rest reads +1 g
   against gravity). Rotate it onto +z, so x/y become the horizontal plane.
2. Forward: the horizontal direction whose acceleration best explains the change in vehicle
   speed (dv/dt from OBD/GPS), found by least squares on 5 s-smoothed 1 Hz averages, trying
   each speed source and a small time lag (LEVIN, 27 vehicles: median correlation 0.46 -> 0.80; 73 % of trips >= 0.5).
Lateral is perpendicular to forward; its sign is not determined, so only |lateral| is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.80665


def rotation_to_z(v: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector v onto +z (Rodrigues)."""
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(v, z))
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])
    k = np.cross(v, z)
    s = np.linalg.norm(k)
    k = k / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * kx + (1 - c) * kx @ kx


@dataclass
class VehicleFrame:
    longitudinal: np.ndarray  # g, + = accelerating, - = braking
    lateral: np.ndarray  # g, sign undetermined
    vertical: np.ndarray  # g, gravity removed
    forward_corr: float  # correlation of 1 Hz longitudinal accel with dv/dt (quality check)
    gravity: np.ndarray


def _rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Centered moving average that ignores NaNs (works on 1-D or 2-D arrays, along axis 0)."""
    x = np.asarray(x, dtype=float)
    if n <= 1:
        return x
    k = np.ones(n)
    valid = ~np.isnan(x)
    filled = np.where(valid, x, 0.0)
    if x.ndim == 1:
        num, den = np.convolve(filled, k, "same"), np.convolve(valid.astype(float), k, "same")
    else:
        num = np.stack([np.convolve(filled[:, i], k, "same") for i in range(x.shape[1])], 1)
        den = np.stack([np.convolve(valid[:, i].astype(float), k, "same") for i in range(x.shape[1])], 1)
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def to_vehicle_frame(
    acc_t: np.ndarray,
    acc: np.ndarray,
    sec_t: np.ndarray,
    speeds_kmh: dict[str, np.ndarray],
    smooth_s: int = 5,
    lags: range = range(-3, 2),
) -> VehicleFrame:
    """speeds_kmh: one or more 1 Hz speed sources (e.g. OBD and GPS). The forward axis is fitted
    for every source and time lag; the best-correlated fit is kept (sensors are not perfectly
    synchronised, and OBD speed is integer km/h)."""
    ok = ~np.isnan(acc).any(axis=1)
    if ok.sum() < 10:
        nan = np.full(len(acc), np.nan)
        return VehicleFrame(nan, nan, nan, np.nan, np.full(3, np.nan))
    gravity = np.median(acc[ok], axis=0)
    r = rotation_to_z(gravity)
    a = acc @ r.T
    horiz, vert = a[:, :2], a[:, 2] - np.linalg.norm(gravity)

    sec = np.floor(acc_t).astype(int)
    n_sec = len(sec_t)
    h1 = np.full((n_sec, 2), np.nan)
    cnts = np.bincount(sec[ok], minlength=n_sec)[:n_sec]
    for i in range(2):
        sums = np.bincount(sec[ok], weights=horiz[ok, i], minlength=n_sec)[:n_sec]
        h1[:, i] = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    h1 = _rolling_mean(h1, smooth_s)

    best = (-np.inf, None)
    for speed in speeds_kmh.values():
        v = _rolling_mean(speed, smooth_s)
        dvdt = np.gradient(np.nan_to_num(v) / 3.6) / G
        moving = np.nan_to_num(v) > 3
        for lag in lags:
            d = np.roll(dvdt, -lag)
            use = ~np.isnan(h1).any(axis=1) & moving
            if use.sum() < 10:
                continue
            w, *_ = np.linalg.lstsq(h1[use], d[use], rcond=None)
            if np.linalg.norm(w) == 0:
                continue
            fwd = w / np.linalg.norm(w)
            x, y = h1[use] @ fwd, d[use]
            if x.std() == 0 or y.std() == 0:
                continue
            c = float(np.corrcoef(x, y)[0, 1])
            if c > best[0]:
                best = (c, fwd)
    corr, fwd = best
    if fwd is None:
        nan = np.full(len(acc), np.nan)
        return VehicleFrame(nan, nan, vert, np.nan, gravity)
    lat = np.array([-fwd[1], fwd[0]])
    return VehicleFrame(horiz @ fwd, horiz @ lat, vert, corr, gravity)
