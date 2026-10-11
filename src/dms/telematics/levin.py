"""Reader for LEVIN Open Data (Yuñ Solutions, Pune; ~30 cars, 4 months; CC BY-NC-SA).

One CSV per vehicle, one row per second: OBD (speed km/h, rpm, throttle...), GPS speed, and
`accData` = 25 accelerometer samples packed as hex. Decoding follows the official
data_extract.ipynb: each axis is a signed byte, range +-2 g -> value / 64 g. Rows of length 162
start with a 12-hex magnetometer prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLES_PER_ROW = 25
MAX_GAP_S = 60  # split a tripID into separate drives at gaps longer than this


@dataclass
class Trip:
    vehicle: str
    trip_id: str  # "<tripID>-<segment>": a LEVIN tripID can span several days, so it is split at gaps
    seconds: pd.DataFrame  # 1 Hz: t, speed_kmh, gps_speed, rpm, throttle
    acc_t: np.ndarray  # (N,) seconds from trip start, 25 Hz
    acc: np.ndarray  # (N, 3) raw device-frame acceleration in g


def decode_acc(hex_rows: pd.Series) -> np.ndarray:
    """(rows,) hex strings -> (rows, 25, 3) acceleration in g; invalid rows become NaN."""
    out = np.full((len(hex_rows), SAMPLES_PER_ROW, 3), np.nan, dtype=np.float32)
    s = hex_rows.fillna("").astype(str)
    for length, offset in ((150, 0), (162, 12)):
        idx = np.flatnonzero(s.str.len().to_numpy() == length)
        if idx.size:
            payload = "".join(x[offset:] for x in s.iloc[idx])
            raw = np.frombuffer(bytes.fromhex(payload), dtype=np.int8).reshape(idx.size, SAMPLES_PER_ROW, 3)
            out[idx] = raw.astype(np.float32) / 64.0
    return out


def _clean_speed(v: pd.Series) -> pd.Series:
    """OBD speed has spikes (e.g. 191 km/h for one second): clip, then 5 s median filter."""
    return v.clip(0, 160).rolling(5, center=True, min_periods=1).median()


def load_vehicle(path: str | Path) -> list[Trip]:
    path = Path(path)
    df = pd.read_csv(path, index_col=0, dtype={"accData": str}, low_memory=False)
    df["timeStamp"] = pd.to_datetime(df["timeStamp"])
    trips = []
    for trip_id, whole in df.groupby("tripID", sort=True):
        whole = whole.sort_values("timeStamp")
        gap = whole["timeStamp"].diff().dt.total_seconds().fillna(0) > MAX_GAP_S
        for seg, g in whole.groupby(gap.cumsum()):
            trip = _make_trip(path.stem, f"{trip_id}-{seg}", g)
            if trip is not None:
                trips.append(trip)
    return trips


def _make_trip(vehicle: str, trip_id: str, g: pd.DataFrame) -> Trip | None:
    if len(g) < 2:
        return None
    t = (g["timeStamp"] - g["timeStamp"].iloc[0]).dt.total_seconds().to_numpy()
    seconds = pd.DataFrame(
        {
            "t": t,
            "speed_kmh": _clean_speed(g["speed"].astype(float)).to_numpy(),
            "gps_speed": g["gps_speed"].to_numpy(dtype=float),
            "rpm": g["rpm"].to_numpy(dtype=float),
            "throttle": g["tPos"].to_numpy(dtype=float),
        }
    )
    acc = decode_acc(g["accData"])
    if np.isnan(acc).all():
        return None  # segment without accelerometer data
    acc_t = (t[:, None] + np.arange(SAMPLES_PER_ROW)[None, :] / SAMPLES_PER_ROW).ravel()
    return Trip(vehicle, trip_id, seconds, acc_t, acc.reshape(-1, 3))
