"""UAH-DriveSet reader (Romera et al., ITSC 2016): 6 drivers x {normal, drowsy, aggressive} x
{motorway, secondary}, 40 trips, recorded by the DriveSafe iPhone app on the windshield.

Folder: <root>/D<k>/<YYYYMMDDhhmmss>-<km>km-D<k>-<BEHAVIOUR>[n]-<ROAD>/ with
  RAW_ACCELEROMETERS.txt  10 Hz: t, active, ax, ay, az (G, gravity removed), KF ax/ay/az, roll, pitch, yaw
  RAW_GPS.txt              1 Hz: t, speed km/h, lat, lon, alt, vacc, hacc, course, difcourse, ...
  PROC_LANE_DETECTION.txt  ~30 Hz: t, offset from lane centre (m), phi, road width, estimator state (-9 = none)
  PROC_VEHICLE_DETECTION   ~10 Hz: t, distance to vehicle ahead (m), time to impact (s), #vehicles, speed (-1 = none)
  PROC_OPENSTREETMAP_DATA  ~1 Hz: t, max speed, reliability flag, road type, #lanes, current lane, ...
  EVENTS_LIST_LANE_CHANGES  one row per lane change
Only the trip folder name carries the label (the behaviour the driver was asked to simulate).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BEHAVIOURS = ["NORMAL", "DROWSY", "AGGRESSIVE"]
_NAME = re.compile(r"^\d{14}-(\d+)km-(D\d)-(NORMAL|DROWSY|AGGRESSIVE)\d?-(MOTORWAY|SECONDARY)$")


@dataclass
class UAHTrip:
    name: str
    driver: str
    behaviour: str
    road: str
    acc: np.ndarray       # (N, 4): t, ax, ay, az  [G, gravity removed]
    gps: np.ndarray       # (M, 4): t, speed km/h, lat, lon
    lane: np.ndarray      # (K, 3): t, offset from lane centre (m), valid
    vehicle: np.ndarray   # (L, 3): t, distance ahead (m), time to impact (s); NaN = no vehicle
    osm: np.ndarray       # (P, 3): t, max speed km/h (NaN unknown), #lanes
    lane_changes: np.ndarray  # (C,) times


def _load(path: Path, ncols: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            a = np.loadtxt(path, usecols=range(ncols), ndmin=2)
        except (OSError, ValueError):
            a = np.empty((0, ncols))
    return a


def load_trip(folder: str | Path) -> UAHTrip:
    folder = Path(folder)
    m = _NAME.match(folder.name)
    if not m:
        raise ValueError(f"Unexpected trip folder name: {folder.name}")
    _, driver, behaviour, road = m.groups()
    acc = _load(folder / "RAW_ACCELEROMETERS.txt", 5)[:, [0, 2, 3, 4]]
    gps = _load(folder / "RAW_GPS.txt", 4)
    lane_raw = _load(folder / "PROC_LANE_DETECTION.txt", 5)
    valid = (lane_raw[:, 4] >= 0) & (lane_raw[:, 1] > -9)
    lane = np.c_[lane_raw[:, 0], np.where(valid, lane_raw[:, 1], np.nan), valid]
    veh_raw = _load(folder / "PROC_VEHICLE_DETECTION.txt", 3)
    veh = veh_raw.copy()
    veh[veh[:, 1] < 0, 1] = np.nan
    veh[veh[:, 2] < 0, 2] = np.nan
    osm_raw = _load(folder / "PROC_OPENSTREETMAP_DATA.txt", 5)
    osm = np.c_[osm_raw[:, 0], np.where(osm_raw[:, 1] > 0, osm_raw[:, 1], np.nan), osm_raw[:, 4]]
    lc = _load(folder / "EVENTS_LIST_LANE_CHANGES.txt", 1)[:, 0]
    return UAHTrip(folder.name, driver, behaviour, road, acc, gps, lane, veh, osm, lc)


def load_all(root: str | Path) -> list[UAHTrip]:
    return [load_trip(p) for p in sorted(Path(root).glob("D*/*")) if p.is_dir() and _NAME.match(p.name)]
