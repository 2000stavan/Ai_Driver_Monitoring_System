"""First Indian real-driving check: blink and head-pose detectors vs. DashGaze eye-tracker ground truth.

DashGaze sample (1 Indian driver, RHD car, real traffic, 15 min). The video is a side-by-side
triplet; only the driver view (x 0-960) is used. Ground truth comes from the eye-tracking
glasses: `blink id` (blinks) and the glasses IMU `roll` / `pitch` (head pose). Tracker
azimuth / elevation are eye-in-head angles, so they are not head pose.

Two threshold settings are compared:
  defaults   = configs/cabin.yaml
  dmd_tuned  = eye thresholds chosen by leave-one-subject-out tuning on DMD (non-Indian)
The second is a small transfer test: tuned on non-Indian data, tested on an Indian driver.

    python scripts/eval_dashgaze.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from extract_cabin_features import extract  # noqa: E402

from dms.cabin.windows import run_monitor  # noqa: E402
from dms.eval.metrics import event_prf, frame_prf  # noqa: E402
from dms.utils.config import PROJECT_ROOT, load_config  # noqa: E402

SAMPLE = PROJECT_ROOT / "data/raw/cabin/dashgaze/sample"
FEATURES = PROJECT_ROOT / "data/processed/dashgaze/0038_driver.features.csv"
DRIVER_VIEW = (0, 960)
DMD_TUNED_EYES = {"closed_ear_ratio": 0.55, "closed_blink_score": 0.45}


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))


def blink_gt(sensors: pd.DataFrame) -> list[tuple[int, int]]:
    ids = sensors["blink id"].to_numpy()
    segs = []
    for bid in pd.unique(ids[~np.isnan(ids)]):
        idx = np.flatnonzero(ids == bid)
        segs.append((int(idx.min()), int(idx.max())))
    return sorted(segs)


def corr(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 else float("nan")


def main() -> None:
    sensors = pd.read_csv(SAMPLE / "0038_sensors.csv")
    if FEATURES.exists():
        feats = pd.read_csv(FEATURES)
    else:
        print("Extracting driver-view features (one-off, a few minutes) ...", flush=True)
        feats = extract(SAMPLE / "0038_vid.mp4", FEATURES, progress=True, crop_x=DRIVER_VIEW)
    n = min(len(feats), len(sensors))
    feats, sensors = feats.iloc[:n].reset_index(drop=True), sensors.iloc[:n].reset_index(drop=True)
    fps = n / feats["t"].iloc[-1]

    gt_blinks = blink_gt(sensors)
    gt_blink_frames = sensors["blink id"].notna().to_numpy()

    base_cfg = load_config("cabin")
    tuned_cfg = copy.deepcopy(base_cfg)
    tuned_cfg["eyes"].update(DMD_TUNED_EYES)

    rows = []
    for name, cfg in (("defaults", base_cfg), ("dmd_tuned", tuned_cfg)):
        frames, events, mon = run_monitor(feats, cfg)
        pred = [(round(e.start * fps), round(e.end * fps)) for e in events
                if e.type in ("BLINK", "SLOW_BLINK", "MICROSLEEP")]
        ev = event_prf(pred, gt_blinks, tol=3)
        fr = frame_prf(frames["eyes_closed"].to_numpy(), gt_blink_frames, feats["face"].to_numpy(dtype=bool))
        rows.append({
            "thresholds": name,
            "blinks_gt": ev["n_gt"], "blinks_pred": ev["n_pred"],
            "blink_recall": ev["recall"], "blink_precision": ev["precision"], "blink_f1": ev["f1"],
            "closed_frame_f1_vs_blink_frames": fr["f1"],
        })

    face_ok = feats["face"].astype(bool)
    head = {
        "face_detected": float(face_ok.mean()),
        "corr_pitch_vs_glasses_imu": corr(feats.loc[face_ok, "pitch"], sensors.loc[face_ok, "pitch"]),
        "corr_roll_vs_glasses_imu": corr(feats.loc[face_ok, "roll"], sensors.loc[face_ok, "roll"]),
        "eye_in_head_azimuth_p5_p95": (float(sensors["azimuth [deg]"].quantile(0.05)),
                                       float(sensors["azimuth [deg]"].quantile(0.95))),
    }

    res = pd.DataFrame(rows)
    out_dir = PROJECT_ROOT / "reports/results"
    res.to_csv(out_dir / "dashgaze_eval.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 2):
        print(res.to_string(index=False))
    print()
    for k, v in head.items():
        print(f"{k}: {v if isinstance(v, tuple) else round(v, 3)}")


if __name__ == "__main__":
    main()
