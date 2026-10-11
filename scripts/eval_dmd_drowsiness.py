"""Check the rule-based eye / blink / yawn detectors against DMD drowsiness labels (frame level).

DMD drowsiness is acted in a parked car, so this checks the *event detectors*, not real fatigue.

    python scripts/prepare_dmd.py --bundle drowsiness --videos face
    python scripts/eval_dmd_drowsiness.py [--limit 1]

Writes reports/results/dmd_drowsiness_eval.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from dms.baseline.calibration import Calibrator
from dms.cabin.fatigue import CabinMonitor
from dms.eval.dmd import load_dmd
from dms.eval.metrics import event_prf, frame_prf
from dms.utils.config import PROJECT_ROOT, load_config

INTERIM = PROJECT_ROOT / "data/interim/dmd/drowsiness"
FEATURES = PROJECT_ROOT / "data/processed/dmd_drowsiness"
RESULTS = PROJECT_ROOT / "reports/results"
FEATURE_COLS = ["ear", "mar", "blink_score", "jaw_open", "yaw", "pitch", "roll", "n_hands", "hand_mouth", "hand_ear"]


def features_for(rec: Path) -> pd.DataFrame:
    cache = FEATURES / f"{rec.name}.features.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        if "hand_mouth" in df.columns:
            return df
    from extract_cabin_features import extract  # local script module

    print(f"Extracting face + hand features for {rec.name} ...")
    return extract(rec / "rgb_face.mp4", cache, progress=False, hands=True)


def run_monitor(df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, list, CabinMonitor]:
    """Calibrate on the first seconds with a face, then run the monitor over the whole recording."""
    def feats(row) -> dict | None:
        if not row.face or math.isnan(row.ear):
            return None
        return {c: getattr(row, c) for c in FEATURE_COLS}

    cal = Calibrator.from_config(cfg)
    for row in df.itertuples():
        cal.add(row.t, feats(row))
        if cal.done(row.t):
            break
    monitor = CabinMonitor(cfg, cal.result())
    closed = np.zeros(len(df), dtype=bool)
    events = []
    for i, row in enumerate(df.itertuples()):
        events += monitor.update(row.t, feats(row))
        closed[i] = monitor.state.eyes_closed
    return closed, events, monitor


def to_frames(events, kinds, fps) -> list[tuple[int, int]]:
    return [(round(e.start * fps), round(e.end * fps)) for e in events if e.type in kinds]


def evaluate(rec: Path, cfg: dict) -> dict:
    ann = load_dmd(rec / "ann.json")
    df = features_for(rec)
    fps = len(df) / df["t"].iloc[-1] if len(df) > 1 else 29.76
    closed_pred, events, monitor = run_monitor(df, cfg)

    n = min(len(df), ann.n_frames)
    closed_pred = closed_pred[:n]
    gt_closed = ann.mask("eyes_state/close")[:n]
    # Score only frames with a clear open/closed label and a visible face.
    clear = (ann.mask("eyes_state/open") | ann.mask("eyes_state/close"))[:n]
    occluded = ann.mask("occlusion/face_occlusion", "occlusion/face")[:n]
    valid = clear & ~occluded & df["face"].to_numpy()[:n]

    gt_blinks = ann.segments("blinks/blinking")
    gt_yawns = ann.segments("yawning/Yawning with hand", "yawning/Yawning without hand")
    gt_long = [s for s in ann.segments("eyes_state/close") if (s[1] - s[0] + 1) / fps >= cfg["eyes"]["microsleep_min_s"]]

    eyes = frame_prf(closed_pred, gt_closed, valid)
    blinks = event_prf(to_frames(events, {"BLINK", "SLOW_BLINK", "MICROSLEEP"}, fps), gt_blinks, tol=3)
    yawns = event_prf(to_frames(events, {"YAWN"}, fps), gt_yawns, tol=15)
    longc = event_prf(to_frames(events, {"MICROSLEEP"}, fps), gt_long, tol=5)
    return {
        "recording": rec.name,
        "frames": n,
        "face_detected": float(df["face"].mean()),
        "baseline_ear_open": monitor.base.ear_open,
        "eyes_closed_precision": eyes["precision"],
        "eyes_closed_recall": eyes["recall"],
        "eyes_closed_f1": eyes["f1"],
        "blink_recall": blinks["recall"],
        "blink_precision": blinks["precision"],
        "blinks_gt": blinks["n_gt"],
        "blinks_pred": blinks["n_pred"],
        "yawn_recall": yawns["recall"],
        "yawn_precision": yawns["precision"],
        "yawns_gt": yawns["n_gt"],
        "yawns_pred": yawns["n_pred"],
        "long_closure_recall": longc["recall"],
        "long_closure_precision": longc["precision"],
        "long_closures_gt": longc["n_gt"],
        "microsleeps_pred": longc["n_pred"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N recordings")
    args = ap.parse_args()
    cfg = load_config("cabin")
    recs = sorted(p for p in INTERIM.iterdir() if (p / "ann.json").exists() and (p / "rgb_face.mp4").exists())
    if args.limit:
        recs = recs[: args.limit]
    if not recs:
        raise SystemExit("No prepared recordings. Run scripts/prepare_dmd.py --bundle drowsiness first.")
    rows = [evaluate(r, cfg) for r in recs]
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "dmd_drowsiness_eval.csv"
    df.to_csv(out, index=False)
    cols = ["recording", "face_detected", "eyes_closed_f1", "blink_recall", "blink_precision",
            "yawn_recall", "yawn_precision", "long_closure_recall"]
    with pd.option_context("display.width", 200, "display.precision", 2):
        print(df[cols].to_string(index=False))
        print("\nMean over recordings:")
        print(df[cols[1:]].mean().to_string())
    print(f"\nSaved {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    main()
