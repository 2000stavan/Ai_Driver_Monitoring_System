"""Check the head-pose eyes-off-road detector against DMD distraction `gaze_on_road` labels.

Head pose is only a proxy for gaze (eyes can move without the head); L2CS-Net gaze replaces it later.

    python scripts/prepare_dmd.py --bundle distractions --videos face body
    python scripts/eval_dmd_gaze.py

Writes reports/results/dmd_gaze_eval.csv
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from dms.baseline.calibration import Calibrator  # noqa: E402
from dms.cabin.fatigue import CabinMonitor  # noqa: E402
from dms.eval.dmd import load_dmd  # noqa: E402
from dms.eval.metrics import event_prf, frame_prf, roc_auc  # noqa: E402
from dms.utils.config import PROJECT_ROOT, load_config  # noqa: E402

INTERIM = PROJECT_ROOT / "data/interim/dmd/distractions"
FEATURES = PROJECT_ROOT / "data/processed/dmd_distraction"
RESULTS = PROJECT_ROOT / "reports/results"
COLS = ["ear", "mar", "blink_score", "jaw_open", "yaw", "pitch", "roll", "n_hands", "hand_mouth", "hand_ear"]


def features_for(rec: Path) -> pd.DataFrame:
    cache = FEATURES / f"{rec.name}.features.csv"
    if cache.exists():
        return pd.read_csv(cache)
    from extract_cabin_features import extract

    print(f"Extracting face + hand features for {rec.name} ...", flush=True)
    return extract(rec / "rgb_face.mp4", cache, progress=False, hands=True)


def evaluate(rec: Path, cfg: dict) -> dict:
    ann = load_dmd(rec / "ann.json")
    df = features_for(rec)
    fps = len(df) / df["t"].iloc[-1]

    def feats(row):
        if not row.face or math.isnan(row.ear):
            return None
        return {c: getattr(row, c) for c in COLS}

    cal = Calibrator.from_config(cfg)
    for row in df.itertuples():
        cal.add(row.t, feats(row))
        if cal.done(row.t):
            break
    base = cal.result()
    mon = CabinMonitor(cfg, base)
    off = np.zeros(len(df), dtype=bool)
    events = []
    for i, row in enumerate(df.itertuples()):
        events += mon.update(row.t, feats(row))
        off[i] = mon.state.off_road

    h = cfg["head"]
    score = np.maximum((df["yaw"] - base.yaw).abs() / h["off_road_yaw_deg"],
                       (df["pitch"] - base.pitch).abs() / h["off_road_pitch_deg"]).to_numpy()

    n = min(len(df), ann.n_frames)
    gt_off = ann.mask("gaze_on_road/not_looking_road")[:n]
    labelled = (gt_off | ann.mask("gaze_on_road/looking_road")[:n])
    valid = labelled & df["face"].to_numpy()[:n]
    frame = frame_prf(off[:n], gt_off, valid)
    min_f = int(h["off_road_min_s"] * fps)
    gt_long = [s for s in ann.segments("gaze_on_road/not_looking_road") if s[1] - s[0] + 1 >= min_f]
    pred = [(round(e.start * fps), round(e.end * fps)) for e in events if e.type == "EYES_OFF_ROAD"]
    ev = event_prf(pred, gt_long, tol=int(0.5 * fps))
    session = rec.name.split("_")[2]
    return {
        "recording": rec.name,
        "session": session,
        "setup": ann.context.get("setup", ""),
        "face_detected": float(df["face"].mean()),
        "off_road_share_gt": float(gt_off[valid].mean()) if valid.any() else math.nan,
        "head_score_auc": roc_auc(score[:n][valid], gt_off[valid]),
        "frame_precision": frame["precision"],
        "frame_recall": frame["recall"],
        "frame_f1": frame["f1"],
        "event_recall_2s": ev["recall"],
        "event_precision_2s": ev["precision"],
        "events_gt_2s": ev["n_gt"],
        "events_pred": ev["n_pred"],
    }


def main() -> None:
    cfg = load_config("cabin")
    recs = sorted(p for p in INTERIM.iterdir() if (p / "ann.json").exists() and (p / "rgb_face.mp4").exists())
    if not recs:
        raise SystemExit("No prepared recordings. Run scripts/prepare_dmd.py --bundle distractions first.")
    df = pd.DataFrame([evaluate(r, cfg) for r in recs])
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "dmd_gaze_eval.csv"
    df.to_csv(out, index=False)
    cols = ["recording", "setup", "off_road_share_gt", "head_score_auc", "frame_precision", "frame_recall",
            "frame_f1", "event_recall_2s", "event_precision_2s", "events_gt_2s", "events_pred"]
    with pd.option_context("display.width", 250, "display.precision", 2):
        print(df[cols].to_string(index=False))
    print(f"\nSaved {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
