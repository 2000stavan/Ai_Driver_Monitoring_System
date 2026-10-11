"""Eyes-off-road: head pose vs. L2CS-Net gaze vs. both, on DMD distraction `gaze_on_road` labels.

Same procedure for every signal, so the comparison is fair:
  deviation from the driver's own neutral (median of the first 10 s) -> angular deviation (deg)
  -> AUC (threshold-free), and events = deviation > threshold for >= min duration, with the
  threshold / duration chosen leave-one-subject-out (never on the subject being scored).

Needs data/processed/dmd_distraction/*.features.csv (head pose, every frame) and
data/processed/dmd_distraction_gaze/*.gaze.csv (L2CS gaze, ~10 fps).

    python scripts/eval_dmd_offroad.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from dms.eval.dmd import load_dmd
from dms.eval.metrics import event_prf, frame_prf, roc_auc
from dms.utils.config import PROJECT_ROOT

INTERIM = PROJECT_ROOT / "data/interim/dmd/distractions"
FEATS = PROJECT_ROOT / "data/processed/dmd_distraction"
GAZE = PROJECT_ROOT / "data/processed/dmd_distraction_gaze"
OUT = PROJECT_ROOT / "reports/results"
CALIB_S = 10.0
MIN_GT_S = 2.0
THRESHOLDS = {"head": [10, 15, 20, 25, 30], "gaze": [8, 10, 12, 15, 20, 25], "head+gaze": [0.6, 0.8, 1.0, 1.2, 1.5]}
MIN_DURS = [0.5, 1.0, 1.5, 2.0]
GAP_S = 0.5  # dips shorter than this do not split an event


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))


def load(rec: Path) -> dict | None:
    gpath = GAZE / f"{rec.name}.gaze.csv"
    fpath = FEATS / f"{rec.name}.features.csv"
    if not gpath.exists() or not fpath.exists():
        return None
    ann = load_dmd(rec / "ann.json")
    f = pd.read_csv(fpath)
    g = pd.read_csv(gpath)
    n = min(len(f), ann.n_frames)
    f = f.iloc[:n]
    fps = n / f["t"].iloc[n - 1]
    # gaze at ~10 fps -> carry forward to every frame (at most 0.3 s old)
    g = g.sort_values("frame")
    full = pd.merge_asof(f[["frame"]], g[["frame", "gaze_yaw", "gaze_pitch"]], on="frame",
                         direction="backward", tolerance=int(0.3 * fps))
    calib = f["t"] < CALIB_S
    def dev(yaw, pitch):
        by, bp = np.nanmedian(yaw[calib]), np.nanmedian(pitch[calib])
        return np.hypot(yaw - by, pitch - bp)
    head = dev(f["yaw"].to_numpy(), f["pitch"].to_numpy())
    gaze = dev(full["gaze_yaw"].to_numpy(), full["gaze_pitch"].to_numpy())
    combo = np.fmax(head / 25.0, gaze / 15.0)  # each scaled by a typical "looking away" angle
    gt = ann.mask("gaze_on_road/not_looking_road")[:n]
    labelled = (gt | ann.mask("gaze_on_road/looking_road")[:n]) & f["face"].to_numpy(dtype=bool)
    return {"rec": rec.name, "subject": "_".join(rec.name.split("_")[:2]), "setup": ann.context.get("setup", ""),
            "fps": fps, "gt": gt, "valid": labelled,
            "gt_events": [s for s in runs(gt) if (s[1] - s[0] + 1) / fps >= MIN_GT_S],
            "signals": {"head": head, "gaze": gaze, "head+gaze": combo}}


def close_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Merge 'above' runs separated by short dips (debounce), so one glance = one event."""
    out = mask.copy()
    r = runs(mask)
    for (s0, e0), (s1, _) in zip(r, r[1:]):
        if s1 - e0 - 1 <= max_gap:
            out[e0 + 1 : s1] = True
    return out


def score(r: dict, sig: str, thr: float, min_s: float) -> dict:
    x = np.nan_to_num(r["signals"][sig], nan=0.0)
    above = close_gaps(x > thr, int(GAP_S * r["fps"]))
    pred = [s for s in runs(above) if (s[1] - s[0] + 1) / r["fps"] >= min_s]
    ev = event_prf(pred, r["gt_events"], tol=int(0.5 * r["fps"]))
    fr = frame_prf(above, r["gt"], r["valid"])
    f1 = ev["f1"] if not np.isnan(ev["f1"]) else 0.0
    return {"event_recall": ev["recall"], "event_precision": ev["precision"], "event_f1": f1, "frame_f1": fr["f1"]}


def main() -> None:
    recs = [x for x in (load(p) for p in sorted(INTERIM.iterdir())) if x is not None]
    if not recs:
        raise SystemExit("No recordings with both features and gaze yet.")
    subjects = sorted({r["subject"] for r in recs})
    rows = []
    for sig, grid in THRESHOLDS.items():
        table = {}
        for thr, md in itertools.product(grid, MIN_DURS):
            table[(thr, md)] = [score(r, sig, thr, md) for r in recs]
        for held in subjects:
            train = [i for i, r in enumerate(recs) if r["subject"] != held]
            best = max(table, key=lambda k: np.mean([table[k][i]["event_f1"] for i in train]))
            for i, r in enumerate(recs):
                if r["subject"] == held:
                    valid = r["valid"]
                    auc = roc_auc(r["signals"][sig][valid], r["gt"][valid])
                    rows.append({"signal": sig, "recording": r["rec"], "subject": held, "setup": r["setup"],
                                 "threshold": best[0], "min_s": best[1], "auc": auc, **table[best][i]})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "dmd_offroad_eval.csv", index=False)
    cols = ["auc", "frame_f1", "event_recall", "event_precision", "event_f1"]
    summary = df.groupby("signal")[cols].mean().round(2)
    by_setup = df.groupby(["signal", "setup"])[cols].mean().round(2)
    print(f"{len(recs)} recordings, {len(subjects)} subjects (leave-one-subject-out thresholds)\n")
    print(summary.to_string())
    print()
    print(by_setup.to_string())
    print("\nThresholds chosen:", df.groupby("signal")[["threshold", "min_s"]].agg(lambda s: s.mode().iloc[0]).to_dict("index"))


if __name__ == "__main__":
    main()
