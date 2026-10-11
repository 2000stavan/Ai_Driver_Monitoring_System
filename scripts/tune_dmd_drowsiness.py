"""Subject-wise threshold tuning for the eye and yawn detectors on DMD drowsiness.

Leave-one-subject-out: for each subject, pick the thresholds that work best on the *other*
subjects, then measure on the held-out subject. The reported numbers are therefore never
tuned on the data they are measured on (INITIALPLAN §12.2).

    python scripts/tune_dmd_drowsiness.py

Writes reports/results/dmd_drowsiness_tuning.csv and dmd_drowsiness_tuned.md
"""

from __future__ import annotations

import copy
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import eval_dmd_drowsiness as ev  # noqa: E402
from dms.utils.config import PROJECT_ROOT, load_config  # noqa: E402

EYE_GRID = {
    ("eyes", "closed_ear_ratio"): [0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
    ("eyes", "closed_blink_score"): [0.25, 0.3, 0.35, 0.4, 0.45, 0.55],
}
YAWN_GRID = {
    ("mouth", "yawn_mar"): [0.2, 0.25, 0.3, 0.35, 0.45],
    ("mouth", "yawn_jaw_score"): [0.2, 0.25, 0.3, 0.35, 0.45],
    ("mouth", "hand_mouth_dist"): [0.1, 0.15, 0.2, 0.3],
    ("mouth", "yawn_hand_min_s"): [1.5, 2.0, 2.5],
}


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r and not (math.isnan(p) or math.isnan(r)) else 0.0


def configs(base: dict, grid: dict):
    keys = list(grid)
    for values in itertools.product(*grid.values()):
        cfg = copy.deepcopy(base)
        for (section, key), v in zip(keys, values):
            cfg[section][key] = v
        yield dict(zip([f"{s}.{k}" for s, k in keys], values)), cfg


def subject(rec_name: str) -> str:
    return "_".join(rec_name.split("_")[:2])  # e.g. gB_10


def score_all(recs, base, grid, name):
    rows = []
    combos = list(configs(base, grid))
    for i, (params, cfg) in enumerate(combos, 1):
        for r in recs:
            m = ev.evaluate(r, cfg)
            rows.append({"grid": name, **params, "recording": r.name, "subject": subject(r.name),
                         "eyes_f1": m["eyes_closed_f1"],
                         "blink_f1": f1(m["blink_precision"], m["blink_recall"]),
                         "yawn_f1": f1(m["yawn_precision"], m["yawn_recall"]),
                         "yawn_recall": m["yawn_recall"], "yawn_precision": m["yawn_precision"],
                         "blink_recall": m["blink_recall"], "blink_precision": m["blink_precision"]})
        print(f"{name}: {i}/{len(combos)} configs", flush=True)
    return pd.DataFrame(rows)


def loso(df: pd.DataFrame, param_cols: list[str], objective: str) -> pd.DataFrame:
    """For each held-out subject, choose params by mean objective over the other subjects."""
    out = []
    for held in sorted(df["subject"].unique()):
        train = df[df["subject"] != held]
        best = train.groupby(param_cols)[objective].mean().idxmax()
        best = best if isinstance(best, tuple) else (best,)
        sel = df[(df["subject"] == held)]
        for c, v in zip(param_cols, best):
            sel = sel[sel[c] == v]
        for row in sel.to_dict("records"):
            out.append({**row, "held_out": held})
    return pd.DataFrame(out)


def main() -> None:
    base = load_config("cabin")
    recs = sorted(p for p in ev.INTERIM.iterdir() if (p / "ann.json").exists())
    eye_df = score_all(recs, base, EYE_GRID, "eyes")
    eye_df["eye_objective"] = (eye_df["eyes_f1"] + eye_df["blink_f1"]) / 2
    yawn_df = score_all(recs, base, YAWN_GRID, "yawn")

    eye_cols = [f"{s}.{k}" for s, k in EYE_GRID]
    yawn_cols = [f"{s}.{k}" for s, k in YAWN_GRID]
    eye_cv = loso(eye_df, eye_cols, "eye_objective")
    yawn_cv = loso(yawn_df, yawn_cols, "yawn_f1")

    out_dir = PROJECT_ROOT / "reports/results"
    pd.concat([eye_df, yawn_df]).to_csv(out_dir / "dmd_drowsiness_tuning.csv", index=False)

    default_eye = eye_df
    for c in eye_cols:
        default_eye = default_eye[default_eye[c] == base[c.split(".")[0]][c.split(".")[1]]]
    default_yawn = yawn_df
    for c in yawn_cols:
        default_yawn = default_yawn[default_yawn[c] == base[c.split(".")[0]][c.split(".")[1]]]

    eye_choice = eye_cv.groupby(eye_cols).size().sort_values(ascending=False)
    yawn_choice = yawn_cv.groupby(yawn_cols).size().sort_values(ascending=False)
    lines = [
        "# DMD drowsiness — subject-wise tuned detectors (leave-one-subject-out)",
        "",
        f"{eye_df['subject'].nunique()} subjects, {len(recs)} recordings. For each held-out subject, thresholds "
        "were chosen on the other subjects only. Means over held-out recordings.",
        "",
        "| Detector | Untuned defaults | Tuned (held-out subjects) |",
        "|---|---|---|",
        f"| Eyes closed, frame F1 | {default_eye['eyes_f1'].mean():.2f} | {eye_cv['eyes_f1'].mean():.2f} |",
        f"| Blinks, event F1 | {default_eye['blink_f1'].mean():.2f} | {eye_cv['blink_f1'].mean():.2f} |",
        f"| Yawns, event F1 | {default_yawn['yawn_f1'].mean():.2f} | {yawn_cv['yawn_f1'].mean():.2f} |",
        f"| Yawns, recall / precision | {default_yawn['yawn_recall'].mean():.2f} / {default_yawn['yawn_precision'].mean():.2f} "
        f"| {yawn_cv['yawn_recall'].mean():.2f} / {yawn_cv['yawn_precision'].mean():.2f} |",
        "",
        "Thresholds chosen across folds (how many held-out recordings used each setting):",
        "",
        "Eyes: " + "; ".join(f"{dict(zip(eye_cols, k if isinstance(k, tuple) else (k,)))} × {n}" for k, n in eye_choice.items()),
        "",
        "Yawn: " + "; ".join(f"{dict(zip(yawn_cols, k if isinstance(k, tuple) else (k,)))} × {n}" for k, n in yawn_choice.items()),
        "",
        "Caveat: DMD drowsiness is acted and non-Indian; this tunes the event detectors only.",
    ]
    (out_dir / "dmd_drowsiness_tuned.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
