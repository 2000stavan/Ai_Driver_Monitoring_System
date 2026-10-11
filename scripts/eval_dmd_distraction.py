"""Phone call / texting / drinking detection on DMD distraction (face camera = dashboard-phone view).

Per 1 s window (step 0.5 s) we combine:
  objects  : YOLO11 (COCO, pretrained) on the face camera: phone / drink / food confidence, position, size
  hands    : MediaPipe hand-to-ear / hand-to-mouth distances (face widths), hands visible
  head     : head pitch / yaw deviation from the personal neutral (texting -> looking down)
Labels: DMD `driver_actions` -> safe / phone_call / texting / drinking / other (majority in window).

Two detectors, both evaluated leave-one-subject-out:
  rules    : zero-shot thresholds, no training
  xgboost  : learned on the other subjects; feature ablation hands / objects / all

    python scripts/eval_dmd_distraction.py
Writes reports/results/dmd_distraction_eval.csv and dmd_distraction_eval.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dms.eval.dmd import load_dmd
from dms.utils.config import PROJECT_ROOT

INTERIM = PROJECT_ROOT / "data/interim/dmd/distractions"
FEATS = PROJECT_ROOT / "data/processed/dmd_distraction"
OBJS = PROJECT_ROOT / "data/processed/dmd_distraction_objects"
OUT = PROJECT_ROOT / "reports/results"
CLASSES = ["safe", "phone_call", "texting", "drinking", "other"]
ACTION_MAP = {
    "driver_actions/safe_drive": "safe",
    "driver_actions/phonecall_right": "phone_call", "driver_actions/phonecall_left": "phone_call",
    "driver_actions/texting_right": "texting", "driver_actions/texting_left": "texting",
    "driver_actions/drinking": "drinking",
}
WIN_S, STEP_S, CALIB_S = 1.0, 0.5, 10.0
OBJ_COLS = [f"{g}_{k}" for g in ("phone", "drink", "food") for k in ("conf", "cx", "cy", "area")]


def frame_labels(ann, n: int) -> np.ndarray:
    lab = np.full(n, "", dtype=object)
    for t, arr in ann.labels.items():
        if t.startswith("driver_actions/"):
            lab[arr[:n] & (lab == "")] = ACTION_MAP.get(t, "other")
    return lab


def windows(rec: Path) -> pd.DataFrame | None:
    fpath, opath = FEATS / f"{rec.name}.features.csv", OBJS / f"{rec.name}.face.objects.csv"
    if not (fpath.exists() and opath.exists()):
        return None
    ann = load_dmd(rec / "ann.json")
    f = pd.read_csv(fpath)
    n = min(len(f), ann.n_frames)
    f = f.iloc[:n].copy()
    fps = n / f["t"].iloc[n - 1]
    o = pd.read_csv(opath).sort_values("frame")
    f = pd.merge_asof(f.sort_values("frame"), o.drop(columns=["t"]), on="frame", direction="backward",
                      tolerance=int(0.25 * fps))
    calib = (f["t"] < CALIB_S) & f["face"].astype(bool)
    f["pitch_dev"] = f["pitch"] - f.loc[calib, "pitch"].median()
    f["yaw_dev"] = (f["yaw"] - f.loc[calib, "yaw"].median()).abs()
    f["label"] = frame_labels(ann, n)
    win, step = int(WIN_S * fps), int(STEP_S * fps)
    rows = []
    for s in range(0, n - win, step):
        w = f.iloc[s:s + win]
        labs = w["label"][w["label"] != ""]
        if len(labs) < 0.5 * win:
            continue
        r = {"recording": rec.name, "subject": "_".join(rec.name.split("_")[:2]),
             "session": rec.name.split("_")[2], "t": s / fps, "label": labs.mode().iloc[0]}
        for c in ("phone_conf", "drink_conf", "food_conf"):
            r[f"{c}_max"], r[f"{c}_mean"] = w[c].max(), w[c].mean()
        for c in ("phone_cx", "phone_cy", "phone_area", "drink_cy", "drink_area"):
            r[f"{c}_mean"] = w[c].mean()
        r["hand_ear_min"], r["hand_mouth_min"] = w["hand_ear"].min(), w["hand_mouth"].min()
        r["hand_near_ear"] = (w["hand_ear"] < 0.5).mean()
        r["hand_near_mouth"] = (w["hand_mouth"] < 0.3).mean()
        r["hands_mean"] = w["n_hands"].mean()
        r["pitch_dev_mean"], r["yaw_dev_mean"] = w["pitch_dev"].mean(), w["yaw_dev"].mean()
        r["mar_mean"], r["face_ratio"] = w["mar"].mean(), w["face"].mean()
        rows.append(r)
    return pd.DataFrame(rows)


FEATURE_SETS = {
    "hands": ["hand_ear_min", "hand_mouth_min", "hand_near_ear", "hand_near_mouth", "hands_mean",
              "pitch_dev_mean", "yaw_dev_mean", "mar_mean", "face_ratio"],
    "objects": ["phone_conf_max", "phone_conf_mean", "drink_conf_max", "drink_conf_mean", "food_conf_max",
                "food_conf_mean", "phone_cx_mean", "phone_cy_mean", "phone_area_mean", "drink_cy_mean",
                "drink_area_mean"],
}
FEATURE_SETS["all"] = FEATURE_SETS["hands"] + FEATURE_SETS["objects"]


def rules(df: pd.DataFrame, conf: float = 0.3) -> np.ndarray:
    pred = np.full(len(df), "safe", dtype=object)
    phone = df["phone_conf_max"] > conf
    pred[phone & (df["hand_ear_min"] < 0.5)] = "phone_call"
    pred[phone & ~(df["hand_ear_min"] < 0.5)] = "texting"
    pred[(df["drink_conf_max"] > conf) & ~phone] = "drinking"
    return pred


def per_class(y: np.ndarray, p: np.ndarray) -> dict:
    out = {}
    for c in CLASSES:
        tp, fp, fn = np.sum((p == c) & (y == c)), np.sum((p == c) & (y != c)), np.sum((p != c) & (y == c))
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        out[c] = {"precision": prec, "recall": rec,
                  "f1": 2 * prec * rec / (prec + rec) if prec + rec and not np.isnan(prec + rec) else 0.0,
                  "support": int(np.sum(y == c))}
    return out


def main() -> None:
    from xgboost import XGBClassifier

    parts = [w for w in (windows(r) for r in sorted(INTERIM.iterdir())) if w is not None and len(w)]
    if not parts:
        raise SystemExit("No recordings with both face features and face objects yet.")
    data = pd.concat(parts, ignore_index=True)
    subjects = sorted(data["subject"].unique())
    y = data["label"].to_numpy()
    results = {"rules (zero-shot)": rules(data)}
    for name, cols in FEATURE_SETS.items():
        pred = np.empty(len(data), dtype=object)
        for held in subjects:
            tr, te = data["subject"] != held, data["subject"] == held
            classes_tr = [c for c in CLASSES if (y[tr] == c).any()]
            enc = {c: i for i, c in enumerate(classes_tr)}
            ytr = np.array([enc[c] for c in y[tr]])
            counts = np.bincount(ytr, minlength=len(classes_tr))
            weights = (len(ytr) / (len(classes_tr) * np.maximum(counts, 1)))[ytr]  # balance classes
            m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.8,
                              colsample_bytree=0.8, tree_method="hist", n_jobs=4, verbosity=0)
            m.fit(data.loc[tr, cols], ytr, sample_weight=weights)
            pred[te.to_numpy()] = np.array(classes_tr)[m.predict(data.loc[te, cols])]
        results[f"xgboost ({name})"] = pred

    rows = []
    for name, pred in results.items():
        pc = per_class(y, pred)
        rows.append({"detector": name, **{f"{c}_f1": pc[c]["f1"] for c in CLASSES},
                     **{f"{c}_recall": pc[c]["recall"] for c in CLASSES},
                     **{f"{c}_precision": pc[c]["precision"] for c in CLASSES},
                     "macro_f1_target": np.mean([pc[c]["f1"] for c in ("phone_call", "texting", "drinking")])})
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "dmd_distraction_eval.csv", index=False)
    support = {c: int((y == c).sum()) for c in CLASSES}
    best = max(results, key=lambda k: res.set_index("detector").loc[k, "macro_f1_target"])
    cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(results[best], name="predicted"))
    show = ["detector", "phone_call_f1", "texting_f1", "drinking_f1", "safe_f1", "macro_f1_target"]
    md = [
        "# DMD distraction — phone call / texting / drinking (face camera, leave-one-subject-out)",
        "",
        f"{data['recording'].nunique()} recordings, {len(subjects)} subjects, {len(data)} one-second windows. "
        f"Support: {support}",
        "",
        res[show].round(2).to_markdown(index=False),
        "",
        f"Confusion matrix, best detector ({best}):",
        "",
        cm.to_markdown(),
    ]
    (OUT / "dmd_distraction_eval.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
