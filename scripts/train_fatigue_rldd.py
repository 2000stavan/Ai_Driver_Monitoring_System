"""Fatigue classifier: UTA-RLDD (labelled, self-recorded alert / low-vigilance / drowsy) -> INVEDRIFAC (Indian).

    python scripts/train_fatigue_rldd.py

Input: per-frame face features from notebooks/01_rldd_extract_features.ipynb, copied to
data/processed/rldd/<Fold>/<subject>_<label>.csv (15 fps).

1. 60 s windows (10 s step): PERCLOS, blink rate and duration, slow blinks, microsleeps, yawns, nods,
   eye openness relative to the personal baseline, head-pose variability (src/dms/cabin/windows.py).
2. Two calibrations (innovation #4, personal baseline):
     self      each video calibrated on its own first 60 s (no reference available)
     personal  every video of a driver calibrated on the first 60 s of that driver's ALERT video
3. XGBoost, leave-one-subject-out. Tasks: alert vs drowsy (binary) and alert / low vigilance / drowsy.
   A zero-training PERCLOS rule is the reference.
4. The self-calibrated binary model is applied to INVEDRIFAC (no labels): drowsiness score per driver
   and first vs last third of each drive.

Writes reports/results/fatigue_rldd.md and models/fatigue/rldd_binary_self.json
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from dms.cabin.windows import calibrate, window_features
from dms.utils.config import PROJECT_ROOT, load_config

RLDD = PROJECT_ROOT / "data/processed/rldd"
INVEDRIFAC = PROJECT_ROOT / "data/processed/invedrifac"
OUT_MD = PROJECT_ROOT / "reports/results/fatigue_rldd.md"
OUT_MODEL = PROJECT_ROOT / "models/fatigue/rldd_binary_self.json"
LABELS = {"alert": 0, "low_vigilance": 1, "drowsy": 2}
FEATURES = ["perclos", "blink_rate", "slow_blink_rate", "microsleeps", "closure_mean_s", "closure_max_s",
            "yawns", "nods", "ear_rel_mean", "ear_rel_std", "mar_mean", "pitch_rel_mean", "pitch_std",
            "yaw_std", "off_road_ratio"]
MIN_FACE = 0.5


def load_rldd() -> dict[tuple[str, str], pd.DataFrame]:
    recs = {}
    for f in sorted(RLDD.glob("*/*.csv")):
        m = re.match(r"^(\d+)_(alert|low_vigilance|drowsy)$", f.stem)
        if m:
            recs[(f"{f.parent.name}/{m.group(1)}", m.group(2))] = pd.read_csv(f)
    return recs


def build_windows(recs, cfg: dict, mode: str) -> pd.DataFrame:
    parts = []
    for (subject, label), df in recs.items():
        base = None
        if mode == "personal" and (subject, "alert") in recs:
            base = calibrate(recs[(subject, "alert")], cfg)
        try:
            w = window_features(df, cfg, baseline=base)
        except RuntimeError:  # no face to calibrate on
            continue
        w.insert(0, "subject", subject)
        w.insert(1, "label", LABELS[label])
        parts.append(w)
    out = pd.concat(parts, ignore_index=True)
    return out[out["face_ratio"] >= MIN_FACE]


def model(n_classes: int) -> XGBClassifier:
    kw = dict(objective="binary:logistic", eval_metric="logloss") if n_classes == 2 else \
        dict(objective="multi:softprob", eval_metric="mlogloss")
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, random_state=0, n_jobs=4, **kw)


def balanced(y: np.ndarray) -> np.ndarray:
    k = len(np.unique(y))
    counts = np.bincount(y, minlength=k)
    return (len(y) / (k * np.maximum(counts, 1)))[y]


def loso(w: pd.DataFrame, y: np.ndarray, n_classes: int) -> np.ndarray:
    prob = np.zeros((len(w), n_classes))
    for s in w["subject"].unique():
        te = (w["subject"] == s).to_numpy()
        m = model(n_classes)
        m.fit(w.loc[~te, FEATURES], y[~te], sample_weight=balanced(y[~te]))
        prob[te] = m.predict_proba(w.loc[te, FEATURES])
    return prob


def score(w: pd.DataFrame, y: np.ndarray, prob: np.ndarray) -> dict:
    pred = prob.argmax(1)
    vid = pd.DataFrame(prob).assign(key=(w["subject"] + "/" + w["label"].astype(str)).to_numpy(), y=y)
    vid = vid.groupby("key").mean()
    vpred = vid.drop(columns="y").to_numpy().argmax(1)
    out = {"window_macro_F1": f1_score(y, pred, average="macro"),
           "window_bal_acc": balanced_accuracy_score(y, pred),
           "video_accuracy": float((vpred == vid["y"].to_numpy()).mean()),
           "videos_correct": f"{int((vpred == vid['y'].to_numpy()).sum())}/{len(vid)}"}
    if prob.shape[1] == 2:
        out["window_AUC"] = roc_auc_score(y, prob[:, 1])
        out["video_AUC"] = roc_auc_score(vid["y"], vid[1])
    return out


def video_probs(w: pd.DataFrame, y: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"subject": w["subject"].to_numpy(), "y": y, "p": prob[:, 1]}).groupby(
        ["subject", "y"])["p"].mean().reset_index()


def bootstrap_gain(a: pd.DataFrame, b: pd.DataFrame, n: int = 2000, seed: int = 0) -> dict:
    """Resample subjects; AUC of a and b on the same resample. Returns 95 % intervals and P(b > a)."""
    rng = np.random.default_rng(seed)
    subs = a["subject"].unique()
    ra, rb = [], []
    for _ in range(n):
        pick = rng.choice(subs, len(subs))
        da = pd.concat([a[a["subject"] == s] for s in pick])
        db = pd.concat([b[b["subject"] == s] for s in pick])
        if da["y"].nunique() < 2:
            continue
        ra.append(roc_auc_score(da["y"], da["p"]))
        rb.append(roc_auc_score(db["y"], db["p"]))
    ra, rb = np.array(ra), np.array(rb)
    return {"a_ci": np.percentile(ra, [2.5, 97.5]), "b_ci": np.percentile(rb, [2.5, 97.5]),
            "gain_ci": np.percentile(rb - ra, [2.5, 97.5]), "p_gain": float((rb > ra).mean())}


def perclos_rule(w: pd.DataFrame, y: np.ndarray, threshold: float) -> dict:
    p = w["perclos"].fillna(0).to_numpy()
    prob = np.c_[1 - p, p]
    s = score(w, y, np.c_[(p < threshold), (p >= threshold)].astype(float))
    s["window_AUC"] = roc_auc_score(y, p)
    s["video_AUC"] = roc_auc_score(*pd.DataFrame({"y": y, "p": prob[:, 1], "k": (w["subject"] + w["label"].astype(str)).to_numpy()})
                                   .groupby("k").mean()[["y", "p"]].to_numpy().T)
    return s


def invedrifac_profile(m: XGBClassifier, cfg: dict) -> pd.DataFrame:
    rows = []
    for f in sorted(INVEDRIFAC.glob("*.features.csv"), key=lambda p: int(re.findall(r"\d+", p.name)[0])):
        df = pd.read_csv(f)
        try:
            w = window_features(df, cfg)
        except RuntimeError:
            continue
        w = w[w["face_ratio"] >= MIN_FACE] if len(w) else w  # empty for drives shorter than one window
        if len(w) < 3:
            rows.append({"driver": f.name.split(".")[0], "minutes": df["t"].iloc[-1] / 60, "windows": len(w)})
            continue
        p = m.predict_proba(w[FEATURES])[:, 1]
        third = max(len(p) // 3, 1)
        rows.append({"driver": f.name.split(".")[0], "minutes": df["t"].iloc[-1] / 60, "windows": len(w),
                     "p_drowsy_mean": p.mean(), "p_drowsy_first_third": p[:third].mean(),
                     "p_drowsy_last_third": p[-third:].mean(), "perclos_mean": w["perclos"].mean()})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config("cabin")
    recs = load_rldd()
    subjects = sorted({s for s, _ in recs})
    print(f"RLDD: {len(recs)} videos, {len(subjects)} subjects")
    results, wins, vids = {}, {}, {}
    for mode in ("self", "personal"):
        w = build_windows(recs, cfg, mode)
        wins[mode] = w
        yb_mask = w["label"] != 1
        wb = w[yb_mask].reset_index(drop=True)
        yb = (wb["label"] == 2).astype(int).to_numpy()
        if mode == "self":
            results["PERCLOS rule (>= 0.15), no training | alert vs drowsy"] = perclos_rule(wb, yb, 0.15)
        pb = loso(wb, yb, 2)
        vids[mode] = video_probs(wb, yb, pb)
        results[f"XGBoost, {mode} calibration | alert vs drowsy"] = score(wb, yb, pb)
        y3 = w["label"].to_numpy()
        results[f"XGBoost, {mode} calibration | 3 classes"] = score(w.reset_index(drop=True), y3,
                                                                    loso(w.reset_index(drop=True), y3, 3))
        for k in [k for k in results if mode in k or "rule" in k]:
            print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in results[k].items()})
    table = pd.DataFrame(results).T
    boot = bootstrap_gain(vids["self"], vids["personal"])

    # final binary model (self calibration: INVEDRIFAC has no alert reference drive)
    w = wins["self"]
    wb = w[w["label"] != 1]
    yb = (wb["label"] == 2).astype(int).to_numpy()
    final = model(2)
    final.fit(wb[FEATURES], yb, sample_weight=balanced(yb))
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(OUT_MODEL)
    OUT_MODEL.with_suffix(".features.json").write_text(json.dumps(FEATURES))
    imp = pd.Series(final.feature_importances_, index=FEATURES).sort_values(ascending=False)
    inv = invedrifac_profile(final, cfg)

    per_label = w.groupby("label")[["perclos", "blink_rate", "closure_mean_s", "ear_rel_mean"]].median()
    per_label.index = [k for k, v in sorted(LABELS.items(), key=lambda kv: kv[1])]
    valid = inv.dropna(subset=["p_drowsy_mean"]) if "p_drowsy_mean" in inv else inv
    rising = int((valid["p_drowsy_last_third"] > valid["p_drowsy_first_third"]).sum()) if len(valid) else 0
    lines = ["# Fatigue classifier: UTA-RLDD -> INVEDRIFAC (Indian)", "",
             f"UTA-RLDD: {len(recs)} videos, {len(subjects)} subjects (fold(s): "
             f"{', '.join(sorted({s.split('/')[0] for s in subjects}))}); 60 s windows, 10 s step; windows with a face in "
             f"< {MIN_FACE:.0%} of frames dropped. Labels are the subject's own state for the whole video "
             "(self-recorded at home, not driving). XGBoost, class-balanced, **leave-one-subject-out**.", "",
             table.map(lambda v: round(v, 3) if isinstance(v, float) else v).to_markdown(), "",
             f"**Personal baseline vs self calibration** (alert vs drowsy, video AUC, 2,000 bootstrap resamples of "
             f"subjects): self 95 % CI {boot['a_ci'][0]:.2f}-{boot['a_ci'][1]:.2f}, personal "
             f"{boot['b_ci'][0]:.2f}-{boot['b_ci'][1]:.2f}; gain {boot['gain_ci'][0]:+.3f} to {boot['gain_ci'][1]:+.3f}, "
             f"personal better in {boot['p_gain']:.1%} of resamples.", "",
             "Median window features per label (self calibration):", "",
             per_label.round(3).to_markdown(), "",
             "Top features (final binary model): " + ", ".join(f"{k} ({v:.2f})" for k, v in imp.head(6).items()), "",
             "## INVEDRIFAC (Indian drivers, no labels) - profile, NOT a validation", "",
             "Self-calibrated binary model; `p_drowsy` = mean predicted probability of drowsy over the windows.", "",
             inv.round(3).to_markdown(index=False), "",
             f"Drowsiness score higher in the last third than the first third for {rising} of {len(valid)} drivers.", "",
             "## Caveats", "",
             "- RLDD labels are per video and self-reported; low vigilance vs drowsy is a fuzzy line.",
             "- RLDD is recorded with laptops / phones at home, not in a car; INVEDRIFAC is in-car, so this is a "
             "cross-domain transfer. Indian numbers need the INVEDRIFAC labels (scripts/label_tool.py).",
             "- Only the folds downloaded so far are used; more folds (notebook, one fold per session) = more subjects."]
    OUT_MD.write_text("\n".join(lines))
    print(table.round(3).to_string())
    print(inv.round(3).to_string())


if __name__ == "__main__":
    main()
