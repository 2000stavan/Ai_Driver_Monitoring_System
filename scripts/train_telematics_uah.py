"""Driving-behaviour classifier: UAH-DriveSet (labelled, Spain) -> applied to LEVIN (unlabelled, Pune).

    python scripts/train_telematics_uah.py

1. 60 s windows (10 s step) from all 40 UAH trips, labelled with the trip's simulated behaviour
   (normal / drowsy / aggressive).
2. XGBoost, leave-one-driver-out (6 folds), four feature tiers:
     A dynamics        braking / acceleration / jerk derived from 1 Hz speed (no absolute speed)
     B + phone motion  orientation-free dynamic acceleration magnitude
     C + speed level   mean / max / variability of speed
     D + UAH extras    lane offset, lane changes, car-following, speed vs. limit (UAH only)
   Scored per window (macro F1) and per trip (mean probability over its windows).
3. Tier B (portable, no absolute speed) is retrained on all UAH trips and applied to every LEVIN trip.
   LEVIN has no behaviour labels, so this is a profile, not a validation.

Writes reports/results/telematics_uah.md, reports/results/levin_behaviour.csv, models/telematics/uah_tierB.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from xgboost import XGBClassifier

from dms.telematics.features import MOTION_COLS, SPEED_COLS, MotionSeries, SpeedSeries, trip_windows
from dms.telematics.levin import load_vehicle
from dms.telematics.uah import BEHAVIOURS, UAHTrip, load_all
from dms.utils.config import PROJECT_ROOT

UAH = PROJECT_ROOT / "data/raw/telematics/uah_driveset"
LEVIN = PROJECT_ROOT / "data/raw/telematics/levin/csv"
OUT_MD = PROJECT_ROOT / "reports/results/telematics_uah.md"
OUT_LEVIN = PROJECT_ROOT / "reports/results/levin_behaviour.csv"
OUT_MODEL = PROJECT_ROOT / "models/telematics/uah_tierB.json"

DYNAMICS = [c for c in SPEED_COLS if not c.startswith("speed_") and c != "moving_frac"]
SPEED_LEVEL = ["speed_mean", "speed_std", "speed_max", "speed_cv", "moving_frac"]
EXTRAS = ["lane_abs_mean", "lane_std", "lane_valid_frac", "lane_changes_per_min", "veh_frac",
          "headway_min_s", "ttc_lt2_frac", "overspeed_frac", "speed_over_limit_mean"]
TIERS = {
    "A dynamics (speed-derived)": DYNAMICS,
    "B + phone motion (portable)": DYNAMICS + MOTION_COLS,
    "C + speed level": DYNAMICS + MOTION_COLS + SPEED_LEVEL,
    "D + UAH extras (lane, car-following, limit)": DYNAMICS + MOTION_COLS + SPEED_LEVEL + EXTRAS,
}
PORTABLE = "B + phone motion (portable)"


def uah_extras(trip: UAHTrip, speed: SpeedSeries):
    def f(t0: float, t1: float) -> dict:
        ln = trip.lane[(trip.lane[:, 0] >= t0) & (trip.lane[:, 0] < t1)]
        off = ln[:, 1][np.isfinite(ln[:, 1])]
        vh = trip.vehicle[(trip.vehicle[:, 0] >= t0) & (trip.vehicle[:, 0] < t1)]
        sp = np.interp(vh[:, 0], speed.t, speed.v) / 3.6 if len(vh) else np.array([])
        dist = vh[:, 1] if len(vh) else np.array([])
        headway = dist / np.maximum(sp, 1.0) if len(vh) else np.array([])
        ttc = vh[:, 2] if len(vh) else np.array([])
        m = (speed.t >= t0) & (speed.t < t1)
        lim = np.interp(speed.t[m], trip.osm[:, 0], trip.osm[:, 1]) if len(trip.osm) else np.full(m.sum(), np.nan)
        known = np.isfinite(lim)
        return {
            "lane_abs_mean": np.abs(off).mean() if len(off) else np.nan,
            "lane_std": off.std() if len(off) else np.nan,
            "lane_valid_frac": len(off) / max(len(ln), 1),
            "lane_changes_per_min": ((trip.lane_changes >= t0) & (trip.lane_changes < t1)).sum() * 60 / (t1 - t0),
            "veh_frac": np.isfinite(dist).mean() if len(dist) else 0.0,
            "headway_min_s": np.nanmin(headway) if np.isfinite(headway).any() else np.nan,
            "ttc_lt2_frac": (ttc[np.isfinite(ttc)] < 2).mean() if np.isfinite(ttc).any() else 0.0,
            "overspeed_frac": (speed.v[m][known] > lim[known] + 5).mean() if known.any() else np.nan,
            "speed_over_limit_mean": (speed.v[m][known] / lim[known]).mean() if known.any() else np.nan,
        }
    return f


def uah_windows() -> pd.DataFrame:
    parts = []
    for trip in load_all(UAH):
        speed = SpeedSeries(trip.gps[:, 0], trip.gps[:, 1])
        motion = MotionSeries(trip.acc[:, 0], trip.acc[:, 1:4])
        w = trip_windows(speed, motion, uah_extras(trip, speed))
        w.insert(0, "trip", trip.name)
        w.insert(1, "driver", trip.driver)
        w.insert(2, "road", trip.road)
        w.insert(3, "label", BEHAVIOURS.index(trip.behaviour))
        parts.append(w)
    return pd.concat(parts, ignore_index=True)


def model() -> XGBClassifier:
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss",
                         random_state=0, n_jobs=4)


def weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=3)
    return (len(y) / (3 * np.maximum(counts, 1)))[y]


def lodo(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Leave-one-driver-out window probabilities."""
    prob = np.zeros((len(df), 3))
    for d in sorted(df["driver"].unique()):
        te = df["driver"] == d
        m = model()
        y = df.loc[~te, "label"].to_numpy()
        m.fit(df.loc[~te, cols], y, sample_weight=weights(y))
        prob[te.to_numpy()] = m.predict_proba(df.loc[te, cols])
    return pd.DataFrame(prob, columns=BEHAVIOURS, index=df.index)


def scores(df: pd.DataFrame, prob: pd.DataFrame) -> dict:
    y, p = df["label"].to_numpy(), prob.to_numpy().argmax(1)
    trips = pd.concat([df[["trip", "label"]], prob], axis=1).groupby("trip").agg(
        {"label": "first", **{b: "mean" for b in BEHAVIOURS}})
    tp = trips[BEHAVIOURS].to_numpy().argmax(1)
    return {
        "window_macro_F1": f1_score(y, p, average="macro"),
        "window_bal_acc": balanced_accuracy_score(y, p),
        **{f"F1_{b.lower()}": f for b, f in zip(BEHAVIOURS, f1_score(y, p, average=None, labels=[0, 1, 2]))},
        "aggressive_vs_rest_AUC": roc_auc_score(y == 2, prob["AGGRESSIVE"]),
        "trip_accuracy": (tp == trips["label"].to_numpy()).mean(),
        "trips_correct": f"{int((tp == trips['label'].to_numpy()).sum())}/{len(trips)}",
    }


LEVIN_CACHE = PROJECT_ROOT / "data/processed/levin_windows.csv"


def levin_windows() -> pd.DataFrame:
    """All LEVIN windows (cached: reading the 21 vehicle CSVs takes ~25 min)."""
    if LEVIN_CACHE.exists():
        return pd.read_csv(LEVIN_CACHE, dtype={"vehicle": str, "trip": str})
    parts = []
    for path in sorted(LEVIN.glob("*.csv"), key=lambda p: int(p.stem)):
        for trip in load_vehicle(path):
            s = trip.seconds
            if (s["speed_kmh"] > 5).sum() < 300:  # at least 5 min of driving
                continue
            ok = ~np.isnan(trip.acc).any(axis=1)
            if ok.sum() < 1000:
                continue
            gravity = np.median(trip.acc[ok], axis=0)
            speed = SpeedSeries(s["t"].to_numpy(dtype=float), s["speed_kmh"].to_numpy(dtype=float))
            motion = MotionSeries(trip.acc_t[ok], trip.acc[ok] - gravity)
            w = trip_windows(speed, motion)
            w = w[w["moving_frac"] >= 0.5]
            if w.empty:
                continue
            w.insert(0, "vehicle", path.stem)
            w.insert(1, "trip", trip.trip_id)
            w.insert(2, "trip_km", float((s["speed_kmh"] / 3600).sum()))
            parts.append(w)
    df = pd.concat(parts, ignore_index=True)
    LEVIN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LEVIN_CACHE, index=False)
    return df


def quality_gate(lw: pd.DataFrame) -> pd.DataFrame:
    """Drop windows whose speed never changes while moving: a frozen logger value, not smooth driving
    (vehicles 4, 46, 23, 37 have this in most windows; it made them look 'normal')."""
    return lw[~((lw["speed_std"] == 0) & (lw["speed_mean"] > 5))]


def levin_profile(lw: pd.DataFrame, m: XGBClassifier, cols: list[str]) -> pd.DataFrame:
    w = lw.dropna(subset=cols)
    p = pd.DataFrame(m.predict_proba(w[cols]), columns=[f"p_{b.lower()}" for b in BEHAVIOURS], index=w.index)
    trips = pd.concat([w[["vehicle", "trip", "trip_km"]], p], axis=1).groupby(["vehicle", "trip"]).agg(
        km=("trip_km", "first"), windows=("trip_km", "size"), **{c: (c, "mean") for c in p.columns}).reset_index()
    trips["predicted"] = np.array(BEHAVIOURS)[trips[p.columns].to_numpy().argmax(1)]
    return trips


def main() -> None:
    df = uah_windows()
    print(f"UAH: {df['trip'].nunique()} trips, {df['driver'].nunique()} drivers, {len(df)} windows")
    results, cms = {}, {}
    for name, cols in TIERS.items():
        prob = lodo(df, cols)
        results[name] = scores(df, prob)
        cms[name] = confusion_matrix(df["label"], prob.to_numpy().argmax(1), labels=[0, 1, 2])
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in results[name].items()})
    table = pd.DataFrame(results).T

    # portable model on all UAH -> LEVIN
    cols = TIERS[PORTABLE]
    final = model()
    final.fit(df[cols], df["label"], sample_weight=weights(df["label"].to_numpy()))
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(OUT_MODEL)
    OUT_MODEL.with_suffix(".features.json").write_text(json.dumps(cols))
    imp = pd.Series(final.feature_importances_, index=cols).sort_values(ascending=False)

    lw_all = levin_windows()
    lw = quality_gate(lw_all)
    print(f"LEVIN windows: {len(lw_all)}, after frozen-speed gate: {len(lw)}")
    lev = levin_profile(lw, final, cols)
    lev.to_csv(OUT_LEVIN, index=False)
    # Where does the shift come from? Same check with the speed-only tier, and feature medians.
    shift_rows = {}
    for name in ("A dynamics (speed-derived)", PORTABLE):
        mt = model()
        mt.fit(df[TIERS[name]], df["label"], sample_weight=weights(df["label"].to_numpy()))
        shift_rows[name] = levin_profile(lw, mt, TIERS[name])["predicted"].value_counts(normalize=True)
    shift = pd.DataFrame(shift_rows).T.fillna(0)[BEHAVIOURS]
    feat_cmp = pd.DataFrame({
        "UAH normal (median)": df.loc[df["label"] == 0, cols].median(),
        "UAH aggressive (median)": df.loc[df["label"] == 2, cols].median(),
        "LEVIN Pune (median)": lw[cols].median(),
    })
    km_w = lambda g: pd.Series({f"p_{b.lower()}": np.average(g[f"p_{b.lower()}"], weights=g["km"]) for b in BEHAVIOURS})
    per_vehicle = lev.groupby("vehicle").apply(km_w, include_groups=False)
    per_vehicle.insert(0, "trips", lev.groupby("vehicle").size())
    per_vehicle.insert(1, "km", lev.groupby("vehicle")["km"].sum())
    feat_cmp = feat_cmp.loc[imp.index[:8]]
    levin_prof = pd.read_csv(PROJECT_ROOT / "reports/results/levin_trips.csv") if (PROJECT_ROOT / "reports/results/levin_trips.csv").exists() else None

    lines = ["# Driving behaviour: UAH-DriveSet (labelled) -> LEVIN Pune (unlabelled)", "",
             f"UAH-DriveSet: {df['trip'].nunique()} trips, 6 drivers, {len(df)} windows of 60 s (10 s step). "
             "Labels = behaviour the driver was asked to **simulate** (normal / drowsy / aggressive). "
             "XGBoost, class-balanced, **leave-one-driver-out**.", "",
             table.map(lambda v: round(v, 3) if isinstance(v, float) else v).to_markdown(), "",
             "Confusion matrices (rows = truth, cols = predicted; normal / drowsy / aggressive):", ""]
    for name, cm in cms.items():
        lines.append(f"- {name}: `{cm.tolist()}`")
    lines += ["", f"Top features of the portable model (tier B): " +
              ", ".join(f"{k} ({v:.2f})" for k, v in imp.head(8).items()), "",
              "## Applied to LEVIN (Pune, 21 vehicles) - profile, NOT a validation", "",
              f"{len(lev)} trips with at least 5 min of driving; windows with at least 50 % moving; "
              f"{len(lw_all) - len(lw)} of {len(lw_all)} windows dropped by the frozen-speed quality gate. "
              f"Trips predicted: {lev['predicted'].value_counts().to_dict()}.", "",
              "Mean probability per vehicle (km-weighted):", "",
              per_vehicle.round(3).to_markdown(), "",
              "### Is the 'aggressive' result about the drivers or the domain?", "",
              "Share of LEVIN trips per predicted class, speed-only model (A) vs portable model (B):", "",
              shift.round(3).to_markdown(), "",
              "Top portable-model features: UAH vs LEVIN medians:", "",
              feat_cmp.round(3).to_markdown(), ""]
    if levin_prof is not None and "vehicle" in levin_prof.columns:
        lines += ["(Compare with harsh-event rates in `levin_profile.md`.)", ""]
    lines += ["### Reading (2026-10-09)", "",
              "- **Frozen speed** in LEVIN: half of all windows (mostly vehicles 4, 46, 23, 37) have a logged speed that "
              "never changes while moving. Before the gate these looked 'normal'; they are a logger fault and are dropped.",
              "- **Speed dynamics**: typical Pune driving has the braking / acceleration / jerk levels of UAH's *simulated "
              "aggressive* driving (medians above). Stop-go Indian city traffic, not necessarily aggressive drivers.",
              "- **Phone motion does not transfer**: LEVIN dynamic acceleration is ~4x UAH's (rougher roads, a different "
              "logger with 1/64 g resolution). Tier B adds nothing for India.",
              "- So a model trained on Western (Spanish) drives labels ~80 % of Indian trips 'aggressive'. This is the "
              "telematics version of RQ1: Western thresholds and models do not transfer. What would: Indian labels "
              "(own Session B drives) and **driver- / fleet-relative scores** (personal baseline, innovation #4).", "",
              "## Caveats", "",
              "- UAH behaviours are **simulated** by 6 Spanish drivers on Spanish roads; 40 trips is small.",
              "- Tier B leaves out absolute speed on purpose: Indian traffic speeds differ so much from Spanish "
              "motorways that a speed-level feature would mostly measure the road, not the driver.",
              "- Phone motion includes road vibration; Indian roads are rougher, so LEVIN 'aggressive' "
              "probabilities may be inflated by road surface (an internship question: separate road from driver).",
              "- Drowsy driving shows mainly in lane keeping (tier D); a phone without a road camera sees little of it.",
              "- Indian validation needs labelled Indian drives: our controlled Session B drives (recording/PROTOCOL.md)."]
    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines[4:12]))
    print(per_vehicle.round(3).to_string())


if __name__ == "__main__":
    main()
