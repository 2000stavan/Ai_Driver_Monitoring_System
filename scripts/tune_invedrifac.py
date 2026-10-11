"""Indian adaptation of the eye-closure detector (RQ2), leave-one-driver-out on our INVEDRIFAC labels.

    python scripts/tune_invedrifac.py

"Before" = default thresholds (set on Western data / DMD). "After" = thresholds chosen on the other
Indian drivers, applied to the held-out driver (repeated for every driver), so no driver's own labels
are used to tune its own score. Also reported separately for "clear" drivers (< 30 % of clips unclear).

Writes reports/results/invedrifac_adaptation.md
"""

from __future__ import annotations

import copy
import itertools

import numpy as np
import pandas as pd

from dms.cabin.windows import run_monitor
from dms.utils.config import PROJECT_ROOT, load_config

TASKS = PROJECT_ROOT / "data/labels/invedrifac_tasks.csv"
LABELS = PROJECT_ROOT / "data/labels/invedrifac_labels.csv"
FEATS = PROJECT_ROOT / "data/processed/invedrifac"
OUT = PROJECT_ROOT / "reports/results/invedrifac_adaptation.md"
EYE = {"blink", "long_closure"}
EYE_TYPES = {"BLINK", "SLOW_BLINK", "MICROSLEEP"}
GRID = {"closed_ear_ratio": [0.6, 0.65, 0.7, 0.75, 0.8, 0.85], "closed_blink_score": [0.3, 0.4, 0.5, 0.55, 0.65]}


def overlaps(a0, a1, b0, b1, tol=0.1) -> bool:
    return a0 - tol <= b1 and b0 - tol <= a1


def counts(df: pd.DataFrame, t: pd.DataFrame, cfg: dict) -> tuple[int, int, int, int]:
    """(true events, detected true events, detections, correct detections) in the labelled segment."""
    seg0, seg1 = t["t0"].min(), t["t1"].max()
    _, events, _ = run_monitor(df[df["t"] <= seg1 + 2], cfg)
    dets = [e for e in events if e.type in EYE_TYPES and seg0 <= e.start <= seg1]
    clear = t[t["label"] != "unclear"]
    truth = clear[(clear["kind"] == "candidate") & clear["label"].isin(EYE)]
    # detections inside clips the labeller could not judge are not counted either way
    unclear = t[t["label"] == "unclear"]
    dets = [e for e in dets if not any(overlaps(r.t0, r.t1, e.start, e.end, 0) for r in unclear.itertuples())]
    hit = sum(any(overlaps(r.event_start, r.event_end, e.start, e.end) for e in dets) for r in truth.itertuples())
    ok = sum(any(overlaps(r.event_start, r.event_end, e.start, e.end) for r in truth.itertuples()) for e in dets)
    return len(truth), hit, len(dets), ok


def prf(c) -> tuple[float, float, float]:
    n, hit, d, ok = np.sum(c, axis=0)
    r, p = hit / max(n, 1), ok / max(d, 1)
    return r, p, 2 * p * r / max(p + r, 1e-9)


def main() -> None:
    lab = pd.read_csv(TASKS).merge(pd.read_csv(LABELS), on="task_id")
    base = load_config("cabin")
    combos = list(itertools.product(*GRID.values()))
    drivers = sorted(lab["driver"].unique())
    feats = {d: pd.read_csv(FEATS / f"{d}.features.csv") for d in drivers}
    unclear_frac = lab.groupby("driver")["label"].apply(lambda s: (s == "unclear").mean())
    table = {}  # (driver, combo) -> counts
    for combo in combos:
        cfg = copy.deepcopy(base)
        cfg["eyes"].update(dict(zip(GRID, combo)))
        for d in drivers:
            try:
                table[(d, combo)] = counts(feats[d], lab[lab["driver"] == d], cfg)
            except RuntimeError:
                table[(d, combo)] = (0, 0, 0, 0)
    default = (base["eyes"]["closed_ear_ratio"], base["eyes"]["closed_blink_score"])
    if default not in combos:
        cfg = copy.deepcopy(base)
        for d in drivers:
            try:
                table[(d, default)] = counts(feats[d], lab[lab["driver"] == d], cfg)
            except RuntimeError:
                table[(d, default)] = (0, 0, 0, 0)

    rows, chosen = [], {}
    for d in drivers:
        others = [o for o in drivers if o != d]
        best = max(combos, key=lambda c: prf([table[(o, c)] for o in others])[2])
        chosen[d] = best
        rows.append({"driver": d, "unclear_frac": unclear_frac[d], "before": table[(d, default)], "after": table[(d, best)]})
    res = pd.DataFrame(rows)

    def summary(mask) -> dict:
        b, a = prf(list(res.loc[mask, "before"])), prf(list(res.loc[mask, "after"]))
        n = int(np.sum([c[0] for c in res.loc[mask, "before"]]))
        return {"drivers": int(mask.sum()), "eye events": n,
                "before recall": b[0], "before precision": b[1], "before F1": b[2],
                "after recall": a[0], "after precision": a[1], "after F1": a[2]}

    clear = res["unclear_frac"] < 0.3
    s = pd.DataFrame({"all drivers": summary(res.index == res.index), "clear drivers (< 30 % unclear)": summary(clear),
                      "blurry / dark drivers": summary(~clear)}).T
    freq = pd.Series([str(dict(zip(GRID, c))) for c in chosen.values()]).value_counts()
    per = res.assign(**{f"{k}_{m}": res[k].map(lambda c, i=i: c[i]) for k in ("before", "after")
                        for i, m in enumerate(["events", "found", "dets", "correct"])}).drop(columns=["before", "after"])
    lines = ["# INVEDRIFAC: Western default vs. Indian-adapted eye-closure detector", "",
             "Ground truth = our check-and-confirm labels (881 clips, 286 unclear and excluded). "
             f"Default (Western / DMD) thresholds: closed_ear_ratio {default[0]}, closed_blink_score {default[1]}. "
             "**After** = thresholds tuned on the other Indian drivers only (leave-one-driver-out).", "",
             s.map(lambda v: round(v, 3) if isinstance(v, float) else v).to_markdown(), "",
             "Thresholds chosen across the folds: " + "; ".join(f"{k} x{v}" for k, v in freq.items()), "",
             "Per driver (events = labelled eye events, found = detected, dets = detections, correct = detections "
             "matching a labelled event):", "", per.round(2).to_markdown(index=False), "",
             "Cross-check on Western data (DMD drowsiness, 16 recordings): default thresholds eyes-closed F1 0.923, "
             "Indian-adapted 0.785 (long-closure precision 0.69 -> 0.47). Thresholds do not transfer in either "
             "direction, so they are kept as a region profile: `load_config('cabin', profile='india')`.", "",
             "Notes: yawns are not tuned (3 labelled yawns). The candidate generator can miss events: the random "
             "spot checks found a few (see invedrifac_labelled_eval.md), so recall is an upper bound."]
    OUT.write_text("\n".join(lines))
    print(s.round(3).to_string())
    print(freq.to_string())


if __name__ == "__main__":
    main()
