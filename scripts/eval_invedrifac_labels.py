"""Score the default eye / yawn detectors against our own INVEDRIFAC labels (check-and-confirm).

Ground truth = candidate clips you labelled as blink / long_closure / yawn. Spot-check clips that
you labelled as an event are events the liberal candidate generator missed (reported separately).
Clips labelled "unclear" are excluded.

    python scripts/eval_invedrifac_labels.py
Writes reports/results/invedrifac_labelled_eval.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dms.cabin.windows import run_monitor
from dms.utils.config import PROJECT_ROOT, load_config

TASKS = PROJECT_ROOT / "data/labels/invedrifac_tasks.csv"
LABELS = PROJECT_ROOT / "data/labels/invedrifac_labels.csv"
FEATS = PROJECT_ROOT / "data/processed/invedrifac"
OUT = PROJECT_ROOT / "reports/results/invedrifac_labelled_eval.md"
EYE = {"blink", "long_closure"}
TYPE_OF = {"BLINK": "eye", "SLOW_BLINK": "eye", "MICROSLEEP": "eye", "YAWN": "yawn"}


def overlaps(a0, a1, b0, b1, tol=0.1) -> bool:
    return a0 - tol <= b1 and b0 - tol <= a1


def main() -> None:
    tasks = pd.read_csv(TASKS).merge(pd.read_csv(LABELS), on="task_id")
    tasks = tasks[tasks["label"] != "unclear"]
    cfg = load_config("cabin")
    rows = []
    for driver, t in tasks.groupby("driver"):
        frames, events, _ = run_monitor(pd.read_csv(FEATS / f"{driver}.features.csv"), cfg)
        seg0, seg1 = t["t0"].min(), t["t1"].max()
        dets = [e for e in events if e.type in TYPE_OF and seg0 <= e.start <= seg1]
        cand = t[t["kind"] == "candidate"]
        for group, labels in (("eye", EYE), ("yawn", {"yawn"})):
            truth = cand[cand["label"].isin(labels)]
            mine = [e for e in dets if TYPE_OF[e.type] == group]
            hit = sum(any(overlaps(r.event_start, r.event_end, e.start, e.end) for e in mine) for r in truth.itertuples())
            correct = sum(any(overlaps(r.event_start, r.event_end, e.start, e.end) for r in truth.itertuples()) for e in mine)
            rows.append({"driver": driver, "group": group, "true_events": len(truth), "detected": hit,
                         "detections": len(mine), "correct_detections": correct})
        spot = t[t["kind"] == "spot_check"]
        rows.append({"driver": driver, "group": "spot_checks", "true_events": int(spot["label"].isin(EYE | {"yawn"}).sum()),
                     "detected": np.nan, "detections": len(spot), "correct_detections": np.nan})
    d = pd.DataFrame(rows)
    lines = ["# INVEDRIFAC — default detectors vs. our check-and-confirm labels", ""]
    for group in ("eye", "yawn"):
        g = d[d["group"] == group]
        rec = g["detected"].sum() / max(g["true_events"].sum(), 1)
        prec = g["correct_detections"].sum() / max(g["detections"].sum(), 1)
        lines.append(f"- **{group}**: {int(g['true_events'].sum())} labelled events, recall **{rec:.2f}**, "
                     f"precision **{prec:.2f}** ({g['driver'].nunique()} drivers)")
    s = d[d["group"] == "spot_checks"]
    lines.append(f"- **Spot checks**: {int(s['true_events'].sum())} events found in {int(s['detections'].sum())} "
                 "random no-candidate clips (events the candidate generator missed)")
    lines += ["", "Per driver:", "", d.to_markdown(index=False)]
    OUT.write_text("\n".join(lines))
    print("\n".join(lines[:6]))


if __name__ == "__main__":
    main()
