"""Old vs. robust personal calibration (INITIALPLAN §7.1).

  old      : 75th percentile EAR of every face frame in the first 10 s
  robust10 : frontal-pose frames only + MAD outlier removal, 10 s
  old60    : old method, 60 s window
  robust60 : robust method, 60 s window (the reference EAR is measured after 60 s, so no overlap)

DMD drowsiness (labelled): eye / blink / yawn scores must not get worse.
INVEDRIFAC (unlabelled): baseline EAR vs. the driver's typical EAR later in the drive, and
microsleep rate - a long run of false microsleeps means the baseline was set too high.

    python scripts/compare_calibration.py
Writes reports/results/calibration_comparison.md
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from eval_dmd_drowsiness import INTERIM, evaluate  # noqa: E402

from dms.cabin.windows import run_monitor  # noqa: E402
from dms.utils.config import PROJECT_ROOT, load_config  # noqa: E402

INVEDRIFAC = PROJECT_ROOT / "data/processed/invedrifac"
OUT = PROJECT_ROOT / "reports/results/calibration_comparison.md"
VARIANTS = {
    "old": {"seconds": 10, "frontal_deg": 1e9, "mad_k": 1e9},
    "robust10": {"seconds": 10, "frontal_deg": 15.0, "mad_k": 3.0},
    "old60": {"seconds": 60, "frontal_deg": 1e9, "mad_k": 1e9},
    "robust60": {"seconds": 60, "frontal_deg": 15.0, "mad_k": 3.0},
}


def variant_cfg(base: dict, name: str) -> dict:
    cfg = copy.deepcopy(base)
    cfg["calibration"].update(VARIANTS[name])
    return cfg


def main() -> None:
    base = load_config("cabin")
    dmd_rows, inv_rows = [], []
    recs = sorted(p for p in INTERIM.iterdir() if (p / "ann.json").exists())
    for name in VARIANTS:
        cfg = variant_cfg(base, name)
        for r in recs:
            dmd_rows.append({"variant": name, **evaluate(r, cfg)})
        for f in sorted(INVEDRIFAC.glob("*.features.csv")):
            df = pd.read_csv(f)
            try:
                _, events, mon = run_monitor(df, cfg)
            except RuntimeError:
                continue
            vis = df[df["face"].astype(bool) & (df["t"] > 60)]
            hours = df["t"].iloc[-1] / 3600
            inv_rows.append({
                "variant": name, "driver": f.name.split(".")[0],
                "baseline_ear": mon.base.ear_open,
                "typical_ear_after_1min": float(vis["ear"].quantile(0.75)) if len(vis) else np.nan,
                "microsleeps_per_h": sum(e.type == "MICROSLEEP" for e in events) / hours,
                "blinks_per_min": sum(e.type == "BLINK" for e in events) / (hours * 60),
            })
    dmd, inv = pd.DataFrame(dmd_rows), pd.DataFrame(inv_rows)
    inv["baseline_error"] = (inv["baseline_ear"] / inv["typical_ear_after_1min"] - 1).abs()

    cols = ["eyes_closed_f1", "blink_recall", "blink_precision", "yawn_recall", "yawn_precision",
            "long_closure_recall", "long_closure_precision"]
    lines = ["# Personal calibration: old vs. robust", "",
             "## DMD drowsiness (16 recordings, labelled) - mean per recording", "",
             dmd.groupby("variant")[cols].mean().loc[list(VARIANTS)].round(3).to_markdown(), "",
             "## INVEDRIFAC (unlabelled, Indian drivers)", "",
             "`baseline_error` = |baseline EAR / driver's 75th-percentile EAR after the first minute - 1|. "
             "Large = calibration did not represent the driver's normal open eyes.", ""]
    summ = inv.groupby("variant").agg(
        drivers=("driver", "count"),
        median_baseline_error=("baseline_error", "median"),
        drivers_error_over_20pct=("baseline_error", lambda s: int((s > 0.2).sum())),
        median_microsleeps_per_h=("microsleeps_per_h", "median"),
        max_microsleeps_per_h=("microsleeps_per_h", "max"),
    ).loc[list(VARIANTS)]
    lines += [summ.round(3).to_markdown(), "", "Per driver:", "",
              inv.pivot(index="driver", columns="variant", values=["baseline_error", "microsleeps_per_h"])
              .round(2).to_markdown(), ""]
    OUT.write_text("\n".join(lines))
    print("\n".join(lines[:16]))
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
