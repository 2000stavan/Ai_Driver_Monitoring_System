"""Create check-and-confirm labelling tasks for INVEDRIFAC (no public labels).

For each driver, a 1-minute segment is chosen (after the first minute, so calibration and setup
are excluded). A deliberately over-sensitive ("liberal") version of the eye / yawn detector proposes
every candidate event in that segment; on DMD this liberal setting catches 99.3 % of labelled blinks
and 97 % of yawns. Random "spot-check" clips with no candidate are added to estimate what the
liberal detector still misses.

    python scripts/make_label_tasks.py [--minutes 1] [--spot 6]

Writes data/labels/invedrifac_tasks.csv (one row per clip to review).
"""

from __future__ import annotations

import argparse
import copy
import re

import numpy as np
import pandas as pd

from dms.cabin.windows import run_monitor
from dms.utils.config import PROJECT_ROOT, load_config

FEATS = PROJECT_ROOT / "data/processed/invedrifac"
VIDEOS = PROJECT_ROOT / "data/raw/cabin/invedrifac"
OUT = PROJECT_ROOT / "data/labels/invedrifac_tasks.csv"
LIBERAL_EYES = {"closed_ear_ratio": 0.8, "closed_blink_score": 0.35, "blink_min_s": 0.03}
LIBERAL_MOUTH = {"yawn_mar": 0.3, "yawn_jaw_score": 0.3, "yawn_min_s": 1.0}
PAD_S = 0.6  # context shown before / after each candidate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=1.0)
    ap.add_argument("--spot", type=int, default=6, help="random no-candidate spot checks per driver")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    cfg = load_config("cabin")
    cfg["eyes"].update(LIBERAL_EYES)
    cfg["mouth"].update(LIBERAL_MOUTH)
    rows = []
    for f in sorted(FEATS.glob("*.features.csv"), key=lambda p: int(re.findall(r"\d+", p.name)[0])):
        driver = f.name.replace(".features.csv", "")
        video = VIDEOS / (driver.replace("_", " ") + ".avi")
        df = pd.read_csv(f)
        dur = df["t"].iloc[-1]
        seg_len = min(args.minutes * 60, max(dur - 10, 0))
        start = 60.0 if dur >= 60 + seg_len else max(0.0, (dur - seg_len) / 2)
        end = start + seg_len
        try:
            _, events, _ = run_monitor(df, cfg)
        except RuntimeError:
            print(f"{driver}: no face for calibration, skipped")
            continue
        cands = [e for e in events if e.type in ("BLINK", "SLOW_BLINK", "MICROSLEEP", "YAWN") and start <= e.start < end]
        for e in cands:
            rows.append({"driver": driver, "video": str(video.relative_to(PROJECT_ROOT)), "kind": "candidate",
                         "proposed": e.type, "t0": max(0.0, e.start - PAD_S), "t1": e.end + PAD_S,
                         "event_start": e.start, "event_end": e.end})
        # spot checks: 2 s windows that do not overlap any candidate
        busy = [(e.start - 1.0, e.end + 1.0) for e in cands]
        tries = 0
        n_spot = 0
        while n_spot < args.spot and tries < 500:
            tries += 1
            t0 = float(rng.uniform(start, max(start, end - 2.0)))
            if any(a <= t0 + 2.0 and t0 <= b for a, b in busy):
                continue
            busy.append((t0, t0 + 2.0))
            rows.append({"driver": driver, "video": str(video.relative_to(PROJECT_ROOT)), "kind": "spot_check",
                         "proposed": "NONE", "t0": t0, "t1": t0 + 2.0, "event_start": np.nan, "event_end": np.nan})
            n_spot += 1
        print(f"{driver}: segment {start:.0f}-{end:.0f} s, {len(cands)} candidates, {n_spot} spot checks")
    tasks = pd.DataFrame(rows)
    tasks.insert(0, "task_id", range(len(tasks)))
    tasks["segment_note"] = "liberal detector + random spot checks"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(OUT, index=False)
    print(f"\n{len(tasks)} tasks ({(tasks.kind == 'candidate').sum()} candidates, "
          f"{(tasks.kind == 'spot_check').sum()} spot checks) -> {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
