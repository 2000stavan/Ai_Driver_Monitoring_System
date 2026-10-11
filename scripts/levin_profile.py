"""Profile everyday Indian driving in LEVIN (no labels): orientation quality, acceleration
distribution and candidate-event rates per vehicle. Informs Indian thresholds (INITIALPLAN §5).

    python scripts/levin_profile.py

Writes reports/results/levin_trips.csv and levin_profile.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dms.telematics.events import detect_events, smooth
from dms.telematics.levin import load_vehicle
from dms.telematics.orientation import to_vehicle_frame
from dms.utils.config import PROJECT_ROOT, load_config

RAW = PROJECT_ROOT / "data/raw/telematics/levin/csv"
RESULTS = PROJECT_ROOT / "reports/results"
PCTS = [50, 90, 95, 99, 99.9]


def main() -> None:
    cfg = load_config("telematics")
    trip_rows, brake_vals, accel_vals, lat_vals = [], [], [], []
    for path in sorted(RAW.glob("*.csv"), key=lambda p: int(p.stem)):
        for tr in load_vehicle(path):
            s = tr.seconds
            if (s["speed_kmh"] > 3).sum() < cfg["orientation"]["min_moving_s"]:
                continue
            vf = to_vehicle_frame(
                tr.acc_t, tr.acc, s["t"].to_numpy(),
                {"obd": s["speed_kmh"].to_numpy(), "gps": s["gps_speed"].to_numpy()},
            )
            reliable = vf.forward_corr >= cfg["orientation"]["min_forward_corr"]
            speed = np.interp(tr.acc_t, s["t"], s["speed_kmh"])
            km = float((s["speed_kmh"] / 3600).sum())
            row = {"vehicle": tr.vehicle, "trip": tr.trip_id, "minutes": len(s) / 60, "km": km,
                   "forward_corr": vf.forward_corr, "reliable": reliable}
            if reliable:
                ev = detect_events(tr.acc_t, vf.longitudinal, vf.lateral, speed, cfg)
                for k in ("HARSH_BRAKE", "HARSH_ACCEL", "SHARP_TURN"):
                    row[k] = sum(e.type == k for e in ev)
                n = int(cfg["smooth_s"] * 25)
                lon, lat = smooth(vf.longitudinal, n), np.abs(smooth(vf.lateral, n))
                moving = speed >= cfg["min_speed_kmh"]
                # one value per second while moving: strongest braking / acceleration / lateral
                sec = np.floor(tr.acc_t).astype(int)
                df = pd.DataFrame({"sec": sec[moving], "lon": lon[moving], "lat": lat[moving]})
                g = df.groupby("sec")
                brake_vals.append((-g["lon"].min()).clip(lower=0).to_numpy())
                accel_vals.append(g["lon"].max().clip(lower=0).to_numpy())
                lat_vals.append(g["lat"].max().to_numpy())
            trip_rows.append(row)
        print(f"vehicle {path.stem}: done", flush=True)

    trips = pd.DataFrame(trip_rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    trips.to_csv(RESULTS / "levin_trips.csv", index=False)

    rel = trips[trips["reliable"]]
    km = rel["km"].sum()
    pct = pd.DataFrame(
        {name: np.percentile(np.concatenate(v), PCTS) for name, v in
         (("braking_g", brake_vals), ("acceleration_g", accel_vals), ("lateral_g", lat_vals))},
        index=[f"p{p}" for p in PCTS],
    ).round(3)
    rates = {k: 100 * rel[k].sum() / km for k in ("HARSH_BRAKE", "HARSH_ACCEL", "SHARP_TURN")}
    per_vehicle = rel.groupby("vehicle").apply(
        lambda g: pd.Series({"km": g.km.sum(), **{k: 100 * g[k].sum() / max(g.km.sum(), 1e-9)
                                                    for k in ("HARSH_BRAKE", "HARSH_ACCEL", "SHARP_TURN")}}),
        include_groups=False,
    ).round(1)

    md = [
        "# LEVIN everyday-driving profile (India, Pune; unlabelled)",
        "",
        f"- Vehicles: {trips['vehicle'].nunique()} | trips with ≥ {cfg['orientation']['min_moving_s']} s driving: {len(trips)}",
        f"- Orientation reliable (forward corr ≥ {cfg['orientation']['min_forward_corr']}): "
        f"{trips['reliable'].mean():.0%} of trips, median corr {trips['forward_corr'].median():.2f}",
        f"- Distance in reliable trips: {km:.0f} km",
        "",
        "## Strongest acceleration per driving second (0.5 s smoothed), percentiles",
        "",
        pct.to_markdown(),
        "",
        f"## Candidate events with v1 thresholds (brake/accel {cfg['events']['harsh_brake_g']} g, "
        f"lateral {cfg['events']['sharp_turn_g']} g, ≥ {cfg['events']['min_duration_s']} s)",
        "",
        "Per 100 km: " + ", ".join(f"{k} {v:.1f}" for k, v in rates.items()),
        "",
        "Per vehicle (per 100 km):",
        "",
        per_vehicle.sort_values("km", ascending=False).to_markdown(),
        "",
        "Caveats: unlabelled data (these are *candidate* events, not verified harsh events); the "
        "forward-axis fit picks the best of several speed sources / lags, which slightly inflates "
        "the reported correlation.",
    ]
    (RESULTS / "levin_profile.md").write_text("\n".join(md))
    print("\n".join(md[:14]))


if __name__ == "__main__":
    main()
