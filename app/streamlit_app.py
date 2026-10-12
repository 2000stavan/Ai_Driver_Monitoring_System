"""Driver monitoring dashboard (v1 skeleton).

    .venv/bin/streamlit run app/streamlit_app.py

An upload view for any driver-facing video, plus three views, each on a public dataset
(a synchronised cabin + telematics + road drive is future work):
  0. Analyse a video - upload a clip: personal calibration, detected events, key frames
  1. Driver camera  - a DMD recording: our detected events vs. DMD's ground-truth labels
  2. Driving        - a LEVIN trip: speed, vehicle-frame acceleration, harsh events, attributed context
  3. Road camera    - an IDD Lite image: Western-only (Cityscapes) vs. Indian fine-tuned segmentation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from dms.cabin.windows import run_monitor
from dms.eval.dmd import load_dmd
from dms.fusion.attribution import SpeedTrack, attribute, summary
from dms.road.segmentation import L1_CLASSES, IDDLite, cityscapes_to_l1, load_segformer, overlay, predict_logits
from dms.telematics.events import detect_events, smooth
from dms.telematics.levin import load_vehicle
from dms.telematics.orientation import to_vehicle_frame
from dms.utils.config import PROJECT_ROOT, load_config

DMD_SETS = {
    "drowsiness": (PROJECT_ROOT / "data/interim/dmd/drowsiness", PROJECT_ROOT / "data/processed/dmd_drowsiness"),
    "distraction": (PROJECT_ROOT / "data/interim/dmd/distractions", PROJECT_ROOT / "data/processed/dmd_distraction"),
}
LEVIN = PROJECT_ROOT / "data/raw/telematics/levin/csv"
IDD_LITE = PROJECT_ROOT / "data/raw/road/idd_lite/idd20k_lite"
ROAD_MODEL = PROJECT_ROOT / "models/road/segformer_idd_lite.pt"
UPLOADS = PROJECT_ROOT / "outputs/uploads"
COLORS = {"BLINK": "#9aa5b1", "SLOW_BLINK": "#f0ad4e", "MICROSLEEP": "#d9534f", "YAWN": "#5bc0de",
          "NOD": "#8e6cc9", "EYES_OFF_ROAD": "#e67e22", "DROWSY": "#c0392b",
          "HARSH_BRAKE": "#d9534f", "HARSH_ACCEL": "#f0ad4e", "SHARP_TURN": "#8e6cc9"}

st.set_page_config(page_title="Driver Monitoring", layout="wide")
st.title("Driver monitoring — v1 dashboard")
st.caption("Prototype. Each view runs on a different public dataset (cabin: DMD, driving: LEVIN Pune, road: IDD Lite). One synchronised drive across all three is future work.")


def timeline(lanes: dict[str, list[tuple[float, float, str]]], title: str) -> go.Figure:
    fig = go.Figure()
    for i, (lane, items) in enumerate(lanes.items()):
        for start, end, kind in items:
            fig.add_trace(go.Scatter(x=[start, max(end, start + 0.2)], y=[lane, lane], mode="lines",
                                     line={"width": 14, "color": COLORS.get(kind, "#4a7fb5")},
                                     name=kind, legendgroup=kind, showlegend=False,
                                     hovertemplate=f"{kind}<br>%{{x:.1f}} s<extra></extra>"))
    fig.update_layout(title=title, height=110 + 45 * len(lanes), margin={"l": 10, "r": 10, "t": 40, "b": 30},
                      xaxis_title="time (s)", yaxis={"categoryorder": "array", "categoryarray": list(lanes)[::-1]})
    return fig


def segs(mask: np.ndarray, fps: float, kind: str) -> list[tuple[float, float, str]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return [(s / fps, (e - 1) / fps, kind) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


tab_upload, tab_cabin, tab_drive, tab_road = st.tabs(["Analyse a video", "Driver camera", "Driving behaviour", "Road camera"])

with tab_upload:
    import hashlib
    import sys

    import cv2

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from extract_cabin_features import extract  # noqa: E402

    st.write("Upload a short video of a driver's face (phone or dashcam, face visible). The first part of the clip "
             "is used as the driver's personal baseline, so start with a few seconds of normal, alert driving.")
    up = st.file_uploader("Driver-facing video", type=["mp4", "mov", "m4v", "avi", "mkv"])
    profile = st.radio("Thresholds", ["default", "india"], horizontal=True,
                       help="india = eye thresholds adapted on Indian drivers (INVEDRIFAC)")
    if up is not None:
        data = up.getvalue()
        key = hashlib.md5(data).hexdigest()
        store = st.session_state.setdefault("uploads", {})
        if key not in store:
            UPLOADS.mkdir(parents=True, exist_ok=True)
            vid = UPLOADS / f"{key}{Path(up.name).suffix.lower()}"
            vid.write_bytes(data)
            with st.spinner("Tracking face and hands in every frame ... (about a third of the video length)"):
                store[key] = (vid, extract(vid, None, progress=False, hands=True, target_fps=15))
        vid, feats = store[key]
        if not feats["face"].any():
            st.error("No face found in this video.")
        else:
            dur = float(feats["t"].iloc[-1])
            calib = float(np.clip(dur * 0.25, 5, 60))
            cfg = load_config("cabin", profile=None if profile == "default" else profile)
            try:
                frames_u, events_u, mon_u = run_monitor(feats, cfg, calib_s=calib)
            except RuntimeError:
                st.error("The face is not visible long enough at the start to calibrate.")
                st.stop()
            kinds = ["BLINK", "SLOW_BLINK", "MICROSLEEP", "YAWN", "NOD", "EYES_OFF_ROAD", "DROWSY"]
            n = {k: sum(e.type == k for e in events_u) for k in kinds}
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Duration", f"{dur:.0f} s")
            c2.metric("Face found", f"{feats['face'].mean():.0%}")
            c3.metric("Blinks / min", f"{n['BLINK'] / max(dur / 60, 1e-9):.0f}")
            c4.metric("Microsleeps", n["MICROSLEEP"])
            c5.metric("Yawns", n["YAWN"])
            c6.metric("Eyes off road", n["EYES_OFF_ROAD"])
            st.caption(f"Personal baseline from the first {calib:.0f} s: open-eye ratio {mon_u.base.ear_open:.3f}.")
            lanes = {}
            for e in events_u:
                lanes.setdefault(e.type, []).append((e.start, e.end, e.type))
            if lanes:
                st.plotly_chart(timeline(lanes, "Detected events"), width="stretch")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=frames_u["t"], y=frames_u["ear"] / mon_u.base.ear_open,
                                     name="eye openness (1 = your normal)", line={"width": 1}))
            fig.add_trace(go.Scatter(x=frames_u["t"], y=frames_u["mar"], name="mouth opening", line={"width": 1}))
            fig.add_vrect(x0=0, x1=calib, fillcolor="#5bc0de", opacity=0.12, line_width=0,
                          annotation_text="calibration", annotation_position="top left")
            fig.update_layout(height=260, margin={"l": 10, "r": 10, "t": 30, "b": 30}, xaxis_title="time (s)")
            st.plotly_chart(fig, width="stretch")
            shown = [e for e in events_u if e.type in ("MICROSLEEP", "SLOW_BLINK", "YAWN", "EYES_OFF_ROAD", "NOD")]
            shown = sorted(shown, key=lambda e: -(e.end - e.start))[:8]
            if shown:
                st.subheader("What it saw")
                cap = cv2.VideoCapture(str(vid))
                cols = st.columns(4)
                for i, e in enumerate(sorted(shown, key=lambda e: e.start)):
                    cap.set(cv2.CAP_PROP_POS_MSEC, (e.start + e.end) / 2 * 1000)
                    ok, img = cap.read()
                    if ok:
                        cols[i % 4].image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                          caption=f"{e.type}  {e.start:.1f}-{e.end:.1f} s", width="stretch")
                cap.release()
            if events_u:
                table = pd.DataFrame([{"event": e.type, "start_s": e.start, "end_s": e.end, "duration_s": e.end - e.start}
                                      for e in events_u])
                st.dataframe(table.round(2), width="stretch")
                st.download_button("Download events (CSV)", table.to_csv(index=False), "events.csv", "text/csv")

with tab_cabin:
    which = st.radio("DMD bundle", list(DMD_SETS), horizontal=True)
    interim, processed = DMD_SETS[which]
    recs = sorted(p.name for p in processed.glob("*.features.csv")) if processed.exists() else []
    if not recs:
        st.info("No processed recordings yet. Run scripts/eval_dmd_drowsiness.py or eval_dmd_gaze.py first.")
    else:
        rec = st.selectbox("Recording", [r.replace(".features.csv", "") for r in recs])
        df = pd.read_csv(processed / f"{rec}.features.csv")
        frames, events, mon = run_monitor(df, load_config("cabin"))
        fps = len(df) / df["t"].iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", f"{df['t'].iloc[-1] / 60:.1f} min")
        c2.metric("Face detected", f"{df['face'].mean():.0%}")
        c3.metric("Blinks / min", f"{sum(e.type == 'BLINK' for e in events) / (df['t'].iloc[-1] / 60):.1f}")
        c4.metric("Microsleeps", sum(e.type == "MICROSLEEP" for e in events))
        ours = {}
        for e in events:
            ours.setdefault(f"ours: {e.type}", []).append((e.start, e.end, e.type))
        gt = {}
        ann_path = interim / rec / "ann.json"
        if ann_path.exists():
            ann = load_dmd(ann_path)
            wanted = {"eyes_state/close": "MICROSLEEP", "blinks/blinking": "BLINK",
                      "yawning/Yawning with hand": "YAWN", "yawning/Yawning without hand": "YAWN",
                      "gaze_on_road/not_looking_road": "EYES_OFF_ROAD",
                      "driver_actions/phonecall_right": "phone", "driver_actions/phonecall_left": "phone",
                      "driver_actions/texting_right": "texting", "driver_actions/texting_left": "texting",
                      "driver_actions/drinking": "drinking"}
            for label, kind in wanted.items():
                if label in ann.labels and ann.labels[label].any():
                    gt[f"truth: {label.split('/')[-1]}"] = segs(ann.labels[label], fps, kind)
        st.plotly_chart(timeline({**ours, **gt}, "Detected events (ours) vs. DMD labels (truth)"),
                        width="stretch")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frames["t"], y=frames["ear"] / mon.base.ear_open, name="eye openness (rel.)",
                                 line={"width": 1}))
        fig.add_trace(go.Scatter(x=frames["t"], y=frames["mar"], name="mouth ratio", line={"width": 1}))
        fig.update_layout(height=260, margin={"l": 10, "r": 10, "t": 30, "b": 30}, xaxis_title="time (s)")
        st.plotly_chart(fig, width="stretch")

with tab_drive:
    vehicles = sorted((p.stem for p in LEVIN.glob("*.csv")), key=int) if LEVIN.exists() else []
    if not vehicles:
        st.info("LEVIN not found under data/raw/telematics/levin/csv.")
    else:
        cfg = load_config("telematics")
        veh = st.selectbox("LEVIN vehicle (Pune)", vehicles)

        @st.cache_data(show_spinner="Loading vehicle ...")
        def trips_of(v: str):
            return [t for t in load_vehicle(LEVIN / f"{v}.csv") if (t.seconds["speed_kmh"] > 3).sum() >= 60]

        trips = trips_of(veh)
        if not trips:
            st.warning("No trip with at least 60 s of driving.")
        else:
            labels = [f"trip {t.trip_id} · {len(t.seconds) / 60:.0f} min · {(t.seconds['speed_kmh'] / 3600).sum():.1f} km"
                      for t in trips]
            tr = trips[labels.index(st.selectbox("Trip", labels))]
            s = tr.seconds
            vf = to_vehicle_frame(tr.acc_t, tr.acc, s["t"].to_numpy(),
                                  {"obd": s["speed_kmh"].to_numpy(), "gps": s["gps_speed"].to_numpy()})
            speed_at = np.interp(tr.acc_t, s["t"], s["speed_kmh"])
            events = detect_events(tr.acc_t, vf.longitudinal, vf.lateral, speed_at, cfg)
            att = attribute(events, SpeedTrack(s["t"].to_numpy(), s["speed_kmh"].to_numpy()))
            km = float((s["speed_kmh"] / 3600).sum())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Distance", f"{km:.1f} km")
            c2.metric("Harsh brakes / 100 km", f"{100 * sum(e.type == 'HARSH_BRAKE' for e in events) / max(km, 1e-9):.0f}")
            c3.metric("Harsh accels / 100 km", f"{100 * sum(e.type == 'HARSH_ACCEL' for e in events) / max(km, 1e-9):.0f}")
            c4.metric("Orientation quality", f"{vf.forward_corr:.2f}",
                      help="Correlation of forward acceleration with speed change; < 0.5 = unreliable")
            if vf.forward_corr < cfg["orientation"]["min_forward_corr"]:
                st.warning("Phone orientation estimate is unreliable for this trip; events may be wrong.")
            n = int(cfg["smooth_s"] * 25)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s["t"], y=s["speed_kmh"], name="speed (km/h)", yaxis="y2", line={"width": 1}))
            fig.add_trace(go.Scatter(x=tr.acc_t[::5], y=smooth(vf.longitudinal, n)[::5], name="longitudinal (g)",
                                     line={"width": 1}))
            for e in events:
                fig.add_vrect(x0=e.start, x1=e.end + 0.3, fillcolor=COLORS.get(e.type, "#999"), opacity=0.35,
                              line_width=0)
            fig.update_layout(height=320, margin={"l": 10, "r": 10, "t": 30, "b": 30}, xaxis_title="time (s)",
                              yaxis={"title": "g"}, yaxis2={"title": "km/h", "overlaying": "y", "side": "right"})
            st.plotly_chart(fig, width="stretch")
            st.subheader("Attributed events")
            st.write(summary(att) or "No risky events in this trip.")
            if att:
                st.dataframe(pd.DataFrame([a.to_dict() for a in att]).round(2), width="stretch")

with tab_road:
    if not IDD_LITE.exists():
        st.info("IDD Lite not found under data/raw/road/idd_lite.")
    elif not ROAD_MODEL.exists():
        st.info("No fine-tuned road model yet. Run scripts/train_road_seg.py first.")
    else:
        import torch

        @st.cache_resource(show_spinner="Loading road models ...")
        def road_models():
            ckpt = torch.load(ROAD_MODEL, map_location="cpu")
            western = load_segformer(ckpt["checkpoint"], cache_dir=PROJECT_ROOT / "models/segformer").eval()
            indian = load_segformer(ckpt["checkpoint"], num_labels=len(L1_CLASSES),
                                    cache_dir=PROJECT_ROOT / "models/segformer").eval()
            indian.load_state_dict(ckpt["state_dict"])
            return western, indian, tuple(ckpt["size"]), ckpt["val_miou"]

        western, indian, size, miou = road_models()
        val = IDDLite(IDD_LITE, "val", size)
        i = st.slider("IDD Lite validation image", 0, len(val) - 1, 0)
        x, y = val[i]
        rgb = np.asarray(Image.open(val.items[i][0]).convert("RGB").resize(size[::-1]))
        with torch.no_grad():
            p_w = cityscapes_to_l1(predict_logits(western, x[None]).argmax(1))[0].numpy()
            p_i = predict_logits(indian, x[None]).argmax(1)[0].numpy()
        valid = y.numpy() != 255
        c1, c2, c3 = st.columns(3)
        c1.image(rgb, caption="Indian road (IDD Lite)", width="stretch")
        c2.image(overlay(rgb, p_w), caption=f"Western only (Cityscapes): pixel acc {(p_w == y.numpy())[valid].mean():.0%}",
                 width="stretch")
        c3.image(overlay(rgb, p_i), caption=f"Indian fine-tuned: pixel acc {(p_i == y.numpy())[valid].mean():.0%}",
                 width="stretch")
        d_w = ((p_w == 0) & (y.numpy() == 0)).sum() / max(((p_w == 0) | (y.numpy() == 0)).sum(), 1)
        d_i = ((p_i == 0) & (y.numpy() == 0)).sum() / max(((p_i == 0) | (y.numpy() == 0)).sum(), 1)
        st.write(f"Drivable-area IoU on this image: Western {d_w:.2f} vs Indian {d_i:.2f}. "
                 f"Fine-tuned model val mIoU {miou:.3f}.")
        st.caption("Colours: purple drivable, pink non-drivable, red living things, blue vehicles, "
                   "orange roadside objects, green far objects, steel blue sky.")
