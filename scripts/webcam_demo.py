"""Live cabin demo: blinks, eye closure, PERCLOS, yawns (also behind a hand), nods, eyes off road.

    python scripts/webcam_demo.py                      # laptop webcam
    python scripts/webcam_demo.py --demo-timing        # DROWSY reacts in ~15 s (shorter PERCLOS window)
    python scripts/webcam_demo.py --india              # eye thresholds adapted on Indian drivers
    python scripts/webcam_demo.py --video clip.mp4     # any recording, no camera needed

1. Look at the screen normally while it calibrates (your personal baseline).
2. Then try: blink, close your eyes for 2 s, yawn (also behind your hand), nod, look away for 3 s,
   act drowsy (slow, heavy blinks) for ~15 s with --demo-timing.
Keys: q = quit, c = recalibrate. Events are saved to outputs/webcam_events.csv on exit.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from collections import deque

import cv2
import numpy as np
import pandas as pd

from dms.baseline.calibration import Calibrator
from dms.cabin.fatigue import CabinMonitor
from dms.cabin.features import LEFT_EYE, MOUTH_CORNERS, MOUTH_PAIRS, RIGHT_EYE, frame_features
from dms.cabin.hands import HandTracker, hand_face_features
from dms.cabin.landmarks import FaceTracker
from dms.utils.config import PROJECT_ROOT, load_config, resolve

# BGR colours
GREEN, RED, YELLOW, WHITE = (80, 200, 90), (60, 60, 230), (0, 200, 240), (240, 240, 240)
GREY, DARK, PANEL, ORANGE = (150, 150, 150), (30, 30, 30), (40, 36, 34), (0, 150, 255)
VIEW_H, PANEL_W = 720, 400
EVENT_NAMES = {"BLINK": "Blink", "SLOW_BLINK": "Slow blink", "MICROSLEEP": "Microsleep", "YAWN": "Yawn",
               "NOD": "Nod", "EYES_OFF_ROAD": "Eyes off road", "DROWSY": "Drowsy"}
SOUNDS = {"EYES CLOSED - MICROSLEEP": "Sosumi", "DROWSY": "Sosumi", "EYES OFF ROAD": "Funk", "YAWNING": "Tink"}


def text(img, s, org, scale=0.6, color=WHITE, thick=1, outline=True):
    if outline:
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_face(img, face, closed, scale):
    color = RED if closed else GREEN
    for idx in (LEFT_EYE, RIGHT_EYE):
        for p in (face.points_px(idx) * scale).astype(int):
            cv2.circle(img, tuple(p), 3, color, -1)
    mouth = MOUTH_CORNERS + [i for pair in MOUTH_PAIRS for i in pair]
    for p in (face.points_px(mouth) * scale).astype(int):
        cv2.circle(img, tuple(p), 3, YELLOW, -1)


def beep(name: str) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["afplay", f"/System/Library/Sounds/{name}.aiff"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def pill(panel, label, value, ok, x, y, w=118):
    cv2.rectangle(panel, (x, y), (x + w, y + 56), (60, 56, 54), -1, cv2.LINE_AA)
    text(panel, label, (x + 8, y + 18), 0.42, GREY, outline=False)
    text(panel, value, (x + 8, y + 45), 0.62, GREEN if ok else RED, 2, outline=False)


def counter(panel, label, n, x, y, warn=False):
    text(panel, str(n), (x, y), 1.1, (RED if warn and n else WHITE), 2, outline=False)
    text(panel, label, (x, y + 20), 0.42, GREY, outline=False)


def draw_panel(state, profile, timing, history, recent, calibrating, progress):
    p = np.full((VIEW_H, PANEL_W, 3), PANEL, np.uint8)
    text(p, "DRIVER MONITOR", (20, 40), 0.85, WHITE, 2, outline=False)
    text(p, f"thresholds: {profile}   timing: {timing}", (20, 64), 0.45, GREY, outline=False)
    if calibrating:
        text(p, "Calibrating your baseline", (20, 120), 0.65, YELLOW, 1, outline=False)
        text(p, "Look ahead normally", (20, 148), 0.55, GREY, outline=False)
        cv2.rectangle(p, (20, 170), (PANEL_W - 20, 190), (70, 70, 70), -1)
        cv2.rectangle(p, (20, 170), (20 + int((PANEL_W - 40) * progress), 190), YELLOW, -1)
        return p
    st = state
    pill(p, "EYES", "CLOSED" if st.eyes_closed else "OPEN", not st.eyes_closed, 20, 90)
    pill(p, "MOUTH", "OPEN" if st.mouth_open else "normal", not st.mouth_open, 142, 90)
    pill(p, "HEAD", "AWAY" if st.off_road else "on road", not st.off_road, 264, 90)
    c = st.counts
    for i, (lab, key, warn) in enumerate([("blinks", "BLINK", False), ("slow blinks", "SLOW_BLINK", True),
                                          ("microsleeps", "MICROSLEEP", True), ("yawns", "YAWN", True),
                                          ("nods", "NOD", True), ("eyes off road", "EYES_OFF_ROAD", True)]):
        counter(p, lab, c.get(key, 0), 22 + 128 * (i % 3), 205 + 72 * (i // 3), warn)
    # PERCLOS bar
    y = 340
    text(p, "Drowsiness (PERCLOS: % of time eyes closed)", (20, y), 0.45, GREY, outline=False)
    cv2.rectangle(p, (20, y + 12), (PANEL_W - 20, y + 32), (70, 70, 70), -1)
    if not math.isnan(st.perclos):
        frac = min(st.perclos / 0.4, 1.0)
        cv2.rectangle(p, (20, y + 12), (20 + int((PANEL_W - 40) * frac), y + 32), RED if st.drowsy else GREEN, -1)
        text(p, f"{st.perclos:.0%}", (PANEL_W - 70, y + 52), 0.5, WHITE, outline=False)
    else:
        text(p, "warming up ...", (20, y + 52), 0.45, GREY, outline=False)
    tx = 20 + int((PANEL_W - 40) * 0.15 / 0.4)
    cv2.line(p, (tx, y + 8), (tx, y + 36), YELLOW, 2)
    # eye openness graph (last 15 s)
    gy, gh = 430, 120
    text(p, "Eye openness, last 15 s (1 = your normal)", (20, gy - 8), 0.45, GREY, outline=False)
    cv2.rectangle(p, (20, gy), (PANEL_W - 20, gy + gh), DARK, -1)
    if len(history) > 1:
        t_end = history[-1][0]
        pts = [(20 + int((PANEL_W - 40) * (1 - (t_end - t) / 15)), gy + gh - int(gh * min(v, 1.4) / 1.4))
               for t, v in history if t_end - t <= 15 and not math.isnan(v)]
        if len(pts) > 1:
            cv2.polylines(p, [np.array(pts, np.int32)], False, GREEN, 2, cv2.LINE_AA)
    thr = gy + gh - int(gh * 0.65 / 1.4)
    cv2.line(p, (20, thr), (PANEL_W - 20, thr), RED, 1)
    text(p, "closed below", (PANEL_W - 120, thr + 16), 0.38, RED, outline=False)
    # recent events
    text(p, "Recent events", (20, 590), 0.5, GREY, outline=False)
    for i, (t, kind) in enumerate(list(recent)[-4:][::-1]):
        col = WHITE if kind == "BLINK" else ORANGE
        text(p, f"{t:6.1f} s   {EVENT_NAMES.get(kind, kind)}", (20, 618 + 24 * i), 0.5, col, outline=False)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--calib", type=float, help="calibration seconds (default from configs/cabin.yaml)")
    ap.add_argument("--video", help="play a video file instead of using the webcam")
    ap.add_argument("--speed", type=float, default=1.0, help="video playback speed (1 = real time)")
    ap.add_argument("--india", action="store_true", help="eye thresholds adapted on Indian drivers")
    ap.add_argument("--demo-timing", action="store_true",
                    help="20 s PERCLOS window so DROWSY reacts within ~15 s (default: 60 s)")
    ap.add_argument("--no-sound", action="store_true")
    ap.add_argument("--no-hands", action="store_true", help="skip hand tracking (faster on slow machines)")
    args = ap.parse_args()

    profile = "india" if args.india else "default"
    cfg = load_config("cabin", profile="india" if args.india else None)
    if args.demo_timing:
        cfg["perclos"]["window_s"] = 20
    timing = "demo (20 s window)" if args.demo_timing else "standard"
    calib_s = args.calib or cfg["calibration"]["live_seconds"]
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        if args.video:
            raise SystemExit(f"Could not open video: {args.video}")
        raise SystemExit("Could not open the webcam. On macOS, allow camera access for your terminal app "
                         "(System Settings > Privacy & Security > Camera).")
    video_fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if args.video else None
    frame_i = 0

    tracker = FaceTracker(resolve(cfg["model_path"]))
    hand_tracker = None if args.no_hands else HandTracker(resolve(cfg["hand_model_path"]))
    calibrator, monitor = Calibrator.from_config(cfg, calib_s), None
    log, t0 = [], time.monotonic()
    history: deque = deque(maxlen=900)
    recent: deque = deque(maxlen=8)
    last_sound: dict[str, float] = {}
    toast: tuple[str, float] | None = None
    hand_feats: dict = {}
    n = 0

    window = "Driver monitoring - live demo"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped sending frames.")
                break
            if video_fps:
                t = frame_i / video_fps
                frame_i += 1
            else:
                t = time.monotonic() - t0
            n += 1
            face = tracker.process(frame, t)
            feats = frame_features(face) if face is not None else None
            if feats is not None and hand_tracker is not None:
                if n % 3 == 0 or not hand_feats:  # hands every 3rd frame, carried forward
                    hand_feats = hand_face_features(hand_tracker.process(frame, t), face)
                feats.update(hand_feats)

            view = frame.copy() if video_fps else cv2.flip(frame, 1)  # mirror only the webcam view
            scale = VIEW_H / view.shape[0]
            view = cv2.resize(view, (int(view.shape[1] * scale), VIEW_H))
            if face is not None:
                draw_face(view, face if video_fps else _mirror(face), monitor is not None and monitor.state.eyes_closed,
                          scale)

            if monitor is None:
                calibrator.add(t, feats)
                text(view, "Look ahead normally - calibrating", (20, 45), 0.9, YELLOW, 2)
                if face is None:
                    text(view, "No face detected", (20, 85), 0.8, RED, 2)
                if calibrator.done(t):
                    base = calibrator.result()
                    monitor = CabinMonitor(cfg, base)
                    print(f"Calibrated: open-eye EAR={base.ear_open:.3f}, yaw={base.yaw:.1f}, pitch={base.pitch:.1f}")
                panel = draw_panel(None, profile, timing, history, recent, True, calibrator.progress(t))
            else:
                for ev in monitor.update(t, feats):
                    print(f"[{ev.start:7.1f}s - {ev.end:7.1f}s] {ev.type:14s} ({ev.duration:.2f}s)")
                    log.append(ev.to_dict())
                    recent.append((ev.start, ev.type))
                    if ev.type == "EYES_OFF_ROAD":
                        monitor.state.counts["EYES_OFF_ROAD"] = monitor.state.counts.get("EYES_OFF_ROAD", 0) + 1
                    if ev.type != "BLINK":
                        toast = (f"{EVENT_NAMES.get(ev.type, ev.type)} detected", t)
                ear_rel = feats["ear"] / monitor.base.ear_open if feats is not None else math.nan
                history.append((t, ear_rel))
                alerts = monitor.live_alerts(t)
                if not monitor.state.face_visible:
                    alerts = alerts + ["NO FACE"]
                for i, a in enumerate(alerts):  # red banner(s) across the top of the video
                    y0 = 12 + 62 * i
                    cv2.rectangle(view, (12, y0), (view.shape[1] - 12, y0 + 52), (30, 30, 200), -1)
                    text(view, a, (28, y0 + 38), 1.05, WHITE, 2, outline=False)
                    key = a.split(" (")[0]
                    if not args.no_sound and key in SOUNDS and t - last_sound.get(key, -99) > 2.5:
                        beep(SOUNDS[key])
                        last_sound[key] = t
                if toast and t - toast[1] < 2.0:  # short confirmation after an event ends
                    w = view.shape[1]
                    cv2.rectangle(view, (w // 2 - 230, VIEW_H - 80), (w // 2 + 230, VIEW_H - 28), (20, 120, 230), -1)
                    text(view, toast[0], (w // 2 - 210, VIEW_H - 44), 0.95, WHITE, 2, outline=False)
                panel = draw_panel(monitor.state, profile, timing, history, recent, False, 1.0)

            cv2.putText(panel, "q: quit    c: recalibrate", (20, VIEW_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1,
                        cv2.LINE_AA)
            cv2.imshow(window, np.hstack([view, panel]))

            if video_fps:
                lag = (t / args.speed) - (time.monotonic() - t0)
                key = cv2.waitKey(max(1, int(lag * 1000))) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:  # window closed with the mouse
                break
            if key == ord("c"):
                calibrator, monitor = Calibrator.from_config(cfg, calib_s), None
                history.clear()
                recent.clear()

    except KeyboardInterrupt:
        print("Stopped with Ctrl+C.")
    finally:
        cap.release()
        tracker.close()
        if hand_tracker is not None:
            hand_tracker.close()
        cv2.destroyAllWindows()
        if monitor is None:
            print("Calibration did not finish, so no events were recorded "
                  "(keep your face in view for the whole calibration).")
        else:
            print("Counts:", monitor.state.counts)
        out = PROJECT_ROOT / "outputs/webcam_events.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["type", "start", "end", "confidence", "info"]
        pd.DataFrame(log, columns=cols).to_csv(out, index=False)
        print(f"Saved {len(log)} events to {out.relative_to(PROJECT_ROOT)}")


def _mirror(face):
    """Copy of a FaceFrame with x flipped, so landmarks line up with the mirrored display."""
    from dataclasses import replace

    lm = face.landmarks.copy()
    lm[:, 0] = 1.0 - lm[:, 0]
    return replace(face, landmarks=lm)


if __name__ == "__main__":
    main()
