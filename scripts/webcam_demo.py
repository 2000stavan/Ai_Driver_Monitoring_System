"""Live cabin demo: blinks, eye closure, PERCLOS, yawns, nods, head turned away.

    python scripts/webcam_demo.py [--camera 0] [--calib 15]          # laptop webcam
    python scripts/webcam_demo.py --video path/to/face_video.mp4      # any recording, no camera needed

1. Look at the screen normally while it calibrates (your personal baseline).
2. Then try: blink, close your eyes for 2 s, yawn, nod, look away for 3 s.
Keys: q = quit, c = recalibrate. Events are saved to outputs/webcam_events.csv on exit.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import pandas as pd

from dms.baseline.calibration import Calibrator
from dms.cabin.fatigue import CabinMonitor
from dms.cabin.features import LEFT_EYE, MOUTH_CORNERS, MOUTH_PAIRS, RIGHT_EYE, frame_features
from dms.cabin.landmarks import FaceTracker
from dms.utils.config import PROJECT_ROOT, load_config, resolve

GREEN, RED, YELLOW, WHITE = (60, 200, 60), (40, 40, 230), (0, 200, 230), (240, 240, 240)


def put(img, text, y, color=WHITE, scale=0.6, x=12):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_face(img, face, closed):
    color = RED if closed else GREEN
    for idx in (LEFT_EYE, RIGHT_EYE):
        for p in face.points_px(idx).astype(int):
            cv2.circle(img, tuple(p), 2, color, -1)
    mouth = MOUTH_CORNERS + [i for pair in MOUTH_PAIRS for i in pair]
    for p in face.points_px(mouth).astype(int):
        cv2.circle(img, tuple(p), 2, YELLOW, -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--calib", type=float, help="calibration seconds (default from configs/cabin.yaml)")
    ap.add_argument("--video", help="play a video file instead of using the webcam")
    ap.add_argument("--speed", type=float, default=1.0, help="video playback speed (1 = real time)")
    args = ap.parse_args()

    cfg = load_config("cabin")
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
    calibrator, monitor = Calibrator.from_config(cfg, calib_s), None
    log, t0, fps_t, fps = [], time.monotonic(), time.monotonic(), 0.0

    window = "DMS - cabin demo"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped sending frames.")
                break
            if video_fps:  # file: time comes from the frame number, display paced to real time
                t = frame_i / video_fps
                frame_i += 1
            else:
                t = time.monotonic() - t0
            face = tracker.process(frame, t)
            feats = frame_features(face) if face is not None else None

            view = frame.copy() if video_fps else cv2.flip(frame, 1)  # mirror only the webcam view
            if face is not None:
                draw_face(view, face if video_fps else _mirror(face), monitor is not None and monitor.state.eyes_closed)

            if monitor is None:
                calibrator.add(t, feats)
                put(view, f"CALIBRATING - look ahead normally  {calibrator.progress(t):.0%}", 30, YELLOW, 0.7)
                if face is None:
                    put(view, "No face detected", 60, RED)
                if calibrator.done(t):
                    base = calibrator.result()
                    monitor = CabinMonitor(cfg, base)
                    print(f"Calibrated: open-eye EAR={base.ear_open:.3f}, yaw={base.yaw:.1f}, pitch={base.pitch:.1f}")
            else:
                for ev in monitor.update(t, feats):
                    print(f"[{ev.start:7.1f}s - {ev.end:7.1f}s] {ev.type:14s} ({ev.duration:.2f}s)")
                    log.append(ev.to_dict())
                st = monitor.state
                y = 30
                if feats is not None:
                    put(view, f"EAR {feats['ear']:.3f} (open {monitor.base.ear_open:.3f})   MAR {feats['mar']:.2f}", y)
                    y += 25
                    put(view, f"yaw {feats['yaw'] - monitor.base.yaw:+5.1f}  pitch {feats['pitch'] - monitor.base.pitch:+5.1f}", y)
                    y += 25
                perclos = "warming up" if np.isnan(st.perclos) else f"{st.perclos:.0%}"
                put(view, f"PERCLOS {perclos}", y)
                y += 25
                c = st.counts
                put(view, f"blinks {c['BLINK']}  slow {c['SLOW_BLINK']}  microsleeps {c['MICROSLEEP']}  "
                          f"yawns {c['YAWN']}  nods {c['NOD']}", y)
                for i, alert in enumerate(monitor.live_alerts(t)):
                    put(view, alert, view.shape[0] - 30 - 35 * i, RED, 0.9)
                if not st.face_visible:
                    put(view, "No face detected", view.shape[0] - 30, RED, 0.9)

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-6)
            fps_t = now
            put(view, f"{fps:4.1f} fps   q: quit  c: recalibrate", view.shape[0] - 8, WHITE, 0.45, x=view.shape[1] - 330)
            cv2.imshow(window, view)

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

    except KeyboardInterrupt:
        print("Stopped with Ctrl+C.")
    finally:
        cap.release()
        tracker.close()
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
