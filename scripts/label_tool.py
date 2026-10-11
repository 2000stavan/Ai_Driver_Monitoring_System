"""Check-and-confirm labelling tool for INVEDRIFAC eye / yawn events.

    .venv/bin/python scripts/label_tool.py

Each task is a short clip (a proposed event with 0.6 s of context, or a random spot check),
played in slow motion on a loop. INVEDRIFAC is 640x480 and soft, so the window shows:
  top     a large, sharpened view of the EYES, following the face frame by frame
  bottom  the full frame (context) and the whole face (mouth / yawns)

Keys (what really happens in the clip):
  1  blink                         (eyes close briefly, < ~0.5 s)
  2  long closure / microsleep     (eyes stay closed ~0.5 s or longer)
  3  yawn
  0  nothing                       (no blink / closure / yawn)
  9  unclear                       (face hidden, too dark, can't tell)
  space  pause / play     a / d  one frame back / forward (while paused)
  r  replay at normal speed    b  back to previous task    q  save and quit
Rule: "closed" = no iris visible. Looking down with droopy but open eyes is NOT a closure.
Labels are saved after every key press (data/labels/invedrifac_labels.csv); restarting resumes.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd

from dms.cabin.gaze import face_box
from dms.cabin.landmarks import FaceTracker
from dms.utils.config import PROJECT_ROOT, load_config, resolve

TASKS = PROJECT_ROOT / "data/labels/invedrifac_tasks.csv"
LABELS = PROJECT_ROOT / "data/labels/invedrifac_labels.csv"
KEYS = {ord("1"): "blink", ord("2"): "long_closure", ord("3"): "yawn", ord("0"): "none", ord("9"): "unclear"}
WIN = "INVEDRIFAC labelling"
EYE_IDX = [33, 133, 160, 144, 362, 263, 385, 380, 70, 300]  # eye corners, lids, brows
EYE_W, EYE_H = 960, 300
HELP = ["1 blink   2 long closure (>= 0.5 s)   3 yawn   0 nothing   9 unclear",
        "space pause   a/d step   r normal speed   b back   q quit",
        "Closed = no iris visible. Looking down with open (droopy) eyes = NOT closed."]


def load_clip(video: str, t0: float, t1: float) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(PROJECT_ROOT / video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
    frames = []
    for _ in range(int((t1 - t0) * fps) + 1):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


class FaceLocator:
    """Tracks the face on every frame of a clip: eye boxes (per frame) and one face box."""

    def __init__(self):
        self.tracker = FaceTracker(resolve(load_config("cabin")["model_path"]))
        self.t = 0.0

    def track(self, frames: list[np.ndarray]):
        eye_boxes, face = [], None
        self.t += 10.0  # the VIDEO-mode tracker needs increasing timestamps; gap = new clip
        for i, f in enumerate(frames):
            fr = self.tracker.process(f, self.t + i / 30.0)
            if fr is None:
                eye_boxes.append(None)
                continue
            if face is None:
                face = face_box(fr, margin=0.1)
            p = fr.points_px(EYE_IDX)
            eye_boxes.append((p[:, 0].min(), p[:, 0].max(), p[:, 1].mean()))
        self.t += len(frames) / 30.0
        return _steady_eye_boxes(eye_boxes, frames[0].shape), face


def _steady_eye_boxes(raw, shape):
    """Fixed-size eye crop (EYE_W:EYE_H aspect) centred on a smoothed eye position; gaps filled."""
    h, w = shape[:2]
    good = [b for b in raw if b is not None]
    if not good:
        return [None] * len(raw)
    width = float(np.median([b[1] - b[0] for b in good])) * 1.35
    height = width * EYE_H / EYE_W
    cx = np.array([(b[0] + b[1]) / 2 if b else np.nan for b in raw])
    cy = np.array([b[2] if b else np.nan for b in raw])
    idx = np.arange(len(raw))
    ok = ~np.isnan(cx)
    cx, cy = np.interp(idx, idx[ok], cx[ok]), np.interp(idx, idx[ok], cy[ok])
    k = np.ones(5) / 5
    cx = np.convolve(np.pad(cx, 2, mode="edge"), k, "valid")
    cy = np.convolve(np.pad(cy, 2, mode="edge"), k, "valid")
    boxes = []
    for x, y in zip(cx, cy):
        x0 = int(np.clip(x - width / 2, 0, w - width))
        y0 = int(np.clip(y - height / 2, 0, h - height))
        boxes.append((x0, y0, x0 + int(width), y0 + int(height)))
    return boxes


def enhance(crop: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    big = cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(big, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 8)).apply(lab[..., 0])
    big = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(big, (0, 0), 3)
    return cv2.addWeighted(big, 1.6, blur, -0.6, 0)  # unsharp mask


def compose(frame, eye_box, face, info, i, n, progress, paused) -> np.ndarray:
    h, w = frame.shape[:2]
    if eye_box is not None:
        x0, y0, x1, y1 = eye_box
        eyes = enhance(frame[y0:y1, x0:x1], (EYE_W, EYE_H))
    else:
        eyes = np.zeros((EYE_H, EYE_W, 3), np.uint8)
        cv2.putText(eyes, "face not found in this frame", (20, EYE_H // 2), 0, 1.0, (0, 0, 255), 2)
    bottom_h = 330
    full = cv2.resize(frame, (int(w * bottom_h / h), bottom_h))
    if face is not None:
        fx0, fy0, fx1, fy1 = face
        fc = frame[fy0:fy1, fx0:fx1]
    else:
        fc = frame[int(h * 0.1): int(h * 0.8), int(w * 0.2): int(w * 0.8)]
    face_img = enhance(fc, (int(bottom_h * fc.shape[1] / max(fc.shape[0], 1)), bottom_h))
    bottom = np.hstack([full, face_img])
    width = max(EYE_W, bottom.shape[1])
    canvas = np.zeros((EYE_H + bottom_h + 110, width, 3), np.uint8)
    canvas[:EYE_H, :EYE_W] = eyes
    canvas[EYE_H: EYE_H + bottom_h, : bottom.shape[1]] = bottom
    y = EYE_H + bottom_h
    cv2.rectangle(canvas, (0, y + 4), (int((i + 1) / max(n, 1) * width), y + 10), (0, 200, 255), -1)
    status = f"{info}   frame {i + 1}/{n}" + ("   PAUSED" if paused else "")
    cv2.putText(canvas, status, (10, y + 30), 0, 0.6, (255, 255, 255), 1)
    cv2.putText(canvas, progress, (10, y + 52), 0, 0.5, (180, 180, 180), 1)
    for k, line in enumerate(HELP):
        cv2.putText(canvas, line, (10, y + 72 + 17 * k), 0, 0.45, (120, 220, 255) if k == 2 else (200, 200, 200), 1)
    return canvas


def save(labelled: dict) -> None:
    out = pd.DataFrame({"task_id": list(labelled), "label": list(labelled.values())})
    out["labelled_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(LABELS, index=False)


def main() -> None:
    tasks = pd.read_csv(TASKS)
    done = pd.read_csv(LABELS) if LABELS.exists() else pd.DataFrame(columns=["task_id", "label", "labelled_at"])
    labelled = dict(zip(done["task_id"], done["label"]))
    order = list(tasks["task_id"])
    pos = next((k for k, tid in enumerate(order) if tid not in labelled), len(order))
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    locator = FaceLocator()
    while pos < len(order):
        task = tasks[tasks["task_id"] == order[pos]].iloc[0]
        frames, fps = load_clip(task["video"], task["t0"], task["t1"])
        if not frames:
            labelled[task["task_id"]] = "unclear"
            pos += 1
            continue
        eye_boxes, face = locator.track(frames)
        # The proposed type is NOT shown, so it cannot bias the answer.
        info = f"{task['driver']}  t={task['t0']:.1f}s  clip {len(frames) / fps:.1f}s"
        speed, paused, i, key = 0.35, False, 0, -1
        while True:
            progress = f"task {pos + 1}/{len(order)}   labelled {len(labelled)}   speed {speed:.2f}x"
            cv2.imshow(WIN, compose(frames[i], eye_boxes[i], face, info, i, len(frames), progress, paused))
            wait = 0 if paused else max(1, int(1000 / (fps * speed)))
            if not paused and i == len(frames) - 1:
                wait += 500  # short pause between loops
            k = cv2.waitKey(wait) & 0xFF
            if k == ord(" "):
                paused = not paused
            elif k == ord("a"):
                paused, i = True, max(0, i - 1)
            elif k == ord("d"):
                paused, i = True, min(len(frames) - 1, i + 1)
            elif k == ord("r"):
                speed, paused, i = 1.0, False, 0
            elif k in KEYS or k in (ord("b"), ord("q")):
                key = k
                break
            elif not paused:
                i = (i + 1) % len(frames)
        if key == ord("q"):
            break
        if key == ord("b"):
            pos = max(0, pos - 1)
            continue
        labelled[task["task_id"]] = KEYS[key]
        save(labelled)
        pos += 1
    cv2.destroyAllWindows()
    print(f"Labelled {len(labelled)} / {len(order)} tasks -> {LABELS.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
