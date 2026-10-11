"""Does low-light enhancement help the face landmarker on Indian cabin video?

Samples frames from every INVEDRIFAC video and measures the face-detection rate with each
preprocessing method (forced on every frame, so the effect is visible even on bright videos).
No labels are needed: a face that is found and tracked is the precondition for every cabin feature.

    python scripts/eval_low_light.py [--per-video 200]
Writes reports/results/low_light_eval.md
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd

from dms.cabin.enhance import brightness, clahe, gamma
from dms.cabin.features import frame_features
from dms.cabin.landmarks import FaceTracker
from dms.utils.config import PROJECT_ROOT, load_config, resolve

VIDEOS = PROJECT_ROOT / "data/raw/cabin/invedrifac"
OUT = PROJECT_ROOT / "reports/results/low_light_eval.md"


def auto_gamma(f: np.ndarray, target: float = 110.0) -> np.ndarray:
    m = max(brightness(f), 1.0) / 255.0
    g = float(np.clip(np.log(m) / np.log(target / 255.0), 1.0, 4.0))
    return gamma(f, round(g, 1))


METHODS = {
    "none": lambda f: f,
    "gamma": auto_gamma,
    "clahe": clahe,
    "gamma_clahe": lambda f: clahe(auto_gamma(f)),
}


def sample_frames(video, n: int, skip_s: float = 60.0) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = int(skip_s * fps) if total > 2 * skip_s * fps else 0
    # runs of 5 consecutive frames so the VIDEO-mode tracker behaves as on real video
    starts = np.linspace(start, total - 6, max(n // 5, 1)).astype(int)
    frames = []
    for s in starts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s))
        for _ in range(5):
            ok, f = cap.read()
            if ok:
                frames.append(f)
    cap.release()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-video", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config("cabin")
    rows = []
    for video in sorted(VIDEOS.glob("*.avi"), key=lambda p: int("".join(c for c in p.stem if c.isdigit()))):
        frames = sample_frames(video, args.per_video)
        row = {"driver": video.stem.replace(" ", "_"), "brightness": np.mean([brightness(f) for f in frames])}
        for name, fn in METHODS.items():
            with FaceTracker(resolve(cfg["model_path"])) as tracker:
                found, ears = 0, []
                for i, f in enumerate(frames):
                    face = tracker.process(fn(f), i / 15.0)
                    if face is not None:
                        found += 1
                        ears.append(frame_features(face)["ear"])
            row[f"face_{name}"] = found / max(len(frames), 1)
            row[f"ear_{name}"] = float(np.median(ears)) if ears else np.nan
        rows.append(row)
        print({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items() if not k.startswith("ear")})
    d = pd.DataFrame(rows)
    dark = d["brightness"] < 70
    face_cols = [f"face_{m}" for m in METHODS]
    lines = ["# Low-light preprocessing - face detection on INVEDRIFAC", "",
             f"{args.per_video} sampled frames per video (after the first minute), each method forced on every frame.", "",
             "| videos | n | " + " | ".join(METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]
    for label, mask in (("all", d.index == d.index), ("dark (mean luma < 70)", dark), ("bright", ~dark)):
        lines.append(f"| {label} | {int(mask.sum())} | " + " | ".join(f"{d.loc[mask, c].mean():.3f}" for c in face_cols) + " |")
    lines += ["", "Per video (face-detection rate; brightness = mean luma 0-255):", "",
              d[["driver", "brightness", *face_cols]].round(3).to_markdown(index=False), "",
              "Median EAR with each method (enhancement must not distort the eye measurement):", "",
              d[["driver", *[f"ear_{m}" for m in METHODS]]].round(3).to_markdown(index=False)]
    OUT.write_text("\n".join(lines))
    print("\n".join(lines[:10]))


if __name__ == "__main__":
    main()
