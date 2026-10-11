"""Run L2CS-Net gaze on a video (face crop from MediaPipe landmarks) and save per-frame gaze.

    python scripts/extract_gaze.py video.mp4 --out gaze.csv [--fps 10] [--crop 0 960]

Gaze is computed at --fps (default 10) and the output has one row per processed frame:
frame, t, gaze_yaw, gaze_pitch (degrees, camera frame: includes head rotation).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import pandas as pd

from dms.cabin.gaze import GazeEstimator
from dms.cabin.landmarks import FaceTracker
from dms.utils.config import load_config, resolve


def extract_gaze(video: Path, out: Path | None = None, target_fps: float = 10.0,
                 crop_x: tuple[int, int] | None = None, batch: int = 64, progress: bool = False) -> pd.DataFrame:
    cfg = load_config("cabin")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(fps / target_fps))
    gaze = GazeEstimator(resolve(cfg["gaze_model_path"]))
    rows, frames, faces, idx = [], [], [], []
    t0 = time.time()

    def flush():
        if frames:
            for i, (yaw, pitch) in zip(idx, gaze.estimate(frames, faces)):
                rows.append({"frame": i, "t": i / fps, "gaze_yaw": float(yaw), "gaze_pitch": float(pitch)})
            frames.clear(), faces.clear(), idx.clear()

    with FaceTracker(resolve(cfg["model_path"])) as tracker:
        i = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % stride == 0:
                _, frame = cap.retrieve()
                if crop_x is not None:
                    frame = frame[:, crop_x[0]: crop_x[1]]
                face = tracker.process(frame, i / fps)
                if face is not None:
                    frames.append(frame), faces.append(face), idx.append(i)
                    if len(frames) == batch:
                        flush()
                        if progress:
                            print(f"  frame {i} ({len(rows) / (time.time() - t0):.0f} gaze/s)", flush=True)
            i += 1
    flush()
    cap.release()
    df = pd.DataFrame(rows, columns=["frame", "t", "gaze_yaw", "gaze_pitch"])
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--crop", type=int, nargs=2)
    args = ap.parse_args()
    out = args.out or args.video.with_suffix(".gaze.csv")
    df = extract_gaze(args.video, out, args.fps, tuple(args.crop) if args.crop else None, progress=True)
    print(f"Saved {len(df)} gaze rows to {out}")


if __name__ == "__main__":
    main()
