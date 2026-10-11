"""Run the pretrained face landmarker on a video and save per-frame features to CSV.

This is the "video -> small feature table" step: the CSV is kept, the video can be deleted.

    python scripts/extract_cabin_features.py path/to/video.mp4 [--out features.csv]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import pandas as pd

from dms.cabin.features import frame_features
from dms.cabin.hands import HandTracker, hand_face_features
from dms.cabin.landmarks import FaceTracker
from dms.utils.config import load_config, resolve


def extract(
    video: Path,
    out: Path | None = None,
    progress: bool = True,
    hands: bool = False,
    target_fps: float | None = None,
    hand_fps: float = 10.0,
    crop_x: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Per-frame features. target_fps < source fps skips frames (e.g. 15 fps still resolves blinks).
    Hands are searched at hand_fps and carried forward in between (the hand search is the slow part).
    crop_x=(x0, x1) keeps only those columns (e.g. the driver view of a side-by-side video)."""
    cfg = load_config("cabin")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, round(fps / target_fps)) if target_fps else 1
    hand_stride = max(1, round(min(fps / stride, fps) / hand_fps))
    hand_feats: dict = {}
    rows = []
    t0 = time.time()
    hand_tracker = HandTracker(resolve(cfg["hand_model_path"])) if hands else None
    with FaceTracker(resolve(cfg["model_path"])) as tracker:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            if i % stride:
                i += 1
                continue
            if crop_x is not None:
                frame = frame[:, crop_x[0] : crop_x[1]]
            face = tracker.process(frame, t)
            row = {"frame": i, "t": t, "face": face is not None}
            if face is not None:
                row.update(frame_features(face))
            if hand_tracker is not None:
                if len(rows) % hand_stride == 0 or not hand_feats:
                    hand_feats = hand_face_features(hand_tracker.process(frame, t), face)
                row.update(hand_feats)
            rows.append(row)
            i += 1
            if progress and len(rows) % 500 == 0:
                print(f"  {i}/{total} frames ({len(rows) / (time.time() - t0):.0f} processed/s)")
    cap.release()
    if hand_tracker is not None:
        hand_tracker.close()
    df = pd.DataFrame(rows)
    df.attrs["fps"] = fps
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--hands", action="store_true", help="also add hand-to-mouth / hand-to-ear features")
    ap.add_argument("--fps", type=float, help="process at this frame rate (skips frames)")
    args = ap.parse_args()
    out = args.out or args.video.with_suffix(".features.csv")
    df = extract(args.video, out, hands=args.hands, target_fps=args.fps)
    print(f"Saved {len(df)} frames ({df['face'].mean():.1%} with a face) to {out}")


if __name__ == "__main__":
    main()
