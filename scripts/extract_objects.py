"""Run YOLO11 (COCO, pretrained) on a video and save per-frame object cues.

    python scripts/extract_objects.py video.mp4 --out objects.csv [--fps 5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from dms.cabin.objects import ObjectDetector
from dms.utils.config import PROJECT_ROOT


def extract_objects(video: Path, out: Path | None = None, target_fps: float = 5.0, batch: int = 16,
                    weights: Path = PROJECT_ROOT / "models/yolo11m.pt") -> pd.DataFrame:
    det = ObjectDetector(weights)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(fps / target_fps))
    rows, frames, idx = [], [], []

    def flush():
        if frames:
            for i, feats in zip(idx, det.detect(frames)):
                rows.append({"frame": i, "t": i / fps, **feats})
            frames.clear(), idx.clear()

    i = 0
    while cap.grab():
        if i % stride == 0:
            _, frame = cap.retrieve()
            frames.append(frame), idx.append(i)
            if len(frames) == batch:
                flush()
        i += 1
    flush()
    cap.release()
    df = pd.DataFrame(rows)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--fps", type=float, default=5.0)
    args = ap.parse_args()
    out = args.out or args.video.with_suffix(".objects.csv")
    df = extract_objects(args.video, out, args.fps)
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
