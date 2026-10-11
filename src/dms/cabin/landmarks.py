"""Thin wrapper around the pretrained MediaPipe Face Landmarker (Tasks API, mediapipe >= 1.x).

Used as-is, no training (INITIALPLAN §4.1). Outputs 478 landmarks, 52 blendshapes and a
facial transformation matrix (used for head pose).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision


@dataclass
class FaceFrame:
    landmarks: np.ndarray  # (478, 3) normalised x, y in [0, 1], z relative depth
    blendshapes: dict[str, float]
    transform: np.ndarray | None  # (4, 4) canonical-face -> camera transform
    width: int
    height: int

    def points_px(self, idx) -> np.ndarray:
        """Landmarks in pixel coordinates (x * width, y * height)."""
        pts = self.landmarks[idx, :2]
        return pts * np.array([self.width, self.height], dtype=np.float32)


class FaceTracker:
    """Runs the landmarker on a stream of BGR frames (video or webcam)."""

    def __init__(self, model_path: str | Path, min_confidence: float = 0.5):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, timestamp_s: float) -> FaceFrame | None:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(timestamp_s * 1000), self._last_ts + 1)  # must strictly increase
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None
        lms = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]], dtype=np.float32)
        shapes = {}
        if result.face_blendshapes:
            shapes = {c.category_name: c.score for c in result.face_blendshapes[0]}
        transform = None
        if result.facial_transformation_matrixes:
            transform = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32)
        return FaceFrame(lms, shapes, transform, w, h)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
