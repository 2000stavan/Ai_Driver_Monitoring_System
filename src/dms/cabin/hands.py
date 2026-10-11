"""Hand cues from the pretrained MediaPipe Hand Landmarker (INITIALPLAN §4.5).

Distances from the hands to the mouth and ears, normalised by face width, so the same
thresholds work at any camera distance. Used for yawning with a hand over the mouth, and later
for phone use (hand at ear), eating and drinking (hand at mouth).
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from dms.cabin.landmarks import FaceFrame

MOUTH_CENTRE = [13, 14]  # inner upper / lower lip
EAR_LEFT, EAR_RIGHT = 454, 234  # face-contour points next to the ears (subject's left / right)


class HandTracker:
    def __init__(self, model_path: str | Path, num_hands: int = 2, min_confidence: float = 0.4):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, timestamp_s: float) -> list[np.ndarray]:
        """Returns one (21, 2) array of pixel coordinates per detected hand."""
        h, w = frame_bgr.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        ts = max(int(timestamp_s * 1000), self._last_ts + 1)
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        return [
            np.array([[p.x * w, p.y * h] for p in hand], dtype=np.float32)
            for hand in result.hand_landmarks
        ]

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def hand_face_features(hands: list[np.ndarray], face: FaceFrame | None) -> dict[str, float]:
    """Minimum hand-to-mouth and hand-to-ear distance in face widths (nan if no hand or no face)."""
    out = {"n_hands": float(len(hands)), "hand_mouth": math.nan, "hand_ear": math.nan}
    if not hands or face is None:
        return out
    ears = face.points_px([EAR_LEFT, EAR_RIGHT])
    face_w = float(np.linalg.norm(ears[0] - ears[1]))
    if face_w <= 0:
        return out
    mouth = face.points_px(MOUTH_CENTRE).mean(axis=0)
    pts = np.concatenate(hands)
    out["hand_mouth"] = float(np.min(np.linalg.norm(pts - mouth, axis=1)) / face_w)
    out["hand_ear"] = float(min(np.min(np.linalg.norm(pts - e, axis=1)) for e in ears) / face_w)
    return out
