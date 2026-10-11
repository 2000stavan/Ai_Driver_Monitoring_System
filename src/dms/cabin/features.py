"""Per-frame cabin features (feature engineering, INITIALPLAN §4.2).

EAR  = eye aspect ratio (Soukupová & Čech, 2016), computed on MediaPipe landmarks.
MAR  = mouth aspect ratio on the inner-lip landmarks.
Head = yaw / pitch / roll from MediaPipe's facial transformation matrix.
"""

from __future__ import annotations

import math

import numpy as np

from dms.cabin.landmarks import FaceFrame

# MediaPipe Face Mesh indices, ordered p1..p6 as in the EAR paper:
# p1/p4 = eye corners, (p2, p6) and (p3, p5) = vertical lid pairs.
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Inner lips: corners (78, 308) and three vertical pairs.
MOUTH_CORNERS = [78, 308]
MOUTH_PAIRS = [(81, 178), (13, 14), (311, 402)]


def eye_aspect_ratio(p: np.ndarray) -> float:
    """p: (6, 2) points p1..p6 in pixels."""
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return float(vertical / (2.0 * horizontal)) if horizontal > 0 else 0.0


def mouth_aspect_ratio(face: FaceFrame) -> float:
    corners = face.points_px(MOUTH_CORNERS)
    width = np.linalg.norm(corners[0] - corners[1])
    if width <= 0:
        return 0.0
    heights = [np.linalg.norm(np.diff(face.points_px(list(pair)), axis=0)) for pair in MOUTH_PAIRS]
    return float(np.mean(heights) / width)


def head_pose_deg(transform: np.ndarray | None) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees from a 4x4 transform. Used relative to a personal baseline,
    so only consistency matters, not the absolute sign convention."""
    if transform is None:
        return (math.nan, math.nan, math.nan)
    r = transform[:3, :3]
    sy = math.hypot(r[0, 0], r[1, 0])
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], sy))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return (yaw, pitch, roll)


def frame_features(face: FaceFrame) -> dict[str, float]:
    ear_l = eye_aspect_ratio(face.points_px(LEFT_EYE))
    ear_r = eye_aspect_ratio(face.points_px(RIGHT_EYE))
    yaw, pitch, roll = head_pose_deg(face.transform)
    bs = face.blendshapes
    return {
        "ear_left": ear_l,
        "ear_right": ear_r,
        "ear": (ear_l + ear_r) / 2.0,
        "mar": mouth_aspect_ratio(face),
        "blink_score": (bs.get("eyeBlinkLeft", math.nan) + bs.get("eyeBlinkRight", math.nan)) / 2.0,
        "jaw_open": bs.get("jawOpen", math.nan),
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
    }
