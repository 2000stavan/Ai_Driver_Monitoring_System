"""Low-light preprocessing for the cabin camera (INITIALPLAN §15: night / dark cabins).

Indian cabin footage is often dark (no IR camera, night drives, tinted glass). MediaPipe's face
detector is trained mostly on well-lit faces, so a dark frame is brightened before landmarking.
Frames that are already bright are left untouched, so daytime results do not change.

  gamma : lifts dark tones (out = in ** (1 / gamma)) - cheap, keeps colours
  clahe : local contrast equalisation on the L channel (LAB) - recovers faces in uneven light
"""

from __future__ import annotations

import cv2
import numpy as np

_LUT_CACHE: dict[float, np.ndarray] = {}


def brightness(frame_bgr: np.ndarray) -> float:
    """Mean luma 0-255 on a downscaled grey image."""
    small = cv2.resize(frame_bgr, (64, 48), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


def gamma(frame_bgr: np.ndarray, g: float) -> np.ndarray:
    lut = _LUT_CACHE.get(g)
    if lut is None:
        lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / g)) * 255.0, 0, 255).astype(np.uint8)
        _LUT_CACHE[g] = lut
    return cv2.LUT(frame_bgr, lut)


def clahe(frame_bgr: np.ndarray, clip: float = 3.0, tiles: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class LowLightEnhancer:
    """Brightens a frame only when its mean luma is below `dark_below`."""

    def __init__(self, method: str = "gamma_clahe", dark_below: float = 70.0, target: float = 110.0):
        if method not in {"none", "gamma", "clahe", "gamma_clahe"}:
            raise ValueError(method)
        self.method, self.dark_below, self.target = method, dark_below, target

    def __call__(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
        """Returns (frame, was_enhanced)."""
        if self.method == "none":
            return frame_bgr, False
        b = brightness(frame_bgr)
        if b >= self.dark_below:
            return frame_bgr, False
        out = frame_bgr
        if "gamma" in self.method:
            # choose gamma so the mean moves roughly to `target` (mean ** (1/g) = target)
            m = max(b, 1.0) / 255.0
            g = float(np.clip(np.log(m) / np.log(self.target / 255.0), 1.0, 4.0))
            out = gamma(out, round(g, 1))
        if "clahe" in self.method:
            out = clahe(out)
        return out, True
