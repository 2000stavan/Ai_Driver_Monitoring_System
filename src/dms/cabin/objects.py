"""In-cabin object cues from the pretrained YOLO11 (COCO) detector (INITIALPLAN §4.5).

COCO already contains the objects that matter for driver distraction: cell phone, bottle, cup,
wine glass and several foods. Per frame we keep the strongest detection of each group (confidence,
centre, size, normalised to the image), which a classifier combines with hand cues.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

GROUPS = {
    "phone": [67],                       # cell phone
    "drink": [39, 40, 41],               # bottle, wine glass, cup
    "food": [46, 47, 48, 49, 52, 53, 54, 55],  # banana, apple, sandwich, orange, hot dog, pizza, donut, cake
    "person": [0],
}
ALL_CLASSES = sorted({c for ids in GROUPS.values() for c in ids})


class ObjectDetector:
    def __init__(self, weights: str | Path, conf: float = 0.1, imgsz: int = 640, device: str | None = None):
        from ultralytics import YOLO

        import torch

        self.model = YOLO(str(weights))
        self.conf, self.imgsz = conf, imgsz
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

    def detect(self, frames_bgr: list[np.ndarray]) -> list[dict[str, float]]:
        """One feature dict per frame: <group>_conf, _cx, _cy, _area (nan when not detected)."""
        results = self.model.predict(frames_bgr, conf=self.conf, imgsz=self.imgsz, classes=ALL_CLASSES,
                                     device=self.device, verbose=False)
        out = []
        for res in results:
            h, w = res.orig_shape
            boxes = res.boxes
            cls = boxes.cls.cpu().numpy().astype(int) if len(boxes) else np.array([], dtype=int)
            conf = boxes.conf.cpu().numpy() if len(boxes) else np.array([])
            xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.zeros((0, 4))
            row = {}
            for group, ids in GROUPS.items():
                sel = np.flatnonzero(np.isin(cls, ids))
                if sel.size:
                    k = sel[np.argmax(conf[sel])]
                    x0, y0, x1, y1 = xyxy[k]
                    row.update({f"{group}_conf": float(conf[k]), f"{group}_cx": float((x0 + x1) / 2 / w),
                                f"{group}_cy": float((y0 + y1) / 2 / h),
                                f"{group}_area": float((x1 - x0) * (y1 - y0) / (w * h))})
                else:
                    row.update({f"{group}_conf": 0.0, f"{group}_cx": np.nan, f"{group}_cy": np.nan,
                                f"{group}_area": 0.0})
            out.append(row)
        return out
