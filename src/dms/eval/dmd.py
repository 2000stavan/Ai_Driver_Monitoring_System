"""Reader for DMD OpenLABEL annotation files.

Labels are frame intervals per action type, indexed on the annotation timeline. Each camera
stream has a `sync.frame_shift`; for the face camera it is 0, so annotation frame i is face-video
frame i. Other streams (body: e.g. 54, hands: e.g. 74) must be shifted before use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DMDAnnotation:
    path: Path
    n_frames: int
    labels: dict[str, np.ndarray]  # action type -> boolean array over annotation frames
    frame_shift: dict[str, int]  # stream name -> sync frame shift
    context: dict[str, str]

    def mask(self, *types: str) -> np.ndarray:
        """OR of the given action types (missing types count as all-False)."""
        out = np.zeros(self.n_frames, dtype=bool)
        for t in types:
            if t in self.labels:
                out |= self.labels[t]
        return out

    def segments(self, *types: str) -> list[tuple[int, int]]:
        """Contiguous (start, end) frame runs of the OR of the given types, end inclusive."""
        m = self.mask(*types).astype(np.int8)
        d = np.diff(np.concatenate([[0], m, [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
        return list(zip(starts.tolist(), ends.tolist()))


def load_dmd(path: str | Path) -> DMDAnnotation:
    path = Path(path)
    root = json.loads(path.read_text())["openlabel"]
    fi = root["frame_intervals"][0]
    n = fi["frame_end"] + 1
    labels: dict[str, np.ndarray] = {}
    for action in root.get("actions", {}).values():
        arr = labels.setdefault(action["type"], np.zeros(n, dtype=bool))
        for iv in action.get("frame_intervals", []):
            arr[iv["frame_start"] : iv["frame_end"] + 1] = True
    shifts = {
        name: s.get("stream_properties", {}).get("sync", {}).get("frame_shift", 0)
        for name, s in root.get("streams", {}).items()
    }
    context = {}
    for c in root.get("contexts", {}).values():
        for item in c.get("context_data", {}).get("text", []):
            context[item["name"]] = item["val"]
    return DMDAnnotation(path, n, labels, shifts, context)
