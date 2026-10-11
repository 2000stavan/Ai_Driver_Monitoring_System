"""Road scene segmentation (Module C, INITIALPLAN §6.2): SegFormer, Cityscapes -> Indian roads (IDD).

v1 uses IDD Lite: 1,403 train / 204 val images at 320x227 with IDD's 7 level-1 classes. The same
code runs on full IDD Segmentation once its level-1 label PNGs are generated (IDD public-code
`createLabels.py --id-type level1Id`), only the data root and image size change.

The Western-only baseline is the Cityscapes model used as-is, with its 19 classes merged into the
7 IDD level-1 classes (CITYSCAPES_TO_L1). Comparing it with the IDD fine-tuned model is the road
half of RQ1/RQ2: how much does a Western road model lose on Indian roads, and how much does
Indian data recover.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

L1_CLASSES = ["drivable", "non_drivable", "living_things", "vehicles", "roadside_objects", "far_objects", "sky"]
IGNORE = 255
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Cityscapes train ids -> IDD level-1 ids (IDD paper, Table 1 hierarchy)
CITYSCAPES_TO_L1 = {
    0: 0,   # road -> drivable
    1: 1,   # sidewalk -> non-drivable
    2: 5,   # building -> far objects
    3: 4,   # wall -> roadside objects
    4: 4,   # fence
    5: 4,   # pole
    6: 4,   # traffic light
    7: 4,   # traffic sign
    8: 5,   # vegetation -> far objects
    9: 1,   # terrain -> non-drivable
    10: 6,  # sky
    11: 2,  # person -> living things
    12: 2,  # rider
    13: 3,  # car -> vehicles
    14: 3,  # truck
    15: 3,  # bus
    16: 3,  # train
    17: 3,  # motorcycle
    18: 3,  # bicycle
}


class IDDLite(torch.utils.data.Dataset):
    """IDD Lite split: leftImg8bit/<split>/<seq>/<id>_image.jpg + gtFine/<split>/<seq>/<id>_label.png."""

    def __init__(self, root: str | Path, split: str, size: tuple[int, int] = (448, 640), train: bool = False):
        root = Path(root)
        self.items = []
        for img in sorted((root / "leftImg8bit" / split).glob("*/*_image.jpg")):
            lbl = root / "gtFine" / split / img.parent.name / img.name.replace("_image.jpg", "_label.png")
            self.items.append((img, lbl if lbl.exists() else None))
        if not self.items:
            raise FileNotFoundError(f"No images under {root / 'leftImg8bit' / split}")
        self.size, self.train = size, train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        img_path, lbl_path = self.items[i]
        img = Image.open(img_path).convert("RGB")
        lbl = Image.open(lbl_path) if lbl_path else Image.new("L", img.size, IGNORE)
        h, w = self.size
        if self.train:
            # scale jitter + random crop back to (h, w), horizontal flip, mild colour jitter.
            # (Flip is fine for pixel classes; it is NOT used for anything that depends on traffic side.)
            s = random.uniform(1.0, 1.5)
            sh, sw = int(h * s), int(w * s)
            img, lbl = img.resize((sw, sh), Image.BILINEAR), lbl.resize((sw, sh), Image.NEAREST)
            y0, x0 = random.randint(0, sh - h), random.randint(0, sw - w)
            img, lbl = img.crop((x0, y0, x0 + w, y0 + h)), lbl.crop((x0, y0, x0 + w, y0 + h))
            if random.random() < 0.5:
                img, lbl = img.transpose(Image.FLIP_LEFT_RIGHT), lbl.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img, lbl = img.resize((w, h), Image.BILINEAR), lbl.resize((w, h), Image.NEAREST)
        x = np.asarray(img, dtype=np.float32) / 255.0
        if self.train:
            x = np.clip(x * random.uniform(0.8, 1.2) + random.uniform(-0.08, 0.08), 0, 1)
        x = (x - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(np.asarray(lbl, dtype=np.int64).copy())


def load_segformer(checkpoint: str, num_labels: int | None = None, cache_dir: str | Path | None = None):
    """A Cityscapes SegFormer (num_labels=None keeps its 19-class head) or a new head with num_labels."""
    from transformers import SegformerForSemanticSegmentation

    kw = {"cache_dir": str(cache_dir) if cache_dir else None}
    if num_labels is not None:
        kw.update(num_labels=num_labels, ignore_mismatched_sizes=True,
                  id2label=dict(enumerate(L1_CLASSES)), label2id={c: i for i, c in enumerate(L1_CLASSES)})
    return SegformerForSemanticSegmentation.from_pretrained(checkpoint, **kw)


def predict_logits(model, x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) logits upsampled to the input size (SegFormer predicts at 1/4 resolution)."""
    logits = model(pixel_values=x).logits
    return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def cityscapes_to_l1(pred19: torch.Tensor) -> torch.Tensor:
    lut = torch.full((256,), IGNORE, dtype=torch.long, device=pred19.device)
    for cs, l1 in CITYSCAPES_TO_L1.items():
        lut[cs] = l1
    return lut[pred19]


class ConfusionMatrix:
    def __init__(self, n: int):
        self.n = n
        self.m = np.zeros((n, n), dtype=np.int64)

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        valid = gt != IGNORE
        idx = gt[valid].cpu().numpy() * self.n + pred[valid].cpu().numpy()
        self.m += np.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)

    def iou(self) -> np.ndarray:
        tp = np.diag(self.m)
        denom = self.m.sum(0) + self.m.sum(1) - tp
        return np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)

    def pixel_acc(self) -> float:
        return float(np.diag(self.m).sum() / max(self.m.sum(), 1))


def overlay(image_rgb: np.ndarray, pred: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Colour the prediction over the image (for the dashboard)."""
    palette = np.array([[128, 64, 128], [244, 35, 232], [220, 20, 60], [0, 0, 142],
                        [250, 170, 30], [107, 142, 35], [70, 130, 180]], dtype=np.uint8)
    colour = palette[np.clip(pred, 0, len(palette) - 1)]
    return (image_rgb * (1 - alpha) + colour * alpha).astype(np.uint8)
