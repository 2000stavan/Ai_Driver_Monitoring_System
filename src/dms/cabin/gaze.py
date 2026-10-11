"""Eye-gaze estimation with the pretrained L2CS-Net (INITIALPLAN §4.4).

L2CS-Net (Abdelrahman et al., 2022; MIT licence): ResNet50 + two 90-bin heads (4° bins over
[-180°, 180°]); the continuous angle is the softmax expectation. Weights: official Gaze360
checkpoint, re-hosted by py-feat as safetensors (huggingface.co/py-feat/l2cs, values unchanged).

Preprocessing follows the upstream pipeline: face crop -> RGB -> 224x224 -> Resize(448) ->
ImageNet normalisation. The upstream *pipeline* calls the first head's output "pitch", but the
layer is named fc_yaw_gaze, and on DashGaze (true gaze projected on the road view) the first output
tracks **horizontal** gaze (r = +0.46 vs +0.25 for the second). So the first output is yaw:
`head_order="yaw_pitch"` (default).

The face crop comes from the MediaPipe landmarks we already compute, so no extra face detector.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

from dms.cabin.landmarks import FaceFrame

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class L2CS(nn.Module):
    """Same layer names as upstream, so the official weights load unchanged."""

    def __init__(self, layers=(3, 4, 6, 3), num_bins: int = 90):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_yaw_gaze = nn.Linear(512 * Bottleneck.expansion, num_bins)
        self.fc_pitch_gaze = nn.Linear(512 * Bottleneck.expansion, num_bins)
        self.fc_finetune = nn.Linear(512 * Bottleneck.expansion + 3, 3)  # unused (upstream vestige)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        layers += [Bottleneck(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc_yaw_gaze(x), self.fc_pitch_gaze(x)


def face_box(face: FaceFrame, margin: float = 0.15) -> tuple[int, int, int, int]:
    """Square-ish crop around the face landmarks, similar in extent to a face-detector box."""
    pts = face.landmarks[:, :2] * np.array([face.width, face.height])
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w, h = x1 - x0, y1 - y0
    x0, x1 = x0 - margin * w, x1 + margin * w
    y0, y1 = y0 - margin * h, y1 + margin * h
    return (max(0, int(x0)), max(0, int(y0)), min(face.width, int(x1)), min(face.height, int(y1)))


class GazeEstimator:
    def __init__(self, weights: str | Path, device: str | None = None, head_order: str = "yaw_pitch"):
        from safetensors.torch import load_file

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = L2CS()
        state = load_file(str(weights))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        real_missing = [k for k in missing if not k.startswith("fc_finetune")]
        if real_missing or unexpected:
            raise RuntimeError(f"L2CS weights do not match: missing={real_missing[:5]} unexpected={unexpected[:5]}")
        self.model.to(self.device).eval()
        self.bins = torch.arange(90, dtype=torch.float32, device=self.device)
        self.head_order = head_order

    def _prep(self, frame_bgr: np.ndarray, box) -> np.ndarray:
        x0, y0, x1, y1 = box
        crop = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (224, 224))
        crop = cv2.resize(crop, (448, 448), interpolation=cv2.INTER_LINEAR)
        return ((crop.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)

    @torch.no_grad()
    def estimate(self, frames_bgr: list[np.ndarray], faces: list[FaceFrame]) -> np.ndarray:
        """Returns (N, 2) array of (gaze_pitch_deg, gaze_yaw_deg)."""
        if not frames_bgr:
            return np.zeros((0, 2), dtype=np.float32)
        batch = np.stack([self._prep(f, face_box(face)) for f, face in zip(frames_bgr, faces)])
        a, b = self.model(torch.from_numpy(batch).to(self.device))
        a = (torch.softmax(a, 1) * self.bins).sum(1) * 4 - 180
        b = (torch.softmax(b, 1) * self.bins).sum(1) * 4 - 180
        a, b = a.cpu().numpy(), b.cpu().numpy()
        yaw, pitch = (a, b) if self.head_order == "yaw_pitch" else (b, a)
        return np.stack([pitch, yaw], axis=1)
