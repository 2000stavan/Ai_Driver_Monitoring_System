# Phase 1 — first Indian real-driving check (DashGaze sample) — 2026-10-07

**Data:** DashGaze sample, **1 Indian driver**, right-hand-drive car, real traffic, 15.3 min (23,237 frames, 25.3 fps). Driver view = left 960 px of the triplet video. Ground truth from the **eye-tracking glasses** the driver wears: 443 blinks (`blink id`), glasses IMU roll / pitch.
**Model:** pretrained MediaPipe Face Landmarker + rule-based monitor (`configs/cabin.yaml`), personal calibration on the first 10 s. No training on Indian data.

## Blinks (event level, ±3 frames)

| Thresholds | GT blinks | Detected | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Defaults | 443 | 318 | **0.74** | **0.94** | **0.83** |
| Tuned on DMD (non-Indian, leave-one-subject-out choice) | 443 | 287 | 0.69 | 0.94 | 0.80 |

For comparison, on DMD (non-Indian, acted, parked, no tracker glasses) blink recall was 0.97 and precision 0.97.

- Face found in **96 %** of frames.
- **Precision stays high (0.94)**: almost everything we call a blink is a real blink.
- **Recall drops (0.97 → 0.74)**: about a quarter of real blinks are missed. Likely causes, not yet separated: the tracker glasses' thick frames partly cover the eyes; real-driving lighting; a different person; and the tracker counts partial blinks (GT blink median 0.40 s, p10 0.20 s) that may never cross our "closed" threshold.
- **Thresholds tuned on non-Indian DMD did slightly worse** on this Indian driver (F1 0.80 vs 0.83). With one driver this is only a hint, but it is the kind of transfer gap RQ1 asks about.
- Frame-level overlap with the tracker's blink frames is low (F1 ≈ 0.1) because the tracker marks the whole blink window (closing + opening) while we mark only fully closed frames; the event-level comparison above is the meaningful one.

## Head pose vs. glasses IMU — inconclusive

Correlation of our head pitch / roll with the glasses IMU is weak: raw −0.29 / +0.35, detrended (10 s) −0.37 / +0.27 (the sign flip on pitch is a convention difference). The IMU angles are relative to gravity and include car motion and drift, and head movement in this drive is small, so this neither validates nor rejects our head-pose estimate. Eye-in-head gaze is > 15° from centre in 4.4 % of frames: eyes move without the head, which is why L2CS-Net gaze is needed.

## Caveats
One driver, always wearing tracker glasses: indicative only. The full DashGaze set (28 drivers) and INVEDRIFAC are needed for real Indian numbers.
