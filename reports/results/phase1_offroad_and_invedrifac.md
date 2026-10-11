# Phase 1 — eyes-off-road with L2CS gaze (DMD) and INVEDRIFAC robustness profile — 2026-10-07

## 1. Eyes-off-road: head pose vs. L2CS-Net gaze vs. both (DMD distraction, 14 subjects, 49 recordings)
Deviation from the driver's own neutral (first 10 s); thresholds and minimum duration chosen leave-one-subject-out.

| Signal | AUC | Frame F1 | Event recall (≥ 2 s) | Event precision |
|---|---|---|---|---|
| Head pose (MediaPipe) | 0.78 | 0.37 | 0.68 | 0.16 |
| **Gaze (L2CS-Net, pretrained Gaze360)** | **0.85** | **0.45** | **0.78** | **0.21** |
| Head + gaze | 0.85 | 0.45 | 0.76 | 0.20 |

- Gaze clearly beats head pose (AUC +0.07, frame F1 +0.08, event recall +10 points); adding head pose to gaze adds nothing, as expected because L2CS gaze already includes head rotation.
- Event precision is low, especially while driving (0.05): most real off-road moments in s1 are short glances (< 2 s, e.g. mirrors) that do not count as ground-truth events, while our detector still fires on longer gaze shifts. Frame-level F1 is the fairer measure here.
- Remaining gap → gaze-zone classifier (mirror vs. dashboard vs. off-road) fine-tuned on DashGaze, and learned temporal models (v2 / internship).

## 2. INVEDRIFAC (24 Indian drivers, 6.6 h, no labels) — robustness and plausibility
Pretrained MediaPipe + rule-based monitor at 15 fps, default thresholds, 10 s personal calibration. **No ground truth**, so these are plausibility checks, not accuracy. Per-driver table: `invedrifac_profile.csv`.

- **Face found in 91 % of frames (median)**; failures concentrate in hard conditions: **Sub 04 1 %** (very dark night clip), Sub 05 74 %, Sub 15 79 %.
- **Blink rate median 11.6 / min**: physiologically plausible (typical 10–20 / min).
- **Implausible outputs reveal two failure modes:**
  - **Calibration failure — Sub 26:** baseline "open-eye" EAR 0.52 (others 0.28–0.42) → 285 microsleeps and PERCLOS 69 %. The 10 s calibration captured unusual frames; calibration needs outlier rejection and a longer window.
  - **Occlusion / short clip — Sub 02:** 112 blinks / min on a 1.5 min clip with the lens partly blocked.
  - Several drivers with very low blink rates (Sub 09, 11, 14: ~4–5 / min) may be misses caused by the low, upward-looking camera angle.
- **Zero yawns in 6.6 h** — either none occur or the mouth rules do not transfer to this camera geometry; cannot tell without labels.

**Conclusion:** Indian in-vehicle footage exposes failure modes (darkness, low camera angle, calibration) that DMD did not. Proper Indian accuracy numbers need labels → ask the authors for ground truth and label a subset ourselves (check-and-confirm tool).
