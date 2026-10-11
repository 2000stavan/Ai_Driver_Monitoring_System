# DMD drowsiness — subject-wise tuned detectors (leave-one-subject-out)

13 subjects, 16 recordings. For each held-out subject, thresholds were chosen on the other subjects only. Means over held-out recordings.

| Detector | Untuned defaults | Tuned (held-out subjects) |
|---|---|---|
| Eyes closed, frame F1 | 0.88 | 0.93 |
| Blinks, event F1 | 0.97 | 0.95 |
| Yawns, event F1 | 0.91 | 0.94 |
| Yawns, recall / precision | 0.88 / 0.97 | 0.92 / 0.97 |

Thresholds chosen across folds (how many held-out recordings used each setting):

Eyes: {'eyes.closed_ear_ratio': 0.4, 'eyes.closed_blink_score': 0.4} × 11; {'eyes.closed_ear_ratio': 0.55, 'eyes.closed_blink_score': 0.45} × 4; {'eyes.closed_ear_ratio': 0.45, 'eyes.closed_blink_score': 0.4} × 1

Yawn: {'mouth.yawn_mar': 0.2, 'mouth.yawn_jaw_score': 0.2, 'mouth.hand_mouth_dist': 0.2, 'mouth.yawn_hand_min_s': 2.0} × 11; {'mouth.yawn_mar': 0.2, 'mouth.yawn_jaw_score': 0.2, 'mouth.hand_mouth_dist': 0.2, 'mouth.yawn_hand_min_s': 1.5} × 2; {'mouth.yawn_mar': 0.2, 'mouth.yawn_jaw_score': 0.2, 'mouth.hand_mouth_dist': 0.2, 'mouth.yawn_hand_min_s': 2.5} × 2; {'mouth.yawn_mar': 0.2, 'mouth.yawn_jaw_score': 0.2, 'mouth.hand_mouth_dist': 0.15, 'mouth.yawn_hand_min_s': 1.5} × 1

Caveat: DMD drowsiness is acted and non-Indian; this tunes the event detectors only.
## Decision (2026-10-07): do not adopt these values yet
With the wider grid, most folds again chose the **lowest values in the grid** (EAR ratio 0.40, blink score 0.40; mouth MAR / jaw 0.20). Edge-of-grid choices on acted, exaggerated DMD yawns and closures risk fitting DMD rather than real drivers, and the earlier DMD-tuned eye thresholds were already slightly *worse* on the Indian DashGaze driver (blink F1 0.80 vs 0.83 with defaults). `configs/cabin.yaml` keeps the defaults; final thresholds will be chosen on Indian data (INVEDRIFAC / full DashGaze) with subject-wise splits.
