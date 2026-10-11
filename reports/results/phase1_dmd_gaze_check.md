# Phase 1 — head-pose "eyes-off-road" proxy vs. DMD gaze_on_road labels (2026-10-07)

**Data:** DMD distraction, all 14 subjects, 49 recordings (s1 real driving "Car Moving"; s2/s3 parked "Car Stopped"), face camera, frame-level `gaze_on_road/{looking, not_looking}`.
**Detector:** head yaw / pitch deviation from the personal neutral pose (30° / 20°), ≥ 2 s for an event. No eye-gaze model yet.

| Setup | GT off-road share | Head-deviation AUC | Frame P / R / F1 | Event (≥ 2 s) recall / precision |
|---|---|---|---|---|
| Car moving (s1) | 14 % | 0.72 | 0.34 / 0.31 / 0.32 | 0.02 / 0.02 |
| Car stopped (s2, s3) | 14 % | 0.80 | 0.49 / 0.34 / 0.36 | 0.19 / 0.51 |
| **All** | | **0.77** | F1 **0.35** | **0.14 / 0.30** |

**Conclusion:** head pose alone is a poor eyes-off-road detector. It ranks off-road moments above on-road ones better than chance (AUC 0.77), but most off-road glances are made largely with the eyes, so a head-turn threshold misses them (especially while driving, event recall 2 %). Together with the DashGaze finding (eyes > 15° from centre without matching head motion), this is the evidence for adding the **L2CS-Net eye-gaze model** next (INITIALPLAN §4.4).
