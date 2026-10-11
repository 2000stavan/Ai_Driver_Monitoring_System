# Phase 5 (started) — phone / texting / drinking on DMD distraction — working notes (2026-10-07)

## Setup
- **Camera choice: face camera, not body camera.** In DMD's side-view body camera the phone is hidden behind the hand/head during calls and small when texting (body-camera phone AUC only 0.67 on gA-1 s2). The frontal face camera shows phone-at-ear and bottle-at-mouth clearly and is the closest match to our own dashboard-phone setup.
- Features per 1 s window: YOLO11m (COCO, pretrained, 5 fps) phone / drink / food confidence + position + size; MediaPipe hand-to-ear / hand-to-mouth distances; head pitch / yaw deviation; mouth ratio.
- Detectors: zero-shot rules; XGBoost (class-balanced) with leave-one-subject-out; feature ablation hands / objects / all.
- Labels: DMD `driver_actions` → safe / phone_call / texting / drinking / other. DMD's "drinking" covers the whole episode, including holding the bottle in the lap.
- Body-camera sync: `frame_shift` varies per recording (e.g. 5 frames for gA-1 s2); negligible for minute-long actions.

## Interim result (2 subjects, 6 recordings, 3,096 windows — pipeline check only)
| Detector | Phone call F1 | Texting F1 | Drinking F1 |
|---|---|---|---|
| Rules (zero-shot) | 0.92 | 0.00 | 0.85 |
| XGBoost hands | 0.91 | 0.26 | 0.37 |
| XGBoost objects | 0.67 | 0.25 | 0.86 |
| XGBoost all | 0.92 | 0.24 | 0.87 |

**Texting is the weak class:** the phone is held low, below the face camera's view. Options: body camera as an extra view, a wider-angle / lower-mounted dashboard phone in our own recordings, gaze-down cues (L2CS). To decide after the full run.

## Full result (2026-10-07): 14 subjects, 49 recordings, 26,041 one-second windows, leave-one-subject-out
| Detector | Phone call F1 | Texting F1 | Drinking F1 | Safe F1 |
|---|---|---|---|---|
| Rules (zero-shot) | 0.94 | 0.02 | 0.81 | 0.69 |
| XGBoost hands | 0.96 | 0.38 | 0.57 | 0.70 |
| XGBoost objects | 0.97 | 0.22 | 0.85 | 0.03 |
| **XGBoost all** | **0.99** | **0.38** | **0.89** | **0.70** |

- **Phone calls are essentially solved from a frontal camera** (phone + hand at ear): 5,965 / 6,051 windows correct.
- **Drinking: F1 0.89** (785 / 846 found); errors are mostly bottle-in-lap moments that DMD still labels as drinking.
- **Texting is the open problem: F1 0.38** (recall 58 %, precision ~28 %). Texting is confused with *safe* and *other* (radio, reaching): the phone is held below the frame, so the camera mainly sees "looking down". This directly motivates (a) a wider / lower dashboard-phone view in our own recordings (recording/PROTOCOL.md already asks for it), and (b) better gaze (L2CS) and a second view — internship-level work.
- Objects and hand cues are complementary: neither alone reaches the combined score.
- Caveats: non-Indian, left-hand-drive car, s2/s3 acted while parked. Indian RHD performance can only be measured on our own recordings.

## Pending (superseded)
Full run on all 14 subjects / 49 recordings once YOLO face-camera extraction finishes (`outputs/run_objects.sh`), then re-run `scripts/eval_dmd_distraction.py`. Eating cannot be tested on DMD (no eating class).
