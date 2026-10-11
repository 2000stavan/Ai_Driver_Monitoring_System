# INITIAL PLAN — AI-based Driver Monitoring System for Indian Roads

> **Status:** v1 base plan, frozen 2026-10-07. **Source of truth** for models, datasets, downloads and phases. This is the plan we **start building from**. It will be revised as results come in. Everything marked **v2** is a planned optimisation, not part of the first build.
>
> **Goal:** the best possible output, not the fastest build. Time is not a constraint, but compute (free Colab/Kaggle) and dataset licences are.
>
> Related files: [`PLAN.md`](PLAN.md) (research design, rationale, limitations, decision log), [`DATASETS.md`](DATASETS.md) (full dataset survey with links and licences).

---

## Contents
1. [Problem and goal](#1-problem-and-goal)
2. [Core strategy](#2-core-strategy)
3. [System architecture](#3-system-architecture)
4. [Module A — Cabin (fatigue + distraction)](#4-module-a--cabin)
5. [Module B — Telematics (braking, acceleration, speeding)](#5-module-b--telematics)
6. [Module C — Road (lanes + hazards)](#6-module-c--road)
7. [Personal baseline and context-aware event attribution](#7-personal-baseline-and-context-aware-event-attribution)
8. [Own data collection](#8-own-data-collection)
9. [Datasets: master table](#9-datasets-master-table)
10. [Datasets considered and not used](#10-datasets-considered-and-not-used)
11. [Downloads: where, how, how long](#11-downloads-where-how-how-long)
12. [Training and evaluation](#12-training-and-evaluation)
13. [Compute and storage](#13-compute-and-storage)
14. [Phases](#14-phases)
15. [Risks, limitations, licences](#15-risks-limitations-licences)
16. [v2 optimisation backlog](#16-v2-optimisation-backlog)
17. [Deliverables](#17-deliverables)

---

## 1. Problem and goal

**From the project brief:** existing driver monitoring systems are trained mostly on Western demographic data and may not capture Indian driver behaviour. Driver state shows up in:

| Signal | Source | Must detect |
|---|---|---|
| Face, gaze, eyelids | Cabin (driver-facing) video | Fatigue, distraction |
| Vehicle kinematics | Telematics (phone IMU + GPS) | Excessive braking, acceleration, speeding (+ aggressive-driving proxy) |
| Lane behaviour, surroundings | Road (forward-facing) video | Lane changing, lane keeping, road hazards |

Required work: data annotation and labelling, feature engineering, supervised ML, testing and validation.

**Our goal:** build a working system for all three signals, **trained and validated on Indian data**, and measure how well non-Indian-trained models transfer to Indian conditions and how much Indian fine-tuning improves them.

### Research questions
| RQ | Question |
|---|---|
| RQ1 | How much does a non-Indian-trained driver monitoring model degrade on Indian driver / cabin data? |
| RQ2 | How much is recovered by Indian fine-tuning and Indian RHD cabin data? |
| RQ3 | Do personal baselines and multimodal context reduce false alarms? |
| Road | How much does Indian fine-tuning improve generic road models on Indian roads? (separate from RQ1) |

---

## 2. Core strategy

### 2.1 Pretrained models + fine-tuning, not training from scratch
```
PRETRAINED MODEL (general knowledge from millions of images)
        │
        ▼
FINE-TUNE on large public data (mostly non-Indian)   ── keep this as the "before" model (RQ1)
        │
        ▼
FINE-TUNE on Indian / RHD data                      ── the "after" model (RQ2)
        │
        ▼
TEST only on held-out Indian drivers / drives       ── the number that counts
```

**Why:** with thousands (not millions) of Indian examples, a model trained from scratch over-fits and generalises worse. A pretrained model already understands faces, hands, objects and roads, and only has to learn the Indian specifics. Result: higher accuracy, less data, less compute.

**Exception:** telematics. No suitable pretrained model exists and the data is small tabular time series, so small models (XGBoost, then a small temporal network) are trained directly. They train in seconds.

### 2.2 Guiding principles
1. **India-first validation.** Every final number is on Indian drivers the model never saw.
2. **Features where they work, deep models where they don't.** Interpretable features (eye openness, PERCLOS, jerk, lateral offset) for fatigue and telematics; deep models for objects, actions, lanes and road segmentation.
3. **Use all relevant data, keep only what helps.** Diverse pre-training improves robustness, but every dataset must earn its place in a **dataset ablation** on the Indian test set.
4. **Know the data type and its role.** Type: 🚗 naturalistic · 🅿️ staged/parked · 🎮 simulator · 🧪 lab · 🤖 machine-labelled · 📟 ADAS reference labels. Role: **pre-train** (non-Indian or supporting), **Indian fine-tune**, **Indian test** (held-out), **reference** (weak labels / context labels). Staged, lab and machine-labelled data are never used as Indian test evidence.
5. **Everything becomes a timestamped event** `(start, end, type, confidence)`, so modules are independent and can be fused.
6. **Rules first, learning later** for fusion.
7. **Don't invent ground truth.** ADAS alerts are reference labels, explanations are context labels, unknown speed limits stay unknown.
8. **Safety and consent above data.** No distraction or drowsiness is ever acted out on a real road.

### 2.3 What the dataset survey found
| Need | Indian public data | Consequence |
|---|---|---|
| Fatigue | ✅ INVEDRIFAC (30 drivers, in-vehicle, day + NIR night) | Indian fine-tuning + testing |
| Gaze | ✅ DashGaze (28 drivers, naturalistic); DGAZE is lab data (supporting only) | Indian fine-tuning + testing on DashGaze |
| Driver + road synced | ✅ DAAD, DAAD-X, SIDDI | Attribution and lane-change testing |
| Telematics | ⚠️ Pune (7 drivers) and LEVIN (30 cars): both **unlabelled** | Indian distributions only; event labels come from own controlled drives (+ UAH-DriveSet, non-Indian) |
| Road | ✅ IDD family, BharatPothole, speed-breaker sets | Indian from the start |
| **Phone / eating / drinking in RHD cabins** | ❌ No public dataset identified in our survey provides sufficient Indian RHD coverage with explicit, per-driver labels for phone use, eating and drinking | **Our own recordings** (INITIALPLAN §8) |

---

## 3. System architecture

The three modules run **in parallel and independently**; inside the road module, hazard, lane and drivable-area branches are also independent. Everything meets only at the event layer.

```
CABIN VIDEO
  ├─► MediaPipe Face ──► features ──► fatigue model ─────────────► DROWSY, MICROSLEEP, YAWN
  ├─► L2CS-Net gaze ──► gaze-zone classifier ────────────────────► EYES_OFF_ROAD, MIRROR_CHECK
  └─► YOLO11 objects + MediaPipe Hands + action classifier ──────► PHONE_USE, EATING, DRINKING

IMU + GPS
  └─► reorient ─► window features ─► XGBoost ─► + OSM speed limits ► HARSH_BRAKE/ACCEL, SHARP_TURN,
                                                                     SPEEDING / SPEED_LIMIT_UNKNOWN, RISK SCORE

ROAD VIDEO
  ├─► YOLO11 hazards ────────────────────────────────────────────► HAZARD_AHEAD, POTHOLE, SPEED_BREAKER
  ├─► Lane detector (CLRNet / UFLD v2) ─► lane offset ───────────► LANE_DEPARTURE (marked roads)
  ├─► SegFormer drivable area ─► position vs. road edges ────────► ROAD_EDGE_DRIFT (unmarked roads)
  └─► lateral motion over time (lane offset / road mask / flow) ─► LANE_CHANGE

                                   │ all events, timestamped
                                   ▼
                    Personal baseline (#4): driver-relative thresholds
                                   ▼
                    Context-aware event attribution (#1): rules
                                   ▼
             Event log (time, type, context, severity) ─► Streamlit dashboard
```

---

## 4. Module A — Cabin

### 4.1 Face, eyes, mouth, head pose
| | Choice |
|---|---|
| **Model** | **MediaPipe Face Landmarker** (478 3D landmarks, iris, 52 blendshapes incl. eye-blink and jaw-open scores) |
| **Training** | None; used as-is |
| **Why** | Pretrained on large, diverse face data; real-time on CPU (and later on phones); gives blink/jaw scores directly; free and well maintained |
| **Alternatives not chosen** | **dlib 68-point** (older, less robust to head turns, no iris). **OpenFace 2** (good action units, but heavier and harder to install/deploy). **Separate RetinaFace + landmark + gaze models** (more pieces to maintain, no clear gain for v1) |
| **Risk** | May track worse on NIR night images, heavy occlusion, sunglasses → measured on INVEDRIFAC; backed up by the eye-state CNN (4.3) |

### 4.2 Fatigue (alert / drowsy)
| | Choice |
|---|---|
| **Features** | Per frame: EAR, MAR, blendshape blink/jaw scores, head pitch/yaw/roll. Per 30–60 s window: PERCLOS, blink rate, blink duration, longest eye closure, yawn count, nod count, trends |
| **Events** | Blink (EAR below personal threshold), microsleep (eyes closed ≥ ~1 s), yawn (MAR high ≥ ~2 s), nod (sharp pitch drop + recovery). Thresholds are starting values, tuned on Indian data |
| **Model (v1)** | **XGBoost** on window features **+ GRU/TCN** over the sequence of window features; final = calibrated combination |
| **Why** | Fatigue is a slow, temporal state; PERCLOS-style features are the established, explainable signal; small models suit small Indian data; matches the brief's "feature engineering + supervised ML" |
| **Alternatives not chosen** | **End-to-end CNN/video model on raw faces** (data-hungry, black box, overfits demographic cues, which is the very problem we study). **Fixed-threshold rules only** (no learning; worse across people) |
| **Pre-training data** | **UTA-RLDD** (real drowsiness, 60 people, alert / low-vigilance / drowsy), **DDD** (frames from RLDD), **YawDD** (yawning), **DMD** drowsiness bundle |
| **Indian data** | **INVEDRIFAC** (fine-tune + test), **D3S** (extra test) |
| **Outputs** | `DROWSY`, `MICROSLEEP`, `YAWN` |

### 4.3 Eye state (backup for hard cases)
| | Choice |
|---|---|
| **Model** | **EfficientNet-B0** (ImageNet, via timm) fine-tuned on eye crops |
| **Why** | Landmark EAR fails with sunglasses, glare, NIR; a small CNN on eye crops is robust and cheap |
| **Alternatives not chosen** | **MobileNetV3** (slightly less accurate; kept for v2 phone deployment). **Large ViTs** (no gain on tiny 24–64 px eye crops) |
| **Data** | **MRL Eye** (85k IR eye crops, glasses, reflections), **CEW** (in-the-wild) → **INVEDRIFAC** eye crops |

### 4.4 Gaze and eyes-off-road
| | Choice |
|---|---|
| **Model** | **L2CS-Net** (pretrained on Gaze360 + MPIIGaze) → per-frame gaze angles → **gaze-zone classifier** (road / mirrors / dashboard / down / off-road) |
| **Why** | Strong open-source appearance-based gaze model; zone classification is what driver monitoring actually needs |
| **Alternatives not chosen** | **MediaPipe iris only** (coarse, sensitive to head pose; used as a feature, not the main signal). **ETH-XGaze / GazeTR models** (comparable; L2CS has simpler code and weights). **Training gaze from scratch** (no reason, given DashGaze size) |
| **Data** | Pre-train: **DMD** gaze zones. Indian fine-tune + test: **DashGaze** 🚗 (subject-wise). **DGAZE** 🧪 is lab data (road video shown to drivers): supplementary pre-training only, added in v2, **never used for Indian validation** |
| **Outputs** | `EYES_OFF_ROAD` (> ~2 s, suppressed when stationary), `MIRROR_CHECK` (used by fusion) |

### 4.5 Phone use, eating, drinking
| | Choice |
|---|---|
| **Object cues** | **YOLO11-m** (Ultralytics, COCO-pretrained: already knows *cell phone, cup, bottle, wine glass, sandwich, banana, apple, pizza, donut*), fine-tuned on cabin frames |
| **Hand cues** | **MediaPipe Hands**: hand-to-mouth / hand-to-ear distance and duration |
| **Action classifier (v1)** | **EfficientNet-B2 / ConvNeXt-Tiny** (ImageNet) on the driver crop + object and hand features, then a temporal filter (several consistent seconds) |
| **Why** | Combining *what object* (YOLO), *where the hand is* (Hands) and *what the scene looks like* (classifier) is more robust than any one alone, and works with modest data |
| **Alternatives not chosen (v1)** | **Video transformers (VideoMAE, TimeSformer)** and **SlowFast / X3D**: likely best accuracy but heavy for free compute; **moved to v2**. **Image classifier alone** (confuses "hand near face" with phone/eating). **Object detector alone** (a phone on the seat is not phone *use*) |
| **Pre-training data** | **SynDD1** 🅿️ (eating, drinking, phone, texting, yawning), **DMD** 🚗 RGB release (phone, texting, drinking), **3MDAD** 🚗 (phone, drinking, day/night, real driving), **Drive&Act** 🎮, **State Farm** 🅿️, **Driver Distraction** 🚗🅿️, **SAM-DD** (if access granted). All mapped to one taxonomy: `safe / phone_call / texting / eating / drinking / other` |
| **RHD domain bridging** | Mirror augmentation (camera-side variation only) → **Bangladesh** (South-Asian RHD proxy, **not Indian data**) → **own Indian RHD recordings** (the only evidence of Indian RHD performance) |
| **Test** | **Own Indian RHD recordings only** (subject-wise) |
| **Labelling aid** | **YOLO-World / Grounding DINO** pre-label own clips ("person drinking from a bottle"); humans correct in CVAT |
| **Outputs** | `PHONE_USE`, `TEXTING`, `EATING`, `DRINKING` |

---

## 5. Module B — Telematics

| Step | Method | Why |
|---|---|---|
| Pre-processing | Resample to a fixed rate, low-pass filter | Phone sensors are noisy and irregular |
| Axis reorientation | Gravity → "down"; dominant acceleration during straight driving → "forward" | Phone orientation varies; events must be in the vehicle frame |
| Candidate events | Longitudinal / lateral acceleration above a starting threshold (~0.3 g) | Cheap first pass |
| Features | Peak g, jerk, duration, yaw rate, speed change, pre/post context | Standard, interpretable kinematics |
| **Model (v1)** | **XGBoost** event classifier; **1D-CNN / TCN** compared as an alternative. Labels: **own controlled drives** (manoeuvres performed at noted times in an empty lot) + manually checked LEVIN/Pune candidates; UAH-DriveSet for non-Indian pre-training | Best for small tabular data; fast; explainable via feature importance |
| Indian context | **LEVIN** distribution of everyday Indian driving informs thresholds and score normalisation | Stop-and-go traffic makes hard braking more common; Western fixed thresholds over-flag |
| Candidate-event mining | Flag candidate events in LEVIN automatically; **only a manually checked subset** is used as labels. LEVIN itself provides no event labels | Pune alone is small |
| **Speeding** | OSM `maxspeed` where present → `SPEEDING`; otherwise `SPEED_LIMIT_UNKNOWN`; optional `ABOVE_NATIONAL_MAXIMUM` | Never invent a legal limit |
| **Driving Risk Score** | Harsh events / km + time above a *known* limit + sharp-turn / weaving rate, normalised on LEVIN | Engineering metric; compared with Pune labels as an aggressive-driving proxy |

| | Choice |
|---|---|
| **Alternatives not chosen** | **Pretrained time-series foundation models** (not proven for this, overkill for tiny data). **LSTM from scratch as main model** (needs more data than we have; kept as comparison). **Fixed Western g-thresholds only** (ignore Indian driving context) |
| **Data** | **Pune VIIT** 🚗 (7 drivers, 7 days × start/return trips, phone fixed horizontally; **no labels**: normal/aggressive/risky in the paper are the authors' model outputs), **LEVIN** 🚗 (30 cars, 4 months, unlabelled: normal-driving distributions + candidate-event mining), **UAH-DriveSet** 🚗 (Spain; non-Indian baseline), own synced drives, OpenStreetMap |
| **Outputs** | `HARSH_BRAKE`, `HARSH_ACCEL`, `SHARP_TURN`, `SPEEDING`, `SPEED_LIMIT_UNKNOWN`, `DRIVING_RISK_SCORE` |

---

## 6. Module C — Road

### 6.1 Hazards
| | Choice |
|---|---|
| **Model** | **YOLO11-m** (COCO-pretrained), fine-tuned |
| **Why** | Best speed/accuracy trade-off with the simplest training pipeline; COCO already covers people, vehicles, animals |
| **Alternatives not chosen** | **RT-DETR** (competitive accuracy, heavier, slower to train on free GPUs; v2 candidate). **Faster R-CNN** (slower, no accuracy advantage here). **Training from scratch** (no reason) |
| **Data** | **IDD Detection** (Indian classes: autorickshaw, animal, rider…), **BharatPothole**, **Indian Roads** (Kaggle), **speed-breaker set** (Mendeley), **IDD-X** (speed breaker, pothole + ego-relative importance) |
| **Relevance** | Object in the ego path (from the road mask) and close (box size/position) → `HAZARD_AHEAD(type)` |
| **Outputs** | `HAZARD_AHEAD(type)`, `POTHOLE`, `SPEED_BREAKER` |

### 6.2 Drivable area (unmarked roads)
| | Choice |
|---|---|
| **Model** | **SegFormer-B2** (Hugging Face, already fine-tuned on Cityscapes) → fine-tuned on IDD |
| **Why** | Strong, efficient transformer segmenter; Cityscapes weights already understand road scenes |
| **Alternatives not chosen** | **DeepLabV3+** (older, usually lower mIoU). **Mask2Former** (higher accuracy, much heavier; v2). **SAM** (segments anything but doesn't know "drivable road" without prompts) |
| **Data** | **IDD Lite** (prototype) → **IDD Segmentation (IDD20K)**; **IDD-AW** for adverse-weather testing |
| **Use** | Ego position vs. drivable-region edges → `ROAD_EDGE_DRIFT`. **Not** used as a lane centre (it may include opposing traffic or the shoulder) |

### 6.3 Lane lines (marked roads)
| | Choice |
|---|---|
| **Model** | **CLRNet** or **UFLD v2** with CULane weights; pick the better one on Indian frames |
| **Why** | Strong open lane-detection baselines; CULane weights already cover occlusion and night scenes |
| **Alternatives not chosen** | **LaneATT / CondLaneNet** (comparable or older). **Classical Hough-line methods** (fail on worn Indian markings). **Downloading CULane itself** (not needed: we use its pretrained weights) |
| **Indian adaptation** | Label a few hundred marked-road frames from DAAD / IDD / own drives and fine-tune |
| **Outputs** | Lane centre → lateral offset → `LANE_DEPARTURE` |

### 6.4 Lane change (all roads)
- Sideways ego motion over time (lane offset trend on marked roads; road-mask position and optical flow on unmarked roads) → `LANE_CHANGE(left/right)`.
- Compared with **DAAD** manoeuvre labels. **IDD-CRS** 📟 is used separately as ADAS-derived reference evidence only where its alert types apply (e.g. lane-departure warnings for `LANE_DEPARTURE`, forward-collision warnings for hazards), not as lane-change labels.

---

## 7. Personal baseline and context-aware event attribution

### 7.1 Personal baseline (#4)
- First 3–5 minutes of each drive → driver's normal EAR, blink rate, neutral head pose, gaze centre, typical braking.
- Implemented (v1.1): robust calibration on the first 60 s (dominant head pose, frontal frames only, MAD outlier removal); see `reports/results/calibration_comparison.md`.
- Features become **driver-relative z-scores**.
- **Why:** faces, eye shapes and driving styles differ; fixed thresholds cause false alarms. Ablation measures the effect (RQ3).

### 7.2 Context-aware event attribution (#1, rule-based in v1)
| Pattern | Attributed context | Validation |
|---|---|---|
| `HARSH_BRAKE` + `EYES_OFF_ROAD` in previous 2 s | Likely distraction-related braking | DAAD, own synced drives |
| `HARSH_BRAKE` + `HAZARD_AHEAD` + attentive driver | Likely hazard response (not penalised) | DAAD; IDD-CRS collision-warning clips where applicable |
| `LANE_CHANGE` without `MIRROR_CHECK` in previous 3 s | Lane change without mirror check | DAAD, DAAD-X |
| `EYES_OFF_ROAD` while speed ≈ 0 | Suppressed (stationary) | DashGaze, DAAD |
| `DROWSY` + `LANE_DEPARTURE` | Critical fatigue alert | Deployment rule only |
| `PHONE_USE` while speed > 0 | Escalated distraction alert | Deployment rule only |

- **Why rules first:** no large, synced, labelled Indian dataset exists for learning fusion; rules are transparent and become the baseline for learned fusion (v2 / internship).
- **Not causal inference:** attribution from co-occurring signals, compared with **DAAD-X** explanation annotations (context labels).

---

## 8. Own data collection

**Why:** no public dataset has explicit, per-driver phone / eating / drinking labels in Indian RHD cabins. This is our main annotation contribution.

| Item | v1 plan |
|---|---|
| Participants | **20–30** volunteers, mixed age and gender; glasses/sunglasses, head coverings and facial hair where naturally present |
| Vehicles | **2–3 cars** (hatchback, sedan, SUV): different cabins and camera angles |
| Setting | **Parked cars only** for all distraction actions; dashboard-mounted phone camera at the RHD angle |
| Priority classes | **1. Eating** (snack, roll, fruit; one/both hands) · **2. Phone** (call left/right ear, texting, scrolling) · **3. Drinking** (bottle, cup, chai glass) |
| Controls | Normal posture, looking ahead, mirror checks, dashboard/radio |
| **Hard negatives** | Adjusting glasses, touching face, scratching chin, hand on head |
| Conditions | Day, harsh glare, night; **left/right hand used for phone/eating/drinking actions** (the vehicle is always RHD); different phones |
| Synced drives | **10+ hours** of normal Mumbai driving: cabin phone + road phone/dashcam + Sensor Logger (IMU + GPS). Harsh manoeuvres only in an empty lot at low speed |
| Sync | Clap/flash at start + phone timestamps |
| Labelling | CVAT / Label Studio; pre-labels from YOLO-World / Grounding DINO; temporal segments + frame boxes |
| Agreement | Second annotator on ~10 %: **temporal IoU**, segment-level precision/recall/F1 with onset/offset tolerance; Cohen's κ for frame labels |
| Ethics | Written consent, research-only use, no faces published without permission; check whether the college needs ethics approval |
| **Safety** | **No distraction or drowsiness on real roads.** Ever. |

---

## 9. Datasets: master table

**Type:** 🚗 naturalistic · 🅿️ staged/parked · 🎮 simulator · 🧪 lab · 🤖 machine-labelled · 📟 ADAS reference labels.
**Role:** **PT** = pre-training (non-Indian or supporting) · **IFT** = Indian fine-tuning · **IT** = Indian test (held-out drivers) · **REF** = reference / context labels (weak, not ground truth) · **BR** = domain bridge (not Indian).
**Stage:** **v1** = first build · **v2** = optimisation.

### Cabin
| Dataset | Origin · type | Role | Used for | Why we use it | Stage | Access |
|---|---|---|---|---|---|---|
| **INVEDRIFAC** | India · 🚗 | **IFT + IT** | Fatigue, eye state | Only sizeable Indian in-vehicle driver-face set (30 drivers, day + NIR night) | v1 | Request |
| **DashGaze** | India · 🚗 | **IFT + IT** | Gaze / eyes-off-road | Naturalistic Indian driving, 28 drivers, synced views, dual-camera phone setup | v1 | Sample open; **full set by request** (email authors) |
| D3S | India · staged | IT | Extra fatigue test | Small Indian set; extra diversity | v1 | Open |
| **UTA-RLDD** | Non-Indian · 🚗 | PT | Fatigue ("before" model) | Real (not acted) drowsiness, 60 people, three levels; 111 GB in 5 folds, processed fold by fold into features | v1 | Request (Google Drive) |
| DDD | Non-Indian · 🚗 | PT (fallback) | Fatigue | Frames extracted from RLDD; **only used if RLDD access fails** | v1 | Kaggle |
| YawDD | Non-Indian · 🚗 | PT | Yawning | Real in-car yawning vs. talking/singing | v1 | IEEE account |
| **DMD** (RGB release) | Non-Indian · 🚗🅿️ | PT | Drowsiness bundle: frame-level **eye state, blink and yawn** labels (acted in a parked car, session s5) to tune and check event detectors, **not** real drowsiness. Distraction bundle: phone call / texting (left + right), drinking, plus **gaze-on-road** and hands-on-wheel labels; session s1 recorded **while actually driving** (drinking, radio, reaching, talking), s2/s3 in a parked car (phone, texting, drinking, hair/makeup, reaching) | Largest modern DMS taxonomy; face + body camera RGB + OpenLABEL JSON | v1 | Form (per bundle) |
| **MRL Eye** | Non-Indian · 🧪 | PT | Eye state | 85k IR eye crops incl. glasses and reflections | v1 | Kaggle |
| CEW | Non-Indian · in-the-wild | PT | Eye state | Unconstrained eye images | v1 | Open |
| **SynDD1** | Non-Indian · 🅿️ | PT | Phone, eating, drinking | Best public class match incl. **eating**; only the dashboard view is needed | v1 | **Request** (files no longer public on Mendeley; email authors) |
| **3MDAD** | Non-Indian · 🚗 | PT | Phone, drinking | Real driving, day + night (IR), two views | v1 | Email authors |
| **Bangladesh** | South Asia (RHD) · 🚗 | **BR** | Phone (drinking only inside a broad "Other" class) | Closest public match to Indian RHD cabins; **not Indian data** | v1 | Open (Zenodo) |
| **Own Indian RHD recordings** | India · 🅿️ | **IFT + IT** | Phone, eating, drinking, hard negatives | The gap no identified public dataset fills | v1 | We collect |
| DGAZE | India · 🧪 | PT (supporting) | Gaze | Indian drivers but lab setup; **never Indian validation** | v2 | Open |
| Drive&Act | Non-Indian · 🎮 | PT | Fine-grained actions | Eating/drinking/phone detail | v2 | Request |
| State Farm | Non-Indian · 🅿️ | PT | Phone, drinking (no eating class) | Large classic benchmark | v2 | Kaggle (accept rules) |
| Driver Distraction (Morocco) | Non-Indian · 🚗🅿️ | PT | Phone, drinking | Extra drivers and cabins | v2 | Password link |
| SAM-DD | Non-Indian · 🅿️ | PT | Distraction actions | Extra variety | v2 | Access-dependent |
| NTHU-DDD | Non-Indian · 🎮 | PT | Fatigue | Night / glasses variety | v2 | Email (often slow) |

### Telematics
| Dataset | Origin · type | Role | Used for | Why | Stage | Access |
|---|---|---|---|---|---|---|
| **Pune VIIT** | India · 🚗 | IFT (distributions, candidate events) | 7 drivers, start/return trips over 7 days; accel + gyro at 50 Hz, GPS in day folders; phone fixed horizontally | Indian smartphone driving; **unlabelled** (verified 2026-10-07) | v1 | Open (Mendeley) |
| **LEVIN** | India · 🚗 | IFT (distributions only) | Normal-driving distributions; candidate-event mining (only a manually checked subset becomes labels) | 30 Indian cars over 4 months; **no event labels of its own** | v1 | Open (GitHub) |
| UAH-DriveSet | Spain · 🚗 | PT | **Labelled** normal / aggressive / drowsy trips + per-event logs; non-Indian baseline | Only labelled smartphone driving set in the plan | v1 | Open |
| OpenStreetMap | — | — | Speed limits | Only open source of limits; unknown stays unknown | v1 | Overpass API |
| **Own synced + controlled drives** | India · 🚗 | **IFT + IT** | **Only source of ground-truth Indian telematics event labels** (harsh brake / accel / turn performed at noted times in an empty lot) + synced events for fusion | No public Indian telematics dataset has event labels | v1 | We collect |

### Road
| Dataset | Origin · type | Role | Used for | Why | Stage | Access |
|---|---|---|---|---|---|---|
| **IDD Lite** | India · 🚗 | IFT | Prototyping | ~50 MB, fast iteration | v1 | Register |
| **IDD Detection** | India · 🚗 | **IFT + IT** (official val split) | Hazards | Indian classes and traffic | v1 | Register |
| **IDD Segmentation** | India · 🚗 | **IFT + IT** (official val split) | Drivable area | Drivable / non-drivable labels for Indian roads | v1 | Register |
| **BharatPothole** | India · 🚗 | IFT + IT | Potholes | 7k+ Indian dashcam frames | v1 | Kaggle |
| Indian Roads | India | IFT | Speed breakers, potholes | Marked and unmarked speed breakers | v1 | Kaggle |
| Speed-breaker set | India | IFT | Speed breakers | 969 extra images | v1 | Open (Mendeley) |
| IDD-AW | India · 🚗 | IT | Adverse-weather robustness | Rain, fog, night | v1 | Open |
| IDD-X | India · 🚗 | IFT | Hazards + ego-relative importance | Speed breaker, pothole, importance labels | v2 | Request (india-data.org) |
| ThirdEye subset | India · 🤖 | PT only | Extra pre-training | Huge Delhi NCR set; labels unverified, **never a test set** | v2 | Open (Hugging Face) |

### Synced cabin + road (attribution, lane change)
| Dataset | Origin · type | Role | Used for | Why | Stage | Access |
|---|---|---|---|---|---|---|
| **DAAD** | India · 🚗 | **IT** | Lane change, mirror checks, attribution | In-cabin + exterior views around manoeuvres; ~240 GB (6 views). **Skipped in v1 unless DAAD-X is insufficient**, then Driver + Front views only | v1 (conditional) | Open (IIIT, very slow) |
| **DAAD-X** | India · 🚗 | **REF** (context labels) | Attribution comparison | Explanation/context annotations, not causal ground truth | v1 | Open (direct link; downloading) |
| SIDDI | India · 🚗 | IT | Rural / semi-urban robustness | Driver eyes + road in small towns | v1 | Open (Drive) |
| IDD-CRS | India · 📟 | **REF** (where alert types apply) | Lane-departure / collision-warning reference | ADAS alerts give temporal reference labels; can be wrong | v1 (if granted) | Email CVIT |
| Brain4Cars | Non-Indian · 🚗 | PT | Manoeuvres | Driver + road around lane changes | v2 | Open |

---

## 10. Datasets considered and not used

| Dataset | Reason |
|---|---|
| CULane (raw data) | We use lane models already trained on it; downloading adds nothing |
| myEye2Wheeler, MOTOR, IndiGo, I2WDD | Two-wheelers; our system targets cars |
| Indian lane + traffic-sign set (PMC 2025) | Only available on request from authors; revisit in v2 |
| IDD-3D, Amrita LiDAR | LiDAR; not used by a camera/phone system |
| IDD 117K, IDD Temporal, IDD Multimodal | Full IDD Detection/Segmentation is enough for v1; revisit if road accuracy plateaus |
| V-SenseDrive (Pakistan), InCarEmo (China), DriverMVT | Not Indian; references only |
| DROZY, HDD, AIDE, DADA-2000 | Fatigue prediction / learned causal modelling, internship-phase topics |
| AUC Distracted Driver | Similar to State Farm; request-only |
| FatigueFace-1156, Phone-sensor normal/aggressive (Mendeley, country unconfirmed) | Small or provenance unclear |
| Multi-Class Driver Behavior (Mendeley mirror) | Same Bangladesh data as Zenodo |

---

## 11. Downloads: where, how, how long

**Measured:** IIIT Hyderabad servers ≈ **1–2.3 MB/s** (≈ 2–5 h per 18 GB). Mendeley, Zenodo, Google Drive and Kaggle from Colab are usually much faster. Sizes marked "?" are not published; check each page before downloading.

| Route | Use for |
|---|---|
| **Mac** | Small datasets, samples, webcam demo |
| **Kaggle "Add data"** | Kaggle-hosted sets: no download at all |
| **Colab → Google Drive (2 TB)** | Every large dataset: `wget -c` straight into Drive (survives disconnects), store as archives, extract on Colab when needed |

| Dataset | Size | Route | Expected time | When |
|---|---|---|---|---|
| DashGaze sample | small | Mac | ✅ have it | now |
| Pune VIIT, LEVIN, speed-breaker set, D3S, CEW | small | Mac | minutes each | now |
| MRL Eye, DDD, BharatPothole, Indian Roads | medium | Kaggle | 0 | when needed |
| **DAAD-X** | **18.6 GB** | Colab → Drive | **~2 h** (in progress) | now |
| **SynDD1** | large (?) | Colab → Drive | likely < 1 h | next |
| **DashGaze full** | large (?) | Colab → Drive | depends on host | next |
| Bangladesh | 2.6 GB | Colab → Drive | minutes | next |
| SIDDI | ? | Drive "Add shortcut" | instant | anytime |
| UAH-DriveSet | medium (?) | Colab → Drive | < 1 h | Phase 2 |
| YawDD | medium (?) | Colab → Drive | < 1 h | after IEEE account |
| IDD Lite → IDD Detection + Segmentation | 50 MB → large | Mac → Colab → Drive | minutes → hours (IIIT server) | after registration |
| IDD-AW | large | Colab → Drive | hours (IIIT server) | Phase 6 |
| INVEDRIFAC, UTA-RLDD, DMD, 3MDAD, DAAD, IDD-CRS | large | Colab → Drive | hours each | after approval |

**Request today** (approval is the slowest step): IDD, INVEDRIFAC, UTA-RLDD, DMD, YawDD (IEEE account); emails for SynDD1, DashGaze (full), 3MDAD and IDD-CRS. DAAD is skipped in v1 (DAAD-X covers it).

**Exact parts to download, links and save locations:** [`DOWNLOADS.md`](DOWNLOADS.md).

---

## 12. Training and evaluation

### 12.1 Training recipe per model
1. Start from pretrained weights.
2. Fine-tune on public (mostly non-Indian) data → **"before" model**.
3. Fine-tune on Indian / RHD data → **"after" model**.
4. Freeze the backbone first; unfreeze gradually only if validation improves.
5. Augmentation: brightness/contrast (glare, night), blur (vibration), random crops, NIR-style greyscale for cabin images, and **mirror augmentation** (camera-side variation only; it is augmentation, **not evidence of RHD adaptation**: only the own Indian RHD test set establishes that).
6. Hyperparameter search on validation drivers (never test drivers).
7. Track every run in Weights & Biases.

### 12.2 Splits
- **Subject-wise always:** no driver in both training and test.
- Few drivers (Pune): **leave-one-driver-out**; report the exact driver count.
- Road: official IDD train/val splits; no overlapping drive sequences.

### 12.3 Metrics
| Module | Metrics |
|---|---|
| Cabin | Precision, recall, F1 per class; **false alarms per hour**; detection latency |
| Telematics | Event-level F1; confusion matrix; per-driver results |
| Road | mAP@50 / mAP@50-95; mIoU; lane F1; agreement with IDD-CRS reference labels |
| Attribution | Agreement with DAAD-X / manual labels; false-alarm reduction vs. no fusion |

**Operating point:** alert thresholds are chosen for a target false-alarm rate (e.g. ≤ 1 per hour), not for accuracy alone.

### 12.4 Experiments
| # | Experiment | Answers |
|---|---|---|
| 1 | "Before" (non-Indian) cabin models on Indian data | RQ1 |
| 2 | "After" (Indian fine-tuned + Indian RHD cabin data) on held-out Indian drivers | RQ2 |
| 3 | RHD ablation: baseline → + mirror augmentation → + Bangladesh proxy → + own Indian RHD data (always tested on held-out own Indian RHD drivers) | RQ2 |
| 4 | Personal baseline: fixed vs. driver-relative thresholds | RQ3 |
| 5 | Attribution: false alarms with vs. without context rules | RQ3 |
| 6 | **Dataset ablation:** remove one pre-training dataset at a time | Which data actually helps |
| 7 | Subgroups: lighting, glasses, head coverings, traffic density, urban vs. rural | Tests hypotheses |
| 8 | Road: COCO/Cityscapes-pretrained vs. IDD fine-tuned | Road adaptation |

All results with bootstrap confidence intervals over drivers.

---

## 13. Compute and storage

| Resource | Use |
|---|---|
| **Mac** | Coding, MediaPipe feature extraction, webcam demo, small models (XGBoost) |
| **Google Colab (free)** | Downloads into Drive, medium training runs |
| **Kaggle (free, ~30 GPU h/week)** | Longer training runs in parallel with Colab |
| **Google Drive (2 TB)** | Raw dataset archives, extracted features, checkpoints |
| **Colab Pro (optional)** | Larger models in v2 (VideoMAE, SegFormer-B5, YOLO11-l) without constant disconnects |

**Storage rules**
- Store datasets as **archives** on Drive; copy one archive to Colab's local disk and extract there (reading many small files from Drive is slow).
- Extract features once (video → landmark CSVs, sampled frames) and reuse them.
- Save checkpoints to Drive every epoch so runs survive disconnects.
- Code lives in GitHub; notebooks only `git clone` and run scripts. Data never goes into git.

**Software:** Python 3.11, MediaPipe, OpenCV, PyTorch, timm, Ultralytics (YOLO11, YOLO-World), Hugging Face Transformers (SegFormer), L2CS-Net, CLRNet / UFLD v2, scikit-learn, XGBoost, NumPy/pandas/SciPy, OSM Overpass, folium, CVAT / Label Studio, Weights & Biases, Streamlit.

---

## 14. Phases

| Phase | Work | Needs | Done when |
|---|---|---|---|
| **0. Setup** | Repo, `.gitignore`, environment, W&B; **send all access requests**; small downloads; queue large downloads to Drive; **dataset audit** (driver counts, classes actually present, licences) | — | Repo runs on Colab/Kaggle; audit table filled |
| **1. Cabin base** | MediaPipe pipeline, feature extraction, webcam demo; L2CS gaze; fatigue XGBoost/GRU on RLDD/DDD/YawDD ("before") | No data for the demo; DDD/RLDD for training | Live demo: blinks, yawns, drowsiness, eyes-off-road |
| **2. Telematics base** | Reorientation, features, XGBoost on Pune; LEVIN distributions; OSM speeding | Pune, LEVIN | Leave-one-driver-out F1 |
| **3. Road base** | YOLO11 on IDD Lite + potholes/speed breakers; SegFormer on IDD Lite; lane model inference | IDD Lite, BharatPothole | mAP / mIoU; demo on Indian dashcam video |
| **4. Own recordings** *(parallel with 1–3)* | Consent forms, protocol, recordings, pre-labelling, labelling, agreement | Volunteers, cars | Labelled RHD clips + synced drives |
| **5. Distraction** | Unified taxonomy; YOLO + Hands + classifier on SynDD1/DMD/3MDAD → Bangladesh → own clips | SynDD1, own clips | RHD ablation done |
| **6. Indian adaptation** | INVEDRIFAC fatigue, DashGaze gaze, full IDD road; experiments 1–2, 6–8 | Approvals | "Before vs. after" tables |
| **7. Personal baseline** | Calibration + ablation | Phases 1–2 | Experiment 4 done |
| **8. Context attribution** | Event timeline + rules; DAAD/DAAD-X/SIDDI/own drives | DAAD-X, DAAD | Experiment 5 done |
| **9. Dashboard** | Streamlit: upload drive → timeline, map, context, scores | All modules | End-to-end demo |
| **10. Write-up** | README, results, limitations, report, demo video, internship pitch | Everything | Ready to send |

---

## 15. Risks, limitations, licences

Full list of limitations: [`PLAN.md` §8](PLAN.md#8-limitations-stated-openly). Key risks and mitigations:

| Risk / limitation | Mitigation / consequence |
|---|---|
| Access requests slow or denied | Requested in Phase 0; v1 still works with DashGaze, DAAD-X, SynDD1, DDD, own data |
| Downloaded releases differ from papers | Phase 0 audit; plan around what is actually present |
| Different label schemes across distraction datasets | One unified taxonomy; dataset ablation |
| Small Indian test sets | Subject-wise / leave-one-out; confidence intervals |
| Phone/eating/drinking recorded only while parked | Phone-use-at-speed is a deployment rule, not validated end to end |
| Drowsiness never recorded on real roads | Fatigue + lane-departure rule not validated on own data |
| ADAS alerts (IDD-CRS) can be wrong | Treated as reference labels |
| DAAD-X explanations are human annotations | Context labels, not causal ground truth |
| Sparse OSM speed limits in India | Many segments `SPEED_LIMIT_UNKNOWN` |
| Unmarked-road lane keeping unsolved | Only road-edge drift reported there |
| Free compute limits | Small/medium models in v1; checkpointing; Kaggle + Colab in parallel; Colab Pro optional |
| **Licences**: DMD (CC BY-NC-ND), LEVIN, BharatPothole (BY-NC-SA), State Farm, Driver Distraction, SAM-DD, 3MDAD (research only) | Research use only; data never redistributed; would need re-licensing for any commercial product |

---

## 16. v2 optimisation backlog

Applied after v1 works end to end, in order of expected gain, each kept only if it improves Indian test results:

1. **Video models for actions:** VideoMAE / X3D (Kinetics-pretrained) replacing the image classifier.
2. **Bigger road models:** YOLO11-l or RT-DETR; SegFormer-B5 or Mask2Former.
3. **More pre-training data:** DGAZE, Drive&Act, State Farm, Driver Distraction, SAM-DD, NTHU-DDD, IDD-X, ThirdEye subset, Brain4Cars.
4. **Ensembles** and test-time augmentation for the best models.
5. **Active learning** on own recordings (label the clips the model is least sure about).
6. **Hard-negative mining** from false alarms.
7. **Lane model fine-tuning** on more labelled Indian marked-road frames.
8. **Learned fusion** replacing rules (internship topic).
9. **On-device versions** (MobileNet / YOLO-n / TFLite) for the Android app (internship topic).

### Internship research proposals (not built now)
Learned context attribution · IMU-as-labeller for speed breakers/potholes · fatigue prediction minutes ahead · single-phone on-device system · true lane keeping on unmarked roads · Indian speed-sign recognition · public Indian RHD distraction benchmark.

---

## 17. Deliverables

1. GitHub repo with clean, documented code
2. "Before vs. after" results for every cabin model, road adaptation results, all ablations (RQ1–RQ3)
3. Own Indian RHD distraction recordings with annotation protocol and agreement scores
4. 2–3 minute demo video (webcam + dashcam + dashboard)
5. Streamlit dashboard
6. Technical report: dataset survey, the Indian RHD distraction gap, results, limitations
7. Internship proposal section (§16)
