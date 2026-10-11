# PLAN — Research design and rationale

> **How the documents fit together**
> - **[`INITIALPLAN.md`](INITIALPLAN.md): source of truth** for models, datasets, roles, downloads, training, evaluation and phases (v1, frozen 2026-10-07).
> - **This file:** *why* the project is designed this way: problem, research questions, principles, scope, rejected system-level alternatives, limitations and the decision log.
> - **[`DATASETS.md`](DATASETS.md):** full dataset survey (links, licences, access).
> - **[`DOWNLOADS.md`](DOWNLOADS.md):** v1 download checklist (exact parts, links, save locations).
>
> If anything here ever disagrees with INITIALPLAN.md, **INITIALPLAN.md wins**. Shared text (research questions, guiding principles, data-gap table) is kept word-for-word identical; the full limitations list lives here (§8).

---

## 1. Problem

**From the project brief:** existing driver monitoring systems are trained mostly on Western demographic data and may not capture Indian driver behaviour. Driver state shows up in facial expressions, vehicle kinematics and lane-keeping behaviour. The project develops AI models for cabin video (face, gaze, eyelids → fatigue, distraction), telematics (braking, acceleration, speeding) and road video (lane changing, lane keeping, road hazards), with data annotation, feature engineering, supervised ML, and testing and validation.

**One line:** a context-aware driver monitoring system **trained and validated on Indian data**, which **measures** how well non-Indian-trained models transfer to Indian driving conditions and how much Indian fine-tuning improves them.

### Requirement → where it is met (INITIALPLAN.md)
| Requirement | Section |
|---|---|
| Fatigue + distraction from face, gaze, eyelids | §4 Module A |
| Braking, acceleration, speeding (+ aggressive-driving proxy) | §5 Module B |
| Lane changing, lane keeping, road hazards | §6 Module C |
| Indian driver behaviour | §2 strategy, §9 dataset roles, §12 evaluation |
| Annotation and labelling | §8 own data collection |
| Feature engineering + supervised ML | §4.2, §5 |
| Testing and validating | §12 |

---

## 2. Research questions

| RQ | Question |
|---|---|
| RQ1 | How much does a non-Indian-trained driver monitoring model degrade on Indian driver / cabin data? |
| RQ2 | How much is recovered by Indian fine-tuning and Indian RHD cabin data? |
| RQ3 | Do personal baselines and multimodal context reduce false alarms? |
| Road | How much does Indian fine-tuning improve generic road models on Indian roads? (separate from RQ1) |

The road question is kept separate because road models start from generic (COCO / Cityscapes) weights and are Indian-trained from the start; they are not a "Western driver demographic" model.

---

## 3. Guiding principles

1. **India-first validation.** Every final number is on Indian drivers the model never saw.
2. **Features where they work, deep models where they don't.** Interpretable features (eye openness, PERCLOS, jerk, lateral offset) for fatigue and telematics; deep models for objects, actions, lanes and road segmentation.
3. **Use all relevant data, keep only what helps.** Diverse pre-training improves robustness, but every dataset must earn its place in a **dataset ablation** on the Indian test set.
4. **Know the data type and its role.** Type: 🚗 naturalistic · 🅿️ staged/parked · 🎮 simulator · 🧪 lab · 🤖 machine-labelled · 📟 ADAS reference labels. Role: **pre-train** (non-Indian or supporting), **Indian fine-tune**, **Indian test** (held-out), **reference** (weak labels / context labels). Staged, lab and machine-labelled data are never used as Indian test evidence.
5. **Everything becomes a timestamped event** `(start, end, type, confidence)`, so modules are independent and can be fused.
6. **Rules first, learning later** for fusion.
7. **Don't invent ground truth.** ADAS alerts are reference labels, explanations are context labels, unknown speed limits stay unknown.
8. **Safety and consent above data.** No distraction or drowsiness is ever acted out on a real road.

---

## 4. Core strategy (summary)

```
Pretrained model → fine-tune on public (mostly non-Indian) data = "before" model (RQ1)
                 → fine-tune on Indian data + Indian RHD cabin data = "after" model (RQ2)
                 → test only on held-out Indian drivers / drives
```

- **Why pretrained models:** with thousands (not millions) of Indian examples, models trained from scratch over-fit; pretrained models already understand faces, hands, objects and roads and only need the Indian specifics.
- **Exception:** telematics. No suitable pretrained model exists; small models (XGBoost, then a small temporal network) are trained directly.
- **Mirror augmentation** (horizontal flip) is augmentation only. It is **not evidence of RHD adaptation**; only the own Indian RHD test set establishes that.
- **Bangladesh** data is a South-Asian RHD **domain bridge**, not Indian data.

Model-by-model choices and rejected alternatives: INITIALPLAN.md §4–§6.

---

## 5. What the dataset survey found

| Need | Indian public data | Consequence |
|---|---|---|
| Fatigue | ✅ INVEDRIFAC (30 drivers, in-vehicle, day + NIR night) | Indian fine-tuning + testing |
| Gaze | ✅ DashGaze (28 drivers, naturalistic); DGAZE is lab data (supporting only) | Indian fine-tuning + testing on DashGaze |
| Driver + road synced | ✅ DAAD, DAAD-X, SIDDI | Attribution and lane-change testing |
| Telematics | ⚠️ Pune (7 drivers) and LEVIN (30 cars): both **unlabelled** | Indian distributions only; event labels come from own controlled drives (+ UAH-DriveSet, non-Indian) |
| Road | ✅ IDD family, BharatPothole, speed-breaker sets | Indian from the start |
| **Phone / eating / drinking in RHD cabins** | ❌ No public dataset identified in our survey provides sufficient Indian RHD coverage with explicit, per-driver labels for phone use, eating and drinking | **Our own recordings** (INITIALPLAN §8) |

So the own recordings are **not** "an Indian driver dataset because nothing exists". They are a **targeted Indian RHD benchmark** for the distraction classes existing public Indian datasets do not adequately cover.

---

## 6. Scope

### Built in v1 (prototype for the internship application)
1. **Cabin:** fatigue, eyes-off-road, phone use, eating, drinking
2. **Telematics:** harsh braking/acceleration, sharp turns, speeding (where the limit is known), Driving Risk Score
3. **Road:** hazards, lane keeping on marked roads, road-edge drift on unmarked roads, lane changes
4. **Personal baseline (#4)**
5. **Context-aware event attribution (#1), rule-based**
6. **Transfer evaluation:** non-Indian "before" vs. Indian "after"
7. **Own Indian RHD distraction recordings**
8. **Streamlit dashboard**

### Proposed for the internship (not built)
| Idea | Why it is research-worthy |
|---|---|
| Learned multimodal context attribution (compared with DAAD-X context annotations + own data) | Needs a larger synced Indian dataset |
| IMU-as-labeller: accelerometer bumps auto-label speed breakers / potholes in video | Cross-modal self-supervision for under-labelled Indian hazards |
| Fatigue **prediction** minutes ahead | Needs long alert→drowsy recordings and a simulator |
| Whole system on one Android phone (front cam, rear cam, IMU), on-device | Low-cost deployment; DashGaze shows the dual-camera phone setup works |
| True lane keeping on unmarked roads ("virtual lanes" from traffic flow) | Open problem on Indian roads |
| Indian speed-limit sign recognition | Removes `SPEED_LIMIT_UNKNOWN` |
| Public Indian RHD distraction benchmark | Covers classes no identified public Indian dataset labels explicitly |

v1 → v2 optimisation backlog: INITIALPLAN.md §16.

---

## 7. System-level alternatives we rejected

| Alternative | Why not |
|---|---|
| One end-to-end deep video model for everything | Needs huge labelled Indian video that doesn't exist; a black box; ignores the brief's "feature engineering" |
| Train only on non-Indian data | That is exactly the problem in the brief |
| Train everything from scratch | Over-fits with thousands of examples; pretrained + fine-tuning is more accurate |
| Collect a full Indian dataset ourselves | Indian fatigue, gaze, telematics and road data already exist; we collect only the real gap |
| Use the drivable-area centre as the lane centre | Wrong on multi-lane and undivided roads; marked and unmarked roads need different methods |
| Guess speed limits from road type | Would invent legal ground truth; unknown limits stay unknown |
| Learned fusion right away | No large synced, labelled Indian dataset yet; rules are transparent and become the baseline |
| Fixed global thresholds | Faces and driving styles vary; personal baselines reduce false alarms |
| Call the risk score "aggression" | It is an engineering metric; compared with Pune labels only as an aggressive-driving proxy |

---

## 8. Limitations (stated openly)

| Limitation | Consequence |
|---|---|
| Indian phone/eating/drinking examples are recorded **only in a stationary vehicle** | Phone use *at speed* is a deployment rule, **not** validated end to end |
| Drowsiness is never recorded on real roads | The `DROWSY + LANE_DEPARTURE` rule is not validated on own data |
| Pune VIIT and LEVIN have **no event labels** | Indian telematics ground truth comes only from own controlled drives (small); results are indicative, not a robust Indian benchmark |
| LEVIN is unlabelled | Used for normal-driving distributions and candidate-event mining only; only a manually checked subset becomes labels |
| IDD-CRS labels come from a commercial ADAS unit | Reference / weak labels, used only where alert types apply; the ADAS can be wrong |
| DAAD-X explanations are human annotations | Context labels, not causal ground truth |
| SynDD1, Drive&Act, State Farm are staged / parked / simulator | Pre-training only |
| DGAZE is lab data | Supporting pre-training only; never Indian validation |
| Bangladesh is South-Asian, not Indian | Domain bridge only |
| Mirror augmentation does not reproduce RHD cabins | Only the own Indian RHD test set measures RHD performance |
| Current DMD public release is RGB-only with fewer participants; eating not listed | Use only classes present after download |
| OSM speed limits are sparse in India | Many segments are `SPEED_LIMIT_UNKNOWN` |
| Unmarked-road lane keeping is not solved | Only road-edge drift is reported there |
| Own dataset is 20–30 people, 2–3 cars | Subgroup results are exploratory |
| Licences: DMD (CC BY-NC-ND), LEVIN and BharatPothole (BY-NC-SA), State Farm, Driver Distraction, SAM-DD, 3MDAD (research only) | Research use only; data never redistributed |

---

## 9. Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-10-05 | India-first training and validation in every module | Core requirement of the brief |
| 2026-10-05 | Own recordings limited to phone / eating / drinking in RHD cabins | Survey found Indian coverage for fatigue, gaze, telematics and roads |
| 2026-10-05 | DGAZE marked lab / supporting | Drivers watch projected road video; not naturalistic |
| 2026-10-05 | DAAD-X = context labels; IDD-CRS = ADAS reference labels | Neither is causal or absolute ground truth |
| 2026-10-05 | Marked vs. unmarked lanes handled separately | Drivable-area centre ≠ lane centre |
| 2026-10-05 | No inferred speed limits (`SPEED_LIMIT_UNKNOWN`) | Avoid inventing legal ground truth |
| 2026-10-05 | "Aggression" → Driving Risk Score | Not a validated aggression measure |
| 2026-10-05 | DMD licence corrected to CC BY-NC-ND; eating not in public release | Verified on the official site |
| 2026-10-06 | Pretrained models + fine-tuning everywhere except telematics | Best accuracy with limited Indian data |
| 2026-10-06 | Goal = best possible output; v1 base + v2 optimisation backlog | Time is not a constraint; compute and licences are |
| 2026-10-07 | Own data scaled to 20–30 people, 2–3 cars, 10+ h synced drives | Best-quality target |
| 2026-10-07 | Dataset roles (PT / IFT / IT / REF / BR) added everywhere; road branches drawn in parallel; wording precision fixes | External review of v1 |
| 2026-10-07 | v1 frozen; INITIALPLAN.md is the source of truth | Start Phase 0 |
| 2026-10-07 | Pune VIIT has **no labels** (paper's normal/aggressive/risky are model outputs; S/R = start/return trip; 7 drivers). Own controlled drives become the Indian telematics label source; UAH-DriveSet the labelled pre-training set | Inspection of the downloaded release + PMC8914310 |
| 2026-10-07 | SynDD1 and full DashGaze are request-only; DDD becomes RLDD fallback; DAAD skipped in v1; only needed parts downloaded (DOWNLOADS.md) | Re-check of dataset access and sizes |
| 2026-10-09 | Personal calibration made robust: 60 s window, dominant-pose frontal frames only, MAD outlier removal (live demo keeps 15 s) | INVEDRIFAC drivers settle in for 30–40 s with the head turned; DMD eye F1 0.885 → 0.923, broken baselines 3 → 0 (`reports/results/calibration_comparison.md`) |
| 2026-10-09 | Low-light enhancement off by default; unrecoverable videos excluded by the face-visibility gate | Gamma / CLAHE gave no gain on INVEDRIFAC; Sub 04 has no visible face (`reports/results/low_light_eval.md`) |
| 2026-10-09 | Road segmentation v1 = SegFormer-B0 on IDD Lite on the Mac; Western-only (Cityscapes merged to IDD level-1) vs Indian fine-tuned. B2 on full IDD and YOLO hazards on IDD Detection left for the Kaggle GPU / internship | Unstable internet; IDD Lite is the only road data on the Mac |
| 2026-10-09 | Behaviour classifier trained on UAH-DriveSet with portable features only (speed-derived dynamics + orientation-free phone motion); lane / car-following / limit features kept as a UAH-only ablation; frozen-speed windows gated out of LEVIN | UAH phone axes are inconsistent across trips; LEVIN has no lane/GPS track; half of LEVIN windows have a frozen speed logger (`reports/results/telematics_uah.md`) |
| 2026-10-10 | Fatigue classifier v1 on RLDD Fold 1 only (12 subjects); RLDD processed 2 videos in parallel on Colab (~3 h / fold); personal baseline taken from the driver's alert video | Fold 1 is enough for a first leave-one-subject-out result; personal baseline gave the largest gain (video AUC 0.81 → 0.94) (`reports/results/fatigue_rldd.md`) |
| 2026-10-11 | INVEDRIFAC labelled by us (881 clips; 286 unclear from blur / darkness, excluded). Eye thresholds adapted on Indian drivers (leave-one-driver-out) and kept as an `india` region profile, not a global default | Western thresholds: F1 0.50 on Indian drivers; India-adapted 0.80; but India-adapted drops DMD from 0.92 to 0.79 (`reports/results/invedrifac_adaptation.md`). Labelling tool upgraded (eye zoom, frame stepping) after the first version proved too unclear |
| 2026-10-11 | Own recordings moved out of v1 into the internship plan; v1 Indian validation = INVEDRIFAC (fatigue), DashGaze sample (blinks / gaze), IDD Lite (road), LEVIN (telematics profile) | Scope and time; protocol and consent form are ready for the internship |
| 2026-10-12 | RLDD Fold 2 added (24 subjects) to test whether the personal-baseline gain is real; bootstrap interval added to the fatigue report | With 12 subjects the gain's 95 % interval touched zero (−0.01 to +0.26); with 24 it is +0.03 to +0.23. Absolute AUCs fell (0.94 → 0.86), the 12-subject estimate was optimistic |
