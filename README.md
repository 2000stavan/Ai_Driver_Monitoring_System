# AI-based Driver Monitoring System for Indian Roads

Detects driver fatigue and distraction (cabin camera), harsh braking / acceleration / sharp turns (phone IMU + GPS), and road scene understanding (road camera). Models start from pretrained weights, are evaluated or fine-tuned on public data, and are **checked on Indian data** wherever it exists, because systems built on Western data miss Indian drivers, cabins (right-hand drive) and roads.

| Document | What it is |
|---|---|
| [`INITIALPLAN.md`](INITIALPLAN.md) | **Source of truth**: models, datasets, roles, phases (v1) |
| [`PLAN.md`](PLAN.md) | Research design, rationale, limitations, decision log |
| [`DATASETS.md`](DATASETS.md) | Full dataset survey (links, licences, access) |
| [`DOWNLOADS.md`](DOWNLOADS.md) | v1 download checklist (exact parts, save locations) |
| [`reports/dataset_audit.md`](reports/dataset_audit.md) | What each downloaded dataset actually contains |
| [`reports/results/`](reports/results/) | Every result below, with per-recording / per-driver tables |
| [`recording/`](recording/) | Own-recording protocol, consent form, dataset request emails |

## Results so far (v1)

**Cabin: fatigue events** (pretrained MediaPipe Face Landmarker + driver-relative rules, no training)

| Check | Data | Result |
|---|---|---|
| Eyes closed (frame level) | DMD drowsiness, 16 recordings | F1 **0.92** |
| Blinks | DMD | recall 0.95, precision 0.98 |
| Long closures (≥ 1 s) | DMD | recall 0.98 |
| Yawns (incl. hand over mouth) | DMD | recall 0.88, precision 0.97 |
| Blinks, **Indian driver** | DashGaze sample (eye-tracker truth) | recall 0.74, precision 0.94 |
| **Indian drivers, Western thresholds** | INVEDRIFAC, 24 drivers, our 881 labelled clips (286 unclear, excluded) | eye-event F1 **0.50** (recall 0.36) |
| **Indian drivers, India-adapted thresholds** (leave-one-driver-out) | same | eye-event F1 **0.80** (clear videos 0.88); [`invedrifac_adaptation.md`](reports/results/invedrifac_adaptation.md) |
| Robust personal calibration | DMD + 24 INVEDRIFAC Indian drivers | eye F1 0.885 → 0.923; drivers with a broken baseline 3 → 0 |

**Cabin: fatigue classifier** (UTA-RLDD Folds 1–2: 24 subjects, 72 videos, XGBoost on 60 s windows, leave-one-subject-out; [`fatigue_rldd.md`](reports/results/fatigue_rldd.md))

| Method | Alert vs drowsy: video AUC | Videos correct | 3 classes: window F1 |
|---|---|---|---|
| PERCLOS rule (no training) | 0.79 | 27/48 | — |
| XGBoost, each video calibrated on itself | 0.74 | 33/48 | 0.38 |
| XGBoost, **personal baseline** (calibrated on the driver's alert video) | **0.86** | **39/48** | **0.50** |

The personal baseline (innovation #4) is the single biggest gain: +0.03 to +0.23 video AUC (95 % bootstrap interval over subjects; better in 99 % of resamples). Without it, the trained model does not beat the simple PERCLOS rule. Applied to the Indian INVEDRIFAC drivers it gives a per-driver drowsiness score (no labels yet, so not a validation).

**Cabin: distraction** (YOLO11m COCO + MediaPipe Hands + L2CS-Net gaze → XGBoost, leave-one-subject-out, DMD 14 subjects)

| Check | Result |
|---|---|
| Eyes off road | AUC 0.85 (gaze) vs 0.78 (head pose only) |
| Phone call / drinking / texting | F1 **0.99 / 0.89 / 0.38** (texting: phone below the camera view) |

**Telematics** (LEVIN, Pune: 21 vehicles, 15,357 km): phone-to-vehicle axis alignment, harsh brake / accel / turn detection, context attribution of each event. 0.3 g ≈ 99th percentile; 21.6 harsh brakes / 100 km.

**Driving behaviour classifier** (XGBoost on UAH-DriveSet, 40 trips, 6 drivers, leave-one-driver-out; [`telematics_uah.md`](reports/results/telematics_uah.md)):

| Features | Window macro F1 | Trips correct | Aggressive vs rest AUC |
|---|---|---|---|
| Speed-derived dynamics only | 0.48 | 22/40 | 0.82 |
| + phone motion (portable) | 0.47 | 28/40 | 0.79 |
| + speed level | 0.55 | 27/40 | 0.91 |
| + lane offset, car-following, speed limit | **0.80** | **39/40** | **0.95** |

A phone alone sees little of drowsy driving; road-camera information (lane, car ahead) is what makes behaviour classification work, which supports Module C. Applied to LEVIN (Pune), the Western-trained model calls ~80 % of trips "aggressive": typical Pune stop-go driving has the acceleration profile of UAH's *simulated aggressive* driving, and phone vibration is ~4× higher. This is RQ1 for telematics; Indian labels (own drives) and driver-relative scores are the fix. Also found: half of LEVIN's windows have a frozen speed logger (now gated out).

**Road** (SegFormer-B0, IDD Lite val, 7 IDD level-1 classes): Western-only (Cityscapes, zero-shot) **mIoU 0.565** → Indian fine-tuned **0.711** (drivable area IoU 0.85 → 0.94; roadside objects 0.25 → 0.52; non-drivable 0.23 → 0.47). The Western model loses most on exactly the classes that look different on Indian roads. Details: [`road_seg_idd_lite.md`](reports/results/road_seg_idd_lite.md).

**Thresholds are regional:** the India-adapted setting drops DMD eyes-closed F1 from 0.92 to 0.79, so it is kept as a region profile (`load_config("cabin", profile="india")`), not a global change.

**Negative results kept on record:** low-light enhancement (gamma / CLAHE) did not improve face detection on INVEDRIFAC ([`low_light_eval.md`](reports/results/low_light_eval.md)); DMD-tuned thresholds were not adopted because they sat at the grid edges and did worse on the Indian DashGaze driver ([`dmd_drowsiness_tuned.md`](reports/results/dmd_drowsiness_tuned.md)).

## Setup

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install -e ".[ml,app,dev]"
mkdir -p models && curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
.venv/bin/python -m pytest
```

Uses MediaPipe ≥ 1.x (Tasks API; the old `mp.solutions` API no longer exists). Other weights (hand landmarker, YOLO11m, L2CS-Net Gaze360, SegFormer) are listed in INITIALPLAN §11.

## Run

| What | Command |
|---|---|
| Live webcam demo (no dataset needed; look ahead 15 s to calibrate) | `.venv/bin/python scripts/webcam_demo.py` |
| Dashboard (driver camera, driving behaviour, road camera) | `.venv/bin/streamlit run app/streamlit_app.py` |
| Video → per-frame features | `.venv/bin/python scripts/extract_cabin_features.py video.mp4 --hands` |
| DMD fatigue check | `scripts/prepare_dmd.py --bundle drowsiness --videos face` then `scripts/eval_dmd_drowsiness.py` |
| DMD distraction | `scripts/extract_gaze.py`, `scripts/extract_objects.py`, `scripts/eval_dmd_distraction.py`, `scripts/eval_dmd_offroad.py` |
| Indian checks | `scripts/eval_dashgaze.py`, `scripts/compare_calibration.py`, `scripts/eval_low_light.py` |
| Label INVEDRIFAC (check-and-confirm) | `scripts/label_tool.py`, then `scripts/eval_invedrifac_labels.py`, `scripts/tune_invedrifac.py` |
| Telematics profile (LEVIN) | `scripts/levin_profile.py` |
| Behaviour classifier (UAH → LEVIN) | `scripts/train_telematics_uah.py` |
| Fatigue classifier (RLDD → INVEDRIFAC) | `scripts/train_fatigue_rldd.py` (features from `notebooks/01_rldd_extract_features.ipynb` in `data/processed/rldd/`) |
| Road segmentation (Western vs Indian) | `scripts/train_road_seg.py` |
| UTA-RLDD features on Colab | `notebooks/01_rldd_extract_features.ipynb` |
| IDD download to Google Drive on Colab | `notebooks/02_idd_download.ipynb` |

## Layout

```
configs/          thresholds and settings (cabin.yaml, telematics.yaml)
src/dms/
  cabin/          face landmarks, EAR/MAR/head pose, fatigue monitor, hands, gaze (L2CS), objects (YOLO), low light
  baseline/       robust personal calibration (#4)
  telematics/     LEVIN + UAH readers, phone-to-vehicle orientation, harsh events, behaviour features
  road/           SegFormer road segmentation (IDD level-1 classes)
  fusion/         context-aware event attribution (#1, rule-based)
  eval/           metrics, DMD reader
  events.py       timestamped Event: the common output of every module
scripts/          runnable entry points (one per experiment)
notebooks/        Colab notebooks for the large datasets
app/              Streamlit dashboard
recording/        own-recording protocol, consent form, dataset request emails
tests/            unit tests
data/, models/   datasets and weights (git-ignored)
reports/          dataset audit and results
```

## Status

| Phase (INITIALPLAN §14) | State |
|---|---|
| 0. Setup | ✅ Repo, environment, dataset survey + audit, access requests |
| 1. Cabin base | ✅ Fatigue events, gaze, webcam demo, robust calibration, fatigue classifier on RLDD Folds 1–2 (24 subjects) · ⏳ Folds 3–5 (optional) |
| 2. Telematics base | ✅ LEVIN pipeline + profile, UAH classifier + LEVIN domain-shift check · ⏳ Indian validation waits for own controlled drives |
| 3. Road base | ✅ SegFormer Western vs Indian on IDD Lite · ⏳ YOLO hazards wait for IDD Detection (Colab) |
| 4. Own recordings | ⏭️ Not in v1 (protocol + consent form ready for the internship) |
| 5. Distraction | ✅ Phone / drinking on DMD · ⏳ texting, eating, RHD clips |
| 6. Indian adaptation | ✅ INVEDRIFAC labelled (881 clips) → eye-event F1 0.50 → 0.80 · ⏳ DashGaze full (requested), full IDD |
| 7. Personal baseline | ✅ Calibration ablation; personal baseline raises RLDD drowsy-video AUC 0.74 → 0.86 (24 subjects, gain significant) |
| 8. Context attribution | ✅ Rule-based version + tests |
| 9. Dashboard | ✅ Three views on separate datasets · ⏳ one synced drive |

## Deliberately left for the internship

v1 is a working, measured baseline. These are open on purpose: each is a research question, not just engineering.

1. **Learned fusion / context attribution:** replace the hand-written rules (`src/dms/fusion/attribution.py`) with a model trained on synced drives (DAAD-X, own Session C drives).
2. **IMU-as-labeller:** use phone vertical acceleration spikes to auto-label speed breakers and potholes in road video.
3. **Fatigue prediction minutes ahead:** sequence models (GRU/TCN) on the window features in `src/dms/cabin/windows.py`, predicting drowsiness before it shows.
4. **Single Android phone system:** on-device MediaPipe + YOLO-n + IMU in one app.
5. **Lane keeping on unmarked Indian roads:** drivable-area-edge drift from the segmentation model instead of painted lanes.
6. **Indian speed-sign recognition** to replace sparse OpenStreetMap speed limits.
7. **Own Indian recordings + public right-hand-drive distraction benchmark** (phone / eating / drinking in Indian cabins, controlled harsh manoeuvres for Indian telematics labels, synced drives); protocol and consent form ready in `recording/`.
8. **Driver-relative telematics scores:** separate road and traffic from the driver (Western-trained models call most Indian driving aggressive; see `telematics_uah.md`).
9. **Night driving:** IR cabin camera and learned low-light enhancement (simple enhancement did not help).
10. **Texting and eating detection:** a lower or wider camera view, plus video models (VideoMAE / X3D).
11. **Full-scale road models:** YOLO11 hazards on IDD Detection + potholes / speed breakers, SegFormer-B2 on full IDD at full resolution (Kaggle GPU).

## Licences

Several datasets are research / non-commercial only (DMD: CC BY-NC-ND 4.0; LEVIN and BharatPothole: CC BY-NC-SA; others). Data is never committed or redistributed.
