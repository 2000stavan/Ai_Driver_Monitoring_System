# DOWNLOADS — what to get, which part, where to put it (v1)

Checked 2026-10-07. Tick boxes as you go. Roles and reasons: [`INITIALPLAN.md`](INITIALPLAN.md) §9.

**Where things go**
- **Mac:** `~/Desktop/Driver_Monitoring_System/data/raw/<module>/<dataset>/` (git-ignored)
- **Drive:** `My Drive/DMS/raw/<dataset>/` (archives, via Colab `wget -c` straight into Drive so downloads resume after disconnects)
- **Kaggle:** nothing to download; click **Add Data** in a Kaggle notebook when needed

**Why we only take parts:** pretrained models (MediaPipe, L2CS-Net, YOLO11, EfficientNet, SegFormer, CLRNet/UFLD) already know faces, objects and roads. We only need enough labelled examples to fine-tune them plus Indian test data, and videos are turned into small feature tables and then deleted.

---

## 1. Request today (approval is the slow part)

| ☐ | Dataset | Link | Part we need | Then save to |
|---|---|---|---|---|
| ☐ | **INVEDRIFAC** | https://sites.google.com/site/invedrifac/ → Drive folder (request **Viewer**, college account) | **All** (Indian fatigue fine-tune + test) | Drive `DMS/raw/invedrifac/` |
| ☐ | **UTA-RLDD** | https://sites.google.com/view/utarldd/home | **All 5 folds, one at a time** (download → features → delete video) | Colab only; keep features in Drive `DMS/processed/rldd/` |
| ☐ | **DMD** | https://dmd.vicomtech.org/ (form) | **Distraction** + **Drowsiness** bundles only. Skip Gaze & Hands | Drive `DMS/raw/dmd/` |
| ☐ | **YawDD** | https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset (free IEEE account) | **YawDD.rar.gz** (4.94 GB). Skip "Participants Information" unless needed for screenshots | Drive `DMS/raw/yawdd/` |
| ☐ | **SynDD1** | no email: AI City Challenge 2024 Track 3 (99 drivers, same team) via online form + licence: https://www.aicitychallenge.org/2024-data-and-evaluation/ | **Dashboard camera view only**; only safe / eating / drinking / phone call / texting segments | Drive `DMS/raw/syndd1/` |
| ⏭️ | **DashGaze (full)** | public on the India Data portal (201 GB, login; found 2026-10-11): **no email needed**; download in parts during the internship | All sessions if granted (Indian gaze test); the sample is enough to start | Drive `DMS/raw/dashgaze/` |
| ☐ | **3MDAD** | official Kaggle copy (no email needed): https://www.kaggle.com/datasets/anouarbenkhalifa/3mdad-multimodal-driver-action-distraction | **Daytime RGB, frontal view** (night/IR in v2) | Drive `DMS/raw/3mdad/` |
| ☐ | **IDD** | https://idd.insaan.iiit.ac.in/ (register) | **IDD Lite** → **IDD Detection** → **IDD Segmentation Part I** (Part II only if needed) | Lite: Mac `data/raw/road/idd_lite/`; others: Drive `DMS/raw/idd/` |
| ☐ | **IDD-CRS** | https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2025/IDDCRS_camera_ready.pdf → email CVIT | If granted: **alert clips only** (~2,300), not all 90 h | Drive `DMS/raw/idd_crs/` |
| — | DAAD | https://cvit.iiit.ac.in/research/projects/cvit-projects/daad | **Skip in v1** (6 views × 39.6 GB). Only if DAAD-X is not enough: **Driver View + Front View** | Drive `DMS/raw/daad/` |

## 2. Download now to the Mac (small)

| ☐ | Dataset | Link | Part | Save to |
|---|---|---|---|---|
| ✅ | DashGaze sample | (Drive sample folder) | All 5 files | `data/raw/cabin/dashgaze/sample/` |
| ☐ | **Pune VIIT** | https://data.mendeley.com/datasets/9vr83n7z5j/2 | All | `data/raw/telematics/pune_viit/` |
| ☐ | **LEVIN** | https://github.com/YunSolutions/levin-openData | All (CSV / SQLite; pick one format) | `data/raw/telematics/levin/` |
| ☐ | Speed-breaker set | https://data.mendeley.com/datasets/bvpt9xdjz8/1 | All | `data/raw/road/speed_breaker/` |
| ☐ | D3S | https://github.com/bindujiit/Driver-Drowsiness-Dataset-D3S- | **Raw videos + annotated set** (skip extracted frames; we extract our own) | `data/raw/cabin/d3s/` |
| ☐ | CEW | https://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html | All | `data/raw/cabin/cew/` |

## 3. Download via Colab → Drive (large, background)

| ☐ | Dataset | Link | Part | Save to | Expected time |
|---|---|---|---|---|---|
| ⏸️ | **DAAD-X** | https://cvit.iiit.ac.in/images/datasets/daadx/daadx.tar.gz | Whole archive (single file, 18.6 GB) | Drive `DMS/raw/daadx.tar.gz` | ~2–5 h (IIIT server); resumes with `wget -c` |
| ☐ | **Bangladesh** | https://zenodo.org/records/14908802 | All (2.6 GB) | Drive `DMS/raw/bangladesh/` | minutes |
| ☐ | UAH-DriveSet | http://www.robesafe.com/personal/eduardo.romera/uah-driveset/ | **Sensor files only; skip the videos** | Drive `DMS/raw/uah/` | < 1 h |
| ☐ | IDD-AW | https://iddaw.github.io/ | **Validation split only** (test use) | Drive `DMS/raw/idd_aw/` | hours (IIIT server); Phase 6 |

## 4. No download

| ☐ | Dataset | Link | How |
|---|---|---|---|
| ☐ | **SIDDI** | https://drive.google.com/drive/folders/1roseEnOTW7dPQ_dtyIUZb5bVYuCyiRKW?usp=sharing | Right-click → **Add shortcut to Drive** |
| — | **MRL Eye** | https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset | Kaggle **Add Data** |
| — | **BharatPothole** | https://www.kaggle.com/datasets/surbhisaswatimohanty/bharatpothole | Kaggle **Add Data** |
| — | Indian Roads | https://www.kaggle.com/datasets/mitangshu11/indian-roads-dataset | Kaggle **Add Data** |
| — | DDD (fallback only) | https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd | Only if UTA-RLDD access fails (DDD is frames from RLDD) |

## 5. Not needed in v1

| Dataset | Why |
|---|---|
| CULane | Lane models come with CULane-trained weights |
| DAAD (full) | DAAD-X covers v1 needs |
| DMD Gaze & Hands bundle | Gaze comes from L2CS-Net (pretrained) + DashGaze |
| IDD Segmentation Part II, IDD 117K, IDD Temporal, IDD Multimodal | Pretrained SegFormer needs less data; add only if results plateau |
| SynDD1 rear-view and side-window cameras | Dashboard view matches our setup |
| UAH videos | Telematics uses sensor data only |
| DGAZE, IDD-X, Drive&Act, State Farm, Driver Distraction, NTHU-DDD, SAM-DD, ThirdEye, Brain4Cars | v2 optimisation (see INITIALPLAN §16) |
