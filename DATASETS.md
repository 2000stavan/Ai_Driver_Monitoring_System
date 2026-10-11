# Datasets — AI-based Driver Monitoring System (India)

Roles, stages and v1/v2 choices: see [`INITIALPLAN.md`](INITIALPLAN.md) §9 (source of truth). Exact parts to download: [`DOWNLOADS.md`](DOWNLOADS.md). Last checked: 2026-10-05 (web search + user's `driver_monitoring_datasets.xlsx` and `indian_driver_monitoring_datasets_detailed.xlsx`).
**(verify)** = licence / access not confirmed online — read the official terms before use.
⭐ = top priority for this project.

**Data type tags:** 🚗 naturalistic (real driving) · 🅿️ staged / parked vehicle · 🎮 simulator · 🧪 lab · 🤖 machine-generated labels · 📟 ADAS-derived reference labels

## Summary

| Module | Indian data available? | Plan |
|---|---|---|
| Cabin — fatigue | ✅ **INVEDRIFAC** (30 drivers, in-vehicle, day + NIR night) | Pre-train Western → fine-tune/validate on INVEDRIFAC + own recordings |
| Cabin — gaze | ✅ **DashGaze** 🚗 (28 drivers, naturalistic) primary; DGAZE 🧪 (lab, supplementary) | Indian gaze model |
| Cabin — phone / eating / drinking | ❌ No strong public Indian dataset with explicit, subject-labelled phone / eating / drinking in RHD cabins. ⚠️ South-Asian RHD proxy: Bangladesh (phone; drinking only inside a broad "Other" class) | Core pre-training: **SynDD1, 3MDAD, DMD (RGB)**; optional: Drive&Act, State Farm, Driver Distraction, SAM-DD → Bangladesh → **own Indian RHD recordings (eating = priority 1)** |
| Cabin + road synced (context attribution, lane change) | ✅ **DAAD**, **DAAD-X** (explanation/context annotations, not causal ground truth), **SIDDI** (driver eyes + road, rural/semi-urban) | Basic fusion rules (#1) + lane-change validation |
| Telematics | ⚠️ **Pune VIIT** (7 drivers) and **LEVIN** (30 cars): both **unlabelled** | Indian distributions + candidate events; labels from own controlled drives; UAH-DriveSet for labelled pre-training |
| Road — hazards | ✅ IDD, IDD-X, BharatPothole, Indian Roads, speed-breaker set | Train directly on Indian data |
| Road — lanes / drivable area | ✅ IDD Segmentation, IDD-AW, e-Yantra Indian Roads; **IDD-CRS** 📟 (ADAS-derived reference labels) | Marked roads: lane boundaries → lane keeping. Unmarked: drivable-region deviation only. Compare with IDD-CRS reference labels |

---

## 1. Cabin — driver camera

### Indian
| Dataset | What | Access | Licence | Use |
|---|---|---|---|---|
| ⭐ **INVEDRIFAC** (IIT Kharagpur) | 30 drivers (20–40 y), in-vehicle, day + NIR night; head orientations, expressions, occlusions, illumination variation; images + videos with ground truth | [sites.google.com/site/invedrifac](https://sites.google.com/site/invedrifac/) — likely by request | (verify) | **Primary Indian fatigue fine-tuning + validation** |
| ⭐ **DashGaze** 🚗 (IIIT-H, IROS 2025) | 900k+ frames, 10 h, **28 drivers, real driving**; synced road + driver + egocentric views via dual-camera phone; gaze labels | [github.com/ThrupthiAnn/DashGaze](https://github.com/ThrupthiAnn/DashGaze): sample public; **full set not yet public**, email authors | (verify) | **Primary Indian naturalistic gaze dataset**; validates the phone-based setup |
| ⭐ **DAAD** (IIIT-H, ECCV 2024) | In-cabin + exterior views, egocentric gaze, sequences before/during manoeuvres (turns, lane changes, halts); varied weather/light | [cvit.iiit.ac.in/.../daad](https://cvit.iiit.ac.in/research/projects/cvit-projects/daad) | (verify) | **Synced cabin + road** → fusion rules, lane-change + mirror-check |
| DGAZE 🧪 (IIIT-H) | 227k driver–road image pairs, 20 drivers; **lab setting** (road video shown to the driver), not naturalistic driving | [cvit.iiit.ac.in/.../dgaze](https://cvit.iiit.ac.in/research/projects/cvit-projects/dgaze) | (verify) | Supplementary gaze pre-training only |
| ⭐ **DAAD-X** (IIIT-H) | Subset of DAAD with hierarchical **explanation/context annotations** for manoeuvres from gaze + ego-vehicle views (in-cabin Aria gaze + out-cabin) | [cvit.iiit.ac.in/.../daadx](https://cvit.iiit.ac.in/research/projects/cvit-projects/daadx) — direct `.tar.gz` download | (verify) | **Context labels** for context-aware event attribution (#1); not causal ground truth |
| **SIDDI** (IIIT Sri City, Smart Transportation Group) | In-vehicle driver eyes/focus + road video from small towns/outskirts of south India (Sri City, Sullurpeta, Kalahasthi): lane-marking visibility, potholes, rural/semi-urban traffic | [Google Drive (public)](https://drive.google.com/drive/folders/1roseEnOTW7dPQ_dtyIUZb5bVYuCyiRKW?usp=sharing), [group page](https://iiits.ac.in/research/research-groups/smart-transportation-group/) | (verify) | Rural/semi-urban Indian test set; check labels before relying on it |
| D3S (JIIT) | 3 subjects; eye-close, yawn, neutral, happy | [GitHub](https://github.com/bindujiit/Driver-Drowsiness-Dataset-D3S-) | None stated | Extra Indian test clips only |
| Multi-Class Driver Behavior (Bangladesh, Nov 2024) — *South-Asian proxy* | 7,286 images, Dhaka; real driving, private cars + buses; safe / turning / texting / talking on phone / others ("others" is broad: drinking, sleeping, talking to someone behind, so **not a clean drinking label**); **RHD vehicles** | [Zenodo 14908802](https://zenodo.org/records/14908802), [Mendeley mzb4b6dff3](https://data.mendeley.com/datasets/mzb4b6dff3/1) | CC BY 4.0 | South-Asian RHD **domain bridge** for phone use (not Indian data); no separate eating/drinking class |
| **Own Indian recordings** | RHD phone-camera clips: phone use, eating, drowsy acting, mirror checks; varied props/lighting | We collect | Ours (consent) | Fill the phone/eating gap; held-out validation |

### Western / international (pre-training)
| Dataset | What | Access | Licence | Use |
|---|---|---|---|---|
| ⭐ MRL Eye | 84,898 IR eye crops, open/closed, glasses, reflections | [mrl.cs.vsb.cz](https://mrl.cs.vsb.cz/eyedataset.html) / [Kaggle](https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset) | (verify) | Eye-state classifier |
| CEW (Closed Eyes in the Wild) | Open/closed eyes, unconstrained | [NUAA](https://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html) | (verify) | Robust eye-state |
| OC-Dataset, Glasses Dataset | Open/closed; glasses vs none | [GitHub](https://github.com/mahmoud-elsayed/OC-Dataset), [Bindu Verma](https://sites.google.com/view/dr-bindu-verma/datasets) | (verify) | Eye-state robustness |
| Driver Drowsiness Dataset (DDD) | Frames from UTA-RLDD | [Kaggle](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) | (verify) | Quick baseline |
| ⭐ UTA-RLDD | ~30 h, 60 participants (incl. Indo-Aryan, Dravidian); alert / low-vigilance / drowsy | [sites.google.com/view/utarldd](https://sites.google.com/view/utarldd/home) | Research (verify) | Temporal fatigue model |
| ⭐ YawDD | 322 in-car videos; yawning / talking / normal | [IEEE DataPort](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset) (free account) | Open access | Yawning |
| ⭐ DMD 🚗🎮 | Originally 41 h, 37 drivers, RGB/IR/depth. **Current public release: RGB only, some participants removed, simulator recordings withdrawn.** Bundles: gaze & hands, drowsiness, distraction (phone, texting, drinking, radio, reaching, hair/makeup, passenger). **Eating not listed**. Drowsiness bundle (README, 14/05/2026): session s5 only, **acted** in an unmoving real car; labels = occlusion, eye state (open/closed/opening/closing), blinks, yawning; 3 cameras (face/body/hands) + mosaic + OpenLABEL JSON per subject; 20 % of data withheld for benchmarking. Distraction bundle (README, 14/05/2026): s1 = real driving (safe, reach side, radio, **drinking**, talking to passenger), s2 = parked (phone call L/R, texting L/R, drinking, hair/makeup, reach side), s3 = parked (reach backseat); levels: occlusion, gaze on road, talking, hands on wheel, hand on gear, objects (cellphone, comb, bottle), driver actions (14 classes) | [dmd.vicomtech.org](https://dmd.vicomtech.org/) (form, 18+) | **CC BY-NC-ND 4.0**, academic only | Core pre-training for phone / drinking / gaze / drowsiness; use only classes present after download |
| Drive&Act 🎮 (Fraunhofer, ICCV 2019) | 12 h, multi-view incl. NIR; fine-grained activities: **eating, drinking**, phone, texting; hierarchical labels. Recorded in a static simulator | [driveandact.com](https://www.driveandact.com/) — request | Non-commercial (verify) | Optional: large fine-grained cabin-action pre-training (incl. eating/drinking, smartphone) |
| State Farm Distracted Driver 🅿️ | 22k labelled images; US, **LHD**; cars were **not actually driven** (towed). Classes: safe, texting L/R, phone L/R, radio, drinking, reaching, hair/makeup, passenger. **No eating class** | Kaggle competition | Competition rules (research only) | Optional phone / drinking pre-training |
| ⭐ **SynDD1** 🅿️ (Iowa State, AI City Challenge) | Staged recordings in a **stationary** car, 3 IR cameras (dashboard, rear-view, right window); 18 behaviours incl. **eating, drinking**, phone call, texting, yawning; 11 gaze zones | [Mendeley 10.17632/ptcp7rp3wb.4](https://data.mendeley.com/datasets/ptcp7rp3wb/4): **files now request-only**, email shaiqur@iastate.edu / anujs@iastate.edu | CC BY | **Core pre-training: best public class match for eating + drinking + phone** (staged, not naturalistic) |
| ⭐ **3MDAD** 🚗 (Tunisia) | Real driving (Sousse); 60+ drivers; daytime RGB + depth, nighttime IR + depth; frontal + side views; 16 in-vehicle actions incl. phone call, texting, drinking, fatigue-related | [HAL paper](https://hal.archives-ouvertes.fr/hal-03321535) — request via authors | Research only (verify; reported CC BY-NC-ND) | **Core pre-training** for phone / drinking in real driving |
| SAM-DD | RGB + depth distraction dataset (drinking, phoning, texting, hair/makeup, glasses, reaching) | Access-dependent | Research only, not redistributable (verify) | Optional pre-training |
| Driver Distraction Dataset 🚗🅿️ (Ezzouhri, Morocco) | ~38k images, 9 drivers, real + parked; 10 State Farm-style classes incl. phone, drinking | [GitHub](https://github.com/AmalEzzouhri/Driver-Distraction-Dataset) (password link) | Academic only | Optional phone / drinking pre-training |
| AUC Distracted Driver | 10 classes (State Farm style), Egypt, **LHD** | Request from authors | Research (verify) | Extra distraction baseline |
| NTHU-DDD | Drowsiness, yawning, glasses, night | [NTHU](https://cv.cs.nthu.edu.tw/~morris/3Ddriver/) — email request | (verify) | Optional — access often slow |
| FatigueFace-1156 | 1,156 images, 8 fatigue classes | [IEEE DataPort](https://ieee-dataport.org/documents/fatigueface-1156-facial-fatigue-dataset-drowsiness-detection-based-eye-closure-yawning) | (verify) | Optional |

---

## 2. Telematics — braking, acceleration, speeding, aggression

| Dataset | What | Access | Licence | Use |
|---|---|---|---|---|
| ⭐ **Pune VIIT Smartphone Driver Behaviour** 🚗 | Accel + gyro @ 50 Hz, Figo/Swift/Nexon, real traffic; **normal / aggressive / risky**. **Driver count unclear** (v1 page: 3 drivers; the paper mentions seven sub-folders, likely one per trip) → count after download | [Mendeley v1](https://data.mendeley.com/datasets/9vr83n7z5j/1), [v2](https://data.mendeley.com/datasets/9vr83n7z5j/2) | CC BY 4.0 | **Labelled Indian training/validation** (leave-one-driver-out); small, so results are indicative |
| ⭐ **LEVIN Open Data** (Yuñ Solutions, Pune) | ~30 cars, 4 months; accel 25 Hz, gyro, magnetometer, GPS, OBD-II (RPM, throttle, speed) — **unlabelled** | [GitHub](https://github.com/YunSolutions/levin-openData) | CC BY-NC-SA | Indian normal-driving distributions, threshold context, risk-score normalisation. **Not** aggression ground truth |
| Phone sensor normal/aggressive (Mendeley) | Accel, gyro, GPS, speed; normal (0) / aggressive (1) | [Mendeley 5stn873wft](https://data.mendeley.com/datasets/5stn873wft/1) | CC BY 4.0 | Extra labelled data — **country unconfirmed** |
| UAH-DriveSet (Spain) | 500+ min, 6 drivers; normal / drowsy / aggressive | [robesafe.com](http://www.robesafe.com/personal/eduardo.romera/uah-driveset/) | Research (verify) | Western baseline for India-vs-West comparison |
| OpenStreetMap `maxspeed` | Speed limits | Overpass API | ODbL | Speeding where tagged; otherwise `SPEED_LIMIT_UNKNOWN` (**no inferred limits**) |
| **Own recordings** | Phone IMU + GPS synced with video | We collect | Ours | Synced events for fusion |

---

## 3. Road — lane keeping, lane changing, hazards

| Dataset | What | Access | Licence | Use |
|---|---|---|---|---|
| ⭐ **IDD Detection** | ~40k+ images, boxes; autorickshaw, animal, rider… | [idd.insaan.iiit.ac.in](https://idd.insaan.iiit.ac.in/) (register) | (verify) | Hazard detector (YOLO) |
| ⭐ **IDD Segmentation (IDD20K)** | 20k images, drivable / non-drivable road, curbs… | same | (verify) | Drivable-area lane keeping |
| IDD Lite | ~50 MB, 7 classes | same | (verify) | Quick Colab experiments |
| ⭐ **IDD-X** | Front + rear dual-view videos; 697k boxes; classes incl. **speed breaker, pothole**; 19 ego-behaviour explanations (cut-in, overtake, deviate, slow down…) | [idd-x.github.io](https://idd-x.github.io/) → india-data.org request | (verify) | Hazards + **why the ego vehicle braked/deviated** (fusion) |
| IDD-AW | Adverse weather / night segmentation | [iddaw.github.io](https://iddaw.github.io/) | (verify) | Robustness testing |
| ⭐ **IDD-CRS** 📟 (IIIT-H, 2025) | Critical road scenarios from a dashcam + an **ADAS unit** whose alerts (forward collision, pedestrian collision, headway, **lane departure**) give temporal **reference (weak) labels**; high-speed lane changes, unsafe approaches; Hyderabad (~90 h, 5,400 one-minute videos, ~2,300 alert clips per paper / user sheet). A critical-event / action dataset, **not** a lane-segmentation dataset | [paper](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2025/IDDCRS_camera_ready.pdf) — access route not found, contact CVIT | (verify) | Reference evidence only where alert types apply (lane-departure warnings, collision warnings); **not** lane-change labels |
| IDD 117K | Large IDD road image release (~72 GB per user sheet) | [insaan.iiit.ac.in](https://insaan.iiit.ac.in/) | (verify) | Only if IDD Detection is not enough |
| IRD — Indian traffic lights | 3,000+ images, 8,000+ annotations; supported/suspended lights; Delhi/Chandigarh | [sites.google.com/view/ird-dataset](https://sites.google.com/view/ird-dataset/home) | (verify) | Optional (signal context) |
| IDD Temporal / Multimodal | Temporal frames; stereo + GPS + LiDAR + OBD | [inai.iiit.ac.in](https://inai.iiit.ac.in/domains/idd.html) | (verify) | Internship phase |
| ⭐ **BharatPothole** (NISER) | 7,000+ Indian dashcam frames, potholes | [Kaggle](https://www.kaggle.com/datasets/surbhisaswatimohanty/bharatpothole) | CC BY-NC-SA 4.0 | Pothole detection |
| Indian Roads Dataset | 4000+ images, potholes + marked/unmarked speed breakers, YOLO format | [Kaggle](https://www.kaggle.com/datasets/mitangshu11/indian-roads-dataset) | (verify) | Speed-breaker detection |
| Marked Speed Breaker Dataset (India) | 969 images | [Mendeley bvpt9xdjz8](https://data.mendeley.com/datasets/bvpt9xdjz8/1) | (verify) | Speed-breaker detection |
| ThirdEye Indian Road Dataset 🤖 | 8,437 clips, 210 GB, Delhi NCR, GPS; **machine-generated, unverified labels** | [Hugging Face](https://huggingface.co/datasets/thirdeyelabs/indian-road-dataset) | CC BY 4.0 | Subset only — test video + GPS, not ground truth |
| Indian Roads (e-Yantra, IIT Bombay) | 3,227 images, semantic segmentation | [Dataset Ninja](https://datasetninja.com/indian-roads-semantic-segmentation) | (verify) | Extra road segmentation |
| Road Lane & Traffic Density India | Indian dashcam videos, inconsistent markings | [IEEE DataPort](https://ieee-dataport.org/node/15060) | (verify) | Lane testing |
| DATS_2022 / ITD / IITM-HeTra | Indian traffic object detection (10k / 17.6k / 6.3k vehicles) | [DATS](https://pmc.ncbi.nlm.nih.gov/articles/PMC9309657/), [ITD](https://ieeexplore.ieee.org/document/10427394/), [HeTra](https://www.kaggle.com/datasets/deepak242424/iitmhetra) | (verify) | Optional extra detection data |
| CULane | Lane markings (China) | Public | Research | Pre-train lane-boundary model for **marked** roads |

---

## RHD domain-shift note
Western distraction sets (State Farm, AUC, DMD, Drive&Act) are **left-hand drive**: driver on the left, camera angle, phone/cup hand and occlusions all differ from Indian RHD cars. Mitigations, run as an ablation: (1) **mirror augmentation** (horizontal flip: it does not reproduce RHD camera geometry, steering-wheel position, cabin layout, mirrors, lighting or hand–object relationships, so it is not real RHD adaptation), (2) Bangladesh RHD proxy, (3) own Indian RHD recordings as the held-out test. Hypotheses to *measure*, not assume: turbans/dupattas/beards near chin–ear–neck, tropical glare, road vibration.

---

## Checked and not used (or internship only)
| Dataset | Reason |
|---|---|
| myEye2Wheeler, MOTOR, IndiGo, I2WDD | Indian but **two-wheeler** — related work only |
| Indian lane + traffic sign (22,400 imgs, PMC 2025) | Available only on request from authors |
| V-SenseDrive (Pakistan), InCarEmo (China), DriverMVT | Non-Indian; references only |
| IDD-3D, Amrita LiDAR | LiDAR — not needed for a phone/camera system |
| DROZY, Brain4Cars, HDD (Honda), AIDE, DADA-2000 | Internship phase (fatigue prediction, learned fusion, causal event modelling) |

---

## Action checklist
1. [ ] **Request access now** (slow): INVEDRIFAC, DAAD, IDD (+ IDD-X via india-data.org), DMD, UTA-RLDD, 3MDAD; optional Drive&Act, SAM-DD; email CVIT about IDD-CRS
2. [ ] Clone/download open sets: **DashGaze**, **DAAD-X**, **SynDD1**, SIDDI, LEVIN, Pune VIIT, BharatPothole, Indian Roads, Bangladesh (Zenodo), MRL Eye, YawDD (IEEE account)
3. [ ] Upload non-Kaggle sets as **private Kaggle datasets** for Colab/Kaggle
4. [ ] **Audit every download**: unique driver count, classes actually present (esp. DMD eating, Pune drivers), licence. Record in a dataset audit table
5. [ ] Check every **(verify)** licence; record citations in `CITATIONS.bib`
6. [ ] Draft consent form + recording protocol (priority: eating → phone → drinking; controls; hard negatives)
