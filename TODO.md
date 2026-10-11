# My pending tasks (do once internet is stable)

Tick each box when done and tell Claude, so the matching code step can run.

## 1. Colab jobs (need stable internet; keep the tab open)
- [ ] **Notebook A, IDD download**: `notebooks/02_idd_download.ipynb` (CPU runtime). Get the two links after logging in at https://idd.insaan.iiit.ac.in/
  - IDD Detection + IDD Segmentation Part I → Drive `DMS/raw/idd/` (~3–6 h with the parallel `wget` cell)
- [x] **Notebook B, RLDD Fold 1** (done 2026-10-10; Folds 2–5 optional: set `FOLDS = ['Fold2']` etc., run cells 1→2→3→4b, ~3 h each, then copy the new `FoldN` folder to `data/processed/rldd/`): `notebooks/01_rldd_extract_features.ipynb` (CPU runtime)
  - First upload `dist/dms_code.zip` → Drive `DMS/code/dms_code.zip` (rebuilt 2026-10-09; re-upload it)
  - RLDD: https://sites.google.com/view/utarldd/home (Drive shortcut `MyDrive/Drowsiness`) (~2 h per fold)

## 2. Downloads
| Done | Dataset | Link | What to get | Put it in |
|---|---|---|---|---|
| ✅ | **UAH-DriveSet** (done 2026-10-09) | http://www.robesafe.com/personal/eduardo.romera/uah-driveset/ | Scroll to **Download**, accept the licence yourself, download the ZIP (it includes videos; after unzipping keep the sensor `.txt` files, videos can be deleted) | Mac `data/raw/telematics/uah_driveset/` |
| ☐ | Bangladesh driver dataset | https://zenodo.org/records/14908802 | All (2.6 GB) | Drive `DMS/raw/bangladesh/` |
| ☐ | YawDD | https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset (free IEEE account) | `YawDD.rar.gz` (4.9 GB) | Drive `DMS/raw/yawdd/` |
| ☐ | Speed-breaker set | https://data.mendeley.com/datasets/bvpt9xdjz8/1 | All | Mac `data/raw/road/speed_breaker/` |
| ☐ | D3S | https://github.com/bindujiit/Driver-Drowsiness-Dataset-D3S- | Raw videos + annotated set (skip extracted frames) | Mac `data/raw/cabin/d3s/` |
| ⏸️ | DAAD-X | https://cvit.iiit.ac.in/images/datasets/daadx/daadx.tar.gz | Whole archive (18.6 GB), via Colab `wget -c` | Drive `DMS/raw/daadx.tar.gz` |
| — | MRL Eye, BharatPothole | https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset · https://www.kaggle.com/datasets/surbhisaswatimohanty/bharatpothole | Nothing to download: added on Kaggle later (**Add Data**) | — |

## 3. Dataset access (2026-10-11: no emails needed now; see `recording/DATASET_REQUEST_EMAILS.md`)
- ⏭️ **DashGaze full**: public on the India Data portal (201 GB, login; found 2026-10-11): **no email needed**; download in parts during the internship
- [x] **SynDD1** → AI City Challenge 2024 Track 3 request form submitted 2026-10-11 (wait for the download email; internship use). Use AI City Challenge 2024 Track 3 (99 drivers) via the online request form + licence: https://www.aicitychallenge.org/2024-data-and-evaluation/
- [ ] **3MDAD**: official Kaggle copy, no email needed: https://www.kaggle.com/datasets/anouarbenkhalifa/3mdad-multimodal-driver-action-distraction
- ⏭️ **IDD-CRS**: not public (checked IIIT IDD site + India Data, 2026-10-11); not used in v1. Email draft kept for the internship (ravi.mishra@research.iiit.ac.in, cc ravi.kiran@iiit.ac.in)
- [ ] Check inbox for the **INVEDRIFAC** authors' reply (labels)

## 4. Offline (no internet needed)
- [x] **Label INVEDRIFAC clips** (done 2026-10-11) (881 clips, ~1–1.5 h, resumable; `q` to stop):
  `.venv/bin/python scripts/label_tool.py`
- ⏭️ Own recordings (Sessions A/B): **not in v1** (decided 2026-10-11); protocol + consent form kept in `recording/` for the internship.

## What each task unlocks (Claude's side)
| When this is done | Claude builds / runs |
|---|---|
| ~~RLDD Fold 1~~ | ✅ Done: fatigue classifier (`reports/results/fatigue_rldd.md`); more folds = rerun `scripts/train_fatigue_rldd.py` |
| ~~INVEDRIFAC labels~~ | ✅ Done: Western vs India-adapted detector (`reports/results/invedrifac_adaptation.md`) |
| ~~UAH-DriveSet~~ | ✅ Done: behaviour classifier (`reports/results/telematics_uah.md`) |
| Own Session B drives | Indian telematics validation |
| Bangladesh / YawDD / D3S | Extra South Asian cabin tests |
