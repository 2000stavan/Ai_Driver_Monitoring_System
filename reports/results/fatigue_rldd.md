# Fatigue classifier: UTA-RLDD -> INVEDRIFAC (Indian)

UTA-RLDD: 72 videos, 24 subjects (fold(s): Fold1, Fold2); 60 s windows, 10 s step; windows with a face in < 50% of frames dropped. Labels are the subject's own state for the whole video (self-recorded at home, not driving). XGBoost, class-balanced, **leave-one-subject-out**.

|                                                       |   window_macro_F1 |   window_bal_acc |   video_accuracy | videos_correct   |   window_AUC |   video_AUC |
|:------------------------------------------------------|------------------:|-----------------:|-----------------:|:-----------------|-------------:|------------:|
| PERCLOS rule (>= 0.15), no training | alert vs drowsy |             0.548 |            0.596 |            0.562 | 27/48            |        0.749 |       0.793 |
| XGBoost, self calibration | alert vs drowsy           |             0.632 |            0.634 |            0.688 | 33/48            |        0.705 |       0.736 |
| XGBoost, self calibration | 3 classes                 |             0.378 |            0.381 |            0.417 | 30/72            |      nan     |     nan     |
| XGBoost, personal calibration | alert vs drowsy       |             0.728 |            0.728 |            0.812 | 39/48            |        0.802 |       0.861 |
| XGBoost, personal calibration | 3 classes             |             0.501 |            0.503 |            0.625 | 45/72            |      nan     |     nan     |

**Personal baseline vs self calibration** (alert vs drowsy, video AUC, 2,000 bootstrap resamples of subjects): self 95 % CI 0.65-0.83, personal 0.75-0.95; gain +0.026 to +0.229, personal better in 99.3% of resamples.

Median window features per label (self calibration):

|               |   perclos |   blink_rate |   closure_mean_s |   ear_rel_mean |
|:--------------|----------:|-------------:|-----------------:|---------------:|
| alert         |     0.03  |           10 |            0.184 |          0.918 |
| low_vigilance |     0.054 |           14 |            0.206 |          0.914 |
| drowsy        |     0.08  |           14 |            0.26  |          0.894 |

Top features (final binary model): ear_rel_std (0.16), closure_max_s (0.14), closure_mean_s (0.10), yaw_std (0.09), blink_rate (0.07), ear_rel_mean (0.07)

## INVEDRIFAC (Indian drivers, no labels) - profile, NOT a validation

Self-calibrated binary model; `p_drowsy` = mean predicted probability of drowsy over the windows.

| driver   |   minutes |   windows |   p_drowsy_mean |   p_drowsy_first_third |   p_drowsy_last_third |   perclos_mean |
|:---------|----------:|----------:|----------------:|-----------------------:|----------------------:|---------------:|
| Sub_01   |     3.474 |        15 |           0.404 |                  0.478 |                 0.365 |          0.07  |
| Sub_02   |     1.507 |         4 |           0.342 |                  0.321 |                 0.583 |          0.055 |
| Sub_03   |     5.113 |        25 |           0.635 |                  0.755 |                 0.576 |          0.14  |
| Sub_04   |     3.923 |         0 |         nan     |                nan     |               nan     |        nan     |
| Sub_05   |     0.887 |         0 |         nan     |                nan     |               nan     |        nan     |
| Sub_06   |    20.832 |       118 |           0.33  |                  0.239 |                 0.563 |          0.039 |
| Sub_07   |    26.762 |       150 |           0.129 |                  0.093 |                 0.264 |          0.043 |
| Sub_08   |    14.897 |        82 |           0.211 |                  0.316 |                 0.152 |          0.018 |
| Sub_09   |    24.316 |       127 |           0.02  |                  0.007 |                 0.035 |          0.014 |
| Sub_10   |    20.367 |       117 |           0.282 |                  0.235 |                 0.531 |          0.079 |
| Sub_11   |    13.962 |        76 |           0.142 |                  0.064 |                 0.226 |          0.011 |
| Sub_12   |    16.536 |        89 |           0.151 |                  0.067 |                 0.298 |          0.046 |
| Sub_13   |    25.914 |       145 |           0.065 |                  0.072 |                 0.098 |          0.015 |
| Sub_14   |    19.882 |       106 |           0.093 |                  0.101 |                 0.127 |          0.007 |
| Sub_15   |    11.544 |        52 |           0.251 |                  0.425 |                 0.071 |          0.026 |
| Sub_16   |     2.976 |        12 |           0.617 |                  0.584 |                 0.471 |          0.3   |
| Sub_17   |    16.567 |        94 |           0.608 |                  0.495 |                 0.614 |          0.043 |
| Sub_18   |    15.171 |        81 |           0.125 |                  0.101 |                 0.224 |          0.018 |
| Sub_20   |    17.918 |        99 |           0.625 |                  0.725 |                 0.59  |          0.046 |
| Sub_21   |    21.119 |       117 |           0.13  |                  0.057 |                 0.317 |          0.018 |
| Sub_22   |    22.482 |       129 |           0.148 |                  0.072 |                 0.356 |          0.023 |
| Sub_26   |    27.402 |       150 |           0.142 |                  0.202 |                 0.145 |          0.039 |
| Sub_27   |    28.261 |       152 |           0.061 |                  0.06  |                 0.055 |          0.013 |
| Sub_28   |    28.294 |       162 |           0.059 |                  0.12  |                 0.05  |          0.024 |

Drowsiness score higher in the last third than the first third for 13 of 22 drivers.

## Caveats

- RLDD labels are per video and self-reported; low vigilance vs drowsy is a fuzzy line.
- RLDD is recorded with laptops / phones at home, not in a car; INVEDRIFAC is in-car, so this is a cross-domain transfer. Indian numbers need the INVEDRIFAC labels (scripts/label_tool.py).
- Only the folds downloaded so far are used; more folds (notebook, one fold per session) = more subjects.