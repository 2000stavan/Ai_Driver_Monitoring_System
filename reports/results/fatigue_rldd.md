# Fatigue classifier: UTA-RLDD -> INVEDRIFAC (Indian)

UTA-RLDD: 36 videos, 12 subjects (fold(s): Fold1); 60 s windows, 10 s step; windows with a face in < 50% of frames dropped. Labels are the subject's own state for the whole video (self-recorded at home, not driving). XGBoost, class-balanced, **leave-one-subject-out**.

|                                                       |   window_macro_F1 |   window_bal_acc |   video_accuracy | videos_correct   |   window_AUC |   video_AUC |
|:------------------------------------------------------|------------------:|-----------------:|-----------------:|:-----------------|-------------:|------------:|
| PERCLOS rule (>= 0.15), no training | alert vs drowsy |             0.465 |            0.564 |            0.542 | 13/24            |        0.729 |       0.778 |
| XGBoost, self calibration | alert vs drowsy           |             0.681 |            0.682 |            0.833 | 20/24            |        0.76  |       0.806 |
| XGBoost, self calibration | 3 classes                 |             0.396 |            0.41  |            0.417 | 15/36            |      nan     |     nan     |
| XGBoost, personal calibration | alert vs drowsy       |             0.777 |            0.777 |            0.875 | 21/24            |        0.829 |       0.944 |
| XGBoost, personal calibration | 3 classes             |             0.51  |            0.517 |            0.639 | 23/36            |      nan     |     nan     |

Median window features per label (self calibration):

|               |   perclos |   blink_rate |   closure_mean_s |   ear_rel_mean |
|:--------------|----------:|-------------:|-----------------:|---------------:|
| alert         |     0.027 |           11 |            0.168 |          0.916 |
| low_vigilance |     0.043 |           12 |            0.2   |          0.918 |
| drowsy        |     0.057 |           14 |            0.226 |          0.918 |

Top features (final binary model): closure_mean_s (0.17), ear_rel_std (0.12), closure_max_s (0.09), blink_rate (0.08), ear_rel_mean (0.07), mar_mean (0.06)

## INVEDRIFAC (Indian drivers, no labels) - profile, NOT a validation

Self-calibrated binary model; `p_drowsy` = mean predicted probability of drowsy over the windows.

| driver   |   minutes |   windows |   p_drowsy_mean |   p_drowsy_first_third |   p_drowsy_last_third |   perclos_mean |
|:---------|----------:|----------:|----------------:|-----------------------:|----------------------:|---------------:|
| Sub_01   |     3.474 |        15 |           0.719 |                  0.986 |                 0.808 |          0.07  |
| Sub_02   |     1.507 |         4 |           0.475 |                  0.847 |                 0.835 |          0.055 |
| Sub_03   |     5.113 |        25 |           0.991 |                  0.994 |                 0.993 |          0.14  |
| Sub_04   |     3.923 |         0 |         nan     |                nan     |               nan     |        nan     |
| Sub_05   |     0.887 |         0 |         nan     |                nan     |               nan     |        nan     |
| Sub_06   |    20.832 |       118 |           0.191 |                  0.196 |                 0.305 |          0.039 |
| Sub_07   |    26.762 |       150 |           0.057 |                  0.106 |                 0.055 |          0.043 |
| Sub_08   |    14.897 |        82 |           0.077 |                  0.101 |                 0.063 |          0.018 |
| Sub_09   |    24.316 |       127 |           0.014 |                  0.005 |                 0.028 |          0.014 |
| Sub_10   |    20.367 |       117 |           0.244 |                  0.153 |                 0.527 |          0.079 |
| Sub_11   |    13.962 |        76 |           0.019 |                  0.01  |                 0.017 |          0.011 |
| Sub_12   |    16.536 |        89 |           0.217 |                  0.206 |                 0.232 |          0.046 |
| Sub_13   |    25.914 |       145 |           0.033 |                  0.059 |                 0.034 |          0.015 |
| Sub_14   |    19.882 |       106 |           0.027 |                  0.045 |                 0.032 |          0.007 |
| Sub_15   |    11.544 |        52 |           0.213 |                  0.364 |                 0.01  |          0.026 |
| Sub_16   |     2.976 |        12 |           0.927 |                  0.985 |                 0.821 |          0.3   |
| Sub_17   |    16.567 |        94 |           0.572 |                  0.458 |                 0.656 |          0.043 |
| Sub_18   |    15.171 |        81 |           0.081 |                  0.064 |                 0.157 |          0.018 |
| Sub_20   |    17.918 |        99 |           0.17  |                  0.22  |                 0.085 |          0.046 |
| Sub_21   |    21.119 |       117 |           0.07  |                  0.012 |                 0.191 |          0.018 |
| Sub_22   |    22.482 |       129 |           0.12  |                  0.054 |                 0.29  |          0.023 |
| Sub_26   |    27.402 |       150 |           0.114 |                  0.221 |                 0.089 |          0.039 |
| Sub_27   |    28.261 |       152 |           0.017 |                  0.014 |                 0.018 |          0.013 |
| Sub_28   |    28.294 |       162 |           0.04  |                  0.076 |                 0.042 |          0.024 |

Drowsiness score higher in the last third than the first third for 10 of 22 drivers.

## Caveats

- RLDD labels are per video and self-reported; low vigilance vs drowsy is a fuzzy line.
- RLDD is recorded with laptops / phones at home, not in a car; INVEDRIFAC is in-car, so this is a cross-domain transfer. Indian numbers need the INVEDRIFAC labels (scripts/label_tool.py).
- Only the folds downloaded so far are used; more folds (notebook, one fold per session) = more subjects.