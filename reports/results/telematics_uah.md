# Driving behaviour: UAH-DriveSet (labelled) -> LEVIN Pune (unlabelled)

UAH-DriveSet: 40 trips, 6 drivers, 2872 windows of 60 s (10 s step). Labels = behaviour the driver was asked to **simulate** (normal / drowsy / aggressive). XGBoost, class-balanced, **leave-one-driver-out**.

|                                             |   window_macro_F1 |   window_bal_acc |   F1_normal |   F1_drowsy |   F1_aggressive |   aggressive_vs_rest_AUC |   trip_accuracy | trips_correct   |
|:--------------------------------------------|------------------:|-----------------:|------------:|------------:|----------------:|-------------------------:|----------------:|:----------------|
| A dynamics (speed-derived)                  |             0.479 |            0.488 |       0.464 |       0.367 |           0.608 |                    0.816 |           0.55  | 22/40           |
| B + phone motion (portable)                 |             0.474 |            0.488 |       0.543 |       0.298 |           0.581 |                    0.785 |           0.7   | 28/40           |
| C + speed level                             |             0.55  |            0.556 |       0.557 |       0.362 |           0.73  |                    0.907 |           0.675 | 27/40           |
| D + UAH extras (lane, car-following, limit) |             0.797 |            0.795 |       0.799 |       0.774 |           0.818 |                    0.95  |           0.975 | 39/40           |

Confusion matrices (rows = truth, cols = predicted; normal / drowsy / aggressive):

- A dynamics (speed-derived): `[[544, 464, 209], [433, 334, 159], [152, 98, 479]]`
- B + phone motion (portable): `[[657, 362, 198], [438, 250, 238], [110, 142, 477]]`
- C + speed level: `[[703, 397, 117], [502, 309, 115], [101, 76, 552]]`
- D + UAH extras (lane, car-following, limit): `[[1008, 125, 84], [193, 679, 54], [105, 24, 600]]`

Top features of the portable model (tier B): jerk_std (0.16), jerk_abs_p95 (0.10), dyn_mean (0.08), acc_max (0.07), dyn_frac_gt_0.2 (0.06), dyn_max (0.06), dyn_p95 (0.06), brake_max (0.06)

## Applied to LEVIN (Pune, 21 vehicles) - profile, NOT a validation

2213 trips with at least 5 min of driving; windows with at least 50 % moving; 192856 of 383234 windows dropped by the frozen-speed quality gate. Trips predicted: {'AGGRESSIVE': 1752, 'DROWSY': 453, 'NORMAL': 8}.

Mean probability per vehicle (km-weighted):

|   vehicle |   trips |       km |   p_normal |   p_drowsy |   p_aggressive |
|----------:|--------:|---------:|-----------:|-----------:|---------------:|
|         1 |      35 |  476.912 |      0.057 |      0.339 |          0.604 |
|        11 |      32 |  234.595 |      0.058 |      0.284 |          0.658 |
|        12 |     129 | 1615.48  |      0.06  |      0.37  |          0.57  |
|        13 |      93 | 1364.2   |      0.062 |      0.255 |          0.682 |
|        19 |      37 |   92.105 |      0.053 |      0.515 |          0.432 |
|         2 |      73 |  936.322 |      0.068 |      0.406 |          0.526 |
|        23 |     171 | 1759.64  |      0.067 |      0.366 |          0.567 |
|        34 |       2 |    3.69  |      0.24  |      0.343 |          0.417 |
|        35 |      22 |  244.13  |      0.125 |      0.201 |          0.674 |
|        37 |      64 |  580.395 |      0.061 |      0.421 |          0.519 |
|        38 |      10 |   84.916 |      0.047 |      0.114 |          0.838 |
|         4 |      80 |  368.937 |      0.056 |      0.289 |          0.655 |
|        43 |     322 | 2108.56  |      0.077 |      0.331 |          0.592 |
|        44 |      12 |   48.328 |      0.069 |      0.435 |          0.496 |
|        45 |     381 | 3643.97  |      0.168 |      0.466 |          0.366 |
|        46 |      69 |  388.091 |      0.079 |      0.376 |          0.545 |
|        47 |      42 |  452.272 |      0.165 |      0.2   |          0.635 |
|         5 |      11 |   40.07  |      0.07  |      0.52  |          0.41  |
|        51 |      54 |  363.839 |      0.14  |      0.387 |          0.473 |
|        53 |     561 | 4050     |      0.12  |      0.29  |          0.59  |
|         8 |      13 |  114.614 |      0.086 |      0.638 |          0.277 |

### Is the 'aggressive' result about the drivers or the domain?

Share of LEVIN trips per predicted class, speed-only model (A) vs portable model (B):

|                             |   NORMAL |   DROWSY |   AGGRESSIVE |
|:----------------------------|---------:|---------:|-------------:|
| A dynamics (speed-derived)  |    0.17  |    0.041 |        0.789 |
| B + phone motion (portable) |    0.004 |    0.205 |        0.792 |

Top portable-model features: UAH vs LEVIN medians:

|                 |   UAH normal (median) |   UAH aggressive (median) |   LEVIN Pune (median) |
|:----------------|----------------------:|--------------------------:|----------------------:|
| jerk_std        |                 0.008 |                     0.015 |                 0.014 |
| jerk_abs_p95    |                 0.016 |                     0.029 |                 0.028 |
| dyn_mean        |                 0.044 |                     0.058 |                 0.208 |
| acc_max         |                 0.036 |                     0.062 |                 0.071 |
| dyn_frac_gt_0.2 |                 0     |                     0     |                 0.495 |
| dyn_max         |                 0.096 |                     0.137 |                 0.425 |
| dyn_p95         |                 0.069 |                     0.095 |                 0.318 |
| brake_max       |                 0.039 |                     0.074 |                 0.09  |

(Compare with harsh-event rates in `levin_profile.md`.)

### Reading (2026-10-09)

- **Frozen speed** in LEVIN: half of all windows (mostly vehicles 4, 46, 23, 37) have a logged speed that never changes while moving. Before the gate these looked 'normal'; they are a logger fault and are dropped.
- **Speed dynamics**: typical Pune driving has the braking / acceleration / jerk levels of UAH's *simulated aggressive* driving (medians above). Stop-go Indian city traffic, not necessarily aggressive drivers.
- **Phone motion does not transfer**: LEVIN dynamic acceleration is ~4x UAH's (rougher roads, a different logger with 1/64 g resolution). Tier B adds nothing for India.
- So a model trained on Western (Spanish) drives labels ~80 % of Indian trips 'aggressive'. This is the telematics version of RQ1: Western thresholds and models do not transfer. What would: Indian labels (own Session B drives) and **driver- / fleet-relative scores** (personal baseline, innovation #4).

## Caveats

- UAH behaviours are **simulated** by 6 Spanish drivers on Spanish roads; 40 trips is small.
- Tier B leaves out absolute speed on purpose: Indian traffic speeds differ so much from Spanish motorways that a speed-level feature would mostly measure the road, not the driver.
- Phone motion includes road vibration; Indian roads are rougher, so LEVIN 'aggressive' probabilities may be inflated by road surface (an internship question: separate road from driver).
- Drowsy driving shows mainly in lane keeping (tier D); a phone without a road camera sees little of it.
- Indian validation needs labelled Indian drives: our controlled Session B drives (recording/PROTOCOL.md).