# LEVIN everyday-driving profile (India, Pune; unlabelled)

- Vehicles: 21 | trips with ≥ 60 s driving: 3700
- Orientation reliable (forward corr ≥ 0.5): 76% of trips, median corr 0.80
- Distance in reliable trips: 15357 km

## Strongest acceleration per driving second (0.5 s smoothed), percentiles

|       |   braking_g |   acceleration_g |   lateral_g |
|:------|------------:|-----------------:|------------:|
| p50   |       0.049 |            0.054 |       0.072 |
| p90   |       0.162 |            0.16  |       0.157 |
| p95   |       0.212 |            0.201 |       0.194 |
| p99   |       0.336 |            0.313 |       0.284 |
| p99.9 |       0.53  |            0.503 |       0.422 |

## Candidate events with v1 thresholds (brake/accel 0.3 g, lateral 0.35 g, ≥ 0.5 s)

Per 100 km: HARSH_BRAKE 21.6, HARSH_ACCEL 8.3, SHARP_TURN 2.9

Per vehicle (per 100 km):

|   vehicle |     km |   HARSH_BRAKE |   HARSH_ACCEL |   SHARP_TURN |
|----------:|-------:|--------------:|--------------:|-------------:|
|        53 | 3187.5 |           6   |           0.6 |          0.9 |
|        43 | 1881.2 |          10.8 |           1.3 |          1.4 |
|        46 | 1703.1 |           5.9 |           1.4 |          0.4 |
|        23 | 1513.5 |          19.8 |           1.1 |          2.2 |
|        45 | 1484.2 |          88.1 |          64.1 |          7.6 |
|        12 | 1122.2 |           9.2 |           2.4 |          2.9 |
|        13 |  971.9 |          42.4 |           4.9 |          1.2 |
|         2 |  735.5 |          19.7 |           2.7 |          8.6 |
|         4 |  663.6 |          17.2 |           2.6 |          5   |
|        37 |  400.4 |          41   |           1.7 |         13   |
|        47 |  380.5 |           7.9 |           0.3 |          0.3 |
|         1 |  369.1 |          18.7 |           4.9 |          2.2 |
|        35 |  247.6 |           6.5 |           0.4 |          1.2 |
|        11 |  227.3 |          15.8 |           4.4 |          1.8 |
|        51 |  208.4 |          25.9 |          28.8 |          5.8 |
|        19 |   96.2 |          20.8 |          10.4 |         12.5 |
|        38 |   45.8 |          10.9 |           0   |          4.4 |
|        44 |   44.7 |          29.1 |          29.1 |          2.2 |
|         5 |   36.1 |          49.8 |           8.3 |          8.3 |
|         8 |   35.2 |          22.7 |           0   |          8.5 |
|        34 |    3   |          33.5 |           0   |          0   |

Caveats: unlabelled data (these are *candidate* events, not verified harsh events); the forward-axis fit picks the best of several speed sources / lags, which slightly inflates the reported correlation.

**Note (2026-10-09):** vehicles 4, 46, 23 and 37 have a frozen OBD speed in most minutes (speed never changes while moving). Harsh-event detection uses the accelerometer, but its 5 km/h speed gate and the orientation fit use speed, so rates for these vehicles are less reliable. See `telematics_uah.md`.
