# INVEDRIFAC: Western default vs. Indian-adapted eye-closure detector

Ground truth = our check-and-confirm labels (881 clips, 286 unclear and excluded). Default (Western / DMD) thresholds: closed_ear_ratio 0.65, closed_blink_score 0.55. **After** = thresholds tuned on the other Indian drivers only (leave-one-driver-out).

|                                |   drivers |   eye events |   before recall |   before precision |   before F1 |   after recall |   after precision |   after F1 |
|:-------------------------------|----------:|-------------:|----------------:|-------------------:|------------:|---------------:|------------------:|-----------:|
| all drivers                    |        24 |          398 |           0.359 |              0.806 |       0.497 |          0.822 |             0.782 |      0.802 |
| clear drivers (< 30 % unclear) |        13 |          282 |           0.426 |              0.878 |       0.573 |          0.954 |             0.81  |      0.876 |
| blurry / dark drivers          |        11 |          116 |           0.198 |              0.615 |       0.3   |          0.5   |             0.684 |      0.578 |

Thresholds chosen across the folds: {'closed_ear_ratio': 0.8, 'closed_blink_score': 0.5} x24

Per driver (events = labelled eye events, found = detected, dets = detections, correct = detections matching a labelled event):

| driver   |   unclear_frac |   before_events |   before_found |   before_dets |   before_correct |   after_events |   after_found |   after_dets |   after_correct |
|:---------|---------------:|----------------:|---------------:|--------------:|-----------------:|---------------:|--------------:|-------------:|----------------:|
| Sub_01   |           0.09 |              27 |             16 |            14 |               14 |             27 |            23 |           26 |              23 |
| Sub_02   |           0.59 |              46 |              3 |             4 |                3 |             46 |             5 |            9 |               5 |
| Sub_03   |           0.6  |              10 |              4 |            11 |               10 |             10 |             5 |           14 |              13 |
| Sub_04   |           1    |               0 |              0 |             0 |                0 |              0 |             0 |            0 |               0 |
| Sub_05   |           0.52 |               1 |              0 |             0 |                0 |              1 |             1 |            9 |               1 |
| Sub_06   |           0.18 |              21 |             12 |            14 |               12 |             21 |            19 |           36 |              21 |
| Sub_07   |           0.02 |              58 |              3 |             4 |                4 |             58 |            57 |           59 |              57 |
| Sub_08   |           0    |              16 |             10 |            10 |               10 |             16 |            16 |           16 |              16 |
| Sub_09   |           1    |               0 |              0 |             0 |                0 |              0 |             0 |            0 |               0 |
| Sub_10   |           0.16 |              23 |             16 |            22 |               16 |             23 |            22 |           33 |              22 |
| Sub_11   |           0.1  |              16 |              3 |             5 |                3 |             16 |            15 |           20 |              15 |
| Sub_12   |           0    |              17 |             11 |            12 |               11 |             17 |            17 |           18 |              17 |
| Sub_13   |           0.36 |              26 |              6 |             6 |                6 |             26 |            24 |           24 |              24 |
| Sub_14   |           0    |               7 |              3 |             3 |                3 |              7 |             7 |           10 |               7 |
| Sub_15   |           0.65 |               4 |              0 |             4 |                0 |              4 |             2 |            3 |               2 |
| Sub_16   |           0.53 |              13 |              3 |            14 |                5 |             13 |             6 |           13 |               5 |
| Sub_17   |           0.56 |              10 |              4 |             5 |                5 |             10 |            10 |           10 |              10 |
| Sub_18   |           0    |               7 |              4 |             4 |                4 |              7 |             7 |           14 |               8 |
| Sub_20   |           0.52 |               4 |              3 |             8 |                3 |              4 |             3 |           11 |               3 |
| Sub_21   |           0    |              18 |              3 |             6 |                4 |             18 |            18 |           29 |              19 |
| Sub_22   |           0.03 |              25 |             20 |            23 |               22 |             25 |            25 |           28 |              25 |
| Sub_26   |           0.22 |              13 |              8 |            10 |                8 |             13 |            13 |           17 |              13 |
| Sub_27   |           0.57 |               2 |              0 |             0 |                0 |              2 |             2 |            2 |               2 |
| Sub_28   |           0.27 |              34 |             11 |            12 |               11 |             34 |            30 |           31 |              30 |

Cross-check on Western data (DMD drowsiness, 16 recordings): default thresholds eyes-closed F1 0.923, Indian-adapted 0.785 (long-closure precision 0.69 -> 0.47). Thresholds do not transfer in either direction, so they are kept as a region profile: `load_config('cabin', profile='india')`.

Notes: yawns are not tuned (3 labelled yawns). The candidate generator can miss events: the random spot checks found a few (see invedrifac_labelled_eval.md), so recall is an upper bound.