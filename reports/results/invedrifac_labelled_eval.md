# INVEDRIFAC — default detectors vs. our check-and-confirm labels

- **eye**: 398 labelled events, recall **0.41**, precision **0.59** (22 drivers)
- **yawn**: 3 labelled events, recall **0.00**, precision **0.00** (22 drivers)
- **Spot checks**: 7 events found in 65 random no-candidate clips (events the candidate generator missed)

Per driver:

| driver   | group       |   true_events |   detected |   detections |   correct_detections |
|:---------|:------------|--------------:|-----------:|-------------:|---------------------:|
| Sub_01   | eye         |            27 |         18 |           17 |                   16 |
| Sub_01   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_01   | spot_checks |             0 |        nan |            5 |                  nan |
| Sub_02   | eye         |            46 |         13 |           33 |                   12 |
| Sub_02   | yawn        |             3 |          0 |            0 |                    0 |
| Sub_02   | spot_checks |             0 |        nan |            0 |                  nan |
| Sub_03   | eye         |            10 |          6 |           20 |                   12 |
| Sub_03   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_03   | spot_checks |             0 |        nan |            0 |                  nan |
| Sub_05   | eye         |             1 |          0 |            3 |                    0 |
| Sub_05   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_05   | spot_checks |             0 |        nan |            3 |                  nan |
| Sub_06   | eye         |            21 |         12 |           16 |                   12 |
| Sub_06   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_06   | spot_checks |             0 |        nan |            2 |                  nan |
| Sub_07   | eye         |            58 |          3 |            4 |                    4 |
| Sub_07   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_07   | spot_checks |             0 |        nan |            2 |                  nan |
| Sub_08   | eye         |            16 |         10 |           10 |                   10 |
| Sub_08   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_08   | spot_checks |             0 |        nan |            6 |                  nan |
| Sub_10   | eye         |            23 |         17 |           23 |                   17 |
| Sub_10   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_10   | spot_checks |             0 |        nan |            2 |                  nan |
| Sub_11   | eye         |            16 |          3 |            7 |                    3 |
| Sub_11   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_11   | spot_checks |             4 |        nan |            6 |                  nan |
| Sub_12   | eye         |            17 |         11 |           12 |                   11 |
| Sub_12   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_12   | spot_checks |             0 |        nan |            6 |                  nan |
| Sub_13   | eye         |            26 |          6 |            8 |                    6 |
| Sub_13   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_13   | spot_checks |             0 |        nan |            1 |                  nan |
| Sub_14   | eye         |             7 |          3 |            3 |                    3 |
| Sub_14   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_14   | spot_checks |             2 |        nan |            6 |                  nan |
| Sub_15   | eye         |             4 |          0 |            4 |                    0 |
| Sub_15   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_15   | spot_checks |             0 |        nan |            0 |                  nan |
| Sub_16   | eye         |            13 |          9 |           48 |                   11 |
| Sub_16   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_16   | spot_checks |             0 |        nan |            0 |                  nan |
| Sub_17   | eye         |            10 |          4 |            6 |                    5 |
| Sub_17   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_17   | spot_checks |             1 |        nan |            1 |                  nan |
| Sub_18   | eye         |             7 |          4 |            4 |                    4 |
| Sub_18   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_18   | spot_checks |             0 |        nan |            6 |                  nan |
| Sub_20   | eye         |             4 |          3 |           20 |                    3 |
| Sub_20   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_20   | spot_checks |             0 |        nan |            4 |                  nan |
| Sub_21   | eye         |            18 |          3 |            6 |                    4 |
| Sub_21   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_21   | spot_checks |             0 |        nan |            3 |                  nan |
| Sub_22   | eye         |            25 |         20 |           23 |                   22 |
| Sub_22   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_22   | spot_checks |             0 |        nan |            3 |                  nan |
| Sub_26   | eye         |            13 |          8 |           11 |                    8 |
| Sub_26   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_26   | spot_checks |             0 |        nan |            4 |                  nan |
| Sub_27   | eye         |             2 |          0 |            1 |                    0 |
| Sub_27   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_27   | spot_checks |             0 |        nan |            4 |                  nan |
| Sub_28   | eye         |            34 |         12 |           20 |                   13 |
| Sub_28   | yawn        |             0 |          0 |            0 |                    0 |
| Sub_28   | spot_checks |             0 |        nan |            1 |                  nan |