# DMD distraction — phone call / texting / drinking (face camera, leave-one-subject-out)

49 recordings, 14 subjects, 26041 one-second windows. Support: {'safe': 10241, 'phone_call': 6051, 'texting': 2257, 'drinking': 846, 'other': 6646}

| detector          |   phone_call_f1 |   texting_f1 |   drinking_f1 |   safe_f1 |   macro_f1_target |
|:------------------|----------------:|-------------:|--------------:|----------:|------------------:|
| rules (zero-shot) |            0.94 |         0.02 |          0.81 |      0.69 |              0.59 |
| xgboost (hands)   |            0.96 |         0.38 |          0.57 |      0.7  |              0.64 |
| xgboost (objects) |            0.97 |         0.22 |          0.85 |      0.03 |              0.68 |
| xgboost (all)     |            0.99 |         0.38 |          0.89 |      0.7  |              0.75 |

Confusion matrix, best detector (xgboost (all)):

| true       |   drinking |   other |   phone_call |   safe |   texting |
|:-----------|-----------:|--------:|-------------:|-------:|----------:|
| drinking   |        785 |      25 |            1 |     27 |         8 |
| other      |         87 |    2415 |           56 |   2589 |      1499 |
| phone_call |          1 |      70 |         5965 |     12 |         3 |
| safe       |         49 |     997 |            3 |   7392 |      1800 |
| texting    |          1 |     227 |            2 |    721 |      1306 |