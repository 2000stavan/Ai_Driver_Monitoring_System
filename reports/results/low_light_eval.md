# Low-light preprocessing - face detection on INVEDRIFAC

200 sampled frames per video (after the first minute), each method forced on every frame.

| videos | n | none | gamma | clahe | gamma_clahe |
|---|---|---|---|---|---|
| all | 24 | 0.862 | 0.861 | 0.863 | 0.861 |
| dark (mean luma < 70) | 3 | 0.630 | 0.625 | 0.628 | 0.627 |
| bright | 21 | 0.895 | 0.895 | 0.897 | 0.895 |

Per video (face-detection rate; brightness = mean luma 0-255):

| driver   |   brightness |   face_none |   face_gamma |   face_clahe |   face_gamma_clahe |
|:---------|-------------:|------------:|-------------:|-------------:|-------------------:|
| Sub_01   |      114.302 |       0.95  |        0.95  |        0.94  |              0.94  |
| Sub_02   |       98.256 |       0.744 |        0.75  |        0.756 |              0.756 |
| Sub_03   |       25.719 |       0.905 |        0.89  |        0.89  |              0.89  |
| Sub_04   |       12.663 |       0     |        0     |        0.01  |              0     |
| Sub_05   |       75.034 |       0.745 |        0.745 |        0.78  |              0.76  |
| Sub_06   |      123.198 |       0.925 |        0.925 |        0.925 |              0.925 |
| Sub_07   |      120.434 |       0.9   |        0.9   |        0.88  |              0.88  |
| Sub_08   |      117.512 |       0.925 |        0.925 |        0.92  |              0.92  |
| Sub_09   |      124.022 |       0.87  |        0.87  |        0.87  |              0.87  |
| Sub_10   |      105.032 |       0.915 |        0.915 |        0.91  |              0.9   |
| Sub_11   |      114.143 |       0.9   |        0.9   |        0.9   |              0.9   |
| Sub_12   |      102.816 |       0.91  |        0.91  |        0.91  |              0.905 |
| Sub_13   |      117.575 |       0.93  |        0.93  |        0.94  |              0.94  |
| Sub_14   |      102.027 |       0.9   |        0.895 |        0.905 |              0.905 |
| Sub_15   |      102.341 |       0.71  |        0.71  |        0.725 |              0.725 |
| Sub_16   |       30.259 |       0.985 |        0.985 |        0.985 |              0.99  |
| Sub_17   |      108.117 |       0.995 |        0.995 |        0.995 |              0.99  |
| Sub_18   |      102.655 |       0.93  |        0.935 |        0.93  |              0.93  |
| Sub_20   |      100.692 |       0.915 |        0.915 |        0.92  |              0.915 |
| Sub_21   |      122.3   |       0.935 |        0.935 |        0.93  |              0.93  |
| Sub_22   |      111.169 |       0.935 |        0.93  |        0.935 |              0.93  |
| Sub_26   |      118.452 |       0.925 |        0.925 |        0.925 |              0.925 |
| Sub_27   |      125.297 |       0.9   |        0.9   |        0.9   |              0.9   |
| Sub_28   |      108.372 |       0.94  |        0.94  |        0.94  |              0.94  |

Median EAR with each method (enhancement must not distort the eye measurement):

| driver   |   ear_none |   ear_gamma |   ear_clahe |   ear_gamma_clahe |
|:---------|-----------:|------------:|------------:|------------------:|
| Sub_01   |      0.287 |       0.287 |       0.277 |             0.278 |
| Sub_02   |      0.195 |       0.198 |       0.199 |             0.202 |
| Sub_03   |      0.2   |       0.198 |       0.215 |             0.213 |
| Sub_04   |    nan     |     nan     |       0.236 |           nan     |
| Sub_05   |      0.29  |       0.286 |       0.285 |             0.286 |
| Sub_06   |      0.247 |       0.247 |       0.249 |             0.249 |
| Sub_07   |      0.289 |       0.289 |       0.296 |             0.296 |
| Sub_08   |      0.359 |       0.359 |       0.361 |             0.361 |
| Sub_09   |      0.314 |       0.313 |       0.313 |             0.313 |
| Sub_10   |      0.339 |       0.338 |       0.344 |             0.343 |
| Sub_11   |      0.332 |       0.332 |       0.335 |             0.335 |
| Sub_12   |      0.291 |       0.291 |       0.289 |             0.289 |
| Sub_13   |      0.36  |       0.361 |       0.369 |             0.369 |
| Sub_14   |      0.341 |       0.341 |       0.341 |             0.34  |
| Sub_15   |      0.298 |       0.298 |       0.293 |             0.293 |
| Sub_16   |      0.264 |       0.249 |       0.256 |             0.249 |
| Sub_17   |      0.279 |       0.28  |       0.286 |             0.285 |
| Sub_18   |      0.295 |       0.295 |       0.294 |             0.296 |
| Sub_20   |      0.341 |       0.342 |       0.337 |             0.337 |
| Sub_21   |      0.338 |       0.338 |       0.337 |             0.337 |
| Sub_22   |      0.356 |       0.356 |       0.357 |             0.357 |
| Sub_26   |      0.33  |       0.33  |       0.329 |             0.329 |
| Sub_27   |      0.298 |       0.298 |       0.296 |             0.296 |
| Sub_28   |      0.323 |       0.323 |       0.322 |             0.321 |
## Conclusion (2026-10-09)

- **Enhancement does not help** on INVEDRIFAC: gamma, CLAHE and both give the same face-detection rate as raw
  frames (all videos 0.86 vs 0.86; dark videos 0.63 vs 0.63). MediaPipe already copes with the dim night videos
  (Sub 03, luma 26 -> 91 % faces; Sub 16, luma 30 -> 98 %).
- **Sub 04 is unrecoverable**: the frames are sensor noise / a blocked view with no visible face even after
  stretching (checked visually). It is excluded by the face-visibility quality gate, not "fixed".
- Decision: `LowLightEnhancer` (src/dms/cabin/enhance.py) stays available but **off by default**.
- Left for the internship: an IR / NIR cabin camera (the real fix for night driving) and learned low-light
  enhancement (e.g. Zero-DCE) evaluated on our own night recordings.
