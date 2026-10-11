# Phase 1 — rule-based detectors vs. DMD drowsiness labels (2026-10-07)

**Setup:** pretrained MediaPipe Face Landmarker (no training) → per-frame EAR / MAR / blendshapes / head pose → rule-based monitor with personal calibration (first 10 s) and the **untuned starting thresholds** in `configs/cabin.yaml`. 16 DMD drowsiness recordings, 13 subjects (non-Indian, acted, parked car). Face camera, frame-level labels. Raw numbers: `dmd_drowsiness_eval.csv`.

| Detector | Result (mean over 16 recordings) |
|---|---|
| Face detected | 100 % of frames |
| Eyes closed (frame level, excl. opening/closing frames) | F1 **0.88** (range 0.68–1.00) |
| Blinks (event level, ±3 frames) | recall **0.97**, precision **0.97** |
| Long closures ≥ 1 s (microsleep proxy) | recall **0.99** |
| Yawns (event level, ±15 frames), mouth only | recall **0.40**, precision **1.00** |
| Yawns, mouth + **hand-over-mouth** cue (update below) | recall **0.88**, precision **0.97** |

**Yawn breakdown:** without hand 29/47 (**62 %**), with hand over the mouth 1/38 (**3 %**). When a hand covers the mouth, the mouth signal disappears (median max jawOpen 0.42 vs 0.65 without hand), so mouth-only rules cannot see these yawns.

**Update (same day): hand-over-mouth cue added.** MediaPipe Hand Landmarker (pretrained, run at 10 fps). During yawns with hand, a hand is visible in 88 % of frames and reaches the mouth (median closest distance 0.03 face widths) for a median 4.3 s; in non-yawn frames a hand is within 0.3 face widths of the mouth in only 1.3 % of frames. Rule: hand within 0.3 face widths of the mouth for ≥ 2 s (0.5 s dropout tolerance) = yawn with hand. Yawn recall **0.40 → 0.88**, precision **0.97**. Caveat: the 0.3 / 2 s values were chosen after looking at all 16 recordings, so this number is optimistic until the subject-wise tuning below. In deployment a hand held at the mouth is ambiguous with eating/drinking; the distraction module (objects) resolves that.

**Next steps:**
1. ~~Hand-at-mouth cue~~ done.
2. Tune thresholds on a **subject-wise split** (tune on some subjects, report on the others). The numbers above use untuned defaults, so they are not inflated by tuning on the test set.
3. Weakest eye-closure recordings (gC-13 0.68, gF-23 2nd session 0.72, gB-7 0.76): inspect for glasses / lighting.

**Caveat:** DMD drowsiness is acted and non-Indian. This validates the *event detectors*, not real fatigue or Indian performance.
