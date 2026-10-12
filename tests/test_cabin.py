import math

import numpy as np

from dms.baseline.calibration import Baseline
from dms.cabin.fatigue import CabinMonitor
from dms.cabin.features import eye_aspect_ratio, head_pose_deg
from dms.eval.metrics import event_prf, frame_prf
from dms.utils.config import load_config


def test_ear_open_vs_closed():
    open_eye = np.array([[0, 0], [1, 1], [2, 1], [3, 0], [2, -1], [1, -1]], dtype=float)
    closed_eye = open_eye * np.array([1, 0.1])
    assert math.isclose(eye_aspect_ratio(open_eye), 2 / 3)
    assert eye_aspect_ratio(closed_eye) < 0.1


def test_head_pose_identity_is_zero():
    assert head_pose_deg(np.eye(4)) == (0.0, 0.0, 0.0)


def test_head_pose_yaw():
    a = math.radians(30)
    m = np.eye(4)
    m[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    yaw, pitch, roll = head_pose_deg(m)
    assert math.isclose(yaw, 30, abs_tol=1e-6) and abs(pitch) < 1e-6 and abs(roll) < 1e-6


def _feats(ear=0.3, mar=0.05, yaw=0.0, pitch=0.0):
    return {"ear": ear, "mar": mar, "blink_score": 0.0, "jaw_open": 0.0, "yaw": yaw, "pitch": pitch, "roll": 0.0}


def _run(monitor, frames, fps=30):
    events = []
    for i, f in enumerate(frames):
        events += monitor.update(i / fps, f)
    return events


def test_blink_microsleep_and_yawn_events():
    cfg = load_config("cabin")
    mon = CabinMonitor(cfg, Baseline(ear_open=0.3, mar_rest=0.05, yaw=0.0, pitch=0.0, n_frames=100))
    frames = (
        [_feats()] * 30
        + [_feats(ear=0.05)] * 6  # 0.2 s closure -> blink
        + [_feats()] * 30
        + [_feats(ear=0.05)] * 45  # 1.5 s closure -> microsleep
        + [_feats()] * 30
        + [_feats(mar=0.8)] * 60  # 2 s wide-open mouth -> yawn
        + [_feats()] * 40  # a yawn is confirmed once the mouth stays closed longer than yawn_gap_s
    )
    kinds = [e.type for e in _run(mon, frames)]
    assert kinds.count("BLINK") == 1
    assert kinds.count("MICROSLEEP") == 1
    assert kinds.count("YAWN") == 1


def test_eyes_off_road_needs_minimum_duration():
    cfg = load_config("cabin")
    mon = CabinMonitor(cfg, Baseline(0.3, 0.05, 0.0, 0.0, 100))
    short = [_feats(yaw=45)] * 30 + [_feats()] * 5  # 1 s: too short
    long = [_feats(yaw=45)] * 90 + [_feats()] * 5  # 3 s: counts
    kinds = [e.type for e in _run(mon, short + long)]
    assert kinds.count("EYES_OFF_ROAD") == 1


def test_metrics():
    pred = np.array([1, 1, 0, 0], dtype=bool)
    gt = np.array([1, 0, 1, 0], dtype=bool)
    m = frame_prf(pred, gt)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    e = event_prf([(0, 5), (50, 55)], [(3, 8), (20, 25)])
    assert e["recall"] == 0.5 and e["precision"] == 0.5


def test_roc_auc():
    from dms.eval.metrics import roc_auc

    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1], dtype=bool)) == 1.0
    assert roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1], dtype=bool)) == 0.5


def test_calibration_ignores_turned_head_and_glitches():
    from dms.baseline.calibration import Calibrator

    cal = Calibrator(seconds=10)
    for i in range(300):
        t = i / 30
        if i < 60:  # settling in: head turned 40 deg, eyes look wider
            cal.add(t, _feats(ear=0.5, yaw=40.0))
        elif i % 50 == 0:  # tracking glitch
            cal.add(t, _feats(ear=0.9))
        else:
            cal.add(t, _feats(ear=0.30 + 0.01 * (i % 3), yaw=2.0, pitch=-10.0))
    base = cal.result()
    assert 0.29 <= base.ear_open <= 0.33
    assert abs(base.yaw - 2.0) < 1 and abs(base.pitch + 10.0) < 1
    assert base.n_used < base.n_frames


def test_india_profile_overlays_defaults():
    base, india = load_config("cabin"), load_config("cabin", profile="india")
    assert "profiles" not in base
    assert india["eyes"]["closed_ear_ratio"] == 0.8 and base["eyes"]["closed_ear_ratio"] == 0.65
    assert india["mouth"] == base["mouth"]


def test_yawn_with_flickering_mouth_is_one_yawn_and_head_tilt_back_is_not_a_nod():
    cfg = load_config("cabin")
    mon = CabinMonitor(cfg, Baseline(ear_open=0.3, mar_rest=0.05, yaw=0.0, pitch=0.0, n_frames=100))
    yawn = []
    for _ in range(4):  # mouth reading flickers: 0.5 s open, 0.3 s dip, head tilted back 20 deg
        yawn += [_feats(mar=0.8, pitch=-20.0)] * 15 + [_feats(mar=0.3, pitch=-20.0)] * 9
    frames = [_feats()] * 30 + yawn + [_feats()] * 60
    kinds = [e.type for e in _run(mon, frames)]
    assert kinds.count("YAWN") == 1
    assert kinds.count("NOD") == 0
    nod = [_feats()] * 30 + [_feats(pitch=20.0)] * 15 + [_feats()] * 30  # head drops 0.5 s and recovers
    assert [e.type for e in _run(CabinMonitor(cfg, mon.base), nod)].count("NOD") == 1
