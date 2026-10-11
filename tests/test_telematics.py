import numpy as np
import pandas as pd

from dms.telematics.events import detect_events
from dms.telematics.levin import decode_acc
from dms.telematics.orientation import rotation_to_z, to_vehicle_frame
from dms.utils.config import load_config


def test_decode_acc():
    # one sample (x=+64 -> 1 g, y=-64 -> -1 g, z=0) repeated 25 times
    row = "40c000" * 25
    out = decode_acc(pd.Series([row, "bad"]))
    assert out.shape == (2, 25, 3)
    assert np.allclose(out[0, 0], [1.0, -1.0, 0.0])
    assert np.isnan(out[1]).all()


def test_rotation_to_z():
    for v in ([1, 0, 0], [0, 1, 1], [0, 0, -1], [0.2, -0.3, 0.9]):
        v = np.array(v, dtype=float)
        r = rotation_to_z(v)
        assert np.allclose(r @ (v / np.linalg.norm(v)), [0, 0, 1], atol=1e-9)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)


def test_vehicle_frame_recovers_braking():
    # Device tilted; car brakes at 0.4 g along device x for 3 s while slowing from 40 km/h.
    rate, secs = 25, 60
    t = np.arange(secs * rate) / rate
    lon_true = np.zeros_like(t)
    lon_true[(t >= 20) & (t < 23)] = -0.4
    lon_true[(t >= 40) & (t < 43)] = 0.3
    speed_s = 40 + np.cumsum(np.array([lon_true[i * rate:(i + 1) * rate].mean() for i in range(secs)]) * 9.81 * 3.6)
    tilt = rotation_to_z(np.array([0.3, 0.1, 1.0])).T  # device -> rotated
    acc_vehicle = np.stack([lon_true, np.zeros_like(t), np.ones_like(t)], axis=1)
    acc = acc_vehicle @ tilt.T
    vf = to_vehicle_frame(t, acc, np.arange(secs, dtype=float), {"speed": speed_s})
    assert vf.forward_corr > 0.9
    cfg = load_config("telematics")
    ev = detect_events(t, vf.longitudinal, vf.lateral, np.interp(t, np.arange(secs), speed_s), cfg)
    kinds = [e.type for e in ev]
    assert kinds.count("HARSH_BRAKE") == 1 and kinds.count("HARSH_ACCEL") == 1


def test_speed_derived_braking_features():
    from dms.telematics.features import SpeedSeries, speed_features

    t = np.arange(0, 60, 1.0)
    v = np.where(t < 30, 50.0, np.maximum(50.0 - (t - 30) * 10, 0))  # brake 10 km/h per s (~0.28 g)
    f = speed_features(SpeedSeries(t, v, smooth_s=1), 0, 60)
    assert 0.25 < f["brake_max"] < 0.32
    assert f["frac_brake_gt_0.2"] > 0
    assert f["acc_max"] < 0.05


def test_motion_features_are_orientation_free():
    from dms.telematics.features import MotionSeries, motion_features

    t = np.arange(0, 60, 0.1)
    a = np.zeros((len(t), 3))
    a[:, 0] = 0.15
    rot = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)  # same motion, phone turned
    f1 = motion_features(MotionSeries(t, a), 0, 60)
    f2 = motion_features(MotionSeries(t, a @ rot.T), 0, 60)
    assert np.isclose(f1["dyn_mean"], f2["dyn_mean"]) and np.isclose(f1["dyn_mean"], 0.15)


def test_uah_trip_name_parsing(tmp_path):
    from dms.telematics.uah import load_trip

    d = tmp_path / "D1" / "20151111134545-16km-D1-AGGRESSIVE-SECONDARY"
    d.mkdir(parents=True)
    (d / "RAW_ACCELEROMETERS.txt").write_text("0.1 1 0.01 0.02 0.03 0 0 0 0 0 0\n0.2 1 0.01 0.02 0.03 0 0 0 0 0 0\n")
    (d / "RAW_GPS.txt").write_text("1 50 40 -3 600 4 5 10 0 0 0 0\n2 51 40 -3 600 4 5 10 0 0 0 0\n")
    trip = load_trip(d)
    assert (trip.driver, trip.behaviour, trip.road) == ("D1", "AGGRESSIVE", "SECONDARY")
    assert trip.acc.shape == (2, 4) and trip.gps.shape == (2, 4) and len(trip.lane) == 0
