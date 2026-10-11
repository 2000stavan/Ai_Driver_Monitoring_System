import numpy as np

from dms.events import Event
from dms.fusion.attribution import SpeedTrack, attribute, summary


def ctx(attributed, kind):
    return [a.context for a in attributed if a.event.type == kind]


def test_brake_contexts():
    events = [
        Event("EYES_OFF_ROAD", 8.5, 10.5),
        Event("HARSH_BRAKE", 10.0, 11.0),  # distracted just before -> likely distraction
        Event("HAZARD_AHEAD", 29.0, 31.0),
        Event("HARSH_BRAKE", 30.0, 31.0),  # hazard ahead, attentive -> hazard response
        Event("HARSH_BRAKE", 50.0, 51.0),  # nothing around -> unexplained
    ]
    a = attribute(events)
    assert ctx(a, "HARSH_BRAKE") == ["likely_distraction", "likely_hazard_response", "unexplained_harsh_brake"]


def test_lane_change_mirror_check():
    events = [Event("MIRROR_CHECK", 18.0, 19.0), Event("LANE_CHANGE", 20.0, 23.0), Event("LANE_CHANGE", 60.0, 63.0)]
    assert ctx(attribute(events), "LANE_CHANGE") == ["lane_change_with_mirror_check", "no_mirror_check"]
    assert ctx(attribute(events, mirror_check_available=False), "LANE_CHANGE") == ["mirror_check_unknown"] * 2


def test_stationary_suppression_and_phone_rule():
    speed = SpeedTrack(np.array([0, 10, 20, 30]), np.array([0, 0, 40, 40]))
    events = [Event("EYES_OFF_ROAD", 2.0, 5.0), Event("PHONE_USE", 25.0, 28.0)]
    a = attribute(events, speed)
    assert ctx(a, "EYES_OFF_ROAD") == ["suppressed_stationary"]
    assert ctx(a, "PHONE_USE") == ["phone_while_moving"]
    assert summary(a) == {"phone_while_moving": 1}


def test_critical_fatigue():
    a = attribute([Event("MICROSLEEP", 10.0, 11.5), Event("LANE_DEPARTURE", 12.0, 13.0)])
    assert ctx(a, "MICROSLEEP") == ["critical_fatigue"]
