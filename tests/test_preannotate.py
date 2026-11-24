"""Tracking + moving filter on synthetic detections (no detector needed)."""
import numpy as np

from src.preannotate import moving_mask, track

FPS = 30


def make_scene(n=90):
    """Camera pans 8 px/frame; car A parked (static in world), car B drives 3 px/frame in world."""
    steps = [np.eye(3)]
    for _ in range(1, n):
        M = np.eye(3); M[0, 2] = -8.0; M[1, 2] = -2.0
        steps.append(M)
    steps = np.array(steps)
    cum = np.zeros_like(steps); cum[0] = np.eye(3)
    for t in range(1, n):
        cum[t] = steps[t] @ cum[t - 1]
    dets = []
    rng = np.random.default_rng(1)
    for t in range(n):
        D = []
        for wx, wy, vx in ((900, 400, 0.0), (300, 500, 3.0)):     # world position at t=0, world speed
            p = cum[t] @ np.array([wx + vx * t, wy, 1.0])
            x, y = p[:2] / p[2] + rng.normal(0, 0.7, 2)              # detector jitter
            D.append([x - 20, y - 10, x + 20, y + 10, 0.9, 1])
        dets.append(np.array(D, np.float32))
    return dets, steps, cum


def test_tracks_survive_camera_motion_and_static_cars_are_filtered():
    dets, steps, cum = make_scene()
    tracks = track(dets, steps)
    assert len(tracks) == 2, [len(t.frames) for t in tracks]
    assert all(len(t.frames) == 90 for t in tracks)
    by_x0 = sorted(tracks, key=lambda t: t.frames[0][1][0])
    moving_car, parked_car = by_x0[0], by_x0[1]
    m_moving = moving_mask(moving_car, cum, FPS, 0.5, 0.25, 0.5)
    m_parked = moving_mask(parked_car, cum, FPS, 0.5, 0.25, 0.5)
    assert m_moving.mean() > 0.9
    assert m_parked.mean() == 0.0


def test_cvat_xml_keyframes_bridge_short_gaps():
    import xml.etree.ElementTree as ET
    from src.preannotate import cvat_video_xml
    # track 7: frames 1-13, short gap (14-15 missing), 16-25, long gap, 60-62
    frames = list(range(1, 14)) + list(range(16, 26)) + [60, 61, 62]
    rows = [(f, 7, float(f), 2.0, 3.0, 4.0, 1) for f in frames]
    root = ET.fromstring(cvat_video_xml(rows, n_frames=100, key_step=6, max_gap=5))
    tracks = root.findall("track")
    assert len(tracks) == 1 and tracks[0].get("label") == "vehicle"
    boxes = [(int(b.get("frame")), int(b.get("outside"))) for b in tracks[0].findall("box")]
    # 0-based frames: grid 0,6,12 / first frame after short gap 15 / grid 18 / last 24 / outside 25
    # second segment: 59, 61 (last), outside 62
    assert boxes == [(0, 0), (6, 0), (12, 0), (15, 0), (18, 0), (24, 0), (25, 1),
                     (59, 0), (61, 0), (62, 1)]
