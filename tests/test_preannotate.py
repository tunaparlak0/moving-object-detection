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
