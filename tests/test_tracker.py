import numpy as np

from src.tracker import Tracker

RNG = np.random.default_rng(1)


def shift(dx, dy):
    M = np.eye(3)
    M[0, 2], M[1, 2] = dx, dy
    return M


def test_moving_object_kept_noise_dropped():
    tr = Tracker(min_hits=5, min_disp=6)
    for f in range(1, 41):
        boxes = []
        if f not in (15, 16):                       # short dropout -> gap must be filled
            boxes.append((100 + 3 * f, 200, 20, 20))
        boxes.append((300 + RNG.uniform(-1, 1), 50, 15, 15))      # static flicker blob
        boxes.append((RNG.uniform(0, 600), RNG.uniform(300, 400), 10, 10))  # random 1-frame noise
        tr.update(f, boxes, None)
    res = tr.results()
    ids = {tid for _, tid, _ in res}
    assert len(ids) == 1
    frames = sorted(f for f, _, _ in res)
    assert frames == list(range(1, 41))             # includes the filled gap and pre-confirmation frames
    f15 = next(b for f, _, b in res if f == 15)
    assert abs(f15[0] - (100 + 3 * 15)) < 6


def test_camera_motion_alone_is_not_object_motion():
    """A static world object slides across the image only because the camera pans."""
    tr = Tracker(min_hits=5, min_disp=6)
    for f in range(1, 41):
        tr.update(f, [(50 + 8 * f, 100, 20, 20)], shift(8, 0) if f > 1 else None)
    assert tr.results() == []


def test_mover_under_camera_motion():
    tr = Tracker(min_hits=5, min_disp=6)
    for f in range(1, 41):
        # camera pans +8 px/frame, object also moves +2 px/frame in the world
        tr.update(f, [(50 + 10 * f, 100, 20, 20)], shift(8, 0) if f > 1 else None)
    assert len({tid for _, tid, _ in tr.results()}) == 1
