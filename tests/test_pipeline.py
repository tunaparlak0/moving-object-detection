"""Synthetic moving-camera scene: textured background + camera pan/rotation + one moving object."""
from pathlib import Path

import cv2
import numpy as np

from src.egomotion import estimate_motion
from src.run_pipeline import DEFAULTS, process_video

RNG = np.random.default_rng(0)
BG = cv2.GaussianBlur((RNG.random((1200, 1600)) * 255).astype(np.uint8), (0, 0), 3)
BG = cv2.normalize(BG, None, 0, 255, cv2.NORM_MINMAX)
BG = cv2.cvtColor(BG, cv2.COLOR_GRAY2BGR)
SIZE = (640, 512)  # w, h


def camera(t: int) -> np.ndarray:
    """World -> image affine for frame t (pan + slow rotation)."""
    a = np.deg2rad(0.1 * t)
    c, s = np.cos(a), np.sin(a)
    tx, ty = -200 - 4 * t, -150 - 2 * t
    return np.array([[c, -s, tx], [s, c, ty]], np.float64)


def obj_box(t: int):
    """Object's world position moves independently of the camera."""
    return 500 + 6 * t, 400 + 1 * t, 30, 30


def render(t: int):
    world = BG.copy()
    x, y, w, h = obj_box(t)
    cv2.rectangle(world, (x, y), (x + w, y + h), (255, 255, 255), -1)
    A = camera(t)
    img = cv2.warpAffine(world, A, SIZE, flags=cv2.INTER_LINEAR)
    corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], np.float64)
    pc = corners @ A[:, :2].T + A[:, 2]
    bx, by = pc.min(0)
    bw, bh = pc.max(0) - pc.min(0)
    return img, (bx, by, bw, bh)


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def test_estimate_motion_recovers_camera_transform():
    f0, _ = render(10)
    f1, _ = render(11)
    g0, g1 = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (f0, f1))
    A0, A1 = np.vstack([camera(10), [0, 0, 1]]), np.vstack([camera(11), [0, 0, 1]])
    true_M = A1 @ np.linalg.inv(A0)
    for flow in ("lk", "farneback"):
        for model in ("homography", "affine"):
            est = estimate_motion(g0, g1, flow=flow, model=model)
            assert est.ok
            pts = np.array([[0, 0, 1], [639, 0, 1], [0, 511, 1], [639, 511, 1]], float).T
            p_est = est.M @ pts
            p_true = true_M @ pts
            err = np.abs(p_est[:2] / p_est[2] - p_true[:2] / p_true[2]).max()
            assert err < 1.0, (flow, model, err)


def test_pipeline_detects_moving_object(tmp_path: Path):
    vpath = tmp_path / "syn.mp4"
    wr = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 30, SIZE)
    gts = {}
    for t in range(90):  # 3 s: the adaptive gap (up to 0.5 s) needs neighbours on both sides
        img, box = render(t)
        wr.write(img)
        gts[t + 1] = box
    wr.release()

    for method in ("diff", "mog2"):
        cfg = dict(DEFAULTS, method=method)
        out = tmp_path / method
        process_video(vpath, out, cfg, save_video=False)
        dets = {}
        for line in (out / "detections.txt").read_text().splitlines():
            p = line.split(",")
            dets.setdefault(int(p[0]), []).append(tuple(map(float, p[2:6])))
        frames = [f for f in gts if f in dets]
        hits = sum(any(iou(d, gts[f]) > 0.3 for d in dets[f]) for f in frames)
        false_pos = sum(sum(iou(d, gts[f]) <= 0.3 for d in dets[f]) for f in frames)
        assert len(frames) > 40, method
        assert hits / len(frames) > 0.9, (method, hits, len(frames))
        assert false_pos / len(frames) < 0.2, (method, false_pos)


def test_candidate_gaps_longest_first_and_fixed_by_default():
    from src.run_pipeline import candidate_gaps
    fixed = dict(DEFAULTS, method="diff", gap_s=0.1, gap_max_s=0.0)
    assert candidate_gaps(fixed, 30) == [3]
    adapt = dict(fixed, gap_max_s=0.5)
    assert candidate_gaps(adapt, 30) == [15, 8, 4, 3]       # 0.5, 0.25, 0.125, 0.1 s
    assert candidate_gaps(dict(adapt, method="mog2"), 30) == [3]  # adaptation is diff-only
