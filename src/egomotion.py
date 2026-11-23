"""Camera ego-motion estimation between two frames.

Estimates a global transform M that maps pixel coordinates of `prev` onto `curr`
(so warping `prev` with M aligns its background with `curr`).

Flow methods:
  - "lk":        sparse Lucas-Kanade on Shi-Tomasi corners
  - "farneback": dense Farneback flow, sampled on a regular grid
Motion models:
  - "homography": 8-DoF projective (good for planar / distant scenes, e.g. UAV)
  - "affine":     6-DoF affine
RANSAC rejects correspondences on independently moving objects (outliers).
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionEstimate:
    M: np.ndarray            # 3x3 transform prev -> curr
    inlier_ratio: float      # fraction of correspondences consistent with M
    n_points: int            # correspondences used
    ok: bool                 # False -> estimation failed, M is identity


def _lk_correspondences(prev: np.ndarray, curr: np.ndarray, max_corners: int = 1500):
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=max_corners, qualityLevel=0.01,
                                  minDistance=8, blockSize=7)
    if pts is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, curr, pts, None, winSize=(21, 21), maxLevel=3,
                                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    # forward-backward check removes unreliable tracks
    back, st2, _ = cv2.calcOpticalFlowPyrLK(curr, prev, nxt, None, winSize=(21, 21), maxLevel=3)
    fb_err = np.linalg.norm(pts - back, axis=2).ravel()
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb_err < 1.0)
    return pts.reshape(-1, 2)[good], nxt.reshape(-1, 2)[good]


def dense_flow(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Dense Farneback flow (H x W x 2), flow[y, x] = displacement of prev pixel."""
    return cv2.calcOpticalFlowFarneback(prev, curr, None, pyr_scale=0.5, levels=4, winsize=21,
                                        iterations=3, poly_n=7, poly_sigma=1.5, flags=0)


def _farneback_correspondences(prev: np.ndarray, curr: np.ndarray, step: int = 16):
    flow = dense_flow(prev, curr)
    h, w = prev.shape
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    src = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    dst = src + flow[ys.ravel(), xs.ravel()]
    return src, dst


def estimate_motion(prev: np.ndarray, curr: np.ndarray, flow: str = "lk",
                    model: str = "homography", ransac_thresh: float = 2.0) -> MotionEstimate:
    """prev/curr: grayscale uint8 frames of identical size.
    model="none" is the no-compensation baseline: the camera is assumed static (identity)."""
    if model == "none":
        return MotionEstimate(np.eye(3), 1.0, 0, True)
    if flow == "lk":
        src, dst = _lk_correspondences(prev, curr)
    elif flow == "farneback":
        src, dst = _farneback_correspondences(prev, curr)
    else:
        raise ValueError(f"unknown flow method: {flow}")

    identity = MotionEstimate(np.eye(3), 0.0, len(src), False)
    if len(src) < 8:
        return identity

    if model == "homography":
        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh, maxIters=2000, confidence=0.995)
    elif model == "affine":
        A, mask = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh,
                                       maxIters=2000, confidence=0.995)
        M = None if A is None else np.vstack([A, [0, 0, 1]])
    else:
        raise ValueError(f"unknown motion model: {model}")

    if M is None or mask is None:
        return identity
    return MotionEstimate(M, float(mask.mean()), len(src), True)


def warp_to(frame: np.ndarray, M: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Warp `frame` with M into an image of `size` (w, h). Also returns a validity mask
    (255 where warped pixels come from inside the source frame)."""
    warped = cv2.warpPerspective(frame, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.warpPerspective(np.full(frame.shape[:2], 255, np.uint8), M, size,
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return warped, valid
