"""Foreground extraction on ego-motion compensated frames, and blob -> box conversion.

Both methods receive past frames already warped into the current frame's coordinates,
so static background lines up and only independently moving objects leave residue.

  - compensated_diff: three-frame differencing |I_t - W(I_{t-k})| AND |I_t - W(I_{t+k})|.
    The AND keeps only the object's current position (ghosts at t-k and t+k cancel out).
  - compensated_mog2: a MOG2 model fitted on a window of warped neighbour frames,
    then applied (without learning) to the current frame.
"""
import cv2
import numpy as np


def _diff_mask(curr: np.ndarray, past: np.ndarray, thresh: int) -> np.ndarray:
    d = cv2.absdiff(curr, past)
    _, m = cv2.threshold(d, thresh, 255, cv2.THRESH_BINARY)
    return m


def compensated_diff(curr: np.ndarray, warped_pasts: list[np.ndarray], valid: np.ndarray,
                     thresh: int = 25) -> np.ndarray:
    """curr: blurred gray current frame; warped_pasts: 1 or 2 blurred gray warped frames
    (e.g. t-k and t+k); valid: mask of pixels covered by all warped frames."""
    mask = _diff_mask(curr, warped_pasts[0], thresh)
    for p in warped_pasts[1:]:
        mask = cv2.bitwise_and(mask, _diff_mask(curr, p, thresh))
    return cv2.bitwise_and(mask, valid)


def compensated_mog2(curr: np.ndarray, warped_window: list[np.ndarray], valid: np.ndarray,
                     var_threshold: float = 32.0, bg_ratio: float = 0.7) -> np.ndarray:
    """warped_window: neighbour frames warped to the current frame.
    bg_ratio < 0.9 (OpenCV default) because in a short window a slow object covers
    some pixels in several frames; the dominant mode must still count as background."""
    mog = cv2.createBackgroundSubtractorMOG2(history=len(warped_window), varThreshold=var_threshold,
                                             detectShadows=False)
    mog.setBackgroundRatio(bg_ratio)
    for f in warped_window:
        mog.apply(f)
    mask = mog.apply(curr, learningRate=0)
    return cv2.bitwise_and(mask, valid)


def clean_mask(mask: np.ndarray, open_k: int = 3, close_k: int = 9) -> np.ndarray:
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)


def mask_to_boxes(mask: np.ndarray, min_area: int = 30, max_area_frac: float = 0.05):
    """Connected components -> list of (x, y, w, h) boxes, filtered by area."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    max_area = max_area_frac * mask.shape[0] * mask.shape[1]
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if min_area <= area <= max_area:
            boxes.append((int(x), int(y), int(w), int(h)))
    return boxes
