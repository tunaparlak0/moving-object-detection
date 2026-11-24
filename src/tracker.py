"""Class-agnostic multi-object tracker for ego-motion compensated foreground blobs.

Every foreground box is a candidate; the tracker keeps only blobs that behave like real
moving objects, which removes most single-frame noise (compression, parallax, warp errors).

Per frame t (boxes in processing-resolution coordinates of frame t):
  1. predict: tracks are carried from frame t-1 to t with the camera step transform M_t
     (so camera motion does not look like object motion), then a constant-velocity
     Kalman step adds the object's own motion.
  2. associate: Hungarian on IoU between predicted track boxes and new boxes
     (a centre-distance gate rescues small fast objects whose boxes barely overlap).
  3. update matched tracks, start a track for each unmatched box, age the rest.

Offline decision (the whole video is processed before output is written):
  a track is reported only if it was observed in >= min_hits frames and its centre moved
  >= min_disp px in the ground (camera-compensated) frame. Its boxes are reported for every
  observed frame, including the frames before it was confirmed, and short gaps
  (<= max_gap frames) are filled with the Kalman prediction.

Nothing here knows what a car or a person is: any blob that keeps moving is kept.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

# Kalman state: [cx, cy, w, h, vx, vy]  (vx, vy in px/frame in the current frame's coordinates)
_F = np.eye(6)
_F[0, 4] = _F[1, 5] = 1.0
_Hm = np.eye(4, 6)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: N x 4, b: M x 4 boxes as x, y, w, h."""
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter + 1e-9)


def _apply_h(M: np.ndarray, pt: np.ndarray) -> np.ndarray:
    v = M @ np.array([pt[0], pt[1], 1.0])
    return v[:2] / v[2]


@dataclass
class _Track:
    tid: int
    x: np.ndarray                      # Kalman state (6,)
    P: np.ndarray                      # covariance (6, 6)
    origin: np.ndarray                 # first observed centre, kept in the current frame's coordinates
    obs: dict = field(default_factory=dict)    # frame -> observed box (x, y, w, h)
    pred: dict = field(default_factory=dict)   # frame -> predicted box while unobserved
    miss: int = 0
    max_disp: float = 0.0

    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, w, h])


class Tracker:
    def __init__(self, iou_thr: float = 0.1, gate: float = 1.5, max_gap: int = 5,
                 min_hits: int = 5, min_disp: float = 6.0, q: float = 1.0, r: float = 4.0):
        """iou_thr: min IoU to associate; gate: else centre distance <= gate * box diagonal.
        max_gap: frames a track may go unobserved; min_hits / min_disp: confirmation rule."""
        self.iou_thr, self.gate, self.max_gap = iou_thr, gate, max_gap
        self.min_hits, self.min_disp = min_hits, min_disp
        self.Q = np.diag([q, q, q, q, q / 2, q / 2])
        self.R = np.diag([r, r, r, r])
        self.active: list[_Track] = []
        self.done: list[_Track] = []
        self._next = 1

    # ------------------------------------------------------------------ per frame
    def update(self, frame: int, boxes: list, M: np.ndarray | None = None) -> None:
        """boxes: [(x, y, w, h), ...] in frame `frame`; M: 3x3 transform frame-1 -> frame."""
        for tr in self.active:
            self._predict(tr, M)
        dets = np.array(boxes, float).reshape(-1, 4)
        matched_t, matched_d = self._associate(dets)
        for ti, di in zip(matched_t, matched_d):
            tr = self.active[ti]
            self._correct(tr, dets[di])
            tr.obs[frame] = tuple(dets[di])
            tr.miss = 0
            tr.max_disp = max(tr.max_disp, float(np.hypot(*(tr.x[:2] - tr.origin))))
        for ti, tr in enumerate(self.active):
            if ti not in matched_t:
                tr.miss += 1
                tr.pred[frame] = tuple(tr.box())
        for di in set(range(len(dets))) - set(matched_d):
            self.active.append(self._new(frame, dets[di]))
        keep = []
        for tr in self.active:
            (keep if tr.miss <= self.max_gap else self.done).append(tr)
        self.active = keep

    def _predict(self, tr: _Track, M: np.ndarray | None) -> None:
        if M is not None:   # camera motion: move the track into the new frame's coordinates
            A = M[:2, :2]
            s = float(np.sqrt(abs(np.linalg.det(A))))
            tr.x[:2] = _apply_h(M, tr.x[:2])
            tr.x[2:4] *= s
            tr.x[4:6] = A @ tr.x[4:6]
            tr.origin = _apply_h(M, tr.origin)
        tr.x = _F @ tr.x
        tr.P = _F @ tr.P @ _F.T + self.Q
        tr.x[2:4] = np.maximum(tr.x[2:4], 1.0)

    def _correct(self, tr: _Track, d: np.ndarray) -> None:
        z = np.array([d[0] + d[2] / 2, d[1] + d[3] / 2, d[2], d[3]])
        S = _Hm @ tr.P @ _Hm.T + self.R
        K = tr.P @ _Hm.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ (z - _Hm @ tr.x)
        tr.P = (np.eye(6) - K @ _Hm) @ tr.P

    def _new(self, frame: int, d: np.ndarray) -> _Track:
        c = np.array([d[0] + d[2] / 2, d[1] + d[3] / 2])
        tr = _Track(self._next, np.array([c[0], c[1], d[2], d[3], 0.0, 0.0]),
                    np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0]), c.copy())
        tr.obs[frame] = tuple(d)
        self._next += 1
        return tr

    def _associate(self, dets: np.ndarray) -> tuple[list, list]:
        if not self.active or not len(dets):
            return [], []
        tb = np.array([t.box() for t in self.active])
        iou = _iou(tb, dets)
        tc = tb[:, :2] + tb[:, 2:] / 2
        dc = dets[:, :2] + dets[:, 2:] / 2
        dist = np.linalg.norm(tc[:, None] - dc[None], axis=2)
        diag = np.hypot(tb[:, 2], tb[:, 3])[:, None]
        ok = (iou >= self.iou_thr) | (dist <= self.gate * diag)
        # prefer IoU, break ties / no-overlap cases by normalised distance
        cost = np.where(ok, (1.0 - iou) + 0.1 * dist / (diag + 1e-9), 1e6)
        rows, cols = linear_sum_assignment(cost)
        keep = ok[rows, cols]
        return list(rows[keep]), list(cols[keep])

    # ------------------------------------------------------------------ output
    def confirmed(self) -> list[_Track]:
        return [t for t in self.done + self.active
                if len(t.obs) >= self.min_hits and t.max_disp >= self.min_disp]

    def results(self) -> list[tuple[int, int, tuple]]:
        """[(frame, track_id, (x, y, w, h)), ...] for confirmed tracks, gaps filled."""
        out = []
        for tr in self.confirmed():
            first, last = min(tr.obs), max(tr.obs)
            for f in range(first, last + 1):
                if f in tr.obs:
                    out.append((f, tr.tid, tr.obs[f]))
                elif f in tr.pred:
                    out.append((f, tr.tid, tr.pred[f]))
        return sorted(out)
