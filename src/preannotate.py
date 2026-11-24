"""Semi-automatic ground-truth pre-annotation for CVAT.

Pipeline (independent of our foreground-based detector, to avoid evaluation bias):
  1. Detection: UAVDT-trained YOLO (car/truck/bus from aerial views) + COCO YOLO
     (person/bicycle/motorbike).
  2. Tracking: IoU + Hungarian matching, with track boxes warped by the estimated
     camera motion between frames (ego-motion compensated tracking).
  3. Moving filter: each track's centre is mapped into a common coordinate frame with
     the chained homographies; a track is "moving" in frames where its ground-plane
     speed exceeds a threshold (relative to its own size). Parked cars are dropped.
  4. Export for CVAT:
       vid_XXX_cvat.zip  'CVAT 1.1' (for video) XML with sparse keyframes -> import this one;
                         CVAT interpolates between keyframes, reviewers fix only keyframes
       vid_XXX.txt       dense MOT rows (used by `src.evaluate --drafts`)
     plus, with --preview, a video (green = kept as moving, grey = filtered as static).

The result is a DRAFT: always review and correct it in CVAT.

Usage:
  python -m src.preannotate vid_002
  python -m src.preannotate all
  python -m src.preannotate vid_002 --speed-thresh 0.3   # re-filter using cached detections
"""
import argparse
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.egomotion import estimate_motion
from src.video_io import make_writer

# Detector classes (kept fine-grained in the detection cache) ...
RAW_CLASSES = ["person", "car", "truck", "bus", "bicycle", "motorbike"]
UAVDT_MAP = {"car": "car", "truck": "truck", "bus": "bus"}
COCO_MAP = {"person": "person", "bicycle": "bicycle", "motorcycle": "motorbike"}
# ... are merged into coarse annotation labels: the task is moving vs. static, the
# class only matters for per-type analysis (e.g. recall on small two-wheelers).
LABELS = ["vehicle", "two_wheeler", "person"]
RAW_TO_LABEL = {"person": "person", "car": "vehicle", "truck": "vehicle", "bus": "vehicle",
                "bicycle": "two_wheeler", "motorbike": "two_wheeler"}


def label_of(raw_idx: int) -> int:
    return LABELS.index(RAW_TO_LABEL[RAW_CLASSES[raw_idx]])
MODELS_DIR = Path("models")


# ----------------------------------------------------------------------------- detection
def load_models():
    from ultralytics import YOLO
    uavdt = YOLO(str(MODELS_DIR / "uavdt_yolov9s.pt"))
    coco = YOLO(str(MODELS_DIR / "yolo11s.pt"))
    return uavdt, coco


def detect_frame(models, frame, conf: float, imgsz: int):
    """Return array N x 6: x1, y1, x2, y2, score, label_idx."""
    out = []
    for model, mapping in zip(models, (UAVDT_MAP, COCO_MAP)):
        r = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
        names = r.names
        for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                          r.boxes.cls.cpu().numpy().astype(int)):
            lab = mapping.get(names[c])
            if lab is not None:
                out.append([x1, y1, x2, y2, s, RAW_CLASSES.index(lab)])
    return np.array(out, np.float32).reshape(-1, 6)


def run_detection_and_motion(vpath: Path, conf: float, imgsz: int, proc_max_side: int = 960):
    """Detections per frame + step homographies M_t (frame t-1 -> t) in full resolution."""
    models = load_models()
    cap = cv2.VideoCapture(str(vpath))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = min(1.0, proc_max_side / max(W, H))
    S, S_inv = np.diag([s, s, 1.0]), np.diag([1 / s, 1 / s, 1.0])
    dets, steps, prev = [], [np.eye(3)], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            steps.append(S_inv @ estimate_motion(prev, g).M @ S)
        prev = g
        dets.append(detect_frame(models, frame, conf, imgsz))
        if len(dets) % 100 == 0:
            print(f"  {len(dets)} frames", flush=True)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return dets, np.array(steps), fps, (W, H)


# ----------------------------------------------------------------------------- tracking
def warp_boxes(boxes: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Move boxes (N x 4 xyxy) by the transform of their centres; size kept."""
    if len(boxes) == 0:
        return boxes
    c = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2, np.ones(len(boxes))])
    p = M @ c
    p = p[:2] / p[2]
    d = (p - c[:2]).T
    return boxes + np.hstack([d, d])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


@dataclass
class Track:
    tid: int
    label: int
    box: np.ndarray                     # last (predicted or observed) xyxy
    missed: int = 0
    frames: list = field(default_factory=list)   # (frame_idx, xyxy, observed: bool)


def track(dets: list, steps: np.ndarray, iou_thr=0.3, high=0.4, max_age=15):
    tracks, finished, next_id = [], [], 1
    for t, D in enumerate(dets):
        for tr in tracks:                         # predict with camera motion
            tr.box = warp_boxes(tr.box[None], steps[t])[0]
        unmatched_d = set(range(len(D)))
        if tracks and len(D):
            iou = iou_matrix(np.array([tr.box for tr in tracks]), D[:, :4])
            for i, j in zip(*linear_sum_assignment(-iou)):
                if iou[i, j] >= iou_thr:
                    tr = tracks[i]
                    tr.box, tr.missed = D[j, :4].copy(), 0
                    tr.frames.append((t, tr.box.copy(), True))
                    unmatched_d.discard(j)
        for tr in tracks:
            if not tr.frames or tr.frames[-1][0] != t:
                tr.missed += 1
        alive = []
        for tr in tracks:
            (alive if tr.missed <= max_age else finished).append(tr)
        tracks = alive
        for j in unmatched_d:
            if D[j, 4] >= high:
                tr = Track(next_id, int(D[j, 5]), D[j, :4].copy())
                tr.frames.append((t, tr.box.copy(), True))
                tracks.append(tr)
                next_id += 1
    return finished + tracks


# ----------------------------------------------------------------------------- moving filter
def moving_mask(tr: Track, cum: np.ndarray, fps: float, win_s: float, speed_thresh: float,
                min_run_s: float) -> np.ndarray:
    """Per observed frame of the track: True if the object moves w.r.t. the ground.
    cum[t] maps frame-0 coordinates to frame t (chained homographies)."""
    f = np.array([fr for fr, _, _ in tr.frames])
    b = np.array([bx for _, bx, _ in tr.frames])
    centres = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, np.ones(len(b))])
    # map every observation into frame-0 ("world") coordinates
    world = np.stack([np.linalg.inv(cum[fi]) @ centres[:, k] for k, fi in enumerate(f)])
    world = world[:, :2] / world[:, 2:3]
    diag = np.median(np.hypot(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]))
    w = max(1, int(round(win_s * fps)))
    moving = np.zeros(len(f), bool)
    for k in range(len(f)):
        lo, hi = np.searchsorted(f, f[k] - w), np.searchsorted(f, f[k] + w, side="right") - 1
        if hi <= lo:
            continue
        # express both endpoints in the current frame's coordinates to get pixel speed there
        pts = np.array([[*world[lo], 1.0], [*world[hi], 1.0]]).T
        q = cum[f[k]] @ pts
        q = q[:2] / q[2]
        speed = np.linalg.norm(q[:, 1] - q[:, 0]) / ((f[hi] - f[lo]) / fps)   # px/s
        moving[k] = speed / diag > speed_thresh                                  # box-diagonals / s
    # remove short flickers: keep only moving runs >= min_run_s
    min_run = int(round(min_run_s * fps))
    out = moving.copy()
    k = 0
    while k < len(moving):
        if moving[k]:
            e = k
            while e + 1 < len(moving) and moving[e + 1]:
                e += 1
            if f[e] - f[k] + 1 < min_run:
                out[k:e + 1] = False
            k = e + 1
        else:
            k += 1
    return out


# ----------------------------------------------------------------------------- export
def cvat_video_xml(rows: list, n_frames: int, key_step: int, max_gap: int) -> str:
    """Dense rows (frame 1-based, id, x, y, w, h, cls 1-based) -> 'CVAT for video 1.1' XML.

    One CVAT track per id. Gaps of <= max_gap frames are bridged (CVAT interpolates them);
    a longer gap ends a segment with an outside keyframe and the track reappears later.
    Inside a segment only the first, the last and every key_step-th frame are keyframes, so
    reviewers fix few boxes and CVAT interpolates the rest. (MOT import cannot do this: CVAT
    treats frames missing from a MOT file as 'object absent'.)"""
    by_id = {}
    for r in rows:
        by_id.setdefault(r[1], []).append(r)
    out = ["<?xml version='1.0' encoding='utf-8'?>", "<annotations>", "  <version>1.1</version>"]
    for new_id, tid in enumerate(sorted(by_id)):
        rs = sorted(by_id[tid])
        segs, cur = [], [rs[0]]
        for r in rs[1:]:
            if r[0] - cur[-1][0] <= max_gap + 1:
                cur.append(r)
            else:
                segs.append(cur)
                cur = [r]
        segs.append(cur)
        out.append(f'  <track id="{new_id}" label="{LABELS[rs[0][6] - 1]}" source="file">')
        for seg in segs:
            f0, f1 = seg[0][0], seg[-1][0]
            keys = [r for r in seg if (r[0] - f0) % key_step == 0 or r[0] == f1]
            if key_step > 1:   # a gap may hide the grid frames: keep at least one box per key_step
                keys = sorted({*keys, *[r for i, r in enumerate(seg[1:], 1)
                                        if r[0] - seg[i - 1][0] > 1]})
            for r in keys:
                out.append(_xml_box(r[0] - 1, r[2:6], outside=0))
            if f1 < n_frames:   # end the segment (frame f1 + 1 in 1-based == f1 in 0-based)
                out.append(_xml_box(f1, seg[-1][2:6], outside=1))
        out.append("  </track>")
    out.append("</annotations>")
    return "\n".join(out) + "\n"


def _xml_box(frame0: int, xywh, outside: int) -> str:
    x, y, w, h = xywh
    return (f'    <box frame="{frame0}" keyframe="1" outside="{outside}" occluded="0" '
            f'xtl="{x:.2f}" ytl="{y:.2f}" xbr="{x + w:.2f}" ybr="{y + h:.2f}" z_order="0"></box>')


def export(vpath: Path, tracks, keep: dict, size, fps, out_dir: Path, preview: bool, key_step: int = 1):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for tr in tracks:
        for (fi, bx, _), m in zip(tr.frames, keep[tr.tid]):
            if m:
                x1, y1, x2, y2 = np.clip(bx, 0, [size[0], size[1], size[0], size[1]])
                if x2 - x1 > 2 and y2 - y1 > 2:
                    rows.append((fi + 1, tr.tid, x1, y1, x2 - x1, y2 - y1, label_of(tr.label) + 1))
    rows.sort()
    fmt = "{},{},{:.1f},{:.1f},{:.1f},{:.1f},1,{},1.0\n"
    gt = "".join(fmt.format(*r) for r in rows)
    (out_dir / f"{vpath.stem}.txt").write_text(gt)
    cap = cv2.VideoCapture(str(vpath))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    with zipfile.ZipFile(out_dir / f"{vpath.stem}_cvat.zip", "w") as z:
        z.writestr("annotations.xml", cvat_video_xml(rows, n_frames, key_step, max_gap=max(1, round(0.5 * fps))))

    if preview:
        by_frame = {}
        for tr in tracks:
            for (fi, bx, _), m in zip(tr.frames, keep[tr.tid]):
                by_frame.setdefault(fi, []).append((bx, m, tr.tid, tr.label))
        cap = cv2.VideoCapture(str(vpath))
        s = min(1.0, 960 / max(size))
        pw, ph = int(size[0] * s), int(size[1] * s)
        wr = make_writer(out_dir / f"{vpath.stem}_preview.mp4", fps, (pw, ph))
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
            for bx, m, tid, lab in by_frame.get(fi, []):
                x1, y1, x2, y2 = (bx * s).astype(int)
                col = (0, 220, 0) if m else (150, 150, 150)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2 if m else 1)
                if m:
                    cv2.putText(frame, f"{tid}:{LABELS[label_of(lab)]}", (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
            cv2.putText(frame, str(fi + 1), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            wr.write(frame)
            fi += 1
        cap.release()
        wr.release()
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("--videos", default="data/videos")
    ap.add_argument("--out", default="results/preannot")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--speed-thresh", type=float, default=0.25, help="box diagonals per second")
    ap.add_argument("--win-s", type=float, default=0.5)
    ap.add_argument("--min-run-s", type=float, default=0.5)
    ap.add_argument("--min-track-s", type=float, default=0.5)
    ap.add_argument("--redetect", action="store_true", help="ignore cached detections")
    ap.add_argument("--preview", action="store_true", help="also write vid_XXX_preview.mp4")
    ap.add_argument("--key-s", type=float, default=0.2,
                    help="keyframe spacing (s) in the CVAT-XML zip; 0 = every frame. The .txt stays dense.")
    args = ap.parse_args()

    vdir = Path(args.videos)
    vids = sorted(vdir.glob("*.mp4")) if args.video_id == "all" else list(vdir.glob(f"{args.video_id}.*"))
    out_dir = Path(args.out)
    for v in vids:
        cache = out_dir / "cache" / f"{v.stem}.npz"
        if cache.exists() and not args.redetect:
            z = np.load(cache, allow_pickle=True)
            dets, steps, fps, size = list(z["dets"]), z["steps"], float(z["fps"]), tuple(z["size"])
        else:
            print(f"{v.stem}: detecting ...", flush=True)
            dets, steps, fps, size = run_detection_and_motion(v, args.conf, args.imgsz)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, dets=np.array(dets, dtype=object), steps=steps, fps=fps, size=size)

        cum = np.zeros_like(steps)
        cum[0] = np.eye(3)
        for t in range(1, len(steps)):
            cum[t] = steps[t] @ cum[t - 1]
        tracks = [tr for tr in track(dets, steps) if len(tr.frames) >= args.min_track_s * fps]
        keep = {tr.tid: moving_mask(tr, cum, fps, args.win_s, args.speed_thresh, args.min_run_s) for tr in tracks}
        n_moving = sum(bool(m.any()) for m in keep.values())
        key_step = max(1, round(args.key_s * fps))
        n_rows = export(v, tracks, keep, size, fps, out_dir, args.preview, key_step)
        print(f"{v.stem}: {len(tracks)} tracks, {n_moving} moving, {n_rows} boxes -> {out_dir}/{v.stem}_cvat.zip")


if __name__ == "__main__":
    main()
