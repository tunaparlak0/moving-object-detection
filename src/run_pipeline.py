"""Moving object detection from a moving camera via ego-motion compensation.

For every frame t:
  1. estimate the step transform M_t : frame t-1 -> frame t (optical flow + RANSAC)
  2. chain step transforms to warp neighbour frames (t-k, t+k, ...) into frame t
     (symmetric window -> output for frame t is produced k*N frames later)
  3. foreground = compensated three-frame differencing or MOG2 on the aligned frames
  4. morphology + connected components -> bounding boxes (in original resolution)

Outputs (results/<video_id>/<run_name>/):
  detections.txt  MOT format: frame, -1, x, y, w, h, conf, -1, -1, -1
  motion.csv      per-frame RANSAC inlier ratio / #points (ego-motion quality)
  vis.mp4         frame with boxes | foreground mask   (unless --no-video)

Usage:
  python -m src.run_pipeline vid_001
  python -m src.run_pipeline all --method mog2 --flow farneback --model affine
"""
import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.egomotion import estimate_motion, warp_to
from src.foreground import clean_mask, compensated_diff, compensated_mog2, mask_to_boxes
from src.video_io import make_writer

DEFAULTS = dict(
    proc_max_side=960,   # processing resolution (longest side, px)
    gap_s=0.1,           # temporal gap k between compared frames (seconds)
    method="diff",       # diff | mog2
    flow="lk",           # lk | farneback
    model="homography",  # homography | affine | none (no-compensation baseline)
    diff_thresh=25,
    mog_window=3,        # MOG2: frames per side (t-Nk..t-k and t+k..t+Nk)
    mog_var=32.0,
    mog_bg_ratio=0.7,
    blur=5,
    border_erode=15,     # ignore pixels near the warped image border
    min_area=30,         # at processing resolution
)


def chain(steps: list[np.ndarray]) -> np.ndarray:
    """steps = [M_{j+1}, ..., M_t] (oldest first) -> transform frame j -> frame t."""
    H = np.eye(3)
    for M in steps:
        H = M @ H
    return H


def process_video(vpath: Path, out_dir: Path, cfg: dict, save_video: bool = True) -> dict:
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = min(1.0, cfg["proc_max_side"] / max(W, H))
    pw, ph = int(round(W * s)), int(round(H * s))
    k = max(1, round(cfg["gap_s"] * fps))
    n_side = 1 if cfg["method"] == "diff" else cfg["mog_window"]
    offsets = [-k * i for i in range(n_side, 0, -1)] + [k * i for i in range(1, n_side + 1)]
    delay = max(offsets)          # target frame lags the newest frame by `delay`
    span = delay - min(offsets)   # buffered frames = span + 1

    out_dir.mkdir(parents=True, exist_ok=True)
    det_f = open(out_dir / "detections.txt", "w")
    mot_f = open(out_dir / "motion.csv", "w", newline="")
    mot_w = csv.writer(mot_f)
    mot_w.writerow(["frame", "ok", "inlier_ratio", "n_points"])
    writer = None
    if save_video:
        writer = make_writer(out_dir / "vis.mp4", fps, (pw * 2, ph))
    erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg["border_erode"],) * 2)

    smalls = deque(maxlen=span + 1)   # colour frames at processing size, newest last
    grays = deque(maxlen=span + 1)    # blurred gray frames
    steps = deque(maxlen=span)        # steps[i] maps grays[i] -> grays[i+1]
    raw_prev = None
    frame_idx, n_boxes, t0 = 0, 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA) if s < 1 else frame
        raw = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(raw, (cfg["blur"], cfg["blur"]), 0)

        if raw_prev is not None:
            est = estimate_motion(raw_prev, raw, flow=cfg["flow"], model=cfg["model"])
            steps.append(est.M)
            mot_w.writerow([frame_idx, int(est.ok), f"{est.inlier_ratio:.3f}", est.n_points])
        raw_prev = raw
        smalls.append(small)
        grays.append(g)
        if len(grays) <= span:
            continue  # warm-up: not enough frames around the target yet

        c = span - delay              # buffer index of the target frame
        target_idx = frame_idx - delay
        step_list = list(steps)
        warped, valid = [], np.full((ph, pw), 255, np.uint8)
        for off in offsets:
            j = c + off
            if off < 0:   # past frame j -> target c
                Hj = chain(step_list[j:c])
            else:         # future frame j -> target c
                Hj = np.linalg.inv(chain(step_list[c:j]))
            w_img, w_valid = warp_to(grays[j], Hj, (pw, ph))
            warped.append(w_img)
            valid = cv2.bitwise_and(valid, w_valid)
        valid = cv2.erode(valid, erode_k)
        if cfg["method"] == "diff":
            mask = compensated_diff(grays[c], warped, valid, cfg["diff_thresh"])
        else:
            mask = compensated_mog2(grays[c], warped, valid, cfg["mog_var"], cfg["mog_bg_ratio"])
        mask = clean_mask(mask)
        boxes = mask_to_boxes(mask, min_area=cfg["min_area"])

        for (x, y, w, h) in boxes:
            det_f.write(f"{target_idx},-1,{x / s:.1f},{y / s:.1f},{w / s:.1f},{h / s:.1f},1,-1,-1,-1\n")
        n_boxes += len(boxes)

        if writer:
            vis = smalls[c].copy()
            for (x, y, w, h) in boxes:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(vis, f"{target_idx}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            writer.write(np.hstack([vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))

    cap.release()
    det_f.close()
    mot_f.close()
    if writer:
        writer.release()
    # frames for which an output was produced (others lack enough neighbours) -> used by evaluation
    meta = {"first_frame": span + 1 - delay, "last_frame": frame_idx - delay, "n_frames": frame_idx,
            "fps": fps, "k": k, "scale": s, "cfg": cfg}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    dt = time.time() - t0
    return {"frames": frame_idx, "boxes": n_boxes, "sec": dt, "fps": frame_idx / dt if dt else 0, "k": k}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id", help="e.g. vid_001, or 'all'")
    ap.add_argument("--videos", default="data/videos")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-video", action="store_true")
    for key, val in DEFAULTS.items():
        ap.add_argument(f"--{key.replace('_', '-')}", type=type(val), default=val)
    args = ap.parse_args()
    cfg = {k: getattr(args, k) for k in DEFAULTS}
    run_name = f"{cfg['method']}_{cfg['flow']}_{cfg['model']}"

    vdir = Path(args.videos)
    vids = sorted(vdir.glob("*.mp4")) if args.video_id == "all" else list(vdir.glob(f"{args.video_id}.*"))
    for v in vids:
        r = process_video(v, Path(args.out) / v.stem / run_name, cfg, save_video=not args.no_video)
        print(f"{v.stem}: {r['frames']} frames, {r['boxes']} boxes, k={r['k']}, "
              f"{r['sec']:.1f}s ({r['fps']:.1f} fps) -> {args.out}/{v.stem}/{run_name}")


if __name__ == "__main__":
    main()
