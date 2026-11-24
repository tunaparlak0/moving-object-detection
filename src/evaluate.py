"""Evaluate detections against ground-truth boxes of moving objects.

Per frame, detections are matched one-to-one to GT boxes (Hungarian on IoU, match if IoU >= thr):
  TP = matched pairs, FP = unmatched detections, FN = unmatched GT boxes.
Reported per video and pooled over all videos (micro average):
  Precision, Recall, F1, mean IoU of matches, recall per GT class.

Only frames for which the pipeline can produce output are scored (the first/last few frames
lack temporal neighbours); see meta.json written by run_pipeline.

GT sources:
  data/annotations/vid_XXX.txt     reviewed CVAT export (default)
  results/preannot/vid_XXX.txt     unreviewed drafts (--drafts) -> PRELIMINARY numbers only

Usage:
  python -m src.evaluate                                  # all runs found, reviewed GT
  python -m src.evaluate --drafts --iou 0.3
  python -m src.evaluate --runs mog2_lk_homography mog2_lk_none
  python -m src.evaluate --frame-step 1                   # score every frame (fully reviewed GT)

By default only every 15th frame is scored (CVAT frames 0, 15, 30, ...): reviewers check and fix
exactly those frames (CVAT Settings -> Player -> Player step = 15); the frames in between come
from the interpolated draft and are not trusted.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.run_pipeline import DEFAULTS

LABELS = ["vehicle", "two_wheeler", "person"]


def load_boxes(path: Path, is_gt: bool) -> dict[int, list[tuple]]:
    """MOT rows -> {frame: [(x, y, w, h, cls), ...]}; GT rows with not_ignored == 0 are skipped."""
    out = defaultdict(list)
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        if is_gt and len(p) > 6 and int(float(p[6])) == 0:
            continue
        cls = int(float(p[7])) if is_gt and len(p) > 7 else -1
        out[int(float(p[0]))].append((float(p[2]), float(p[3]), float(p[4]), float(p[5]), cls))
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: N x 4, b: M x 4 boxes as x, y, w, h."""
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter + 1e-9)


def match_frame(gt: list, det: list, thr: float):
    """Return (tp, fp, fn, matched_ious, matched_gt_indices)."""
    if not gt or not det:
        return 0, len(det), len(gt), [], []
    iou = iou_matrix(np.array([g[:4] for g in gt]), np.array([d[:4] for d in det]))
    rows, cols = linear_sum_assignment(-iou)
    keep = iou[rows, cols] >= thr
    tp = int(keep.sum())
    return tp, len(det) - tp, len(gt) - tp, list(iou[rows, cols][keep]), list(rows[keep])


def frame_range(run_dir: Path, video: Path) -> tuple[int, int]:
    """Frames the pipeline produced output for (from meta.json, else recomputed from the config)."""
    meta = run_dir / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        return m["first_frame"], m["last_frame"]
    method = run_dir.name.split("_")[0]
    cap = cv2.VideoCapture(str(video))
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    k = max(1, round(DEFAULTS["gap_s"] * fps))
    n_side = 1 if method == "diff" else DEFAULTS["mog_window"]
    delay, span = k * n_side, 2 * k * n_side
    return span + 1 - delay, n - delay


def scored_frames(first: int, last: int, step: int) -> list[int]:
    """Frames (1-based, MOT) to score. step > 1: only the frames reviewers checked in CVAT,
    i.e. CVAT frames 0, step, 2*step, ... (= MOT frames 1, 1+step, ...)."""
    return [f for f in range(first, last + 1) if step <= 1 or (f - 1) % step == 0]


def evaluate_video(gt: dict, det: dict, first: int, last: int, thr: float, step: int = 1) -> dict:
    s = dict(tp=0, fp=0, fn=0, iou_sum=0.0, frames=0)
    cls_tot, cls_hit = defaultdict(int), defaultdict(int)
    for f in scored_frames(first, last, step):
        g, d = gt.get(f, []), det.get(f, [])
        tp, fp, fn, ious, hit = match_frame(g, d, thr)
        s["tp"] += tp; s["fp"] += fp; s["fn"] += fn; s["iou_sum"] += sum(ious); s["frames"] += 1
        for i, box in enumerate(g):
            cls_tot[box[4]] += 1
            cls_hit[box[4]] += i in hit
    s["cls_tot"], s["cls_hit"] = dict(cls_tot), dict(cls_hit)
    return s


def summarize(s: dict) -> dict:
    p = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0.0
    r = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0.0
    out = dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0,
               mean_iou=s["iou_sum"] / s["tp"] if s["tp"] else 0.0)
    for ci, name in enumerate(LABELS, 1):
        tot = s["cls_tot"].get(ci, 0)
        out[f"recall_{name}"] = s["cls_hit"].get(ci, 0) / tot if tot else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="run names, e.g. mog2_lk_homography (default: all found)")
    ap.add_argument("--videos", default="data/videos")
    ap.add_argument("--results", default="results")
    ap.add_argument("--gt", default="data/annotations")
    ap.add_argument("--drafts", action="store_true", help="use unreviewed pre-annotations as GT (preliminary)")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--frame-step", type=int, default=15,
                    help="score only CVAT frames 0, N, 2N, ... (the reviewed ones); 1 = every frame")
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args()

    gt_dir = Path(args.results) / "preannot" if args.drafts else Path(args.gt)
    gt_tag = "drafts" if args.drafts else "gt"
    videos = [v for v in sorted(Path(args.videos).glob("vid_*.mp4")) if (gt_dir / f"{v.stem}.txt").exists()]
    if not videos:
        raise SystemExit(f"no ground truth found in {gt_dir}")
    runs = args.runs or sorted({d.name for v in videos for d in (Path(args.results) / v.stem).glob("*_*_*") if d.is_dir()})

    rows, pooled = [], {}
    for run in runs:
        tot = dict(tp=0, fp=0, fn=0, iou_sum=0.0, frames=0, cls_tot=defaultdict(int), cls_hit=defaultdict(int))
        for v in videos:
            run_dir = Path(args.results) / v.stem / run
            if not (run_dir / "detections.txt").exists():
                continue
            first, last = frame_range(run_dir, v)
            s = evaluate_video(load_boxes(gt_dir / f"{v.stem}.txt", True),
                               load_boxes(run_dir / "detections.txt", False), first, last, args.iou,
                               args.frame_step)
            rows.append(dict(run=run, video=v.stem, tp=s["tp"], fp=s["fp"], fn=s["fn"], **summarize(s)))
            for key in ("tp", "fp", "fn", "iou_sum", "frames"):
                tot[key] += s[key]
            for c, n in s["cls_tot"].items():
                tot["cls_tot"][c] += n
            for c, n in s["cls_hit"].items():
                tot["cls_hit"][c] += n
        if tot["frames"]:
            pooled[run] = dict(tp=tot["tp"], fp=tot["fp"], fn=tot["fn"], videos=sum(r["run"] == run for r in rows),
                               **summarize(tot))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{gt_tag}_iou{args.iou:g}" + (f"_step{args.frame_step}" if args.frame_step > 1 else "")
    with open(out / f"per_video_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(out / f"summary_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run"] + list(next(iter(pooled.values())).keys()))
        w.writeheader(); w.writerows({"run": k, **v} for k, v in pooled.items())

    if args.drafts:
        print("!! PRELIMINARY: GT = unreviewed pre-annotation drafts !!")
    scope = f"CVAT frames 0, {args.frame_step}, {2 * args.frame_step}, ..." if args.frame_step > 1 else "all frames"
    print(f"IoU >= {args.iou}, {len(videos)} videos, pooled over {scope}")
    print(f"{'run':24} {'P':>6} {'R':>6} {'F1':>6} {'mIoU':>6}  {'R_veh':>6} {'R_2wh':>6} {'R_per':>6}")
    for run, m in pooled.items():
        print(f"{run:24} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['mean_iou']:6.3f}  "
              f"{m['recall_vehicle']:6.3f} {m['recall_two_wheeler']:6.3f} {m['recall_person']:6.3f}")
    print(f"-> {out}/summary_{tag}.csv, per_video_{tag}.csv")


if __name__ == "__main__":
    main()
