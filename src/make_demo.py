"""Side-by-side demo video: detections of two runs on the same video.

Left:  run A (default: no ego-motion compensation)
Right: run B (default: our method)
Ground-truth boxes are drawn thin in white so the viewer can judge both.

Usage:
  python -m src.make_demo vid_022
  python -m src.make_demo vid_022 --left diff_lk_none_adapt --right diff_lk_homography_adapt_trk
Output: results/demo/<video>_demo.mp4
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from src.evaluate import load_boxes
from src.video_io import make_writer


def panel(frame, dets, gt, s, title, color):
    img = frame.copy()
    for x, y, w, h, _ in gt:
        cv2.rectangle(img, (int(x * s), int(y * s)), (int((x + w) * s), int((y + h) * s)), (255, 255, 255), 1)
    for x, y, w, h, _ in dets:
        cv2.rectangle(img, (int(x * s), int(y * s)), (int((x + w) * s), int((y + h) * s)), color, 2)
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, f"{title}  ({len(dets)} boxes)", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("--left", default="diff_lk_none_adapt")
    ap.add_argument("--right", default="diff_lk_homography_adapt_trk")
    ap.add_argument("--left-title", default="Without ego-motion compensation")
    ap.add_argument("--right-title", default="Ours (ego-motion compensated)")
    ap.add_argument("--width", type=int, default=960, help="width of each panel")
    args = ap.parse_args()

    v = args.video_id
    left = load_boxes(Path("results") / v / args.left / "detections.txt", False)
    right = load_boxes(Path("results") / v / args.right / "detections.txt", False)
    gt = load_boxes(Path("data/annotations") / f"{v}.txt", True)
    cap = cv2.VideoCapture(str(next(Path("data/videos").glob(f"{v}.*"))))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = args.width / W
    size = (args.width, int(round(H * s)))
    out = Path("results/demo") / f"{v}_demo.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    wr = make_writer(out, fps, (size[0] * 2, size[1]))
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f += 1
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        g = gt.get(f, [])
        wr.write(np.hstack([panel(frame, left.get(f, []), g, s, args.left_title, (0, 0, 255)),
                            panel(frame, right.get(f, []), g, s, args.right_title, (0, 220, 0))]))
    cap.release()
    wr.release()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
