"""Draw ground-truth boxes on a video to verify annotations.

Usage:
    python -m src.visualize_gt vid_001            # live window (q: quit, space: pause)
    python -m src.visualize_gt vid_001 --save     # write results/vid_001_gt.mp4
"""
import argparse
from pathlib import Path

import cv2

from src.gt_io import load_mot_gt


def color_for(tid: int) -> tuple[int, int, int]:
    return ((tid * 37) % 255, (tid * 91) % 255, (tid * 157) % 255)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("--videos", default="data/videos")
    ap.add_argument("--annotations", default="data/annotations")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    vpath = next(Path(args.videos).glob(f"{args.video_id}.*"))
    gt = load_mot_gt(Path(args.annotations) / f"{args.video_id}.txt")
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = None
    if args.save:
        out = Path("results") / f"{args.video_id}_gt.mp4"
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1  # MOT frames are 1-based
        for tid, x, y, w, h in gt.get(frame_idx, []):
            c = color_for(tid)
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), c, 2)
            cv2.putText(frame, str(tid), (int(x), int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        cv2.putText(frame, f"frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if writer:
            writer.write(frame)
        else:
            cv2.imshow("GT", frame)
            key = cv2.waitKey(int(1000 / fps)) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                cv2.waitKey(0)
    cap.release()
    if writer:
        writer.release()
        print(f"saved {out}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
