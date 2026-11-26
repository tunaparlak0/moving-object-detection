"""Draw ground-truth boxes on a video to verify annotations.

Usage:
    python -m src.visualize_gt vid_001            # live window (q: quit, space: pause)
    python -m src.visualize_gt vid_001 --save     # write results/gt_preview/vid_001_gt.mp4
    python -m src.visualize_gt all --save         # every video that has data/annotations/vid_XXX.txt

Saved videos are downscaled (longest side 1280). The frame counter shows the CVAT frame number
(0-based); reviewed checkpoint frames (0, 15, 30, ...) are marked "CHECK".
"""
import argparse
from pathlib import Path

import cv2

from src.gt_io import load_mot_gt
from src.video_io import make_writer

LABELS = {1: "vehicle", 2: "two_wheeler", 3: "person"}


def color_for(tid: int) -> tuple[int, int, int]:
    return ((tid * 37) % 255, (tid * 91) % 255, (tid * 157) % 255)


def show(vpath: Path, ann: Path, save: bool, max_side: int = 1280) -> None:
    gt = load_mot_gt(ann)
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = min(1.0, max_side / max(W, H))
    size = (int(W * s), int(H * s))
    writer, out = None, None
    if save:
        out = Path("results") / "gt_preview" / f"{vpath.stem}_gt.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = make_writer(out, fps, size)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1  # MOT frames are 1-based; CVAT shows frame_idx - 1
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if s < 1 else frame
        for tid, x, y, w, h in gt.get(frame_idx, []):
            c = color_for(tid)
            p1, p2 = (int(x * s), int(y * s)), (int((x + w) * s), int((y + h) * s))
            cv2.rectangle(frame, p1, p2, c, 2)
            cv2.putText(frame, str(tid), (p1[0], p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        check = (frame_idx - 1) % 15 == 0
        cv2.putText(frame, f"CVAT frame {frame_idx - 1}" + ("  CHECK" if check else ""), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if check else (0, 255, 255), 2)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id", help="e.g. vid_001, or 'all'")
    ap.add_argument("--videos", default="data/videos")
    ap.add_argument("--annotations", default="data/annotations")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    ann_dir = Path(args.annotations)
    if args.video_id == "all":
        vids = [v for v in sorted(Path(args.videos).glob("vid_*.mp4")) if (ann_dir / f"{v.stem}.txt").exists()]
    else:
        vids = [next(Path(args.videos).glob(f"{args.video_id}.*"))]
    for v in vids:
        show(v, ann_dir / f"{v.stem}.txt", args.save)


if __name__ == "__main__":
    main()
