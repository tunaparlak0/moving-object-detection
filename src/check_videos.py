"""Check the dataset against the project requirements.

Usage:
    python -m src.check_videos [--videos data/videos] [--annotations data/annotations]
"""
import argparse
from pathlib import Path

import cv2

MIN_VIDEOS = 25
MIN_SIDE = 512
MIN_DURATION_S = 15.0
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok_open": False}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"ok_open": True, "w": w, "h": h, "fps": fps, "frames": n,
            "duration": n / fps if fps > 0 else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="data/videos")
    ap.add_argument("--annotations", default="data/annotations")
    args = ap.parse_args()

    vids = sorted(p for p in Path(args.videos).iterdir() if p.suffix.lower() in VIDEO_EXTS)
    ann_dir = Path(args.annotations)
    n_ok = 0
    print(f"{'video':<16}{'res':>11}{'fps':>7}{'dur(s)':>8}{'GT':>5}  status")
    for p in vids:
        info = video_info(p)
        has_gt = (ann_dir / f"{p.stem}.txt").exists()
        if not info["ok_open"]:
            print(f"{p.name:<16}{'-':>11}{'-':>7}{'-':>8}{'-':>5}  CANNOT OPEN")
            continue
        problems = []
        if min(info["w"], info["h"]) < MIN_SIDE:
            problems.append(f"res<{MIN_SIDE}")
        if info["duration"] < MIN_DURATION_S:
            problems.append(f"dur<{MIN_DURATION_S:g}s")
        if not has_gt:
            problems.append("no GT")
        status = "OK" if not problems else ", ".join(problems)
        n_ok += not problems
        print(f"{p.name:<16}{info['w']:>5}x{info['h']:<5}{info['fps']:>7.1f}"
              f"{info['duration']:>8.1f}{'yes' if has_gt else 'no':>5}  {status}")

    print(f"\n{n_ok}/{len(vids)} videos fully valid (need >= {MIN_VIDEOS}).")
    return 0 if n_ok >= MIN_VIDEOS else 1


if __name__ == "__main__":
    raise SystemExit(main())
