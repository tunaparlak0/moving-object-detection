"""Video writer that produces browser/VS Code playable H.264 MP4 files."""
import sys

import cv2


def make_writer(path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """H.264 (avc1). On Windows the Media Foundation backend provides the encoder
    (OpenCV's bundled FFmpeg has no H.264 encoder); falls back to mp4v if unavailable."""
    w, h = size
    size = (w - w % 2, h - h % 2)  # H.264 needs even dimensions
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    if sys.platform == "win32":
        wr = cv2.VideoWriter(str(path), cv2.CAP_MSMF, fourcc, fps, size)
    else:
        wr = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not wr.isOpened():
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    return wr
