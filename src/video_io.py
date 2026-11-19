"""Video writer that produces small, browser/VS Code playable H.264 MP4 files.

Primary: frames are piped to the ffmpeg binary bundled with `imageio-ffmpeg`
(libx264, CRF quality control -> compact files).
Fallback: OpenCV VideoWriter (Media Foundation H.264 on Windows, then mp4v).
"""
import subprocess
import sys

import cv2
import numpy as np


class FFmpegWriter:
    """Minimal cv2.VideoWriter-compatible wrapper (write / release / isOpened)."""

    def __init__(self, path, fps: float, size: tuple[int, int], crf: int = 23, preset: str = "veryfast"):
        import imageio_ffmpeg
        self.size = size
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:.3f}", "-i", "-",
               "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def isOpened(self) -> bool:
        return self.proc.poll() is None

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


def make_writer(path, fps: float, size: tuple[int, int]):
    w, h = size
    size = (w - w % 2, h - h % 2)  # H.264 / yuv420p needs even dimensions
    try:
        return FFmpegWriter(path, fps, size)
    except (ImportError, OSError):
        pass
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    wr = (cv2.VideoWriter(str(path), cv2.CAP_MSMF, fourcc, fps, size) if sys.platform == "win32"
          else cv2.VideoWriter(str(path), fourcc, fps, size))
    if not wr.isOpened():
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    return wr
