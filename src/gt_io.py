"""Load ground-truth boxes in MOT format (CVAT 'MOT 1.1' export).

Line format: frame, id, x, y, w, h, not_ignored, class, visibility
"""
from collections import defaultdict
from pathlib import Path


def load_mot_gt(path: str | Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """Return {frame (1-based): [(track_id, x, y, w, h), ...]}, skipping ignored rows."""
    boxes = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        frame, tid = int(float(parts[0])), int(float(parts[1]))
        x, y, w, h = map(float, parts[2:6])
        if len(parts) > 6 and int(float(parts[6])) == 0:
            continue
        boxes[frame].append((tid, x, y, w, h))
    return dict(boxes)
