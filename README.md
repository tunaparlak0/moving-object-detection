# Moving Object Detection from a Moving Camera via Ego-motion Compensation

Project-7: detect and track truly moving objects in videos captured from a moving
platform (UAV / handheld / vehicle) by compensating camera ego-motion.

## Pipeline

1. **Optical flow** – dense (Farneback) / sparse (Lucas-Kanade) motion between frames.
2. **Global motion estimation** – Affine or Homography (RANSAC) on background flow.
3. **Compensation** – warp the previous frame onto the current frame.
4. **Foreground extraction** – frame differencing / MOG2 on compensated frames. For
   differencing the temporal gap is chosen per frame: the longest of 0.5 / 0.25 / 0.1 s whose
   aligned difference stays clean (≤ 1 % of pixels changed), so slow cameras get a long gap
   (whole objects separate) and fast or parallax-heavy shots fall back to a short one.
5. **Post-processing & tracking** – morphology, contours → candidate boxes; class-agnostic
   Kalman/IoU tracker (`src/tracker.py`) keeps only blobs that persist and move in the
   camera-compensated frame. No object classifier is used: anything that moves is detected.
6. **Evaluation** – compare against ground-truth boxes (Precision / Recall / F1 / mean IoU, recall per class), all methods scored on the same frames.

## Dataset requirements

- ≥ 25 original videos, ≥ 512x512, ≥ 15 s each
- Ground-truth bounding boxes for **moving** objects only

## Layout

```
data/videos/        raw videos (not committed)
data/annotations/   ground-truth boxes
src/                pipeline code
results/            outputs (not committed)
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt   # includes CUDA 13.2 PyTorch (extra index in the file)
```

Model weights for GT pre-annotation (not in git, ~35 MB):
```powershell
.venv\Scripts\python -c "from huggingface_hub import hf_hub_download as d; import shutil, os; os.makedirs('models', exist_ok=True); shutil.copy(d('dronefreak/uavdt-yolov9s','best.pt'), 'models/uavdt_yolov9s.pt')"
.venv\Scripts\python -c "from ultralytics.utils.downloads import attempt_download_asset as a; a('models/yolo11s.pt')"
```

Videos are not stored in git; sources are listed in `data/manifest.csv`.

Video Sources: https://www.pexels.com/search/videos/drone%20footage/

## Usage
```powershell
.venv\Scripts\python -m src.check_videos                    # dataset requirements check
.venv\Scripts\python -m src.run_pipeline vid_002            # compensated 3-frame differencing (adaptive gap) + tracker
.venv\Scripts\python -m src.run_pipeline vid_002 --track 0  # raw per-frame blobs (no tracker)
.venv\Scripts\python -m src.run_pipeline vid_002 --gap-max-s 0 --gap-s 0.25  # fixed gap instead of adaptive
.venv\Scripts\python -m src.run_pipeline all --method mog2  # compensated MOG2, all videos
.venv\Scripts\python -m src.run_pipeline vid_002 --flow farneback --model affine
.venv\Scripts\python -m src.preannotate all                # GT draft for CVAT (detector + tracker + moving filter)
.venv\Scripts\python -m src.run_pipeline all --model none   # no-compensation baseline
.venv\Scripts\python -m src.evaluate --iou 0.3              # P/R/F1 vs data/annotations (--drafts: unreviewed drafts)
.venv\Scripts\python -m src.visualize_gt all --save         # labelled previews -> results/gt_preview/
.venv\Scripts\python -m pytest -q tests                     # synthetic moving-camera tests
```
The YOLO detectors are used **only** by `src.preannotate` to speed up ground-truth labelling;
the detection pipeline itself never uses them.

Outputs go to `results/<video>/<method>_<flow>_<model>[_adapt][_trk]/` (`detections.txt`, `motion.csv`,
`vis.mp4`, `meta.json` with the frame gaps chosen).
