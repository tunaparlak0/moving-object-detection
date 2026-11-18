# Moving Object Detection from a Moving Camera via Ego-motion Compensation

Project-7: detect and track truly moving objects in videos captured from a moving
platform (UAV / handheld / vehicle) by compensating camera ego-motion.

## Pipeline
1. **Optical flow** – dense (Farneback) / sparse (Lucas-Kanade) motion between frames.
2. **Global motion estimation** – Affine or Homography (RANSAC) on background flow.
3. **Compensation** – warp the previous frame onto the current frame.
4. **Foreground extraction** – frame differencing / MOG2 on compensated frames.
5. **Post-processing & tracking** – morphology, contours → bounding boxes, Kalman/IoU tracker.
6. **Evaluation** – compare against ground-truth boxes (Precision / Recall / F1 / IoU, MOT metrics).

## Dataset requirements
- ≥ 25 original videos, ≥ 512x512, ≥ 15 s each
- Ground-truth bounding boxes for **moving** objects only

## Layout
```
data/videos/        raw videos (not committed)
data/annotations/   ground-truth boxes
src/                pipeline code
notebooks/          experiments
results/            outputs (not committed)
```

## Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
