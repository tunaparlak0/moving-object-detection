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
.venv\Scripts\python -m src.run_pipeline vid_002            # compensated 3-frame differencing
.venv\Scripts\python -m src.run_pipeline all --method mog2  # compensated MOG2, all videos
.venv\Scripts\python -m src.run_pipeline vid_002 --flow farneback --model affine
.venv\Scripts\python -m src.preannotate all                # GT draft for CVAT (detector + tracker + moving filter)
.venv\Scripts\python -m pytest -q tests                     # synthetic moving-camera tests
```
Outputs go to `results/<video>/<method>_<flow>_<model>/` (`detections.txt`, `motion.csv`, `vis.mp4`).
