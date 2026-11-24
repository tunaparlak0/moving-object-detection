"""Hand-checkable cases for the evaluation matching."""
import numpy as np

from src.egomotion import estimate_motion
from src.evaluate import evaluate_video, iou_matrix, match_frame, summarize

V, P = 1, 3  # class ids: vehicle, person


def test_iou_matrix_known_values():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 10, 10], [20, 20, 5, 5]], float)
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 50 / 150, 0.0], atol=1e-6)


def test_match_frame_counts_tp_fp_fn():
    gt = [(0, 0, 10, 10, V), (100, 100, 10, 10, P)]
    det = [(1, 1, 10, 10, -1), (300, 300, 10, 10, -1)]        # one hit, one false alarm
    tp, fp, fn, ious, hit = match_frame(gt, det, 0.3)
    assert (tp, fp, fn) == (1, 1, 1) and hit == [0] and ious[0] > 0.6


def test_match_is_one_to_one():
    gt = [(0, 0, 10, 10, V)]
    det = [(0, 0, 10, 10, -1), (1, 0, 10, 10, -1)]           # two detections on one object
    assert match_frame(gt, det, 0.3)[:3] == (1, 1, 0)


def test_below_threshold_is_not_a_match():
    gt = [(0, 0, 10, 10, V)]
    det = [(7, 0, 10, 10, -1)]                               # IoU = 30/170 < 0.3
    assert match_frame(gt, det, 0.3)[:3] == (0, 1, 1)


def test_evaluate_video_scores_only_the_given_frame_range_and_per_class_recall():
    gt = {1: [(0, 0, 10, 10, V)], 2: [(0, 0, 10, 10, V), (50, 50, 10, 10, P)], 9: [(0, 0, 10, 10, V)]}
    det = {2: [(0, 0, 10, 10, -1)], 3: [(80, 80, 5, 5, -1)]}
    s = evaluate_video(gt, det, first=2, last=3, thr=0.3)      # frames 1 and 9 ignored
    assert (s["tp"], s["fp"], s["fn"], s["frames"]) == (1, 1, 1, 2)
    m = summarize(s)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["recall_vehicle"] == 1.0 and m["recall_person"] == 0.0


def test_no_compensation_baseline_returns_identity():
    img = np.random.default_rng(0).integers(0, 255, (64, 64), np.uint8)
    est = estimate_motion(img, np.roll(img, 5, axis=1), model="none")
    assert est.ok and np.allclose(est.M, np.eye(3))


def test_scored_frames_match_cvat_player_step():
    from src.evaluate import scored_frames
    # MOT frame = CVAT frame + 1; reviewers check CVAT 0, 15, 30, ...
    assert scored_frames(1, 50, 15) == [1, 16, 31, 46]
    assert scored_frames(5, 50, 15) == [16, 31, 46]
    assert scored_frames(1, 4, 1) == [1, 2, 3, 4]
