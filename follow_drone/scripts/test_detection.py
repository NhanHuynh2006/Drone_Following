"""Test detection + tracking pipeline (no MAVLink, no flight).

Loads camera + PFDet + OC-SORT, displays annotated frames.
Useful to verify model + tracker work before flying.

Usage:
    python scripts/test_detection.py [--show]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import yaml

from follow.camera import CameraThread
from follow.detector import PersonDetector
from follow.ocsort import OCSORT
from follow.target_selector import TargetSelector
from follow.distance import PinholeDistanceEstimator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    detector = PersonDetector(
        weights=cfg['model']['weights'],
        conf_threshold=cfg['model']['conf_threshold'],
        nms_iou=cfg['model']['nms_iou'],
        img_size=cfg['model']['img_size'],
        use_ema=cfg['model']['use_ema'],
        device=cfg['model']['device'],
    )
    camera = CameraThread(
        source=cfg['camera']['source'],
        width=cfg['camera']['width'],
        height=cfg['camera']['height'],
        fps=cfg['camera']['fps'],
    )
    camera.start()
    tracker = OCSORT(**cfg['tracking'])
    selector = TargetSelector(
        strategy=cfg['target_selection']['strategy'],
        lock_on_first=cfg['target_selection']['lock_on_first'],
        image_width=cfg['camera']['width'],
        image_height=cfg['camera']['height'],
    )
    distance_estimator = PinholeDistanceEstimator(
        fy=cfg['camera']['fy'],
        person_height_m=cfg['distance']['person_height_m'],
        min_bbox_height_px=cfg['distance']['min_bbox_height_px'],
        max_distance_m=cfg['distance']['max_distance_m'],
    )

    print("Press 'q' to quit, 'u' to unlock target.\n")
    fps_smooth = 0
    last_t = time.time()

    try:
        while True:
            frame, _ = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            t0 = time.time()
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            target = selector.select(tracks)
            distance = None
            if target:
                distance = distance_estimator.estimate(target['box'])

            elapsed = time.time() - t0
            fps_now = 1.0 / max(0.001, time.time() - last_t)
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps_now if fps_smooth > 0 else fps_now
            last_t = time.time()

            if not args.no_show:
                vis = frame.copy()
                for t in tracks:
                    x1, y1, x2, y2 = [int(v) for v in t['box']]
                    is_target = (target and t['id'] == target['id'])
                    color = (0, 255, 0) if is_target else (200, 200, 200)
                    thick = 3 if is_target else 1
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
                    label = f"ID{t['id']} ({t['score']:.2f})"
                    if is_target and distance:
                        label += f" {distance:.1f}m"
                    cv2.putText(vis, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(vis, f"FPS: {fps_smooth:.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow('Detection Test', vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('u'):
                    selector.unlock()
                    print("Unlocked")
            else:
                if int(time.time()) % 2 == 0:
                    print(f"FPS: {fps_smooth:.1f} | tracks: {len(tracks)} | "
                          f"target: {target['id'] if target else None} | "
                          f"distance: {distance}")
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
