import os
import glob
import numpy as np
import cv2
import torch
import types
import csv
import argparse

try:
    from ultralytics.trackers import BOTSORT
    from ultralytics.utils import IterableSimpleNamespace
except ImportError:
    print("Error: ultralytics is required for tracking. Install via 'pip install ultralytics'.")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-object tracking and size estimation using BoT-SORT")
    parser.add_argument("--txt_dir", type=str, default="output/demo/my_new_images/txt_per_image",
                        help="Directory containing detection text files per frame")
    parser.add_argument("--img_dir", type=str, default="datasets/data/my_new_images/images_preprocessed",
                        help="Directory containing input images")
    parser.add_argument("--out_dir", type=str, default="output/demo/my_new_images/botsort_tracking",
                        help="Directory to save tracking visualization and CSV output")
    parser.add_argument("--track_buffer", type=int, default=120,
                        help="Number of frames to keep a lost track active (default: 120)")
    parser.add_argument("--match_thresh", type=float, default=0.9,
                        help="Matching threshold for track association (default: 0.9)")
    parser.add_argument("--min_hits", type=int, default=5,
                        help="Minimum number of frame detections before confirming a track ID (default: 5)")
    parser.add_argument("--fps", type=float, default=15.0,
                        help="Output video frames per second (default: 15.0)")
    return parser.parse_args()


def parse_points(point_str):
    point_str = point_str.replace('[', '').replace(']', '')
    return [int(x) for x in point_str.split()]


class SafeNamespace(IterableSimpleNamespace):
    """Configuration namespace fallback for optional or dynamic attributes."""
    def __getattr__(self, item):
        if item == 'fuse_score':
            return False
        return None


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter_area = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / max(union_area, 1e-6)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    txt_files = sorted(glob.glob(os.path.join(args.txt_dir, "*.txt")))
    if not txt_files:
        print(f"No detection text files found in {args.txt_dir}")
        return

    tracker_cfg = SafeNamespace(**{
        'tracker_type': 'botsort',
        'track_high_thresh': 0.1,
        'track_low_thresh': 0.05,
        'new_track_thresh': 0.1,
        'track_buffer': args.track_buffer,
        'match_thresh': args.match_thresh,
        'gmc_method': 'sparseOptFlow',
        'proximity_thresh': 0.5,
        'appearance_thresh': 0.25,
        'with_reid': False,
        'name': 'botsort',
        'fuse_score': False
    })

    tracker = BOTSORT(tracker_cfg)
    total_counted = set()
    track_history = {}
    track_diameters = {}

    video_writer = None
    video_out_path = os.path.join(args.out_dir, "tracking_output.mp4")

    for txt_file in txt_files:
        base_name = os.path.basename(txt_file).replace('.txt', '')

        # Locate corresponding image
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
            temp_path = os.path.join(args.img_dir, base_name + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break

        if img_path is None:
            continue

        frame_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue

        if video_writer is None:
            h, w = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_out_path, fourcc, args.fps, (w, h))

        detections = []
        det_diameters = []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 6:
                    try:
                        d_val = float(parts[0])
                    except (ValueError, IndexError):
                        d_val = 0.0
                    conf = float(parts[3])
                    x_pts = parse_points(parts[4])
                    y_pts = parse_points(parts[5])
                    if len(x_pts) > 0 and len(y_pts) > 0:
                        bbox = [min(x_pts), min(y_pts), max(x_pts), max(y_pts), conf, 0]
                        detections.append(bbox)
                        det_diameters.append(d_val)

        if len(detections) > 0:
            det_tensor = torch.tensor(detections).float()
        else:
            det_tensor = torch.empty((0, 6)).float()

        try:
            from ultralytics.engine.results import Boxes
            boxes_obj = Boxes(det_tensor, img.shape[:2])
        except Exception:
            boxes_obj = det_tensor

        try:
            tracks = tracker.update(boxes_obj, img)
        except Exception:
            try:
                tracks = tracker.update(det_tensor, img)
            except Exception:
                tracks = []

        for track in tracks:
            if hasattr(track, 'tlbr'):
                x1, y1, x2, y2 = map(int, track.tlbr)
                track_id = int(track.track_id)
            elif hasattr(track, 'id'):
                x1, y1, x2, y2 = map(int, track.xyxy[0] if hasattr(track, 'xyxy') else track[:4])
                track_id = int(track.id)
            else:
                x1, y1, x2, y2, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])

            track_history[track_id] = track_history.get(track_id, 0) + 1

            # Associate track bounding box with closest frame detection to map estimated size
            track_diam = 0.0
            best_iou = 0.0
            for b_idx, det_box in enumerate(detections):
                iou = calculate_iou([x1, y1, x2, y2], det_box[:4])
                if iou > best_iou:
                    best_iou = iou
                    track_diam = det_diameters[b_idx]

            if track_diam > 0:
                track_diameters.setdefault(track_id, []).append(track_diam)

            # Confirm track once persistence threshold is met
            is_confirmed = track_history[track_id] >= args.min_hits
            if is_confirmed:
                total_counted.add(track_id)
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{track_id}"
            if track_diam > 0:
                label += f" | {track_diam:.0f}mm"
            elif track_id in track_diameters and len(track_diameters[track_id]) > 0:
                label += f" | {np.median(track_diameters[track_id]):.0f}mm"

            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imwrite(os.path.join(args.out_dir, frame_name), img)
        if video_writer is not None:
            video_writer.write(img)

    if video_writer is not None:
        video_writer.release()

    # Export aggregated fruit sizing measurements
    csv_out_path = os.path.join(args.out_dir, "apple_sizes_summary.csv")
    all_diams = []
    with open(csv_out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Apple_ID", "Frames_Tracked", "Estimated_Diameter_mm", "Estimated_Volume_cm3"])
        for tid in sorted(total_counted):
            diams = track_diameters.get(tid, [])
            med_diam = float(np.median(diams)) if len(diams) > 0 else 0.0
            if med_diam > 0:
                all_diams.append(med_diam)
            r_cm = (med_diam / 10.0) / 2.0
            vol_cm3 = (4.0 / 3.0) * np.pi * (r_cm ** 3) if med_diam > 0 else 0.0
            writer.writerow([tid, track_history.get(tid, 0), round(med_diam, 1), round(vol_cm3, 1)])

    print(f"\nBoT-SORT tracking completed.")
    print(f"Total confirmed unique apples: {len(total_counted)}")
    if len(all_diams) > 0:
        print(f"Mean diameter: {np.mean(all_diams):.1f} mm (Min: {np.min(all_diams):.1f} mm, Max: {np.max(all_diams):.1f} mm)")
    print(f"Annotated video saved to: {video_out_path}")
    print(f"Measurements CSV saved to: {csv_out_path}")


if __name__ == '__main__':
    main()
