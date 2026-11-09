import os
import glob
import numpy as np
import cv2
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="IoU-based apple tracking")
    parser.add_argument("--txt_dir", type=str, default="output/demo/my_new_images/txt_per_image",
                        help="Directory containing detection text files")
    parser.add_argument("--img_dir", type=str, default="datasets/data/my_new_images/images",
                        help="Directory containing input images")
    parser.add_argument("--out_dir", type=str, default="output/demo/my_new_images/tracking",
                        help="Directory to save tracking visualization")
    parser.add_argument("--max_missed", type=int, default=5,
                        help="Maximum missed frames before terminating a track")
    parser.add_argument("--iou_thresh", type=float, default=0.05,
                        help="Minimum IoU overlap threshold for track association")
    return parser.parse_args()


def calculate_iou(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    return inter_area / float(box_a_area + box_b_area - inter_area)


def parse_points(point_str):
    point_str = point_str.replace('[', '').replace(']', '')
    return [int(x) for x in point_str.split()]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    txt_files = sorted(glob.glob(os.path.join(args.txt_dir, "*.txt")))
    active_tracks = {}
    next_id = 1
    total_counted = 0

    for txt_file in txt_files:
        base_name = os.path.basename(txt_file).replace('.txt', '')
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
            candidate = os.path.join(args.img_dir, base_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        detections = []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 6:
                    x_pts = parse_points(parts[4])
                    y_pts = parse_points(parts[5])
                    if len(x_pts) > 0 and len(y_pts) > 0:
                        bbox = [min(x_pts), min(y_pts), max(x_pts), max(y_pts)]
                        detections.append(bbox)

        matched_detects = set()
        matched_tracks = set()

        for track_id, track_data in active_tracks.items():
            best_iou = 0.0
            best_det_idx = -1

            for i, det_bbox in enumerate(detections):
                if i in matched_detects:
                    continue
                iou = calculate_iou(track_data['bbox'], det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i

            if best_iou > args.iou_thresh:
                active_tracks[track_id]['bbox'] = detections[best_det_idx]
                active_tracks[track_id]['missed_frames'] = 0
                matched_detects.add(best_det_idx)
                matched_tracks.add(track_id)

        for track_id in list(active_tracks.keys()):
            if track_id not in matched_tracks:
                active_tracks[track_id]['missed_frames'] += 1
                if active_tracks[track_id]['missed_frames'] > args.max_missed:
                    del active_tracks[track_id]

        for i, det_bbox in enumerate(detections):
            if i not in matched_detects:
                active_tracks[next_id] = {'bbox': det_bbox, 'missed_frames': 0}
                next_id += 1
                total_counted += 1

        if img_path and os.path.exists(img_path):
            img = cv2.imread(img_path)
            for t_id, t_data in active_tracks.items():
                if t_data['missed_frames'] == 0:
                    bbox = t_data['bbox']
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(img, f"ID:{t_id}", (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(args.out_dir, os.path.basename(img_path)), img)

    print(f"Tracking completed. Total unique tracks identified: {total_counted}")


if __name__ == '__main__':
    main()
