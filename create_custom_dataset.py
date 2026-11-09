import json
import os
import cv2
import math
import numpy as np
import argparse
from detectron2.structures import BoxMode


def parse_args():
    parser = argparse.ArgumentParser(description="Convert VIA JSON annotations to Detectron2 dataset dicts")
    parser.add_argument("--base_dir", type=str, default="./datasets/data/my_new_images",
                        help="Directory containing gt_json/train/ annotations and output folder")
    parser.add_argument("--images_dir", type=str, default="./datasets/data/my_new_images/images",
                        help="Directory containing raw image files")
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


def create_dataset(base_dir, images_dir):
    json_modal = os.path.join(base_dir, "gt_json", "train", "via_region_data_modal.json")
    json_amodal = os.path.join(base_dir, "gt_json", "train", "via_region_data_amodal.json")

    if not os.path.exists(json_modal) or not os.path.exists(json_amodal):
        raise FileNotFoundError(f"Missing VIA annotation files in {os.path.join(base_dir, 'gt_json', 'train')}")

    with open(json_modal, 'r') as f:
        modal_data = json.load(f)
    with open(json_amodal, 'r') as f:
        amodal_data = json.load(f)

    amodal_by_filename = {v["filename"]: v["regions"] for k, v in amodal_data.items()}
    dataset_dicts = []

    for idx, (k, v) in enumerate(modal_data.items()):
        filename = v["filename"]
        if filename not in amodal_by_filename:
            continue

        image_path = os.path.join(images_dir, filename)
        if not os.path.exists(image_path):
            alt_path = os.path.join(base_dir, "images", filename)
            if os.path.exists(alt_path):
                image_path = alt_path
            else:
                continue

        img = cv2.imread(image_path)
        if img is None:
            continue
        height, width = img.shape[:2]

        record = {
            "file_name": image_path,
            "image_id": idx,
            "height": height,
            "width": width
        }

        depth_name = filename.rsplit("_color.", 1)[0] + "_depth.npy"
        record["depth_file"] = os.path.join(base_dir, "depth_maps", depth_name)

        objs = []
        modal_regions = v["regions"]
        amodal_regions = amodal_by_filename[filename]

        for m_reg in modal_regions:
            try:
                m_shape = m_reg["shape_attributes"]
                px_m = m_shape["all_points_x"]
                py_m = m_shape["all_points_y"]
                poly_m = [(x + 0.5, y + 0.5) for x, y in zip(px_m, py_m)]
                poly_m_flat = [p for x in poly_m for p in x]
                bbox_m = [np.min(px_m), np.min(py_m), np.max(px_m), np.max(py_m)]
            except KeyError:
                continue

            best_iou = 0.0
            best_poly_a_flat = None
            best_bbox_a = None

            for a_reg in amodal_regions:
                try:
                    a_shape = a_reg["shape_attributes"]
                    if a_shape.get("name") == "circle":
                        cx = a_shape["cx"]
                        cy = a_shape["cy"]
                        r = a_shape["r"]
                        bbox_a = [cx - r, cy - r, cx + r, cy + r]
                        num_points = 16
                        px_a = [cx + r * math.cos(2 * math.pi * i / num_points) for i in range(num_points)]
                        py_a = [cy + r * math.sin(2 * math.pi * i / num_points) for i in range(num_points)]
                    else:
                        px_a = a_shape["all_points_x"]
                        py_a = a_shape["all_points_y"]
                        bbox_a = [np.min(px_a), np.min(py_a), np.max(px_a), np.max(py_a)]

                    iou = calculate_iou(bbox_m, bbox_a)
                    if iou > best_iou:
                        best_iou = iou
                        poly_a = [(x + 0.5, y + 0.5) for x, y in zip(px_a, py_a)]
                        best_poly_a_flat = [p for x in poly_a for p in x]
                        best_bbox_a = bbox_a
                except KeyError:
                    continue

            if best_poly_a_flat is not None and len(poly_m_flat) > 4 and len(best_poly_a_flat) > 4:
                obj = {
                    "bbox_visible": bbox_m,
                    "bbox": best_bbox_a,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "visible_mask": [poly_m_flat],
                    "segmentation": [best_poly_a_flat],
                    "category_id": 0,
                }
                objs.append(obj)

        if len(objs) > 0:
            record["annotations"] = objs
            dataset_dicts.append(record)

    output_file = os.path.join(base_dir, 'train_dataset_dicts.npy')
    np.save(output_file, np.array(dataset_dicts))
    print(f"Created {output_file} with {len(dataset_dicts)} annotated images.")


if __name__ == "__main__":
    args = parse_args()
    create_dataset(args.base_dir, args.images_dir)
