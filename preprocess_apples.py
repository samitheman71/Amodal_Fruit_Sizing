import cv2
import os
import glob
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Selective HSV color enhancement for fruit detection")
    parser.add_argument("--input_dir", type=str, default="datasets/data/my_new_images/images",
                        help="Path to input image directory")
    parser.add_argument("--output_dir", type=str, default="datasets/data/my_new_images/images_preprocessed",
                        help="Path to save preprocessed images")
    parser.add_argument("--sat_factor", type=float, default=1.6,
                        help="Saturation boost multiplier for target fruit hue (default: 1.6)")
    parser.add_argument("--val_factor", type=float, default=1.3,
                        help="Brightness (Value) boost multiplier for target fruit hue (default: 1.3)")
    return parser.parse_args()


def preprocess_images(input_dir, output_dir, sat_factor=1.6, val_factor=1.3):
    os.makedirs(output_dir, exist_ok=True)

    img_files = glob.glob(os.path.join(input_dir, "*.*"))
    img_files = [f for f in img_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Found {len(img_files)} images in {input_dir}")

    for img_path in img_files:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Convert to HSV color space for hue-based segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Red hue spans across 0 and 180 in OpenCV HSV representation
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        red_mask = mask1 + mask2

        # Enhance saturation and brightness exclusively within masked target regions
        s_float = s.astype(np.float32)
        v_float = v.astype(np.float32)

        s_float[red_mask > 0] = np.clip(s_float[red_mask > 0] * sat_factor, 0, 255)
        v_float[red_mask > 0] = np.clip(v_float[red_mask > 0] * val_factor, 0, 255)

        hsv[:, :, 1] = s_float.astype(np.uint8)
        hsv[:, :, 2] = v_float.astype(np.uint8)

        final_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, final_img)

    print(f"Preprocessing completed. Output saved to: {output_dir}")


if __name__ == '__main__':
    args = parse_args()
    preprocess_images(args.input_dir, args.output_dir, args.sat_factor, args.val_factor)
