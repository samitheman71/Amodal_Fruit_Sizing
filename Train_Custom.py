import argparse
import numpy as np
import os
import cv2
import torch
import time

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune amodal instance segmentation on custom dataset")
    parser.add_argument("--num_iterations", dest="num_iterations", type=int, default=1000,
                        help="Maximum number of training iterations")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=2,
                        help="Training batch size per step")
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.005,
                        help="Base learning rate")
    parser.add_argument("--dataset_path", dest="dataset_path", type=str, default="./datasets/data/my_new_images/",
                        help="Path to folder containing train_dataset_dicts.npy")
    parser.add_argument("--weights", dest="weights", type=str, default="./output/trial01/model_0002999.pth",
                        help="Path to base pretrained model weights")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default="./output/finetuned/",
                        help="Directory to save model checkpoints")
    return parser.parse_args()


def load_custom_dataset(dataset_path):
    dataset_dicts_file = os.path.join(dataset_path, 'train_dataset_dicts.npy')
    if not os.path.exists(dataset_dicts_file):
        raise FileNotFoundError(f"Dataset dictionary file not found: {dataset_dicts_file}. Run create_custom_dataset.py first.")

    dataset_dicts = np.load(dataset_dicts_file, allow_pickle=True).tolist()
    DatasetCatalog.register("Custom_Apples_train", lambda: dataset_dicts)
    dataset_metadata = MetadataCatalog.get("Custom_Apples_train")
    dataset_metadata.thing_classes = ["apple"]
    return dataset_dicts, dataset_metadata


if __name__ == '__main__':
    args = parse_args()
    dataset_dicts_train, metadata = load_custom_dataset(args.dataset_path)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_orcnn_X_101_32x8d_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("Custom_Apples_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2

    # Initialize from pretrained weights
    if os.path.exists(args.weights):
        cfg.MODEL.WEIGHTS = args.weights
    else:
        print(f"Warning: Base weights file not found at {args.weights}. Falling back to ImageNet backbone initialization.")

    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.MAX_ITER = args.num_iterations
    cfg.SOLVER.STEPS = []

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # Data Augmentation: Random relative cropping to improve small instance scale robustness
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "relative_range"
    cfg.INPUT.CROP.SIZE = [0.6, 0.6]

    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("Starting training...")
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    print(f"Training completed. Final weights saved to {cfg.OUTPUT_DIR}")
