import os
import sys
import argparse
import numpy as np
from pathlib import Path

if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent.parent.parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
SFA3D_PATH = os.path.join(PROJECT_ROOT, "Model", "SFA3D")
sys.path.insert(0, SFA3D_PATH)


from easydict import EasyDict as edict
from sfa.data_process.kitti_dataset import KittiDataset
from sfa.data_process.transformation import camera_to_lidar_box


# CONFIG
parser = argparse.ArgumentParser()
parser.add_argument(
    "--sample_id",
    type=int,
    default=808)

args = parser.parse_args()

SAMPLE_ID = args.sample_id

DATASET_DIR = Path(PROJECT_ROOT / "Model" / "SFA3D" / "dataset" / "kitti")
OUTPUT_PATH = Path(PROJECT_ROOT / "Output" / f"{SAMPLE_ID:06d}_ground_truth.npy")


# SFA3D boundary
MIN_X, MAX_X = 0, 50
MIN_Y, MAX_Y = -25, 25
MIN_Z, MAX_Z = -2.73, 1.27


# DATASET CONFIG
configs = edict()
configs.dataset_dir = DATASET_DIR
configs.input_size = (608, 608)
configs.hm_size = (152, 152)
configs.num_classes = 3
configs.max_objects = 50


# LOAD DATASET
dataset = KittiDataset(
    configs,
    mode="val",
    hflip_prob=0.0,
    num_samples=1)

dataset.sample_id_list = [SAMPLE_ID]

sample_id = dataset.sample_id_list[0]
labels, has_labels = dataset.get_label(sample_id)


# LOAD CALIBRATION
calib = dataset.get_calib(sample_id)


# CONVERT CAMERA BOXES → LiDAR BOXES
ground_truth = []

for label in labels:

    class_id = int(label[0])

    # Only classes supported by SFA3D
    if class_id not in [0, 1, 2]:
        continue

    camera_box = label[1:]

    lidar_box = camera_to_lidar_box(
        camera_box.reshape(1, -1),
        calib.V2C,
        calib.R0,
        calib.P2)[0]


    x = lidar_box[0]
    y = lidar_box[1]
    z = lidar_box[2]

    h = lidar_box[3]
    w = lidar_box[4]
    l = lidar_box[5]

    yaw = lidar_box[6]


    # Apply SFA3D boundary
    if not (
        MIN_X <= x <= MAX_X and
        MIN_Y <= y <= MAX_Y and
        MIN_Z <= z <= MAX_Z
    ):  continue

    ground_truth.append([
        class_id, x, y, z, h, w, l, yaw])


# SAVE
ground_truth = np.array(
    ground_truth,
    dtype=np.float32)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True)

np.save(OUTPUT_PATH, ground_truth)