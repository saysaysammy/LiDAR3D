import os
import sys
import torch
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
from sfa.models.model_utils import create_model
from sfa.data_process.kitti_dataset import KittiDataset
from sfa.utils.evaluation_utils import decode
from sfa.utils.evaluation_utils import post_processing
from sfa.utils.evaluation_utils import convert_det_to_real_values


configs = edict()
configs.arch = "fpn_resnet_18"
configs.num_classes = 3
configs.num_input_features = 4
configs.head_conv = 64
configs.imagenet_pretrained = False
configs.heads = {
    "hm_cen": 3,
    "cen_offset": 2,
    "direction": 2,
    "z_coor": 1,
    "dim": 3}

device = torch.device("cpu")

model = create_model(configs)

checkpoint_path = Path(PROJECT_ROOT / "Model" / "SFA3D" /
    "checkpoints" / "fpn_resnet_18" / "fpn_resnet_18_epoch_300.pth")

checkpoint = torch.load(
    checkpoint_path,
    map_location=device)

model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()


configs.dataset_dir = Path(PROJECT_ROOT / "Model" / "SFA3D" / "dataset" / "kitti")
configs.input_size = (608, 608)
configs.hm_size = (152, 152)
configs.max_objects = 50


# Validation Dataset
dataset = KittiDataset(
    configs,
    mode="val",
    hflip_prob=0.,
    num_samples=1)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--sample_id",
    type=int,
    default=808,
    help="KITTI sample ID")

parser.add_argument(
    "--bin_path",
    type=str,
    default=None,
    help="Path to custom LiDAR .bin file")

args = parser.parse_args()

sample_id = args.sample_id
custom_bin_path = args.bin_path

is_custom_bin = custom_bin_path is not None

if is_custom_bin:

    if not os.path.isfile(custom_bin_path):
        raise FileNotFoundError(
            f"BIN file not found:\n{custom_bin_path}")

    # Read custom LiDAR
    custom_points = np.fromfile(
        custom_bin_path,
        dtype=np.float32)

    if custom_points.size % 4 != 0:
        raise ValueError(
            "Invalid .bin file. "
            "Expected float32 values in groups of 4 (x, y, z, intensity).")

    custom_points = custom_points.reshape(-1, 4)

    dataset.get_lidar = lambda _: custom_points.copy()
else:
    dataset.sample_id_list = [sample_id]


targets, bev_map, info = dataset[0]

bev_map = bev_map.unsqueeze(0)
bev_map = bev_map.to(device).float()


with torch.no_grad():
    outputs = model(bev_map)


configs.K = 50
with torch.no_grad():
    detections = decode(
        outputs["hm_cen"],
        outputs["cen_offset"],
        outputs["direction"],
        outputs["z_coor"],
        outputs["dim"],
        K=configs.K)


# Post Processing
detections = detections.cpu().numpy()

detections = post_processing(
    detections,
    num_classes=configs.num_classes,
    down_ratio=4,
    peak_thresh=0.2)


detections = detections[0]
real_detections = convert_det_to_real_values(
    detections,
    num_classes=configs.num_classes)


# SAVE REAL DETECTIONS FOR PYVISTA
results_dir = Path(PROJECT_ROOT / "Output")
results_dir.mkdir(exist_ok=True)

if is_custom_bin:

    bin_stem = Path(custom_bin_path).stem

    detections_path = (
        results_dir /
        f"custom_{bin_stem}_real_detections.npy")
else:
    sample_id = int(dataset.sample_id_list[0])

    detections_path = (
        results_dir /
        f"{sample_id:06d}_real_detections.npy")

np.save(detections_path, real_detections)