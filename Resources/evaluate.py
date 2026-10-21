import os
import sys
import csv
import numpy as np
import torch
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SFA3D_PATH = os.path.join(PROJECT_ROOT, "Model", "SFA3D")
sys.path.insert(0, SFA3D_PATH)


from easydict import EasyDict as edict
from sfa.data_process.kitti_dataset import KittiDataset
from sfa.models.model_utils import create_model
from sfa.utils.evaluation_utils import (
    decode,
    post_processing,
    convert_det_to_real_values)
from sfa.data_process.transformation import camera_to_lidar_box


# CONFIG
DATASET_DIR = Path(PROJECT_ROOT/ "Model"/ "SFA3D"/ "dataset"/ "kitti")

CHECKPOINT = Path(PROJECT_ROOT / "Model" / "SFA3D" /
    "checkpoints" / "fpn_resnet_18" / "fpn_resnet_18_epoch_300.pth")

NUM_CLASSES = 3
MAX_OBJECTS = 50

INPUT_SIZE = (608, 608)
HM_SIZE = (152, 152)

PEAK_THRESH = 0.2

# Maximum center distance for matching
MATCH_DISTANCE = 2.0

CLASS_NAMES = {
    0: "Pedestrian",
    1: "Car",
    2: "Cyclist"}


# DEVICE
device = torch.device("cpu")

print("=" * 70)
print("SFA3D Evaluation")
print("=" * 70)
print("Device:", device)


# MODEL CONFIG
configs = edict()

configs.dataset_dir = DATASET_DIR
configs.input_size = INPUT_SIZE
configs.hm_size = HM_SIZE
configs.num_classes = NUM_CLASSES
configs.max_objects = MAX_OBJECTS

configs.arch = "fpn_resnet_18"
configs.head_conv = 64
configs.imagenet_pretrained = False

configs.heads = {
    "hm_cen": NUM_CLASSES,
    "cen_offset": 2,
    "direction": 2,
    "z_coor": 1,
    "dim": 3}


# CREATE MODEL
print("\nLoading model...")

model = create_model(configs)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device)

model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("MODEL READY!")


# DATASET
dataset = KittiDataset(
    configs,
    mode="val",
    hflip_prob=0.0)

print("\nValidation samples:", len(dataset))


# MATCHING FUNCTION
def match_detections(predictions, ground_truths):

    """
    Match Prediction and Ground Truth using:

    1. Same class
    2. Center distance <= MATCH_DISTANCE

    Prediction format:
    [class_id, x, y, z, h, w, l, yaw]

    Ground Truth format:
    [class_id, x, y, z, h, w, l, yaw]
    """

    matched_predictions = set()
    matched_ground_truths = set()

    errors = []

    for gt_index, gt in enumerate(ground_truths):

        gt_class = int(gt[0])

        best_prediction = None
        best_distance = float("inf")

        for pred_index, pred in enumerate(predictions):

            if pred_index in matched_predictions:
                continue

            pred_class = int(pred[0])

            # Classes must match
            if pred_class != gt_class:
                continue

            # Center distance in XY plane
            dx = pred[1] - gt[1]
            dy = pred[2] - gt[2]

            distance = np.sqrt(
                dx * dx +
                dy * dy)

            if distance < best_distance:

                best_distance = distance
                best_prediction = pred_index

        # Match accepted
        if (best_prediction is not None
            and best_distance <= MATCH_DISTANCE):

            matched_predictions.add(best_prediction)
            matched_ground_truths.add(gt_index)

            errors.append(best_distance)

    true_positive = len(matched_ground_truths)

    false_positive = (
        len(predictions) - true_positive)

    false_negative = (
        len(ground_truths) - true_positive)

    return (
        true_positive,
        false_positive,
        false_negative,
        errors)



# RESULTS
results = []

total_tp = 0
total_fp = 0
total_fn = 0

all_errors = []


# MAIN EVALUATION LOOP
for index in range(len(dataset)):

    sample_id = dataset.sample_id_list[index]

    print(f"\n[{index + 1:03d}/{len(dataset)}] "
        f"Sample {sample_id:06d}")

    
    # LOAD SAMPLE
    sample = dataset[index]

    # Official validation output:
    # metadata, bev_map, targets
    metadata = sample[0]
    bev_map = sample[1]

    
    # MODEL INPUT
    bev_map = bev_map.unsqueeze(0)

    # Dataset BEV can be float64.
    # Model expects float32.

    bev_map = bev_map.float()
    bev_map = bev_map.to(device)

    
    # INFERENCE
    with torch.no_grad():
        outputs = model(bev_map)


    # DECODE
    detections = decode(
        outputs["hm_cen"],
        outputs["cen_offset"],
        outputs["direction"],
        outputs["z_coor"],
        outputs["dim"],
        K=50)

    
    # POST PROCESSING
    detections = (
        detections
        .detach()
        .cpu()
        .numpy())

    detections = post_processing(
        detections,
        num_classes=NUM_CLASSES,
        down_ratio=4,
        peak_thresh=PEAK_THRESH)

    # Batch size = 1
    detections = detections[0]

    
    # CONVERT TO REAL LIDAR VALUES
    predictions = convert_det_to_real_values(
        detections,
        num_classes=NUM_CLASSES)

    if len(predictions) == 0:

        predictions = np.empty(
            (0, 8),
            dtype=np.float32)


    # GROUND TRUTH
    labels, has_labels = dataset.get_label(sample_id)

    ground_truth = []

    if has_labels:

        calib = dataset.get_calib(sample_id)

        labels = np.asarray(labels)

        # IMPORTANT:
        # Make sure labels are always (N, 8)
        # [class_id, x, y, z, h, w, l, ry]

        if labels.size > 0:

            if labels.ndim == 1:

                labels = labels.reshape(1, -1)

            for label in labels:

                # Flatten just in case
                label = np.asarray(label).reshape(-1)

                # Need:
                # class_id + 7 box values

                if label.size < 8:
                    continue

                class_id = int(label[0])

                # Only SFA3D classes
                if class_id not in CLASS_NAMES:
                    continue

                # KITTI CAMERA BOX
                # [x, y, z, h, w, l, ry]
                camera_box = np.asarray(
                    label[1:8],
                    dtype=np.float32)

                if camera_box.size != 7:
                    continue

                # SFA3D camera_to_lidar_box expects
                # an array of boxes with shape (N, 7).
                # Therefore, make this single box (1, 7).
                camera_box = camera_box.reshape(1, 7)

                lidar_boxes = camera_to_lidar_box(
                    camera_box,
                    calib.V2C,
                    calib.R0,
                    calib.P2)

                # Extract the single converted box
                lidar_box = np.asarray(
                    lidar_boxes[0],
                    dtype=np.float32
                ).reshape(-1)

                if lidar_box.size != 7:
                    continue

                x = float(lidar_box[0])
                y = float(lidar_box[1])
                z = float(lidar_box[2])

                
                # SAME SFA3D BOUNDARY
                if not (
                    0 <= x <= 50
                    and
                    -25 <= y <= 25
                    and
                    -2.73 <= z <= 1.27
                ):
                    continue

                ground_truth.append(
                    np.concatenate([
                        np.array(
                            [class_id],
                            dtype=np.float32
                        ),
                        lidar_box
                    ]))

    
    # CONVERT GT LIST TO ARRAY
    ground_truth = np.asarray(
        ground_truth,
        dtype=np.float32)

    if ground_truth.size == 0:

        ground_truth = np.empty(
            (0, 8),
            dtype=np.float32)

    elif ground_truth.ndim == 1:

        ground_truth = ground_truth.reshape(1, -1)

    
    # MATCH PREDICTIONS WITH GT
    tp, fp, fn, errors = match_detections(
        predictions,
        ground_truth)

    total_tp += tp
    total_fp += fp
    total_fn += fn

    all_errors.extend(errors)


    # PER-FRAME RESULT
    frame_mean_error = (
        np.mean(errors)
        if errors
        else "")

    results.append({
        "sample_id": sample_id,
        "ground_truth": len(ground_truth),
        "predictions": len(predictions),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "mean_position_error": frame_mean_error})

    print(
        f"GT={len(ground_truth)} | "
        f"Pred={len(predictions)} | "
        f"TP={tp} | "
        f"FP={fp} | "
        f"FN={fn}")



# FINAL METRICS
precision = (
    total_tp /
    (total_tp + total_fp)
    if (total_tp + total_fp) > 0
    else 0)

recall = (
    total_tp /
    (total_tp + total_fn)
    if (total_tp + total_fn) > 0
    else 0)

mean_position_error = (
    np.mean(all_errors)
    if len(all_errors) > 0
    else 0)



# SAVE CSV
csv_path = (Path(PROJECT_ROOT/"Output"/"evaluation_results.csv"))

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sample_id",
            "ground_truth",
            "predictions",
            "true_positive",
            "false_positive",
            "false_negative",
            "mean_position_error"])

    writer.writeheader()
    writer.writerows(results)


# FINAL OUTPUT
print("\n")

print("=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

print("Validation frames:", len(dataset))

print("Ground Truth:", total_tp + total_fn)

print("Predictions:", total_tp + total_fp)

print("True Positive:", total_tp)

print("False Positive:", total_fp)

print("False Negative:", total_fn)

print(f"Precision: {precision:.4f}")

print(f"Recall:    {recall:.4f}")

print(f"Mean Position Error: {mean_position_error:.3f} m")

print("=" * 70)

print(f"\nResults saved to: {csv_path}")