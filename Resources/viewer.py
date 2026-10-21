import sys
import numpy as np
import argparse
import pyvista as pv
from pathlib import Path


# CONFIG
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
    help="Custom LiDAR .bin file")

parser.add_argument(
    "--prediction_path",
    type=str,
    default=None,
    help="Custom prediction .npy file")

args = parser.parse_args()

SAMPLE_ID = args.sample_id
CUSTOM_BIN_PATH = args.bin_path
CUSTOM_PREDICTION_PATH = args.prediction_path
IS_CUSTOM = CUSTOM_BIN_PATH is not None

if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent.parent.parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent


# PATHS
if IS_CUSTOM:

    BIN_PATH = Path(CUSTOM_BIN_PATH)

    if CUSTOM_PREDICTION_PATH is not None:

        PREDICTION_PATH = Path(
            CUSTOM_PREDICTION_PATH)

    else:

        PREDICTION_PATH = Path(
        PROJECT_ROOT / "Output" / f"custom_{BIN_PATH.stem}_real_detections.npy")

    GROUND_TRUTH_PATH = None

else:

    BIN_PATH = Path(
        PROJECT_ROOT / "Model" / "SFA3D" / "dataset" /
        "kitti" / "training" / "velodyne" / f"{SAMPLE_ID:06d}.bin")

    PREDICTION_PATH = Path(
        PROJECT_ROOT / "Output" / f"{SAMPLE_ID:06d}_real_detections.npy")

    GROUND_TRUTH_PATH = Path(
        PROJECT_ROOT / "Output" / f"{SAMPLE_ID:06d}_ground_truth.npy")


# CHECK FILES
if not BIN_PATH.exists():

    raise FileNotFoundError(
        f"LiDAR BIN file not found:\n{BIN_PATH}")


if not PREDICTION_PATH.exists():

    raise FileNotFoundError(
        f"Prediction file not found:\n{PREDICTION_PATH}")


if not IS_CUSTOM:

    if not GROUND_TRUTH_PATH.exists():

        raise FileNotFoundError(
            f"Ground Truth file not found:\n"
            f"{GROUND_TRUTH_PATH}")



# SFA3D BOUNDARY
MIN_X, MAX_X = 0, 50
MIN_Y, MAX_Y = -25, 25
MIN_Z, MAX_Z = -2.73, 1.27


# CLASS INFORMATION
CLASS_NAMES = {
    0: "Pedestrian",
    1: "Car",
    2: "Cyclist"}

CLASS_COLORS = {
    0: "blue",
    1: "red",
    2: "green"}


# BOX EDGES
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)]


# CREATE 3D BOX
def create_box(box):

    class_id, x, y, z, h, w, l, yaw = box

    # Local coordinates
    corners = np.array([
        [-l / 2,  w / 2,  0],
        [-l / 2, -w / 2,  0],
        [ l / 2, -w / 2,  0],
        [ l / 2,  w / 2,  0],

        [-l / 2,  w / 2,  h],
        [-l / 2, -w / 2,  h],
        [ l / 2, -w / 2,  h],
        [ l / 2,  w / 2,  h],])

    # Rotation around Z axis
    rotation = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]])

    corners = corners @ rotation.T

    # Translation
    corners[:, 0] += x
    corners[:, 1] += y
    corners[:, 2] += z

    # Create line representation
    lines = []

    for a, b in BOX_EDGES:
        lines.extend([2, a, b])

    return pv.PolyData(
        corners,
        lines=np.array(lines))



# LOAD POINT CLOUD
points = np.fromfile(
    BIN_PATH,
    dtype=np.float32)

if points.size % 4 != 0:

    raise ValueError(
        "Invalid .bin file. "
        "Expected float32 values in groups of 4.")

points = points.reshape(-1, 4)
points_xyz = points[:, :3]


# APPLY SFA3D BOUNDARY
mask = (
    (points[:, 0] >= MIN_X) & (points[:, 0] <= MAX_X) &
    (points[:, 1] >= MIN_Y) & (points[:, 1] <= MAX_Y) &
    (points[:, 2] >= MIN_Z) & (points[:, 2] <= MAX_Z))

points_xyz = points_xyz[mask]


# LOAD PREDICTIONS
predictions = np.load(PREDICTION_PATH)


# LOAD GROUND TRUTH
if IS_CUSTOM:

    ground_truth = np.empty(
        (0, 8),
        dtype=np.float32)
else:
    ground_truth = np.load(
        GROUND_TRUTH_PATH)


# CREATE POINT CLOUD
point_cloud = pv.PolyData(points_xyz)


# CREATE PLOTTER
if IS_CUSTOM:

    # Custom BIN → single panel
    plotter = pv.Plotter(
        window_size=(1000, 500))

else:

    # KITTI → two panels
    plotter = pv.Plotter(
        shape=(1, 2),
        window_size=(1000, 500))


# PANEL 1
# SFA3D PREDICTIONS
plotter.subplot(0, 0)

if IS_CUSTOM:

    plotter.add_text(
        "SFA3D Predictions - Custom BIN",
        position="upper_edge",
        font_size=10)

else:

    plotter.add_text(
        f"SFA3D Predictions - KITTI {SAMPLE_ID:06d}",
        position="upper_edge",
        font_size=10)


plotter.add_mesh(
    point_cloud,
    color="lightgray",
    point_size=2,
    render_points_as_spheres=True)


# Prediction boxes
prediction_labels = []
prediction_count = {0: 0, 1: 0, 2: 0}


for detection in predictions:

    class_id = int(detection[0])

    if class_id not in CLASS_NAMES:
        continue

    prediction_count[class_id] += 1

    color = CLASS_COLORS[class_id]

    box = create_box(detection)

    plotter.add_mesh(
        box,
        color=color,
        line_width=4)

    x = detection[1]
    y = detection[2]
    z = detection[3]

    distance = np.sqrt(x ** 2 + y ** 2)

    label = (
        f"{CLASS_NAMES[class_id]} "
        f"#{prediction_count[class_id]}\n"
        f"{distance:.1f} m")

    prediction_labels.append((
        np.array([x, y, z]), label))


# Prediction labels
for position, label in prediction_labels:

    plotter.add_point_labels(
        np.array([position]),
        [label],
        point_size=8,
        font_size=11,
        shape_opacity=0.65,
        always_visible=True)


# Prediction legend
plotter.add_legend(
    [
        ["Car", "red"],
        ["Pedestrian", "blue"],
        ["Cyclist", "green"]
    ],
    bcolor="white")

plotter.add_axes()


# CUSTOM BIN
if IS_CUSTOM:

    plotter.show()

    raise SystemExit


# PANEL 2
# GROUND TRUTH
plotter.subplot(0, 1)
plotter.add_text(
    f"KITTI Ground Truth - {SAMPLE_ID:06d}",
    position="upper_edge",
    font_size=10)

plotter.add_mesh(
    point_cloud,
    color="lightgray",
    point_size=2,
    render_points_as_spheres=True)

    
# Ground Truth boxes
gt_labels = []
gt_count = {0: 0, 1: 0, 2: 0}

for gt in ground_truth:

    class_id = int(gt[0])

    if class_id not in CLASS_NAMES:
        continue

    gt_count[class_id] += 1

    color = CLASS_COLORS[class_id]

    box = create_box(gt)

    plotter.add_mesh(
        box,
        color=color,
        line_width=4)

    x = gt[1]
    y = gt[2]
    z = gt[3]

    distance = np.sqrt(x ** 2 + y ** 2)

    label = (
        f"GT {CLASS_NAMES[class_id]} "
        f"#{gt_count[class_id]}\n"
        f"{distance:.1f} m")

    gt_labels.append((
        np.array([x, y, z]),label))

    
# Ground Truth labels
for position, label in gt_labels:

        plotter.add_point_labels(
            np.array([position]),
            [label],
            point_size=8,
            font_size=11,
            shape_opacity=0.65,
            always_visible=True)


# Ground Truth legend
plotter.add_legend(
    [
        ["Car", "red"],
        ["Pedestrian", "blue"],
        ["Cyclist", "green"]
    ],
    bcolor="white")

plotter.add_axes()


# LINK CAMERA MOVEMENT
plotter.link_views()
plotter.show()