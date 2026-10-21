import sys
import numpy as np
import pyvista as pv
import argparse
from pathlib import Path

MIN_X, MAX_X = 0, 50
MIN_Y, MAX_Y = -25, 25
MIN_Z, MAX_Z = -2.73, 1.27

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)]

parser = argparse.ArgumentParser(
    description="Export SFA3D results to VTK")

parser.add_argument("--sample_id", type=int, default=None)
parser.add_argument("--bin_path", type=str, default=None)
parser.add_argument("--prediction_path", type=str, default=None)
parser.add_argument("--ground_truth_path", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
args = parser.parse_args()


if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent.parent.parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "Output"
EXPORT_DIR = (
    Path(args.output_dir)
    if args.output_dir
    else OUTPUT_DIR)

EXPORT_DIR.mkdir(parents=True, exist_ok=True)

KITTI_BIN_DIR = (PROJECT_ROOT / "Model" / "SFA3D" /
    "dataset" / "kitti" / "training" / "velodyne")

def create_box(box):
    _, x, y, z, h, w, l, yaw = box

    corners = np.array([
        [-l/2,  w/2, 0], [-l/2, -w/2, 0],
        [ l/2, -w/2, 0], [ l/2,  w/2, 0],
        [-l/2,  w/2, h], [-l/2, -w/2, h],
        [ l/2, -w/2, h], [ l/2,  w/2, h]
    ], dtype=np.float32)

    rotation = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]])

    corners = corners @ rotation.T
    corners[:, 0] += x
    corners[:, 1] += y
    corners[:, 2] += z

    lines = []
    for a, b in BOX_EDGES:
        lines.extend([2, a, b])

    return pv.PolyData(
        corners,
        lines=np.array(lines, dtype=np.int64))


def load_point_cloud(bin_path):
    data = np.fromfile(bin_path, dtype=np.float32)

    if data.size == 0:
        raise ValueError("The BIN file is empty.")

    if data.size % 4 != 0:
        raise ValueError(
            "Invalid BIN file: float count is not divisible by 4.")

    points = data.reshape(-1, 4)[:, :3]

    mask = (
        (points[:, 0] >= MIN_X) & (points[:, 0] <= MAX_X) &
        (points[:, 1] >= MIN_Y) & (points[:, 1] <= MAX_Y) &
        (points[:, 2] >= MIN_Z) & (points[:, 2] <= MAX_Z))

    return points[mask]


def load_boxes(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found:\n{path}")

    data = np.load(path)

    if data.size == 0:
        return np.empty((0, 8), dtype=np.float32)

    if data.ndim != 2 or data.shape[1] != 8:
        raise ValueError(
            f"Invalid NPY format: expected (N, 8), got {data.shape}")

    return data


def boxes_to_polydata(boxes):
    if len(boxes) == 0:
        return pv.PolyData()

    result = None

    for box in boxes:
        mesh = create_box(box)
        mesh["ClassID"] = np.full(
            mesh.n_cells, int(box[0]), dtype=np.int8)

        result = mesh if result is None else result.merge(mesh)

    return result


def export_vtk(bin_path, boxes, output_path):
    points = load_point_cloud(bin_path)

    # Explicit VTK vertices for the point cloud.
    cloud = pv.PolyData(points)
    cloud["PointType"] = np.zeros(
        len(points), dtype=np.int8)

    boxes_mesh = boxes_to_polydata(boxes)

    if boxes_mesh.n_points > 0:
        result = cloud.merge(boxes_mesh)
    else:
        result = cloud

    result.save(output_path)


# Input validation
if args.sample_id is not None and args.bin_path is not None:
    raise ValueError(
        "Use either --sample_id or --bin_path, not both.")

if args.sample_id is None and args.bin_path is None:
    raise ValueError(
        "Provide either --sample_id or --bin_path.")


# KITTI mode
if args.sample_id is not None:

    name = f"{args.sample_id:06d}"

    bin_path = KITTI_BIN_DIR / f"{name}.bin"

    prediction_path = (
        Path(args.prediction_path)
        if args.prediction_path
        else OUTPUT_DIR / f"{name}_real_detections.npy")

    ground_truth_path = (
        Path(args.ground_truth_path)
        if args.ground_truth_path
        else OUTPUT_DIR / f"{name}_ground_truth.npy")

    if not bin_path.exists():
        raise FileNotFoundError(
            f"KITTI BIN file not found:\n{bin_path}")

    predictions = load_boxes(prediction_path)
    ground_truth = load_boxes(ground_truth_path)

    detection_output = EXPORT_DIR / f"{name}_detection.vtk"
    ground_truth_output = EXPORT_DIR / f"{name}_ground_truth.vtk"

    export_vtk(bin_path, predictions, detection_output)
    export_vtk(bin_path, ground_truth, ground_truth_output)


# Custom BIN mode
else:
    bin_path = Path(args.bin_path)

    if not bin_path.exists():
        raise FileNotFoundError(
            f"Custom BIN file not found:\n{bin_path}")

    stem = bin_path.stem

    prediction_path = (
        Path(args.prediction_path)
        if args.prediction_path
        else OUTPUT_DIR / f"custom_{stem}_real_detections.npy")

    predictions = load_boxes(prediction_path)

    output_path = (
        EXPORT_DIR / f"custom_{stem}_detection.vtk")

    export_vtk(bin_path, predictions, output_path)