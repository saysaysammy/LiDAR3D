import sys
import os
import subprocess
import numpy as np
import customtkinter as ctk
from tkinter import filedialog

def resource_path(relative_path):
    if getattr(sys, "frozen", False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)

# Appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


# Main Window
app = ctk.CTk()
app.title("LiDAR 3D - Powered by SFA3D")
app.iconbitmap(resource_path("Resources/LiDAR3D.ico"))
app.geometry("850x650")
app.minsize(850,650)


# Title
title = ctk.CTkLabel(
    app,
    text="LiDAR 3D",
    font=ctk.CTkFont(size=32, weight="bold"))
title.pack(pady=(40, 5))


subtitle = ctk.CTkLabel(
    app,
    text="3D Object Detection using LiDAR",
    font=ctk.CTkFont(size=14))
subtitle.pack(pady=(0, 30))


# Main Frame
main_frame = ctk.CTkFrame(app)
main_frame.pack(
    fill="both",
    expand=True,
    padx=40,
    pady=20)

main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=1)
main_frame.grid_rowconfigure(0, weight=1)


# LEFT SIDE
left_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
left_frame.grid(
    row=0,
    column=0,
    sticky="nsew",
    padx=(35, 20),
    pady=30)


# Input Mode
mode_label = ctk.CTkLabel(
    left_frame,
    text="Input Mode")
mode_label.pack(pady=(0, 10))

mode_var = ctk.StringVar(value="KITTI Sample")


# Fixed input area
input_frame = ctk.CTkFrame(
    left_frame,
    fg_color="transparent",
    height=95)
input_frame.pack_propagate(False)

input_label = ctk.CTkLabel(
    input_frame,
    text="Sample ID")
input_label.pack(pady=(0, 10))

input_row = ctk.CTkFrame(
    input_frame,
    fg_color="transparent")
input_row.pack()

sample_entry = ctk.CTkEntry(
    input_row,
    width=250,
    placeholder_text="Example : 0 to 999")
sample_entry.pack()


# Custom BIN
bin_frame = ctk.CTkFrame(
    input_row,
    fg_color="transparent")

bin_entry = ctk.CTkEntry(
    bin_frame,
    width=150,
    placeholder_text="Select a .bin file")
bin_entry.pack(side="left", padx=(0, 10))


status_label = ctk.CTkLabel(
    left_frame,
    text="Ready",
    font=ctk.CTkFont(size=14))


def browse_bin():
    path = filedialog.askopenfilename(
        title="Select LiDAR BIN file",
        filetypes=[
            ("LiDAR BIN files", "*.bin"),
            ("All files", "*.*")])

    if path:
        bin_entry.delete(0, "end")
        bin_entry.insert(0, path)
        status_label.configure(
            text=f"Selected: {os.path.basename(path)}")


browse_button = ctk.CTkButton(
    bin_frame,
    text="Browse",
    width=90,
    command=browse_bin)
browse_button.pack(side="left")


def update_input_mode(choice=None):
    sample_entry.delete(0, "end")
    bin_entry.delete(0, "end")

    sample_entry.configure(
        placeholder_text="Example : 0 to 999")

    bin_entry.configure(
        placeholder_text="Select a .bin file")
    
    status_label.configure(text="Ready")
    result_label.configure(text="No detection")

    if mode_var.get() == "KITTI Sample":
        input_label.configure(text="Sample ID")
        bin_frame.pack_forget()
        sample_entry.pack()
    else:
        input_label.configure(text="Custom Input")
        sample_entry.pack_forget()
        bin_frame.pack()

    input_frame.focus_set()
        
    
mode_menu = ctk.CTkOptionMenu(
    left_frame,
    variable=mode_var,
    values=["KITTI Sample", "Custom BIN"],
    width=250,
    command=update_input_mode)

mode_menu.pack(pady=(0, 5))
input_frame.pack(fill="x", pady=(15, 0))


# Paths
def get_project_paths():
    if getattr(sys, "frozen", False):
        project_root = os.path.dirname(sys.executable)
        python_path = None
        lidar_path = os.path.join(project_root, "Resources", "lidar", "lidar.exe")
        gt_path = os.path.join(project_root, "Resources", "lidar", "ground_truth.exe")
        viewer_path = os.path.join(project_root, "Resources", "viewer", "viewer.exe")
        vtk_path = os.path.join(project_root, "Resources", "viewer", "export_vtk.exe")
    else:
        project_root = os.path.dirname(os.path.abspath(__file__))
        python_path = os.path.join(project_root, "sfa3d_env", "Scripts", "python.exe")
        lidar_path = os.path.join(project_root, "Resources", "lidar.py")
        gt_path = os.path.join(project_root, "Resources", "ground_truth.py")
        viewer_path = os.path.join(project_root, "Resources", "viewer.py")
        vtk_path = os.path.join(project_root, "Resources", "export_vtk.py")

    output_path = os.path.join(project_root, "Output")
    return (project_root, python_path, lidar_path, gt_path, viewer_path, output_path, vtk_path)

def get_sample_id():
    sample_id = sample_entry.get().strip()

    if not sample_id.isdigit():
        status_label.configure(
            text="Please enter a valid Sample ID")
        return None

    sample_id = int(sample_id)

    velodyne_path = resource_path(f"Model/SFA3D/dataset/kitti/training/velodyne/{sample_id:06d}.bin")
    label_path = resource_path(f"Model/SFA3D/dataset/kitti/training/label_2/{sample_id:06d}.txt")
    calib_path = resource_path(f"Model/SFA3D/dataset/kitti/training/calib/{sample_id:06d}.txt")
    required_files = [velodyne_path, label_path, calib_path]

    missing_files = [
        path for path in required_files
        if not os.path.exists(path)]

    if missing_files:
        result_label.configure(text="Sample Not Found!"
            f"\nSample ID : {sample_id}"
            "\n\nThis sample is not"
            "\navailable in the dataset."
            "\n\nDataset path:" 
            "\nModel/SFA3D/"
            "\ndataset/kitti/training")
        return None
    
    return sample_id

def run_script(script_path, python_path, args, wait=False):
    if getattr(sys, "frozen", False):
        command = [script_path] + args
    else:
        command = [python_path, script_path] + args
    if wait:
        return subprocess.run(command, check=True)
    return subprocess.Popen(command)

def get_prediction_path():
    project_root, _, _, _, _, output_path, _ = get_project_paths()

    if mode_var.get() == "KITTI Sample":
        sample_id = get_sample_id()

        if sample_id is None:
            return None

        return os.path.join(
            output_path,
            f"{sample_id:06d}_real_detections.npy")

    bin_path = bin_entry.get().strip()

    if not bin_path:
        status_label.configure(
            text="Please select a .bin file")
        return None

    if not os.path.isfile(bin_path):
        status_label.configure(
            text="Selected BIN file does not exist")
        return None

    stem = os.path.splitext(
        os.path.basename(bin_path))[0]

    return os.path.join(
        output_path,
        f"custom_{stem}_real_detections.npy")

# Detection Result
result_frame = ctk.CTkFrame(main_frame)
result_frame.grid(
    row=0,
    column=1,
    sticky="nsew",
    padx=(20, 35),
    pady=30)

result_title = ctk.CTkLabel(
    result_frame,
    text="Detection Result :",
    font=ctk.CTkFont(size=20, weight="bold"))
result_title.pack(pady=(60, 25))

result_label = ctk.CTkLabel(
    result_frame,
    text="No detection",
    font=ctk.CTkFont(size=15),
    justify="left")
result_label.pack(pady=10, padx=10)


def show_detection_result(prediction_path):
    try:
        predictions = np.load(prediction_path)

        if predictions.ndim == 1 and predictions.size == 0:
            predictions = predictions.reshape(0, 8)

        if predictions.ndim != 2 or predictions.shape[1] < 8:
            result_label.configure(
                text="Invalid detection output")
            return

        total = len(predictions)

        # SFA3D class IDs:
        class_names = {
            0: "Pedestrian",
            1: "Car",
            2: "Cyclist"}

        counts = {0: 0, 1: 0, 2: 0}

        for detection in predictions:
            class_id = int(detection[0])
            if class_id in counts:
                counts[class_id] += 1

        text = (
            f"Objects detected: {total}\n\n"
            f"Pedestrian: {counts[0]}\n"
            f"Car: {counts[1]}\n"
            f"Cyclist: {counts[2]}")

        result_label.configure(text=text)

    except Exception as e:
        result_label.configure(
            text="Could not read detection result")
        print(e)


# Run Detection
def check_detection(prediction_path, process):
    """Wait for test_lidar.py to finish and then show the result."""

    if os.path.isfile(prediction_path):
        try:
            # Make sure the file can actually be read before showing it.
            np.load(prediction_path)
            show_detection_result(prediction_path)
            status_label.configure(text="Detection completed")
            run_button.configure(state="normal")
            view_button.configure(state="normal")
            vtk_button.configure(state="normal")
            return
        except Exception:
            pass

    if process.poll() is not None:
        status_label.configure(text="Detection failed")
        run_button.configure(state="normal")
        view_button.configure(state="normal")
        vtk_button.configure(state="normal")
        return

    app.after(500, check_detection, prediction_path, process)


def run_detection():
    project_root, python_path, lidar_path, _, _, _, _ = get_project_paths()

    try:
        prediction_path = get_prediction_path()

        if prediction_path is None:
            return

        # Remove an old result so the UI never displays stale detections.
        if os.path.isfile(prediction_path): os.remove(prediction_path)

        if mode_var.get() == "KITTI Sample":
            sample_id = get_sample_id()
            if sample_id is None:
                return

            process = run_script(lidar_path, python_path, ["--sample_id", str(sample_id)])

            status_label.configure(
                text=f"Detecting {sample_id:06d}...")

        else:
            bin_path = bin_entry.get().strip()

            if not bin_path or not os.path.isfile(bin_path):
                status_label.configure(
                    text="Please select a valid .bin file")
                return

            process = run_script(lidar_path, python_path, ["--bin_path", bin_path])

            status_label.configure(
                text=f"Detecting {os.path.basename(bin_path)}...")

        result_label.configure(text="Detecting...")
        run_button.configure(state="disabled")
        view_button.configure(state="disabled")
        vtk_button.configure(state="disabled")

        app.after(500, check_detection, prediction_path, process)

    except Exception as e:
        status_label.configure(
            text="Could not run detection")
        print(e)


# View 3D Result
def view_3d_result():
    project_root, python_path, _, gt_path, viewer_path, _, _ = get_project_paths()
    try:
        if mode_var.get() == "KITTI Sample":
            sample_id = get_sample_id()
            if sample_id is None:
                return

            prediction_path = os.path.join(
                project_root,
                "Output",
                f"{sample_id:06d}_real_detections.npy")

            if not os.path.isfile(prediction_path):
                status_label.configure(
                    text="Run Detection first")
                return

            # Generate Ground Truth before opening the viewer.
            run_script(gt_path, python_path, ["--sample_id", str(sample_id)], wait=True)
            run_script(viewer_path, python_path, ["--sample_id", str(sample_id)])

            status_label.configure(
                text=f"Opening 3D view: {sample_id:06d}")

        else:
            bin_path = bin_entry.get().strip()

            if not bin_path:
                status_label.configure(
                    text="Please select a .bin file")
                return

            if not os.path.isfile(bin_path):
                status_label.configure(
                    text="Selected BIN file does not exist")
                return

            stem = os.path.splitext(
                os.path.basename(bin_path))[0]

            prediction_path = os.path.join(
                project_root,
                "Output",
                f"custom_{stem}_real_detections.npy")

            if not os.path.isfile(prediction_path):
                status_label.configure(
                    text="Run Detection first")
                return

            run_script(viewer_path, python_path, ["--bin_path", bin_path, "--prediction_path", prediction_path])

            status_label.configure(
                text=f"Opening 3D view: {stem}")

    except Exception as e:
        status_label.configure(
            text="Could not open 3D viewer")
        print(e)


# Export VTK
def export_vtk():
    (project_root, python_path, _, gt_path, _,
    output_path, vtk_path) = get_project_paths()

    try:
        if mode_var.get() == "KITTI Sample":

            sample_id = get_sample_id()

            if sample_id is None:
                return

            prediction_path = os.path.join(
                output_path,
                f"{sample_id:06d}_real_detections.npy")

            ground_truth_path = os.path.join(
                output_path,
                f"{sample_id:06d}_ground_truth.npy")

            if not os.path.isfile(prediction_path):
                status_label.configure(
                    text="Run Detection first")
                return

            # Generate Ground Truth if it doesn't exist
            if not os.path.isfile(ground_truth_path):
                run_script(
                        gt_path,
                        python_path,
                        ["--sample_id", str(sample_id)],
                    wait=True)

            # Ask user where to save the VTK files
            save_folder = filedialog.askdirectory(
                title="Select folder to save VTK files")

            if not save_folder:
                return

            # Export VTK files to selected folder
            run_script(vtk_path, python_path, [
                "--sample_id", str(sample_id),
                "--prediction_path", prediction_path,
                "--ground_truth_path", ground_truth_path,
                "--output_dir", save_folder
            ], wait=True)

            status_label.configure(
                text=f"VTK exported to: {os.path.basename(save_folder)}")

        else:

            bin_path = bin_entry.get().strip()

            if not bin_path:
                status_label.configure(
                    text="Please select a .bin file")
                return

            if not os.path.isfile(bin_path):
                status_label.configure(
                    text="Selected BIN file does not exist")
                return

            stem = os.path.splitext(
                os.path.basename(bin_path))[0]

            prediction_path = os.path.join(
                output_path,
                f"custom_{stem}_real_detections.npy")

            if not os.path.isfile(prediction_path):
                status_label.configure(
                    text="Run Detection first")
                return
            
            save_folder = filedialog.askdirectory(
                title="Select folder to save VTK files")

            if not save_folder:
                return
            
            # Export VTK file to selected folder
            run_script(vtk_path, python_path, [
                "--bin_path", bin_path,
                "--prediction_path", prediction_path,
                "--output_dir", save_folder
            ], wait=True)

            status_label.configure(
                text=f"VTK exported to: {os.path.basename(save_folder)}")

    except Exception as e:
        status_label.configure(
            text="Could not export VTK")
        print(e)


# Buttons
run_button = ctk.CTkButton(
    left_frame,
    text="Run Detection",
    width=250,
    height=45,
    command=run_detection)
run_button.pack(pady=(0, 10))


view_button = ctk.CTkButton(
    left_frame,
    text="View 3D Result",
    width=250,
    height=45,
    command=view_3d_result)
view_button.pack(pady=10)


vtk_button = ctk.CTkButton(
    left_frame,
    text="Export as VTK",
    width=250,
    height=45,
    command=export_vtk)
vtk_button.pack(pady=10)

status_label.pack(pady=10)

update_input_mode()
app.mainloop()