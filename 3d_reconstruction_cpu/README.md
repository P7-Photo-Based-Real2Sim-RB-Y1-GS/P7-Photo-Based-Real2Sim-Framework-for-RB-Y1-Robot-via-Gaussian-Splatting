# 3D Reconstruction CPU

A Python program for reconstructing an RGB-D turntable dataset into 3D assets in a CPU-only environment.

This project uses RGB images, depth maps, object masks, camera intrinsics, and frame angle metadata to isolate the target object, estimate the turntable rotation axis, and generate both STL files for simulation and GLB files with visual appearance.

## Features

* Object-only point cloud generation using RGB, depth, and mask data
* Pose generation based on turntable angle metadata
* Automatic estimation of the rotation-axis pivot
* Automatic selection of the rotation direction
* Foreground depth filtering
* Expanded bounding-box depth masking to handle RGB/depth misalignment
* Model-to-frame ICP refinement
* Watertight mesh generation using visual hull reconstruction
* Vertex-colored GLB generation using best-view RGB projection
* Export support for STL, PLY, and GLB files

## Dataset Layout

The user dataset folder must follow the structure below.

```text
<YOUR_DATASET_PATH>/
  rgb/
    000000.png
    000001.png
    ...
  depth/
    000000.png
    000001.png
    ...
  masks/
    000000.png
    000001.png
    ...
  cam_K.txt
  frame_index.csv
  conversion_summary.json   # optional
```

The `frame_index.csv` file must include either `angle_deg` or `angle_deg_unwrapped` for each frame in order to generate stable turntable poses.

## Installation

Python 3.12 is recommended.

```powershell
git clone https://github.com/P7-Photo-Based-Real2Sim-RB-Y1-GS/P7-Photo-Based-Real2Sim-Framework-for-RB-Y1-Robot-via-Gaussian-Splatting.git
cd P7-Photo-Based-Real2Sim-Framework-for-RB-Y1-Robot-via-Gaussian-Splatting\3d_reconstruction_cpu

py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

If virtual environment activation is blocked due to the PowerShell execution policy, you can call the Python executable directly without activation, as shown above with `& .\.venv\Scripts\python.exe`.

## Quick Check

Replace `<YOUR_DATASET_PATH>` with the path to your own dataset folder and run the following command.

```powershell
& .\.venv\Scripts\python.exe .\reconstruct_rgbd_object.py `
  --dataset "<YOUR_DATASET_PATH>" `
  --dry-run
```

Example:

```powershell
& .\.venv\Scripts\python.exe .\reconstruct_rgbd_object.py `
  --dataset "D:\datasets\object_001" `
  --dry-run
```

## Recommended Run

The easiest way to run the reconstruction pipeline is to use the PowerShell wrapper.

```powershell
.\run_reconstruction.ps1 `
  -Dataset "<YOUR_DATASET_PATH>" `
  -Output "<YOUR_OUTPUT_PATH>" `
  -InstallRequirements `
  -CleanOutput
```

Example:

```powershell
.\run_reconstruction.ps1 `
  -Dataset "D:\datasets\object_001" `
  -Output "D:\datasets\object_001\reconstruction_realistic_bestview" `
  -InstallRequirements `
  -CleanOutput
```

If `-Output` is omitted, the results will be generated in `<YOUR_DATASET_PATH>\reconstruction_realistic_bestview`.

## Python Direct Run

You can also run the Python script directly instead of using the PowerShell wrapper.

```powershell
& .\.venv\Scripts\python.exe .\reconstruct_rgbd_object.py `
  --dataset "<YOUR_DATASET_PATH>" `
  --output "<YOUR_OUTPUT_PATH>" `
  --pose-mode turntable `
  --turntable-angle-sign auto `
  --mesh-method visual-hull `
  --export-object-crops `
  --depth-mask-source expanded-bbox `
  --color-object-filter yellow `
  --model-refine-iterations 1 `
  --foreground-depth-filter `
  --foreground-depth-span-m 0.18 `
  --foreground-gap-m 0.08 `
  --foreground-gap-margin-m 0.02 `
  --visual-hull-voxel-size-m 0.002 `
  --visual-hull-frame-step 1 `
  --visual-hull-min-hit-ratio 0.78 `
  --visual-hull-mask-dilate-px 5 `
  --visual-hull-padding-m 0.026 `
  --visual-hull-smooth-iterations 1 `
  --visual-hull-color-mode best-view `
  --visual-hull-color-smooth-iterations 1 `
  --no-loop-correction
```

## Output Files

The output folder typically contains the following files.

```text
estimated_poses_frame_to_ref.csv
frame_stats.csv
object_points_colored.ply
object_mesh_visual_hull.stl
object_mesh_visual_hull_colored.ply
object_mesh_visual_hull_vertex_color.glb
reconstruction_report.json
object_only/
  rgb/
  rgba/
  depth/
  mask/
```

Use `object_mesh_visual_hull.stl` for simulation geometry. If visual appearance is required in a viewer or simulator, use `object_mesh_visual_hull_vertex_color.glb`.

## RGB-Depth Alignment Note

For accurate surface reconstruction, the depth data must be properly aligned to the RGB camera coordinate system. If the depth maps are only resized, they may not match the RGB masks at the pixel level, which can cause BPA or Poisson surface meshes to tear or include unwanted background regions.

Whenever possible, it is recommended to generate a dataset by re-aligning the depth data to the color camera coordinate system from the original RealSense, ROS bag, or db3 data. For more details, refer to [docs/DATASET.md](docs/DATASET.md).

## Git and Large Files

Generated datasets, `.stl`, `.glb`, `.ply`, `.db3`, and `.bag` files can be very large, so it is recommended not to upload them to Git. The `.gitignore` file is configured to exclude these files.
