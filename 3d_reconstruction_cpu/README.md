# 3D Reconstruction CPU

RGB-D turntable dataset을 CPU 환경에서 3D asset으로 복원하는 Python 프로그램입니다.

이 프로젝트는 RGB, depth, object mask, camera intrinsics, frame angle metadata를 사용해 물체만 분리하고, 회전축을 추정한 뒤 STL과 외관 포함 GLB를 생성합니다.

## Features

- RGB/depth/mask 기반 object-only point cloud 생성
- turntable angle metadata 기반 pose 생성
- 회전축 pivot 자동 추정
- 회전 방향 자동 선택
- foreground depth filtering
- RGB/depth misalignment 대응용 expanded bbox depth mask
- model-to-frame ICP refinement
- visual hull 기반 watertight mesh 생성
- best-view RGB projection 기반 vertex color GLB 생성
- STL, PLY, GLB export

## Dataset Layout

사용자 데이터셋 폴더는 아래 구조를 가져야 합니다.

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

`frame_index.csv`에는 각 프레임의 `angle_deg` 또는 `angle_deg_unwrapped` 값이 있어야 turntable pose를 안정적으로 만들 수 있습니다.

## Installation

Python 3.12를 권장합니다.

```powershell
git clone https://github.com/P7-Photo-Based-Real2Sim-RB-Y1-GS/P7-Photo-Based-Real2Sim-Framework-for-RB-Y1-Robot-via-Gaussian-Splatting.git
cd P7-Photo-Based-Real2Sim-Framework-for-RB-Y1-Robot-via-Gaussian-Splatting\3d_reconstruction_cpu

py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

PowerShell 실행 정책 때문에 venv activation이 막힐 수 있습니다. 이 경우 activation 없이 위처럼 `& .\.venv\Scripts\python.exe`를 직접 호출하면 됩니다.

## Quick Check

아래의 `<YOUR_DATASET_PATH>`를 본인 데이터셋 폴더로 바꿔 실행하세요.

```powershell
& .\.venv\Scripts\python.exe .\reconstruct_rgbd_object.py `
  --dataset "<YOUR_DATASET_PATH>" `
  --dry-run
```

예시:

```powershell
& .\.venv\Scripts\python.exe .\reconstruct_rgbd_object.py `
  --dataset "D:\datasets\object_001" `
  --dry-run
```

## Recommended Run

가장 쉬운 실행 방법은 PowerShell wrapper를 사용하는 것입니다.

```powershell
.\run_reconstruction.ps1 `
  -Dataset "<YOUR_DATASET_PATH>" `
  -Output "<YOUR_OUTPUT_PATH>" `
  -InstallRequirements `
  -CleanOutput
```

예시:

```powershell
.\run_reconstruction.ps1 `
  -Dataset "D:\datasets\object_001" `
  -Output "D:\datasets\object_001\reconstruction_realistic_bestview" `
  -InstallRequirements `
  -CleanOutput
```

`-Output`을 생략하면 `<YOUR_DATASET_PATH>\reconstruction_realistic_bestview`에 결과가 생성됩니다.

## Python Direct Run

PowerShell wrapper 대신 Python을 직접 실행할 수도 있습니다.

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

출력 폴더에는 보통 아래 파일들이 생성됩니다.

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

시뮬레이션 geometry에는 `object_mesh_visual_hull.stl`을 사용하고, 외관이 필요한 뷰어/시뮬레이터에는 `object_mesh_visual_hull_vertex_color.glb`를 사용하면 됩니다.

## RGB-Depth Alignment Note

정확한 표면 복원을 위해서는 depth가 RGB camera 좌표계에 정확히 align되어 있어야 합니다. 단순 resize된 depth는 RGB mask와 픽셀 단위로 맞지 않아 BPA/Poisson 같은 표면 mesh가 찢어지거나 배경이 섞일 수 있습니다.

가능하면 원본 RealSense/ROS bag/db3에서 depth를 color camera 좌표계로 다시 align한 데이터셋을 만드는 것이 좋습니다. 자세한 내용은 [docs/DATASET.md](docs/DATASET.md)를 참고하세요.

## Git and Large Files

생성된 dataset, `.stl`, `.glb`, `.ply`, `.db3`, `.bag` 파일은 대용량이므로 Git에 올리지 않는 것을 권장합니다. `.gitignore`는 이런 파일들을 제외하도록 설정되어 있습니다.
