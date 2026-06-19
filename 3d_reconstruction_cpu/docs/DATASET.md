# Dataset Format

The reconstruction script expects a three-pass RealSense RGB-D dataset.

```text
object_001/
  camera_intrinsics.json
  pass_01_level/
    rgb/frame_000000.png
    depth/frame_000000.png
    mask_refined/frame_000000.png
    metadata.csv
  pass_02_high/
    rgb/
    depth/
    mask_refined/
    metadata.csv
  pass_03_low/
    rgb/
    depth/
    mask_refined/
    metadata.csv
```

## Required Data

- `rgb/`: color frames.
- `depth/`: uint16 depth frames, aligned to the color stream.
- `mask_refined/`: binary or grayscale SAM2 object masks.
- `metadata.csv`: frame angle metadata. The script looks for columns such as
  `stage_angle_deg`, `angle_deg`, or `turntable_angle_deg`.
- `camera_intrinsics.json`: color camera intrinsics and depth scale.

## Alignment Requirement

Depth images should be aligned to RGB camera coordinates. The script uses the
color camera intrinsics for back-projection because the captured depth stream is
expected to be aligned to the color stream.

If RGB, depth, and masks are not pixel-aligned, depth from the background may be
included inside the mask. This produces duplicated surfaces, noisy edges, or
over-smoothed meshes.
