# Dataset and RGB-Depth Alignment

## Required Files

The reconstruction script expects a dataset folder like this:

```text
object_001/
  rgb/
  depth/
  masks/
  cam_K.txt
  frame_index.csv
```

## RGB, Depth, and Mask Coordinate System

For high-quality surface reconstruction, RGB, depth, and mask images should refer to the same camera coordinate system.

The ideal setup is:

- RGB image is in color camera coordinates.
- Depth image is aligned to the color camera.
- Mask image is in the same pixel coordinates as RGB.
- `cam_K.txt` is the color camera intrinsic matrix for the aligned images.

If depth was only resized to RGB resolution, it is not truly aligned. This can cause:

- object depth and background depth mixed inside a mask
- missing object edges
- duplicated or warped geometry
- poor BPA/Poisson surface meshes

## Best Fix

Regenerate the dataset from the original capture using calibrated RGB-depth alignment.

For Intel RealSense, the desired operation is conceptually:

```python
align = rs.align(rs.stream.color)
aligned_frames = align.process(frames)
aligned_depth = aligned_frames.get_depth_frame()
color = aligned_frames.get_color_frame()
```

For ROS/RealSense, prefer an already aligned depth topic such as:

```text
/aligned_depth_to_color/image_raw
```

or regenerate aligned depth by using:

- depth camera intrinsics
- color camera intrinsics
- depth-to-color extrinsics or TF transform
- z-buffer projection into the color image plane

## Current Fallback Strategy

When aligned depth is unavailable, this project uses a robust fallback:

- expanded RGB-mask bounding box
- foreground depth clustering
- yellow object color filtering
- turntable pose estimation
- visual hull carving from RGB masks
- best-view RGB projection for GLB vertex colors

This produces a stable watertight object mesh, but it cannot fully recover fine molded surface details that are absent or misregistered in depth.

