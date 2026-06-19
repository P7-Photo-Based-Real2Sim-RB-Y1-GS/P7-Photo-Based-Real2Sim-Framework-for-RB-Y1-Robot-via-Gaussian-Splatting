# Feature-Poor Object Reconstruction Limitations

## Summary

During the project, objects with few visible surface features or weak depth
contrast were harder to reconstruct reliably than objects with stronger color,
texture, or geometric cues.

The final CPU pipeline reduces these problems with SAM2 mask filtering,
circle-fit turntable alignment, ICP pass merging, and ML outlier refinement.
However, these steps cannot restore geometric information that is missing from
the RGB-D observations.

## Observed Failure Modes

- Smooth or texture-poor outer surfaces produced fewer reliable image features.
- Reflective or glossy regions introduced unstable or missing depth.
- Repetitive or uniform surfaces made visual alignment less distinctive.
- Low-quality depth near object edges created floating points and torn surfaces.
- Background leakage through masks produced unwanted surface fragments.

## Why This Happens

Intel RealSense D400-series cameras use active stereo depth. Intel's D400
datasheet explains that the optional IR projector adds a non-visible static IR
pattern to improve depth accuracy in low-texture scenes. This is useful, but it
also indicates that low texture is a known depth-sensing challenge for stereo
systems.

The RealSense D435i product specification lists active stereoscopic depth and an
operating range of roughly 0.3 m to 3 m. Captures outside the stable range, or
captures with poor lighting and reflective surfaces, can lower reconstruction
quality.

RGB-D camera evaluations also report that depth measurements can be affected by
lighting intensity, specular reflection, diffuse reflection, and target distance.
These effects are especially visible when the object has weak external features.

## Why the Current Method Is Still Useful

Compared with BundleSDF and Gaussian Splatting in this CPU-focused Real2Sim
setting:

- SAM2 masks isolate the object before 3D fusion, reducing background geometry.
- Turntable circle fitting avoids relying only on visual feature matching.
- ICP is used only after each pass has been brought into an object-centered
  frame.
- IsolationForest and DBSCAN remove spatial and appearance-based outliers from
  the fused point cloud.
- The output is an explicit mesh (`PLY`, `STL`, `GLB`) suitable for simulation
  geometry or later Isaac Sim conversion.

## References

- Intel RealSense D400 Series Datasheet:
  https://cdrdv2-public.intel.com/841984/Intel-RealSense-D400-Series-Datasheet.pdf
- Intel RealSense D435i product specification:
  https://www.intel.com/content/www/us/en/products/sku/190004/intel-realsense-depth-camera-d435i/specifications.html
- Fan et al., "Depth Ranging Performance Evaluation and Improvement for RGB-D
  Cameras on Field-Based High-Throughput Phenotyping Robots":
  https://arxiv.org/abs/2011.01022
