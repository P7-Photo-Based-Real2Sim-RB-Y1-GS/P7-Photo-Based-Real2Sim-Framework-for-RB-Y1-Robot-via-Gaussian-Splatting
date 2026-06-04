from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class Capture:
    name: str
    root: Path
    rgb_dir: Path
    depth_dir: Path
    mask_dir: Path | None
    frame_index: Path | None
    cam_k: Path | None


@dataclass(frozen=True)
class Frame:
    capture: str
    stem: str
    rgb_path: Path
    depth_path: Path
    mask_path: Path | None
    index: int
    angle_deg: float | None


@dataclass
class FrameCloud:
    frame: Frame
    pcd: object
    points_before_sample: int
    points_after_sample: int
    median_depth_m: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct an RGB-D turntable object dataset into STL and "
            "appearance-preserving 3D assets."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Dataset root. It may contain rgb/depth/masks directly or child captures. "
            "Required unless RGBD_DATASET is set."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: <dataset>/reconstruction_exports",
    )
    parser.add_argument(
        "--pose-mode",
        choices=["register", "turntable", "reuse"],
        default="turntable",
        help=(
            "register: ORB/ICP sequential alignment; turntable: use angle_deg only; "
            "reuse: load an existing poses CSV."
        ),
    )
    parser.add_argument(
        "--poses-csv",
        type=Path,
        default=None,
        help="Pose CSV with t00..t33 columns. Auto-detected in reuse mode if omitted.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Use every Nth frame. Increase for a faster draft run.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit frames after --every filtering. Useful for quick tests.",
    )
    parser.add_argument(
        "--depth-unit-m",
        type=float,
        default=None,
        help="Meters represented by one depth pixel unit. Default reads conversion_summary.json or 0.001.",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=2.0)
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=16,
        help="Mask values above this are kept. Ignored when no masks directory exists.",
    )
    parser.add_argument(
        "--depth-mask-source",
        choices=["rgb-mask", "expanded-bbox"],
        default="expanded-bbox",
        help=(
            "rgb-mask keeps depth only inside the RGB mask. expanded-bbox searches "
            "foreground depth inside an expanded RGB-mask bbox, useful when depth "
            "was resized but not truly registered to RGB."
        ),
    )
    parser.add_argument(
        "--depth-mask-bbox-padding-px",
        type=int,
        default=56,
        help="Padding used by --depth-mask-source expanded-bbox.",
    )
    parser.add_argument(
        "--foreground-depth-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Within each object mask, keep only the near foreground depth cluster. "
            "This removes wall/table depth leaking through RGB-D misalignment or holes."
        ),
    )
    parser.add_argument(
        "--foreground-gap-m",
        type=float,
        default=0.08,
        help="Depth histogram gap, in meters, used to split foreground from background.",
    )
    parser.add_argument(
        "--foreground-gap-margin-m",
        type=float,
        default=0.02,
        help="Extra depth margin kept after the detected foreground/background gap.",
    )
    parser.add_argument(
        "--foreground-depth-span-m",
        type=float,
        default=0.18,
        help=(
            "Maximum depth span kept from the near object surface. Use 0 to disable. "
            "The default is tuned for a shallow box-like object."
        ),
    )
    parser.add_argument(
        "--foreground-min-points",
        type=int,
        default=1000,
        help="Minimum masked depth points required before foreground clustering is applied.",
    )
    parser.add_argument(
        "--keep-largest-object-component",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After depth filtering, keep only the largest connected object component.",
    )
    parser.add_argument(
        "--object-mask-erode-px",
        type=int,
        default=1,
        help="Erode the valid object mask by this many pixels to suppress RGB/depth edge leakage.",
    )
    parser.add_argument(
        "--object-mask-open-px",
        type=int,
        default=3,
        help="Morphological open kernel size for the valid object mask. Use 0 to disable.",
    )
    parser.add_argument(
        "--object-mask-close-px",
        type=int,
        default=5,
        help="Morphological close kernel size for the valid object mask. Use 0 to disable.",
    )
    parser.add_argument(
        "--export-object-crops",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export mask-cropped object-only RGB/RGBA/depth/mask PNGs used for reconstruction.",
    )
    parser.add_argument(
        "--object-crop-padding-px",
        type=int,
        default=24,
        help="Padding around the object bbox when exporting object-only crop PNGs.",
    )
    parser.add_argument(
        "--color-object-filter",
        choices=["off", "auto", "yellow"],
        default="off",
        help=(
            "Optional color refinement after depth masking. auto keeps the dominant "
            "saturated object hue; yellow is tuned for yellow cases."
        ),
    )
    parser.add_argument(
        "--color-filter-min-saturation",
        type=int,
        default=45,
        help="Minimum HSV saturation used by --color-object-filter.",
    )
    parser.add_argument(
        "--color-filter-hue-width-deg",
        type=float,
        default=42.0,
        help="Circular hue window in degrees used by --color-object-filter.",
    )
    parser.add_argument(
        "--points-per-frame",
        type=int,
        default=18000,
        help="Randomly sample each frame cloud to this many points before fusion.",
    )
    parser.add_argument(
        "--registration-points",
        type=int,
        default=3500,
        help="Point budget per frame for registration refinement.",
    )
    parser.add_argument(
        "--voxel-size-m",
        type=float,
        default=0.003,
        help="Final fusion voxel size in meters.",
    )
    parser.add_argument(
        "--mesh-method",
        choices=["box", "visual-hull", "tsdf", "both", "all", "poisson", "bpa"],
        default="both",
        help=(
            "Mesh reconstruction method. Use box for box-like objects when RGB/depth "
            "alignment is rough and raw surface meshing becomes unstable."
        ),
    )
    parser.add_argument(
        "--tsdf-voxel-size-m",
        type=float,
        default=0.0025,
        help="Voxel size for TSDF fusion mesh reconstruction.",
    )
    parser.add_argument(
        "--tsdf-trunc-m",
        type=float,
        default=0.015,
        help="Signed-distance truncation distance for TSDF fusion.",
    )
    parser.add_argument(
        "--visual-hull-voxel-size-m",
        type=float,
        default=0.003,
        help="Voxel size for silhouette visual-hull reconstruction.",
    )
    parser.add_argument(
        "--visual-hull-padding-m",
        type=float,
        default=0.018,
        help="Padding around the fused object bounds for visual-hull carving.",
    )
    parser.add_argument(
        "--visual-hull-min-hit-ratio",
        type=float,
        default=0.92,
        help="Fraction of projected silhouette masks that must contain a voxel.",
    )
    parser.add_argument(
        "--visual-hull-frame-step",
        type=int,
        default=1,
        help="Use every Nth frame for visual-hull carving.",
    )
    parser.add_argument(
        "--visual-hull-mask-dilate-px",
        type=int,
        default=3,
        help="Dilate silhouettes by this many pixels before visual-hull carving.",
    )
    parser.add_argument(
        "--visual-hull-smooth-iterations",
        type=int,
        default=2,
        help="Taubin smoothing iterations applied to the visual-hull mesh.",
    )
    parser.add_argument(
        "--visual-hull-color-mode",
        choices=["uniform", "average", "best-view"],
        default="uniform",
        help=(
            "uniform paints the visual hull with --box-prior-color; average projects "
            "RGB frames onto vertices; best-view chooses the most front-facing RGB "
            "view per vertex for more realistic appearance."
        ),
    )
    parser.add_argument(
        "--visual-hull-color-smooth-iterations",
        type=int,
        default=0,
        help="Lightly smooth visual-hull vertex colors after projection to reduce speckle.",
    )
    parser.add_argument("--poisson-depth", type=int, default=9)
    parser.add_argument(
        "--box-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also export a clean oriented cuboid fitted to the fused foreground cloud. "
            "Use this for box-like objects when raw RGB-D surfaces are incomplete."
        ),
    )
    parser.add_argument(
        "--box-prior-fit",
        choices=["silhouette", "fused-obb"],
        default="silhouette",
        help=(
            "silhouette estimates box width/height/depth from mask size and foreground "
            "depth over the turntable angles; fused-obb uses the fused cloud OBB."
        ),
    )
    parser.add_argument(
        "--box-prior-margin-m",
        type=float,
        default=0.003,
        help="Small safety margin added to each fitted box dimension.",
    )
    parser.add_argument(
        "--box-prior-angle-window-deg",
        type=float,
        default=20.0,
        help="Angle window used to pick front/back and side frames for silhouette box fitting.",
    )
    parser.add_argument(
        "--box-prior-color",
        type=float,
        nargs=3,
        default=(1.0, 0.64, 0.05),
        metavar=("R", "G", "B"),
        help="RGB color in 0..1 used for the fitted box-prior GLB/PLY.",
    )
    parser.add_argument(
        "--poisson-density-quantile",
        type=float,
        default=0.02,
        help="Remove low-density Poisson vertices below this quantile.",
    )
    parser.add_argument(
        "--loop-correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Distribute accepted last-to-first registration drift across a full turn. "
            "Disabled by default because a bad loop closure can warp the entire model."
        ),
    )
    parser.add_argument(
        "--loop-min-fitness",
        type=float,
        default=0.35,
        help="Minimum last-to-first registration fitness required to apply loop correction.",
    )
    parser.add_argument(
        "--loop-max-rmse-m",
        type=float,
        default=0.02,
        help="Maximum last-to-first inlier RMSE in meters allowed for loop correction.",
    )
    parser.add_argument(
        "--loop-max-translation-m",
        type=float,
        default=0.08,
        help="Maximum closure drift translation in meters allowed for loop correction.",
    )
    parser.add_argument(
        "--loop-max-rotation-rad",
        type=float,
        default=0.60,
        help="Maximum closure drift rotation in radians allowed for loop correction.",
    )
    parser.add_argument(
        "--angle-closure-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For single-capture turntable register mode, distribute drift so the final "
            "pose matches the angle metadata. This prevents sequential ICP drift."
        ),
    )
    parser.add_argument(
        "--angle-closure-max-translation-m",
        type=float,
        default=0.35,
        help="Maximum metadata closure translation drift allowed for angle closure correction.",
    )
    parser.add_argument(
        "--angle-closure-max-rotation-rad",
        type=float,
        default=1.20,
        help="Maximum metadata closure rotation drift allowed for angle closure correction.",
    )
    parser.add_argument(
        "--turntable-axis",
        choices=["y", "z"],
        default="y",
        help="Axis used by pose-mode=turntable. Open3D camera coordinates usually use y as vertical.",
    )
    parser.add_argument(
        "--turntable-angle-sign",
        choices=["auto", "negative", "positive"],
        default="auto",
        help=(
            "Direction used to undo turntable rotation. auto evaluates both signs "
            "and keeps the more compact fused object."
        ),
    )
    parser.add_argument(
        "--turntable-pivot-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Optional turntable pivot in camera coordinates, meters. "
            "If omitted, it is estimated from masked RGB-D points."
        ),
    )
    parser.add_argument(
        "--turntable-pivot-fit",
        choices=["angle", "median"],
        default="angle",
        help="angle fits the center trajectory over frame angles; median uses robust per-frame centers.",
    )
    parser.add_argument(
        "--model-refine-iterations",
        type=int,
        default=0,
        help="Refine each frame pose against the current fused object-only model this many times.",
    )
    parser.add_argument(
        "--model-refine-distance-m",
        type=float,
        default=0.012,
        help="Max correspondence distance for model pose refinement.",
    )
    parser.add_argument(
        "--model-refine-min-fitness",
        type=float,
        default=0.15,
        help="Minimum ICP fitness required to accept a model-refinement pose update.",
    )
    parser.add_argument(
        "--model-refine-max-translation-m",
        type=float,
        default=0.035,
        help="Reject model-refinement updates with translation larger than this.",
    )
    parser.add_argument(
        "--model-refine-max-rotation-rad",
        type=float,
        default=0.45,
        help="Reject model-refinement updates with rotation larger than this.",
    )
    parser.add_argument(
        "--fused-dbscan-eps-m",
        type=float,
        default=0.018,
        help="DBSCAN radius for keeping the largest fused object cluster. Use 0 to disable.",
    )
    parser.add_argument(
        "--fused-dbscan-min-points",
        type=int,
        default=35,
        help="DBSCAN minimum points for the largest fused object cluster filter.",
    )
    parser.add_argument(
        "--export-obj",
        action="store_true",
        help="Also export OBJ files. GLB/PLY preserve appearance more reliably.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for point sampling and RANSAC.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect dataset and dependency availability without reconstructing.",
    )
    return parser.parse_args()


def fail(message: str, code: int = 1) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_depth_unit_m(dataset: Path, override: float | None) -> float:
    if override is not None:
        return override
    summary = dataset / "conversion_summary.json"
    if summary.exists():
        data = load_json(summary)
        if "depth_units_m_per_unit" in data:
            return float(data["depth_units_m_per_unit"])
    return 0.001


def find_captures(dataset: Path) -> list[Capture]:
    if not dataset.exists():
        fail(f"Dataset does not exist: {dataset}")
    direct = make_capture(dataset.name, dataset)
    if direct is not None:
        return [direct]

    captures: list[Capture] = []
    for child in sorted(p for p in dataset.iterdir() if p.is_dir()):
        capture = make_capture(child.name, child)
        if capture is not None:
            captures.append(capture)
    if not captures:
        fail(
            "Could not find rgb/depth folders. Expected <dataset>/rgb and "
            "<dataset>/depth, or child folders with that layout."
        )
    return captures


def make_capture(name: str, root: Path) -> Capture | None:
    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    if not rgb_dir.is_dir() or not depth_dir.is_dir():
        return None
    mask_dir = root / "masks"
    frame_index = root / "frame_index.csv"
    cam_k = root / "cam_K.txt"
    return Capture(
        name=name,
        root=root,
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        mask_dir=mask_dir if mask_dir.is_dir() else None,
        frame_index=frame_index if frame_index.exists() else None,
        cam_k=cam_k if cam_k.exists() else None,
    )


def read_frame_index(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        out: dict[str, dict[str, str]] = {}
        for row in reader:
            file_name = row.get("file") or row.get("filename") or ""
            if file_name:
                out[Path(file_name).stem] = row
        return out


def discover_frames(captures: list[Capture], every: int, max_frames: int | None) -> list[Frame]:
    if every < 1:
        fail("--every must be >= 1")
    frames: list[Frame] = []
    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    for capture in captures:
        index_by_stem = read_frame_index(capture.frame_index)
        rgb_files = sorted(p for p in capture.rgb_dir.iterdir() if p.suffix.lower() in image_exts)
        for local_index, rgb_path in enumerate(rgb_files):
            if local_index % every != 0:
                continue
            stem = rgb_path.stem
            depth_path = find_matching_file(capture.depth_dir, stem, image_exts)
            if depth_path is None:
                continue
            mask_path = (
                find_matching_file(capture.mask_dir, stem, image_exts)
                if capture.mask_dir is not None
                else None
            )
            meta = index_by_stem.get(stem, {})
            angle = parse_float(
                meta.get("angle_deg_unwrapped")
                or meta.get("angle_deg")
                or meta.get("angle")
            )
            frames.append(
                Frame(
                    capture=capture.name,
                    stem=stem,
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    mask_path=mask_path,
                    index=len(frames),
                    angle_deg=angle,
                )
            )
            if max_frames is not None and len(frames) >= max_frames:
                return frames
    return frames


def find_matching_file(directory: Path | None, stem: str, exts: Iterable[str]) -> Path | None:
    if directory is None:
        return None
    for ext in exts:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_intrinsics(dataset: Path, captures: list[Capture], first_rgb: Path) -> Intrinsics:
    width, height = Image.open(first_rgb).size
    candidates: list[Path] = []
    candidates.extend(c.cam_k for c in captures if c.cam_k is not None)
    candidates.append(dataset / "cam_K.txt")
    summary = dataset / "conversion_summary.json"

    for path in candidates:
        if path is not None and path.exists():
            values = [float(v) for v in path.read_text(encoding="utf-8").split()]
            if len(values) != 9:
                fail(f"Invalid cam_K.txt, expected 9 numbers: {path}")
            return Intrinsics(
                width=width,
                height=height,
                fx=values[0],
                fy=values[4],
                cx=values[2],
                cy=values[5],
            )

    if summary.exists():
        data = load_json(summary)
        if "cam_K" in data:
            k = data["cam_K"]
            return Intrinsics(
                width=width,
                height=height,
                fx=float(k[0][0]),
                fy=float(k[1][1]),
                cx=float(k[0][2]),
                cy=float(k[1][2]),
            )

    fail("Could not load intrinsics. Add cam_K.txt or conversion_summary.json.")


def dependency_status() -> dict[str, bool]:
    import importlib.util

    return {
        "open3d": importlib.util.find_spec("open3d") is not None,
        "cv2": importlib.util.find_spec("cv2") is not None,
        "trimesh": importlib.util.find_spec("trimesh") is not None,
        "numpy": importlib.util.find_spec("numpy") is not None,
        "PIL": importlib.util.find_spec("PIL") is not None,
    }


def require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        fail(
            "Open3D is not installed. Use Python 3.10-3.12 and run "
            "`python -m pip install -r requirements.txt`."
        )
        raise exc
    return o3d


def require_cv2():
    try:
        import cv2
    except ImportError:
        return None
    return cv2


def make_open3d_intrinsic(o3d, intr: Intrinsics):
    return o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy
    )


def load_rgb_depth_mask(frame: Frame, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    color = np.asarray(Image.open(frame.rgb_path).convert("RGB"))
    depth = np.asarray(Image.open(frame.depth_path))
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = depth.astype(np.uint16, copy=False)
    mask = None
    if frame.mask_path is not None:
        mask = np.asarray(Image.open(frame.mask_path).convert("L")) > args.mask_threshold
    return color, depth, mask


def valid_depth_mask(depth: np.ndarray, mask: np.ndarray | None, args: argparse.Namespace, depth_unit_m: float) -> np.ndarray:
    depth_m = depth.astype(np.float32) * depth_unit_m
    valid = (depth_m >= args.min_depth_m) & (depth_m <= args.max_depth_m)
    if mask is not None:
        valid &= make_depth_roi_mask(mask, args)
    if args.foreground_depth_filter:
        valid = apply_foreground_depth_filter(depth_m, valid, args)
    valid = clean_object_valid_mask(valid, args)
    return valid


def make_depth_roi_mask(mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.depth_mask_source == "rgb-mask":
        return mask
    bbox = mask_bbox(mask, args.depth_mask_bbox_padding_px, mask.shape[1], mask.shape[0])
    if bbox is None:
        return mask
    x0, y0, x1, y1 = bbox
    roi = np.zeros_like(mask, dtype=bool)
    roi[y0:y1, x0:x1] = True
    return roi


def apply_foreground_depth_filter(
    depth_m: np.ndarray,
    valid: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    z = depth_m[valid]
    z = z[np.isfinite(z)]
    if z.size < args.foreground_min_points:
        return valid

    z_sorted = np.sort(z)
    cutoff: float | None = None

    # RGB masks are in color-image coordinates, while this dataset's depth was only
    # resized to color resolution. A large depth gap inside the mask usually means
    # background depth leaked into the object silhouette.
    if z_sorted.size > 2 and args.foreground_gap_m > 0:
        diffs = z_sorted[1:] - z_sorted[:-1]
        start = int(0.05 * diffs.size)
        end = int(0.95 * diffs.size)
        if end > start:
            local = diffs[start:end]
            gap_local_idx = int(np.argmax(local))
            gap_idx = start + gap_local_idx
            gap = float(diffs[gap_idx])
            if gap >= args.foreground_gap_m:
                cutoff = float(z_sorted[gap_idx] + args.foreground_gap_margin_m)

    if args.foreground_depth_span_m > 0:
        near_surface = float(np.quantile(z_sorted, 0.01))
        span_cutoff = near_surface + args.foreground_depth_span_m
        cutoff = span_cutoff if cutoff is None else min(cutoff, span_cutoff)

    if cutoff is None:
        return valid
    return valid & (depth_m <= cutoff)


def clean_object_valid_mask(valid: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if not np.any(valid):
        return valid
    try:
        import cv2
    except ImportError:
        return valid

    mask_u8 = valid.astype(np.uint8) * 255
    if args.object_mask_close_px and args.object_mask_close_px > 1:
        k = int(args.object_mask_close_px)
        kernel = np.ones((k, k), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    if args.object_mask_open_px and args.object_mask_open_px > 1:
        k = int(args.object_mask_open_px)
        kernel = np.ones((k, k), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    if args.object_mask_erode_px and args.object_mask_erode_px > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask_u8 = cv2.erode(mask_u8, kernel, iterations=int(args.object_mask_erode_px))
    if args.keep_largest_object_component:
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            (mask_u8 > 0).astype(np.uint8),
            connectivity=8,
        )
        if count > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if areas.size:
                largest = int(np.argmax(areas)) + 1
                mask_u8 = (labels == largest).astype(np.uint8) * 255
    return mask_u8 > 0


def save_object_crop(
    frame: Frame,
    color: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    crop_root: Path,
    padding_px: int,
) -> None:
    bbox = mask_bbox(valid, padding_px, color.shape[1], color.shape[0])
    if bbox is None:
        return
    x0, y0, x1, y1 = bbox
    valid_crop = valid[y0:y1, x0:x1]
    color_crop = color[y0:y1, x0:x1].copy()
    color_crop[~valid_crop] = 0
    alpha = (valid_crop.astype(np.uint8) * 255)[:, :, None]
    rgba_crop = np.concatenate([color_crop, alpha], axis=2)
    depth_crop = depth[y0:y1, x0:x1].copy()
    depth_crop[~valid_crop] = 0
    mask_crop = valid_crop.astype(np.uint8) * 255

    for subdir in ("rgb", "rgba", "depth", "mask"):
        (crop_root / subdir).mkdir(parents=True, exist_ok=True)
    Image.fromarray(color_crop).save(crop_root / "rgb" / f"{frame.stem}.png")
    Image.fromarray(rgba_crop).save(crop_root / "rgba" / f"{frame.stem}.png")
    Image.fromarray(depth_crop).save(crop_root / "depth" / f"{frame.stem}.png")
    Image.fromarray(mask_crop).save(crop_root / "mask" / f"{frame.stem}.png")


def mask_bbox(mask: np.ndarray, padding_px: int, width: int, height: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    pad = max(int(padding_px), 0)
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + 1 + pad, width)
    y1 = min(int(ys.max()) + 1 + pad, height)
    return x0, y0, x1, y1


def make_point_cloud(
    o3d,
    frame: Frame,
    intrinsic,
    args: argparse.Namespace,
    depth_unit_m: float,
    rng: np.random.Generator,
    point_budget: int,
) -> FrameCloud:
    color, depth, mask = load_rgb_depth_mask(frame, args)
    valid = valid_depth_mask(depth, mask, args, depth_unit_m)
    valid = refine_valid_mask_by_color(color, valid, args)
    crop_root = getattr(args, "_object_crop_root", None)
    if crop_root is not None:
        save_object_crop(
            frame,
            color,
            depth,
            valid,
            crop_root,
            args.object_crop_padding_px,
        )
    filtered_depth = np.where(valid, depth, 0).astype(np.uint16)

    depth_values_m = filtered_depth[filtered_depth > 0].astype(np.float32) * depth_unit_m
    median_depth_m = float(np.median(depth_values_m)) if depth_values_m.size else None

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color),
        o3d.geometry.Image(filtered_depth),
        depth_scale=1.0 / depth_unit_m,
        depth_trunc=args.max_depth_m,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    before = len(pcd.points)
    pcd = sample_point_cloud(pcd, point_budget, rng)
    estimate_normals(o3d, pcd, radius=max(args.voxel_size_m * 4.0, 0.01))
    return FrameCloud(
        frame=frame,
        pcd=pcd,
        points_before_sample=before,
        points_after_sample=len(pcd.points),
        median_depth_m=median_depth_m,
    )


def refine_valid_mask_by_color(
    color: np.ndarray,
    valid: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.color_object_filter == "off" or not np.any(valid):
        return valid
    try:
        import cv2
    except ImportError:
        return valid

    hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.float32) * 2.0
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    saturated = valid & (saturation >= args.color_filter_min_saturation) & (value >= 35)
    if np.count_nonzero(saturated) < max(args.foreground_min_points // 4, 100):
        return valid

    if args.color_object_filter == "yellow":
        hue_center = 42.0
    else:
        hist, edges = np.histogram(hue[saturated], bins=72, range=(0.0, 360.0))
        if not np.any(hist):
            return valid
        peak = int(np.argmax(hist))
        hue_center = float((edges[peak] + edges[peak + 1]) * 0.5)

    hue_distance = circular_hue_distance_deg(hue, hue_center)
    keep = valid & (hue_distance <= args.color_filter_hue_width_deg * 0.5)
    if np.count_nonzero(keep) < max(args.foreground_min_points // 2, 250):
        return valid

    keep_u8 = keep.astype(np.uint8) * 255
    close_px = max(args.object_mask_close_px, 5)
    if close_px > 1:
        kernel = np.ones((int(close_px), int(close_px)), np.uint8)
        keep_u8 = cv2.morphologyEx(keep_u8, cv2.MORPH_CLOSE, kernel)
    if args.keep_largest_object_component:
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            (keep_u8 > 0).astype(np.uint8), connectivity=8
        )
        if count > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if areas.size:
                largest = int(np.argmax(areas)) + 1
                keep_u8 = (labels == largest).astype(np.uint8) * 255
    return (keep_u8 > 0) & valid


def circular_hue_distance_deg(hue: np.ndarray, center: float) -> np.ndarray:
    diff = np.abs(hue - center)
    return np.minimum(diff, 360.0 - diff)


def sample_point_cloud(pcd, point_budget: int, rng: np.random.Generator):
    if point_budget <= 0 or len(pcd.points) <= point_budget:
        return pcd
    idx = rng.choice(len(pcd.points), size=point_budget, replace=False)
    return pcd.select_by_index(idx.tolist())


def estimate_normals(o3d, pcd, radius: float) -> None:
    if len(pcd.points) == 0:
        return
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40)
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(30)
    except RuntimeError:
        pass


def estimate_poses(
    o3d,
    frames: list[Frame],
    frame_clouds: list[FrameCloud],
    intr: Intrinsics,
    args: argparse.Namespace,
    depth_unit_m: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if args.pose_mode == "reuse":
        poses_path = args.poses_csv or auto_detect_poses_csv(args.dataset)
        if poses_path is None:
            fail("--pose-mode reuse requested, but no poses CSV was found.")
        print(f"Loading poses: {poses_path}")
        return load_poses_csv(poses_path, frames)
    if args.pose_mode == "turntable":
        pivot = get_turntable_pivot(frame_clouds, args)
        return estimate_turntable_poses(
            frames, frame_clouds, args.turntable_axis, pivot, args
        )
    return estimate_registered_poses(
        o3d, frames, frame_clouds, intr, args, depth_unit_m, rng
    )


def refine_poses_to_model(
    o3d,
    frame_clouds: list[FrameCloud],
    poses: list[np.ndarray],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if args.model_refine_iterations <= 0:
        return poses

    refine_stats: list[dict[str, object]] = []
    current_poses = [pose.copy() for pose in poses]
    source_voxel = max(args.voxel_size_m * 2.0, 0.006)
    for iteration in range(args.model_refine_iterations):
        print(
            f"Model pose refinement {iteration + 1}/{args.model_refine_iterations}..."
        )
        model, _raw_count = fuse_clouds(o3d, frame_clouds, current_poses, args)
        target = model.voxel_down_sample(source_voxel)
        estimate_normals(o3d, target, radius=max(source_voxel * 3.0, 0.015))

        accepted = 0
        rejected = 0
        fitness_values: list[float] = []
        rmse_values: list[float] = []
        next_poses: list[np.ndarray] = []
        for fc, pose in zip(frame_clouds, current_poses):
            source = sample_point_cloud(fc.pcd, args.registration_points, rng)
            source = source.voxel_down_sample(source_voxel)
            estimate_normals(o3d, source, radius=max(source_voxel * 3.0, 0.015))
            result = refine_absolute_pose_to_model(o3d, source, target, pose, args)
            if result is None:
                next_poses.append(pose)
                rejected += 1
                continue

            delta = result.transformation @ np.linalg.inv(pose)
            delta_translation = float(np.linalg.norm(delta[:3, 3]))
            delta_rotation = rotation_angle(delta[:3, :3])
            accept = (
                result.fitness >= args.model_refine_min_fitness
                and delta_translation <= args.model_refine_max_translation_m
                and delta_rotation <= args.model_refine_max_rotation_rad
            )
            if accept:
                next_poses.append(result.transformation)
                accepted += 1
                fitness_values.append(float(result.fitness))
                rmse_values.append(float(result.inlier_rmse))
            else:
                next_poses.append(pose)
                rejected += 1

        current_poses = next_poses
        stat = {
            "iteration": iteration + 1,
            "accepted": accepted,
            "rejected": rejected,
            "mean_fitness": float(np.mean(fitness_values)) if fitness_values else 0.0,
            "mean_rmse_m": float(np.mean(rmse_values)) if rmse_values else 0.0,
        }
        refine_stats.append(stat)
        print(
            "  accepted="
            f"{accepted}, rejected={rejected}, "
            f"mean_fitness={stat['mean_fitness']:.3f}, "
            f"mean_rmse={stat['mean_rmse_m']:.4f} m"
        )
    args._model_refine = refine_stats
    return current_poses


def refine_absolute_pose_to_model(o3d, source, target, init: np.ndarray, args: argparse.Namespace):
    if len(source.points) < 8 or len(target.points) < 8:
        return None
    try:
        return o3d.pipelines.registration.registration_icp(
            source,
            target,
            args.model_refine_distance_m,
            init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
        )
    except Exception:
        try:
            return o3d.pipelines.registration.registration_icp(
                source,
                target,
                args.model_refine_distance_m,
                init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
            )
        except Exception:
            return None


def auto_detect_poses_csv(dataset: Path) -> Path | None:
    candidates = [
        dataset / "reconstruction_exports" / "estimated_poses_frame_to_ref.csv",
        dataset / "recon_reg_output" / "estimated_poses_frame_to_ref.csv",
        dataset / "recon_output" / "estimated_poses_frame_to_ref.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_poses_csv(path: Path, frames: list[Frame]) -> list[np.ndarray]:
    by_stem: dict[str, np.ndarray] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = [f"t{r}{c}" for r in range(4) for c in range(4)]
        for row in reader:
            if not all(k in row for k in required):
                fail(f"Pose CSV missing t00..t33 columns: {path}")
            stem = row.get("stem") or Path(row.get("file", "")).stem
            mat = np.array([float(row[k]) for k in required], dtype=np.float64).reshape(4, 4)
            by_stem[stem] = mat
    poses: list[np.ndarray] = []
    for frame in frames:
        if frame.stem not in by_stem:
            fail(f"Pose CSV has no transform for frame {frame.stem}")
        poses.append(by_stem[frame.stem])
    return poses


def get_turntable_pivot(frame_clouds: list[FrameCloud], args: argparse.Namespace) -> np.ndarray:
    if args.turntable_pivot_m is not None:
        pivot = np.asarray(args.turntable_pivot_m, dtype=np.float64)
        fit = {"source": "manual", "pivot_m": pivot.tolist()}
    elif args.turntable_pivot_fit == "angle":
        pivot, fit = estimate_turntable_pivot_from_angles(frame_clouds, args.turntable_axis)
    else:
        pivot = estimate_turntable_pivot(frame_clouds)
        fit = {"source": "median_centers", "pivot_m": pivot.tolist()}
    args._turntable_pivot = pivot.tolist()
    args._turntable_pivot_fit = fit
    print(f"Turntable pivot: [{pivot[0]:.4f}, {pivot[1]:.4f}, {pivot[2]:.4f}] m")
    return pivot


def estimate_turntable_pivot_from_angles(
    frame_clouds: list[FrameCloud],
    axis: str,
) -> tuple[np.ndarray, dict[str, object]]:
    centers: list[np.ndarray] = []
    angles: list[float] = []
    for fc in frame_clouds:
        if fc.frame.angle_deg is None:
            continue
        pts = np.asarray(fc.pcd.points)
        if len(pts) < 20:
            continue
        lo = np.quantile(pts, 0.05, axis=0)
        hi = np.quantile(pts, 0.95, axis=0)
        centers.append((lo + hi) * 0.5)
        angles.append(math.radians(fc.frame.angle_deg))
    if len(centers) < 8:
        pivot = estimate_turntable_pivot(frame_clouds)
        return pivot, {
            "source": "median_centers_fallback",
            "reason": "not_enough_angle_centers",
            "pivot_m": pivot.tolist(),
        }

    center_np = np.asarray(centers, dtype=np.float64)
    angle_np = np.asarray(angles, dtype=np.float64)
    design = np.column_stack(
        [
            np.ones(len(angle_np), dtype=np.float64),
            np.cos(angle_np),
            np.sin(angle_np),
        ]
    )
    if axis == "y":
        plane = (0, 2)
        axis_index = 1
    else:
        plane = (0, 1)
        axis_index = 2

    coeff_a = np.linalg.lstsq(design, center_np[:, plane[0]], rcond=None)[0]
    coeff_b = np.linalg.lstsq(design, center_np[:, plane[1]], rcond=None)[0]
    pivot = np.median(center_np, axis=0)
    pivot[plane[0]] = coeff_a[0]
    pivot[plane[1]] = coeff_b[0]
    pivot[axis_index] = float(np.median(center_np[:, axis_index]))

    pred_a = design @ coeff_a
    pred_b = design @ coeff_b
    residual = np.sqrt(
        (pred_a - center_np[:, plane[0]]) ** 2
        + (pred_b - center_np[:, plane[1]]) ** 2
    )
    fit = {
        "source": "angle_center_fit",
        "axis": axis,
        "frames_used": int(len(center_np)),
        "pivot_m": pivot.tolist(),
        "median_residual_m": float(np.median(residual)),
        "p95_residual_m": float(np.quantile(residual, 0.95)),
    }
    return pivot, fit


def estimate_turntable_pivot(frame_clouds: list[FrameCloud]) -> np.ndarray:
    centers: list[np.ndarray] = []
    for fc in frame_clouds:
        pts = np.asarray(fc.pcd.points)
        if len(pts) < 20:
            continue
        lo = np.quantile(pts, 0.05, axis=0)
        hi = np.quantile(pts, 0.95, axis=0)
        centers.append((lo + hi) * 0.5)
    if not centers:
        return np.zeros(3, dtype=np.float64)
    return np.median(np.asarray(centers), axis=0)


def estimate_turntable_poses(
    frames: list[Frame],
    frame_clouds: list[FrameCloud],
    axis: str,
    pivot: np.ndarray,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    if any(frame.angle_deg is None for frame in frames):
        fail("pose-mode=turntable requires angle_deg or angle_deg_unwrapped in frame_index.csv.")

    if args.turntable_angle_sign == "auto":
        candidates = {
            "negative": make_turntable_pose_list(frames, axis, pivot, -1.0),
            "positive": make_turntable_pose_list(frames, axis, pivot, 1.0),
        }
        scores = {
            name: score_pose_compactness(frame_clouds, poses)
            for name, poses in candidates.items()
        }
        best_name = min(scores, key=scores.get)
        args._turntable_direction = {
            "selected": best_name,
            "scores": scores,
            "metric": "robust_quantile_volume",
        }
        print(
            "Turntable direction: "
            f"{best_name} (negative={scores['negative']:.6g}, "
            f"positive={scores['positive']:.6g})"
        )
        return candidates[best_name]

    sign = -1.0 if args.turntable_angle_sign == "negative" else 1.0
    args._turntable_direction = {"selected": args.turntable_angle_sign}
    return make_turntable_pose_list(frames, axis, pivot, sign)


def make_turntable_pose_list(
    frames: list[Frame],
    axis: str,
    pivot: np.ndarray,
    sign: float,
) -> list[np.ndarray]:
    first_angle = frames[0].angle_deg or 0.0
    poses = []
    for frame in frames:
        delta = math.radians((frame.angle_deg or 0.0) - first_angle)
        poses.append(rotation_about_axis_around_pivot(sign * delta, axis, pivot))
    return poses


def score_pose_compactness(
    frame_clouds: list[FrameCloud],
    poses: list[np.ndarray],
    max_points_per_frame: int = 400,
) -> float:
    chunks: list[np.ndarray] = []
    for fc, pose in zip(frame_clouds, poses):
        pts = np.asarray(fc.pcd.points)
        if len(pts) == 0:
            continue
        if len(pts) > max_points_per_frame:
            idx = np.linspace(0, len(pts) - 1, max_points_per_frame).astype(int)
            pts = pts[idx]
        chunks.append(transform_points(pts, pose))
    if not chunks:
        return float("inf")
    pts_all = np.vstack(chunks)
    lo = np.quantile(pts_all, 0.03, axis=0)
    hi = np.quantile(pts_all, 0.97, axis=0)
    extents = np.maximum(hi - lo, 1e-6)
    return float(np.prod(extents))


def estimate_registered_poses(
    o3d,
    frames: list[Frame],
    frame_clouds: list[FrameCloud],
    intr: Intrinsics,
    args: argparse.Namespace,
    depth_unit_m: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    cv2 = require_cv2()
    reg_clouds = [
        FrameCloud(
            frame=fc.frame,
            pcd=sample_point_cloud(fc.pcd, args.registration_points, rng),
            points_before_sample=fc.points_before_sample,
            points_after_sample=min(fc.points_after_sample, args.registration_points),
            median_depth_m=fc.median_depth_m,
        )
        for fc in frame_clouds
    ]
    for fc in reg_clouds:
        estimate_normals(o3d, fc.pcd, radius=max(args.voxel_size_m * 6.0, 0.012))
    pivot = get_turntable_pivot(frame_clouds, args)

    poses = [np.eye(4)]
    pair_stats: list[dict[str, object]] = []
    for i in range(1, len(frames)):
        source = reg_clouds[i].pcd
        target = reg_clouds[i - 1].pcd
        candidates: list[tuple[str, np.ndarray]] = []
        if cv2 is not None:
            orb_t = estimate_orb_transform(
                cv2, frames[i], frames[i - 1], intr, args, depth_unit_m, rng
            )
            if orb_t is not None:
                candidates.append(("orb3d", orb_t))
        candidates.extend(turntable_pair_candidates(frames[i], frames[i - 1], pivot, args.turntable_axis))
        candidates.append(("identity", np.eye(4)))

        best_name = "identity"
        best_result = None
        best_transform = np.eye(4)
        for name, init in candidates:
            result = refine_pair_registration(o3d, source, target, init, args)
            if result is None:
                continue
            if best_result is None or registration_score(result) > registration_score(best_result):
                best_name = name
                best_result = result
                best_transform = result.transformation

        poses.append(poses[-1] @ best_transform)
        pair_stats.append(
            {
                "source": frames[i].stem,
                "target": frames[i - 1].stem,
                "init": best_name,
                "fitness": float(best_result.fitness if best_result is not None else 0.0),
                "inlier_rmse": float(best_result.inlier_rmse if best_result is not None else 0.0),
            }
        )
        if i % 20 == 0 or i == len(frames) - 1:
            print(
                f"Registered {i + 1}/{len(frames)} frames "
                f"(last={best_name}, fitness={pair_stats[-1]['fitness']:.3f})"
            )

    if should_angle_closure_correct(frames, args):
        poses = apply_angle_closure_correction(poses, frames, pivot, args)

    if should_loop_correct(frames, args):
        poses = apply_loop_correction(o3d, reg_clouds, poses, args)

    args._pair_stats = pair_stats
    return poses


def registration_score(result) -> tuple[float, float]:
    return (float(result.fitness), -float(result.inlier_rmse))


def turntable_pair_candidates(
    current: Frame,
    previous: Frame,
    pivot: np.ndarray,
    axis: str,
) -> list[tuple[str, np.ndarray]]:
    if current.angle_deg is None or previous.angle_deg is None:
        return []
    delta = math.radians(current.angle_deg - previous.angle_deg)
    return [
        (f"angle_{axis}_neg_pivot", rotation_about_axis_around_pivot(-delta, axis, pivot)),
        (f"angle_{axis}_pos_pivot", rotation_about_axis_around_pivot(delta, axis, pivot)),
    ]


def refine_pair_registration(o3d, source, target, init: np.ndarray, args: argparse.Namespace):
    if len(source.points) < 3 or len(target.points) < 3:
        return None
    current = init
    last_result = None
    voxel_sizes = [
        max(args.voxel_size_m * 6.0, 0.018),
        max(args.voxel_size_m * 3.0, 0.009),
        max(args.voxel_size_m * 1.5, 0.0045),
    ]
    for voxel in voxel_sizes:
        src = source.voxel_down_sample(voxel)
        tgt = target.voxel_down_sample(voxel)
        if len(src.points) < 3 or len(tgt.points) < 3:
            continue
        estimate_normals(o3d, src, radius=voxel * 3.0)
        estimate_normals(o3d, tgt, radius=voxel * 3.0)
        max_dist = voxel * 2.0
        try:
            last_result = o3d.pipelines.registration.registration_colored_icp(
                src,
                tgt,
                max_dist,
                current,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=35,
                ),
            )
        except Exception:
            try:
                last_result = o3d.pipelines.registration.registration_icp(
                    src,
                    tgt,
                    max_dist,
                    current,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
                )
            except Exception:
                continue
        if last_result is not None:
            current = last_result.transformation
    return last_result


def estimate_orb_transform(
    cv2,
    source_frame: Frame,
    target_frame: Frame,
    intr: Intrinsics,
    args: argparse.Namespace,
    depth_unit_m: float,
    rng: np.random.Generator,
) -> np.ndarray | None:
    src_color, src_depth, src_mask = load_rgb_depth_mask(source_frame, args)
    tgt_color, tgt_depth, tgt_mask = load_rgb_depth_mask(target_frame, args)
    src_valid = valid_depth_mask(src_depth, src_mask, args, depth_unit_m)
    tgt_valid = valid_depth_mask(tgt_depth, tgt_mask, args, depth_unit_m)

    src_gray = cv2.cvtColor(src_color, cv2.COLOR_RGB2GRAY)
    tgt_gray = cv2.cvtColor(tgt_color, cv2.COLOR_RGB2GRAY)
    src_mask_u8 = (src_valid.astype(np.uint8) * 255)
    tgt_mask_u8 = (tgt_valid.astype(np.uint8) * 255)

    orb = cv2.ORB_create(nfeatures=4000, fastThreshold=7)
    kp_src, des_src = orb.detectAndCompute(src_gray, src_mask_u8)
    kp_tgt, des_tgt = orb.detectAndCompute(tgt_gray, tgt_mask_u8)
    if des_src is None or des_tgt is None or len(kp_src) < 8 or len(kp_tgt) < 8:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(des_src, des_tgt, k=2)
    matches = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            matches.append(m)
    matches = sorted(matches, key=lambda m: m.distance)[:500]
    if len(matches) < 8:
        return None

    src_pts: list[np.ndarray] = []
    tgt_pts: list[np.ndarray] = []
    for match in matches:
        src_uv = kp_src[match.queryIdx].pt
        tgt_uv = kp_tgt[match.trainIdx].pt
        src_point = backproject_pixel(src_uv, src_depth, src_valid, intr, depth_unit_m)
        tgt_point = backproject_pixel(tgt_uv, tgt_depth, tgt_valid, intr, depth_unit_m)
        if src_point is not None and tgt_point is not None:
            src_pts.append(src_point)
            tgt_pts.append(tgt_point)

    if len(src_pts) < 8:
        return None
    return ransac_rigid_transform(
        np.asarray(src_pts),
        np.asarray(tgt_pts),
        rng=rng,
        threshold_m=max(args.voxel_size_m * 4.0, 0.012),
        iterations=300,
        min_inliers=8,
    )


def backproject_pixel(
    uv: tuple[float, float],
    depth: np.ndarray,
    valid: np.ndarray,
    intr: Intrinsics,
    depth_unit_m: float,
) -> np.ndarray | None:
    u = int(round(uv[0]))
    v = int(round(uv[1]))
    if v < 0 or v >= depth.shape[0] or u < 0 or u >= depth.shape[1]:
        return None
    if not valid[v, u]:
        return None
    z = float(depth[v, u]) * depth_unit_m
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    return np.array([x, y, z], dtype=np.float64)


def ransac_rigid_transform(
    src: np.ndarray,
    tgt: np.ndarray,
    rng: np.random.Generator,
    threshold_m: float,
    iterations: int,
    min_inliers: int,
) -> np.ndarray | None:
    n = len(src)
    if n < 3:
        return None
    best_inliers: np.ndarray | None = None
    best_count = 0
    for _ in range(iterations):
        sample = rng.choice(n, size=3, replace=False)
        transform = rigid_transform(src[sample], tgt[sample])
        if transform is None:
            continue
        moved = transform_points(src, transform)
        errors = np.linalg.norm(moved - tgt, axis=1)
        inliers = errors < threshold_m
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or best_count < min_inliers:
        return None
    return rigid_transform(src[best_inliers], tgt[best_inliers])


def rigid_transform(src: np.ndarray, tgt: np.ndarray) -> np.ndarray | None:
    if len(src) < 3:
        return None
    src_center = src.mean(axis=0)
    tgt_center = tgt.mean(axis=0)
    src0 = src - src_center
    tgt0 = tgt - tgt_center
    h = src0.T @ tgt0
    try:
        u, _s, vt = np.linalg.svd(h)
    except np.linalg.LinAlgError:
        return None
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = tgt_center - r @ src_center
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (points @ transform[:3, :3].T) + transform[:3, 3]


def should_loop_correct(frames: list[Frame], args: argparse.Namespace) -> bool:
    if not args.loop_correction or len(frames) < 30:
        return False
    angles = [frame.angle_deg for frame in frames if frame.angle_deg is not None]
    if len(angles) < 2:
        return False
    return max(angles) - min(angles) > 300.0


def should_angle_closure_correct(frames: list[Frame], args: argparse.Namespace) -> bool:
    if not args.angle_closure_correction or len(frames) < 30:
        return False
    if len({frame.capture for frame in frames}) != 1:
        args._angle_closure = {
            "accepted": False,
            "reason": "multiple_captures_not_supported_for_single_closure",
        }
        return False
    angles = [frame.angle_deg for frame in frames if frame.angle_deg is not None]
    if len(angles) != len(frames):
        args._angle_closure = {"accepted": False, "reason": "missing_angle_metadata"}
        return False
    return max(angles) - min(angles) > 300.0


def apply_angle_closure_correction(
    poses: list[np.ndarray],
    frames: list[Frame],
    pivot: np.ndarray,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    first_angle = frames[0].angle_deg or 0.0
    last_angle = frames[-1].angle_deg or 0.0
    expected_final = rotation_about_axis_around_pivot(
        -math.radians(last_angle - first_angle),
        args.turntable_axis,
        pivot,
    )
    drift = expected_final @ np.linalg.inv(poses[-1])
    trans_drift = float(np.linalg.norm(drift[:3, 3]))
    rot_drift = rotation_angle(drift[:3, :3])
    angle_closure = {
        "accepted": False,
        "translation_m": trans_drift,
        "rotation_rad": rot_drift,
        "expected_final_angle_delta_deg": last_angle - first_angle,
    }

    rejection_reasons: list[str] = []
    if trans_drift > args.angle_closure_max_translation_m:
        rejection_reasons.append(
            f"translation {trans_drift:.4f} > {args.angle_closure_max_translation_m:.4f}"
        )
    if rot_drift > args.angle_closure_max_rotation_rad:
        rejection_reasons.append(
            f"rotation {rot_drift:.4f} > {args.angle_closure_max_rotation_rad:.4f}"
        )
    if rejection_reasons:
        angle_closure["reason"] = "; ".join(rejection_reasons)
        args._angle_closure = angle_closure
        print(f"Angle closure skipped: {angle_closure['reason']}")
        return poses

    corrected: list[np.ndarray] = []
    denom = max(len(poses) - 1, 1)
    for i, pose in enumerate(poses):
        alpha = i / denom
        corrected.append(interpolate_transform(drift, alpha) @ pose)
    angle_closure["accepted"] = True
    args._angle_closure = angle_closure
    print(
        "Angle closure applied: "
        f"translation={trans_drift:.4f} m, rotation={rot_drift:.4f} rad"
    )
    return corrected


def apply_loop_correction(
    o3d,
    reg_clouds: list[FrameCloud],
    poses: list[np.ndarray],
    args: argparse.Namespace,
) -> list[np.ndarray]:
    print("Estimating loop-closure drift...")
    closure_result = refine_pair_registration(
        o3d,
        reg_clouds[-1].pcd,
        reg_clouds[0].pcd,
        poses[-1],
        args,
    )
    if closure_result is None:
        args._loop_drift = {"accepted": False, "reason": "registration_failed"}
        print("Loop correction skipped: last-to-first registration failed.")
        return poses

    last_to_first = closure_result.transformation
    drift = last_to_first @ np.linalg.inv(poses[-1])
    trans_drift = float(np.linalg.norm(drift[:3, 3]))
    rot_drift = rotation_angle(drift[:3, :3])
    loop_drift = {
        "accepted": False,
        "translation_m": trans_drift,
        "rotation_rad": rot_drift,
        "fitness": float(closure_result.fitness),
        "inlier_rmse": float(closure_result.inlier_rmse),
    }
    rejection_reasons: list[str] = []
    if closure_result.fitness < args.loop_min_fitness:
        rejection_reasons.append(
            f"fitness {closure_result.fitness:.3f} < {args.loop_min_fitness:.3f}"
        )
    if closure_result.inlier_rmse > args.loop_max_rmse_m:
        rejection_reasons.append(
            f"rmse {closure_result.inlier_rmse:.4f} > {args.loop_max_rmse_m:.4f}"
        )
    if trans_drift > args.loop_max_translation_m:
        rejection_reasons.append(
            f"translation {trans_drift:.4f} > {args.loop_max_translation_m:.4f}"
        )
    if rot_drift > args.loop_max_rotation_rad:
        rejection_reasons.append(
            f"rotation {rot_drift:.4f} > {args.loop_max_rotation_rad:.4f}"
        )
    if rejection_reasons:
        loop_drift["reason"] = "; ".join(rejection_reasons)
        args._loop_drift = loop_drift
        print(f"Loop correction skipped: {loop_drift['reason']}")
        return poses

    corrected: list[np.ndarray] = []
    denom = max(len(poses) - 1, 1)
    for i, pose in enumerate(poses):
        alpha = i / denom
        corrected.append(interpolate_transform(drift, alpha) @ pose)
    loop_drift["accepted"] = True
    args._loop_drift = loop_drift
    print(
        "Loop correction applied: "
        f"translation={trans_drift:.4f} m, rotation={rot_drift:.4f} rad, "
        f"fitness={closure_result.fitness:.3f}, rmse={closure_result.inlier_rmse:.4f}"
    )
    return corrected


def interpolate_transform(transform: np.ndarray, alpha: float) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation_exp(rotation_log(transform[:3, :3]) * alpha)
    out[:3, 3] = transform[:3, 3] * alpha
    return out


def rotation_angle(r: np.ndarray) -> float:
    value = (np.trace(r) - 1.0) / 2.0
    return float(math.acos(float(np.clip(value, -1.0, 1.0))))


def rotation_log(r: np.ndarray) -> np.ndarray:
    theta = rotation_angle(r)
    if theta < 1e-12:
        return np.zeros(3)
    denom = 2.0 * math.sin(theta)
    axis = np.array(
        [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]],
        dtype=np.float64,
    ) / denom
    return axis * theta


def rotation_exp(v: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    axis = v / theta
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def rotation_about_axis(angle_rad: float, axis: str) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    t = np.eye(4)
    if axis == "y":
        t[:3, :3] = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    elif axis == "z":
        t[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    else:
        fail(f"Unsupported axis: {axis}")
    return t


def rotation_about_axis_around_pivot(angle_rad: float, axis: str, pivot: np.ndarray) -> np.ndarray:
    rotation = rotation_about_axis(angle_rad, axis)
    to_origin = np.eye(4)
    from_origin = np.eye(4)
    to_origin[:3, 3] = -pivot
    from_origin[:3, 3] = pivot
    return from_origin @ rotation @ to_origin


def fuse_clouds(o3d, frame_clouds: list[FrameCloud], poses: list[np.ndarray], args: argparse.Namespace):
    fused = o3d.geometry.PointCloud()
    for fc, pose in zip(frame_clouds, poses):
        cloud = o3d.geometry.PointCloud(fc.pcd)
        cloud.transform(pose)
        fused += cloud
    raw_count = len(fused.points)
    fused = fused.voxel_down_sample(args.voxel_size_m)
    if len(fused.points) > 20:
        fused, _ = fused.remove_statistical_outlier(nb_neighbors=24, std_ratio=1.8)
    if len(fused.points) > 20:
        fused, _ = fused.remove_radius_outlier(nb_points=6, radius=args.voxel_size_m * 5.0)
    if args.fused_dbscan_eps_m > 0 and len(fused.points) > args.fused_dbscan_min_points:
        labels = np.asarray(
            fused.cluster_dbscan(
                eps=args.fused_dbscan_eps_m,
                min_points=args.fused_dbscan_min_points,
                print_progress=False,
            )
        )
        valid_labels = labels[labels >= 0]
        if valid_labels.size:
            counts = np.bincount(valid_labels)
            largest = int(np.argmax(counts))
            keep = np.where(labels == largest)[0]
            if keep.size >= 8:
                fused = fused.select_by_index(keep.tolist())
                args._fused_cluster = {
                    "method": "dbscan_largest",
                    "eps_m": args.fused_dbscan_eps_m,
                    "min_points": args.fused_dbscan_min_points,
                    "kept_points": int(keep.size),
                    "clusters": int(counts.size),
                }
    estimate_normals(o3d, fused, radius=max(args.voxel_size_m * 5.0, 0.015))
    return fused, raw_count


def make_meshes(
    o3d,
    fused,
    frames: list[Frame],
    poses: list[np.ndarray],
    intr: Intrinsics,
    depth_unit_m: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    meshes: dict[str, object] = {}
    if args.box_prior or args.mesh_method == "box":
        meshes["box_prior"] = make_box_prior_mesh(
            o3d, fused, frames, intr, depth_unit_m, args
        )
    if args.mesh_method in ("visual-hull", "all"):
        meshes["visual_hull"] = make_visual_hull_mesh(o3d, frames, poses, intr, fused, args)
    if args.mesh_method in ("tsdf", "all"):
        meshes["tsdf"] = make_tsdf_mesh(o3d, frames, poses, intr, depth_unit_m, fused, args)
    if args.mesh_method in ("both", "bpa"):
        meshes["bpa"] = make_bpa_mesh(o3d, fused, args)
    if args.mesh_method in ("both", "poisson"):
        meshes["poisson"] = make_poisson_mesh(o3d, fused, args)
    if args.mesh_method == "all":
        meshes["bpa"] = make_bpa_mesh(o3d, fused, args)
        meshes["poisson"] = make_poisson_mesh(o3d, fused, args)
    return meshes


def make_tsdf_mesh(
    o3d,
    frames: list[Frame],
    poses: list[np.ndarray],
    intr: Intrinsics,
    depth_unit_m: float,
    fused,
    args: argparse.Namespace,
):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(args.tsdf_voxel_size_m),
        sdf_trunc=float(args.tsdf_trunc_m),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    intrinsic = make_open3d_intrinsic(o3d, intr)
    integrated = 0
    for frame, pose in zip(frames, poses):
        color, depth, mask = load_rgb_depth_mask(frame, args)
        valid = valid_depth_mask(depth, mask, args, depth_unit_m)
        valid = refine_valid_mask_by_color(color, valid, args)
        if np.count_nonzero(valid) < args.foreground_min_points:
            continue
        filtered_depth = np.where(valid, depth, 0).astype(np.uint16)
        filtered_color = color.copy()
        filtered_color[~valid] = 0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(filtered_color),
            o3d.geometry.Image(filtered_depth),
            depth_scale=1.0 / depth_unit_m,
            depth_trunc=args.max_depth_m,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
        integrated += 1

    if integrated == 0:
        fail("TSDF could not integrate any frames after object masking.")
    mesh = volume.extract_triangle_mesh()
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        fail("TSDF produced an empty mesh.")

    if len(fused.points) >= 8:
        bbox = fused.get_axis_aligned_bounding_box()
        bbox = bbox.scale(1.08, bbox.get_center())
        mesh = mesh.crop(bbox)
    cleanup_mesh(mesh)
    keep_largest_mesh_component(mesh)
    mesh.compute_vertex_normals()
    args._tsdf = {
        "integrated_frames": integrated,
        "voxel_size_m": args.tsdf_voxel_size_m,
        "trunc_m": args.tsdf_trunc_m,
    }
    return mesh


def make_visual_hull_mesh(
    o3d,
    frames: list[Frame],
    poses: list[np.ndarray],
    intr: Intrinsics,
    fused,
    args: argparse.Namespace,
):
    try:
        from skimage import measure
    except ImportError:
        fail(
            "visual-hull mesh requires scikit-image. Run "
            "`python -m pip install -r requirements.txt`."
        )

    selected = [
        (frame, pose)
        for i, (frame, pose) in enumerate(zip(frames, poses))
        if i % max(args.visual_hull_frame_step, 1) == 0 and frame.mask_path is not None
    ]
    if len(selected) < 8:
        fail("visual-hull requires at least 8 frames with masks.")

    lower, upper = visual_hull_bounds(fused, args)
    voxel = float(args.visual_hull_voxel_size_m)
    dims = np.maximum(np.ceil((upper - lower) / voxel).astype(int) + 1, 3)
    axes = [lower[i] + np.arange(dims[i], dtype=np.float32) * voxel for i in range(3)]
    xx, yy, zz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    masks = load_visual_hull_masks(selected, args)
    keep = carve_visual_hull(points, selected, masks, intr, args)
    occupancy = keep.reshape(tuple(dims))
    if np.count_nonzero(occupancy) < 8:
        fail("visual-hull carving produced too few occupied voxels.")

    padded = np.pad(occupancy.astype(np.float32), 1, mode="constant", constant_values=0)
    verts, faces, _normals, _values = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=(voxel, voxel, voxel),
    )
    verts = verts + (lower - voxel)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    cleanup_mesh(mesh)
    keep_largest_mesh_component(mesh)
    if args.visual_hull_smooth_iterations > 0:
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=int(args.visual_hull_smooth_iterations)
        )
        cleanup_mesh(mesh)
    mesh.compute_vertex_normals()
    if args.visual_hull_color_mode == "best-view":
        color_visual_hull_vertices_best_view(o3d, mesh, selected, intr, args)
    elif args.visual_hull_color_mode == "average":
        color_visual_hull_vertices(o3d, mesh, selected, intr, args)
    else:
        mesh.paint_uniform_color(np.clip(np.asarray(args.box_prior_color), 0.0, 1.0))
    if args.visual_hull_color_smooth_iterations > 0:
        smooth_mesh_vertex_colors(mesh, int(args.visual_hull_color_smooth_iterations))
    mesh.compute_vertex_normals()
    args._visual_hull = {
        "frames_used": len(selected),
        "voxel_size_m": voxel,
        "grid_dims": dims.tolist(),
        "occupied_voxels": int(np.count_nonzero(occupancy)),
        "min_hit_ratio": args.visual_hull_min_hit_ratio,
        "color_mode": args.visual_hull_color_mode,
        "color_smooth_iterations": args.visual_hull_color_smooth_iterations,
        "bounds_m": {
            "lower": lower.tolist(),
            "upper": upper.tolist(),
        },
    }
    return mesh


def visual_hull_bounds(fused, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(fused.points, dtype=np.float64)
    if len(points) < 8:
        fail("Not enough fused object points to set visual-hull bounds.")
    lower = np.quantile(points, 0.01, axis=0)
    upper = np.quantile(points, 0.99, axis=0)
    padding = max(float(args.visual_hull_padding_m), float(args.visual_hull_voxel_size_m))
    lower -= padding
    upper += padding
    return lower, upper


def load_visual_hull_masks(
    selected: list[tuple[Frame, np.ndarray]],
    args: argparse.Namespace,
) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError:
        cv2 = None
    out: list[np.ndarray] = []
    for frame, _pose in selected:
        mask = np.asarray(Image.open(frame.mask_path).convert("L")) > args.mask_threshold
        if cv2 is not None and args.visual_hull_mask_dilate_px > 0:
            k = int(args.visual_hull_mask_dilate_px) * 2 + 1
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
        out.append(mask)
    return out


def carve_visual_hull(
    points: np.ndarray,
    selected: list[tuple[Frame, np.ndarray]],
    masks: list[np.ndarray],
    intr: Intrinsics,
    args: argparse.Namespace,
) -> np.ndarray:
    keep = np.zeros(len(points), dtype=bool)
    chunk_size = 60000
    min_ratio = float(args.visual_hull_min_hit_ratio)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        pts = points[start:stop]
        hit_counts = np.zeros(len(pts), dtype=np.uint16)
        visible_counts = np.zeros(len(pts), dtype=np.uint16)
        for (_frame, pose), mask in zip(selected, masks):
            cam = transform_points(pts, np.linalg.inv(pose))
            z = cam[:, 2]
            valid_z = z > args.min_depth_m
            u = np.round(intr.fx * cam[:, 0] / z + intr.cx).astype(np.int32)
            v = np.round(intr.fy * cam[:, 1] / z + intr.cy).astype(np.int32)
            in_image = (
                valid_z
                & (u >= 0)
                & (u < intr.width)
                & (v >= 0)
                & (v < intr.height)
            )
            visible_counts[in_image] += 1
            if np.any(in_image):
                hits = np.zeros(len(pts), dtype=bool)
                hits[in_image] = mask[v[in_image], u[in_image]]
                hit_counts[hits] += 1
        visible = visible_counts > 0
        ratio = np.zeros(len(pts), dtype=np.float32)
        ratio[visible] = hit_counts[visible] / visible_counts[visible]
        keep[start:stop] = visible & (ratio >= min_ratio)
    return keep


def color_visual_hull_vertices(
    o3d,
    mesh,
    selected: list[tuple[Frame, np.ndarray]],
    intr: Intrinsics,
    args: argparse.Namespace,
) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        return
    accum = np.zeros((len(vertices), 3), dtype=np.float64)
    counts = np.zeros(len(vertices), dtype=np.float64)
    step = max(args.visual_hull_frame_step, 1)
    color_selected = selected[:: max(step, 2)]
    for frame, pose in color_selected:
        color = np.asarray(Image.open(frame.rgb_path).convert("RGB"), dtype=np.float64) / 255.0
        mask = np.asarray(Image.open(frame.mask_path).convert("L")) > args.mask_threshold
        cam = transform_points(vertices, np.linalg.inv(pose))
        z = cam[:, 2]
        valid_z = z > args.min_depth_m
        u = np.round(intr.fx * cam[:, 0] / z + intr.cx).astype(np.int32)
        v = np.round(intr.fy * cam[:, 1] / z + intr.cy).astype(np.int32)
        in_image = (
            valid_z
            & (u >= 0)
            & (u < intr.width)
            & (v >= 0)
            & (v < intr.height)
        )
        if not np.any(in_image):
            continue
        visible = np.zeros(len(vertices), dtype=bool)
        visible[in_image] = mask[v[in_image], u[in_image]]
        if not np.any(visible):
            continue
        accum[visible] += color[v[visible], u[visible]]
        counts[visible] += 1.0
    colors = np.tile(np.asarray(args.box_prior_color, dtype=np.float64), (len(vertices), 1))
    colored = counts > 0
    colors[colored] = accum[colored] / counts[colored, None]
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))


def color_visual_hull_vertices_best_view(
    o3d,
    mesh,
    selected: list[tuple[Frame, np.ndarray]],
    intr: Intrinsics,
    args: argparse.Namespace,
) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    if len(vertices) == 0 or len(normals) != len(vertices):
        mesh.paint_uniform_color(np.clip(np.asarray(args.box_prior_color), 0.0, 1.0))
        return

    best_score = np.full(len(vertices), -np.inf, dtype=np.float64)
    best_colors = np.tile(np.asarray(args.box_prior_color, dtype=np.float64), (len(vertices), 1))
    fallback_sum = np.zeros((len(vertices), 3), dtype=np.float64)
    fallback_count = np.zeros(len(vertices), dtype=np.float64)

    for frame, pose in selected:
        color = np.asarray(Image.open(frame.rgb_path).convert("RGB"), dtype=np.float64) / 255.0
        mask = np.asarray(Image.open(frame.mask_path).convert("L")) > args.mask_threshold

        inv_pose = np.linalg.inv(pose)
        cam = transform_points(vertices, inv_pose)
        normals_cam = normals @ inv_pose[:3, :3].T
        z = cam[:, 2]
        valid_z = z > args.min_depth_m
        safe_z = np.where(valid_z, z, 1.0)
        u = np.round(intr.fx * cam[:, 0] / safe_z + intr.cx).astype(np.int32)
        v = np.round(intr.fy * cam[:, 1] / safe_z + intr.cy).astype(np.int32)
        in_image = (
            valid_z
            & (u >= 0)
            & (u < intr.width)
            & (v >= 0)
            & (v < intr.height)
        )
        if not np.any(in_image):
            continue

        in_mask = np.zeros(len(vertices), dtype=bool)
        in_mask[in_image] = mask[v[in_image], u[in_image]]
        if not np.any(in_mask):
            continue

        cam_norm = np.linalg.norm(cam, axis=1)
        cam_norm = np.maximum(cam_norm, 1e-9)
        view_dir = -cam / cam_norm[:, None]
        front_score = np.einsum("ij,ij->i", normals_cam, view_dir)
        usable = in_mask & (front_score > 0.05)

        # Keep a fallback average for vertices whose normals are noisy or never
        # become strongly front-facing under the estimated poses.
        fallback_sum[in_mask] += color[v[in_mask], u[in_mask]]
        fallback_count[in_mask] += 1.0

        if not np.any(usable):
            continue
        score = front_score / np.maximum(z * z, 1e-6)
        better = usable & (score > best_score)
        if np.any(better):
            best_score[better] = score[better]
            best_colors[better] = color[v[better], u[better]]

    missing = ~np.isfinite(best_score)
    fallback = missing & (fallback_count > 0)
    if np.any(fallback):
        best_colors[fallback] = fallback_sum[fallback] / fallback_count[fallback, None]
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(best_colors, 0.0, 1.0))


def smooth_mesh_vertex_colors(mesh, iterations: int) -> None:
    colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if colors.size == 0 or triangles.size == 0 or iterations <= 0:
        return
    vertex_count = len(colors)
    neighbors: list[set[int]] = [set() for _ in range(vertex_count)]
    for tri in triangles:
        a, b, c = [int(v) for v in tri]
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    neighbor_arrays = [np.fromiter(n, dtype=np.int64) if n else np.array([], dtype=np.int64) for n in neighbors]
    smoothed = colors.copy()
    for _ in range(iterations):
        next_colors = smoothed.copy()
        for i, idx in enumerate(neighbor_arrays):
            if idx.size == 0:
                continue
            # Keep most of the projected color so printed logos and dark parts do not
            # dissolve, but borrow a little from adjacent vertices to tame speckle.
            next_colors[i] = 0.72 * smoothed[i] + 0.28 * smoothed[idx].mean(axis=0)
        smoothed = next_colors
    mesh.vertex_colors = mesh.vertex_colors.__class__(np.clip(smoothed, 0.0, 1.0))


def make_box_prior_mesh(
    o3d,
    pcd,
    frames: list[Frame],
    intr: Intrinsics,
    depth_unit_m: float,
    args: argparse.Namespace,
):
    if len(pcd.points) < 8:
        fail("Not enough fused points to create a box-prior mesh.")

    if args.box_prior_fit == "silhouette":
        fit = estimate_silhouette_box_fit(frames, intr, depth_unit_m, args)
        if fit is not None:
            center = robust_cloud_center(pcd)
            args._box_prior = {
                **fit,
                "center_m": center.tolist(),
            }
            return make_axis_aligned_box_mesh(o3d, fit["extents_m"], center, args)
        print("Silhouette box fit failed; falling back to fused cloud OBB.")

    obb = pcd.get_oriented_bounding_box(robust=True)
    extents = np.maximum(
        np.asarray(obb.extent, dtype=np.float64) + args.box_prior_margin_m,
        args.voxel_size_m,
    )
    args._box_prior = {
        "source": "fused_obb",
        "extents_m": extents.tolist(),
        "center_m": np.asarray(obb.center, dtype=np.float64).tolist(),
    }
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=float(extents[0]),
        height=float(extents[1]),
        depth=float(extents[2]),
    )
    mesh.translate(-extents * 0.5)
    mesh.rotate(obb.R, center=(0.0, 0.0, 0.0))
    mesh.translate(obb.center)
    mesh.paint_uniform_color(np.clip(np.asarray(args.box_prior_color), 0.0, 1.0))
    mesh.compute_vertex_normals()
    return mesh


def estimate_silhouette_box_fit(
    frames: list[Frame],
    intr: Intrinsics,
    depth_unit_m: float,
    args: argparse.Namespace,
) -> dict | None:
    records: list[tuple[float, float, float]] = []
    for frame in frames:
        if frame.angle_deg is None or frame.mask_path is None:
            continue
        _, depth, mask = load_rgb_depth_mask(frame, args)
        if mask is None or not np.any(mask):
            continue
        depth_m = depth.astype(np.float32) * depth_unit_m
        valid = valid_depth_mask(depth, mask, args, depth_unit_m)
        if np.count_nonzero(valid) < args.foreground_min_points:
            continue

        ys, xs = np.where(mask)
        if xs.size < 10 or ys.size < 10:
            continue
        median_depth = float(np.median(depth_m[valid]))
        if not math.isfinite(median_depth) or median_depth <= 0:
            continue

        width_px = float(xs.max() - xs.min() + 1)
        height_px = float(ys.max() - ys.min() + 1)
        apparent_width_m = width_px * median_depth / intr.fx
        apparent_height_m = height_px * median_depth / intr.fy
        if apparent_width_m <= 0 or apparent_height_m <= 0:
            continue
        records.append((frame.angle_deg % 180.0, apparent_width_m, apparent_height_m))

    if len(records) < 8:
        return None

    data = np.asarray(records, dtype=np.float64)
    angles = data[:, 0]
    apparent_widths = data[:, 1]
    apparent_heights = data[:, 2]
    window = max(float(args.box_prior_angle_window_deg), 1.0)
    front_distance = np.minimum(angles, 180.0 - angles)
    side_distance = np.abs(angles - 90.0)
    front = front_distance <= window
    side = side_distance <= window

    if np.count_nonzero(front) >= 3 and np.count_nonzero(side) >= 3:
        width_m = float(np.median(apparent_widths[front]))
        depth_m = float(np.median(apparent_widths[side]))
        fit_method = "front_side_median"
    else:
        theta = np.deg2rad(angles)
        design = np.column_stack([np.abs(np.cos(theta)), np.abs(np.sin(theta))])
        width_m, depth_m = np.linalg.lstsq(design, apparent_widths, rcond=None)[0]
        width_m = float(max(width_m, args.voxel_size_m))
        depth_m = float(max(depth_m, args.voxel_size_m))
        fit_method = "least_squares_apparent_width"

    height_m = float(np.median(apparent_heights))
    extents = np.array([width_m, height_m, depth_m], dtype=np.float64)
    if not np.all(np.isfinite(extents)) or np.any(extents <= 0):
        return None
    extents = np.maximum(extents + args.box_prior_margin_m, args.voxel_size_m)
    return {
        "source": "silhouette",
        "method": fit_method,
        "frames_used": int(len(records)),
        "front_frames": int(np.count_nonzero(front)),
        "side_frames": int(np.count_nonzero(side)),
        "angle_window_deg": window,
        "extents_m": extents.tolist(),
        "raw_extents_m": [width_m, height_m, depth_m],
    }


def robust_cloud_center(pcd) -> np.ndarray:
    points = np.asarray(pcd.points, dtype=np.float64)
    lower = np.quantile(points, 0.05, axis=0)
    upper = np.quantile(points, 0.95, axis=0)
    return (lower + upper) * 0.5


def make_axis_aligned_box_mesh(o3d, extents: list[float], center: np.ndarray, args: argparse.Namespace):
    extents_np = np.maximum(np.asarray(extents, dtype=np.float64), args.voxel_size_m)
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=float(extents_np[0]),
        height=float(extents_np[1]),
        depth=float(extents_np[2]),
    )
    mesh.translate(-extents_np * 0.5)
    mesh.translate(center)
    mesh.paint_uniform_color(np.clip(np.asarray(args.box_prior_color), 0.0, 1.0))
    mesh.compute_vertex_normals()
    return mesh


def make_bpa_mesh(o3d, pcd, args: argparse.Namespace):
    distances = pcd.compute_nearest_neighbor_distance()
    avg = float(np.mean(distances)) if len(distances) else args.voxel_size_m
    radii = o3d.utility.DoubleVector([avg * 1.5, avg * 2.5, avg * 4.0])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    cleanup_mesh(mesh)
    keep_largest_mesh_component(mesh)
    return mesh


def make_poisson_mesh(o3d, pcd, args: argparse.Namespace):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.poisson_depth, scale=1.1, linear_fit=False
    )
    density_np = np.asarray(densities)
    if density_np.size and args.poisson_density_quantile > 0:
        threshold = np.quantile(density_np, args.poisson_density_quantile)
        mesh.remove_vertices_by_mask(density_np < threshold)
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.08, bbox.get_center())
    mesh = mesh.crop(bbox)
    cleanup_mesh(mesh)
    keep_largest_mesh_component(mesh)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh


def cleanup_mesh(mesh) -> None:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()


def keep_largest_mesh_component(mesh) -> None:
    if len(mesh.triangles) == 0:
        return
    try:
        labels, counts, _areas = mesh.cluster_connected_triangles()
    except Exception:
        return
    labels_np = np.asarray(labels)
    counts_np = np.asarray(counts)
    if labels_np.size == 0 or counts_np.size <= 1:
        return
    largest = int(np.argmax(counts_np))
    mesh.remove_triangles_by_mask((labels_np != largest).tolist())
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()


def export_assets(
    o3d,
    output_dir: Path,
    fused,
    meshes: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    point_path = output_dir / "object_points_colored.ply"
    o3d.io.write_point_cloud(str(point_path), fused)
    paths["colored_point_cloud_ply"] = str(point_path)

    for name, mesh in meshes.items():
        ply_path = output_dir / f"object_mesh_{name}_colored.ply"
        o3d.io.write_triangle_mesh(str(ply_path), mesh, write_vertex_colors=True)
        paths[f"{name}_colored_ply"] = str(ply_path)

        stl_path = output_dir / f"object_mesh_{name}.stl"
        o3d.io.write_triangle_mesh(str(stl_path), mesh)
        paths[f"{name}_stl"] = str(stl_path)

        glb_path = output_dir / f"object_mesh_{name}_vertex_color.glb"
        if export_glb_with_trimesh(mesh, glb_path):
            paths[f"{name}_glb"] = str(glb_path)

        if args.export_obj:
            obj_path = output_dir / f"object_mesh_{name}.obj"
            o3d.io.write_triangle_mesh(str(obj_path), mesh, write_vertex_colors=True)
            paths[f"{name}_obj"] = str(obj_path)
    return paths


def export_glb_with_trimesh(mesh, path: Path) -> bool:
    try:
        import trimesh
    except ImportError:
        print("trimesh not installed; skipping GLB export.")
        return False
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(faces) == 0:
        return False
    colors = None
    if mesh.has_vertex_colors():
        vertex_colors = np.asarray(mesh.vertex_colors)
        colors = np.clip(vertex_colors * 255.0, 0, 255).astype(np.uint8)
        alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
        colors = np.concatenate([colors, alpha], axis=1)
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
    tm.export(str(path))
    return True


def write_poses_csv(output_dir: Path, frames: list[Frame], poses: list[np.ndarray]) -> Path:
    path = output_dir / "estimated_poses_frame_to_ref.csv"
    fields = ["capture", "stem", "angle_deg"] + [f"t{r}{c}" for r in range(4) for c in range(4)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for frame, pose in zip(frames, poses):
            row = {
                "capture": frame.capture,
                "stem": frame.stem,
                "angle_deg": "" if frame.angle_deg is None else frame.angle_deg,
            }
            for r in range(4):
                for c in range(4):
                    row[f"t{r}{c}"] = float(pose[r, c])
            writer.writerow(row)
    return path


def write_frame_stats_csv(output_dir: Path, frame_clouds: list[FrameCloud]) -> Path:
    path = output_dir / "frame_stats.csv"
    fields = [
        "capture",
        "stem",
        "angle_deg",
        "raw_valid_points",
        "sampled_points",
        "median_depth_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for fc in frame_clouds:
            writer.writerow(
                {
                    "capture": fc.frame.capture,
                    "stem": fc.frame.stem,
                    "angle_deg": "" if fc.frame.angle_deg is None else fc.frame.angle_deg,
                    "raw_valid_points": fc.points_before_sample,
                    "sampled_points": fc.points_after_sample,
                    "median_depth_m": fc.median_depth_m,
                }
            )
    return path


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    captures: list[Capture],
    frames: list[Frame],
    intr: Intrinsics,
    depth_unit_m: float,
    raw_fused_count: int,
    fused,
    meshes: dict[str, object],
    exported_paths: dict[str, str],
    elapsed_sec: float,
) -> Path:
    report = {
        "dataset": str(args.dataset),
        "output": str(output_dir),
        "captures": [c.name for c in captures],
        "frame_count": len(frames),
        "pose_mode": args.pose_mode,
        "intrinsics": {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.cx,
            "cy": intr.cy,
        },
        "depth_unit_m": depth_unit_m,
        "voxel_size_m": args.voxel_size_m,
        "turntable_pivot_m": getattr(args, "_turntable_pivot", None),
        "turntable_pivot_fit": getattr(args, "_turntable_pivot_fit", None),
        "turntable_direction": getattr(args, "_turntable_direction", None),
        "raw_fused_points": raw_fused_count,
        "clean_fused_points": len(fused.points),
        "fused_cluster": getattr(args, "_fused_cluster", None),
        "meshes": {
            name: {
                "vertices": len(mesh.vertices),
                "triangles": len(mesh.triangles),
            }
            for name, mesh in meshes.items()
        },
        "exports": exported_paths,
        "box_prior_fit": getattr(args, "_box_prior", None),
        "visual_hull": getattr(args, "_visual_hull", None),
        "tsdf": getattr(args, "_tsdf", None),
        "model_refine": getattr(args, "_model_refine", None),
        "angle_closure": getattr(args, "_angle_closure", None),
        "loop_drift": getattr(args, "_loop_drift", None),
        "elapsed_sec": elapsed_sec,
    }
    if hasattr(args, "_pair_stats"):
        report["pairwise_registration_stats"] = getattr(args, "_pair_stats")
    path = output_dir / "reconstruction_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def print_dry_run(
    dataset: Path,
    captures: list[Capture],
    frames: list[Frame],
    intr: Intrinsics,
    depth_unit_m: float,
) -> None:
    print(f"Dataset: {dataset}")
    print(f"Captures: {len(captures)}")
    for capture in captures:
        mask_status = "yes" if capture.mask_dir is not None else "no"
        print(f"  - {capture.name}: rgb={capture.rgb_dir}, depth={capture.depth_dir}, masks={mask_status}")
    print(f"Frames: {len(frames)}")
    print(
        "Intrinsics: "
        f"{intr.width}x{intr.height}, fx={intr.fx:.3f}, fy={intr.fy:.3f}, "
        f"cx={intr.cx:.3f}, cy={intr.cy:.3f}"
    )
    print(f"Depth unit: {depth_unit_m} m/unit")
    angles = [f.angle_deg for f in frames if f.angle_deg is not None]
    if angles:
        print(f"Angle span: {min(angles):.3f} to {max(angles):.3f} deg")
    print("Dependencies:")
    for name, ok in dependency_status().items():
        print(f"  - {name}: {'ok' if ok else 'missing'}")


def main() -> None:
    args = parse_args()
    if args.mesh_method == "box":
        args.box_prior = True
    start = time.perf_counter()
    if args.dataset is None:
        env_dataset = os.environ.get("RGBD_DATASET")
        if env_dataset:
            args.dataset = Path(env_dataset)
        else:
            fail(
                "Dataset path is required. Use `--dataset <path>` or set the "
                "RGBD_DATASET environment variable."
            )
    dataset = args.dataset.resolve()
    output_dir = (args.output or (dataset / "reconstruction_exports")).resolve()
    args.dataset = dataset

    captures = find_captures(dataset)
    frames = discover_frames(captures, args.every, args.max_frames)
    if not frames:
        fail("No matching RGB/depth frames found.")
    intr = load_intrinsics(dataset, captures, frames[0].rgb_path)
    depth_unit_m = load_depth_unit_m(dataset, args.depth_unit_m)

    if args.dry_run:
        print_dry_run(dataset, captures, frames, intr, depth_unit_m)
        return

    o3d = require_open3d()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.export_object_crops:
        args._object_crop_root = output_dir / "object_only"
    intrinsic = make_open3d_intrinsic(o3d, intr)
    rng = np.random.default_rng(args.seed)

    print(f"Loading {len(frames)} RGB-D frames...")
    frame_clouds: list[FrameCloud] = []
    for i, frame in enumerate(frames):
        frame_clouds.append(
            make_point_cloud(
                o3d,
                frame,
                intrinsic,
                args,
                depth_unit_m,
                rng,
                args.points_per_frame,
            )
        )
        if (i + 1) % 25 == 0 or i == len(frames) - 1:
            print(f"Loaded {i + 1}/{len(frames)} frames")

    print(f"Estimating poses with mode={args.pose_mode}...")
    poses = estimate_poses(o3d, frames, frame_clouds, intr, args, depth_unit_m, rng)
    poses = refine_poses_to_model(o3d, frame_clouds, poses, args, rng)
    write_poses_csv(output_dir, frames, poses)
    write_frame_stats_csv(output_dir, frame_clouds)

    print("Fusing point clouds...")
    fused, raw_fused_count = fuse_clouds(o3d, frame_clouds, poses, args)
    print(f"Fused points: raw={raw_fused_count}, clean={len(fused.points)}")

    print(f"Building mesh(es): {args.mesh_method}...")
    meshes = make_meshes(o3d, fused, frames, poses, intr, depth_unit_m, args)
    print("Exporting assets...")
    exported = export_assets(o3d, output_dir, fused, meshes, args)
    report_path = write_report(
        output_dir,
        args,
        captures,
        frames,
        intr,
        depth_unit_m,
        raw_fused_count,
        fused,
        meshes,
        exported,
        time.perf_counter() - start,
    )

    print("Done.")
    print(f"Report: {report_path}")
    for key, path in exported.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
