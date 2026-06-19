from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ANGLE_COLUMNS = (
    "stage_angle_deg",
    "angle_deg",
    "angle_deg_unwrapped",
    "turntable_angle_deg",
    "theta_deg",
    "rotation_deg",
)


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale_m: float


@dataclass(frozen=True)
class Frame:
    pass_name: str
    stem: str
    rgb_path: Path
    depth_path: Path
    mask_path: Path
    angle_deg: float


@dataclass
class PassReport:
    pass_name: str
    frames_total: int
    frames_used: int
    frames_skipped: int
    axis_center_x_m: float
    axis_center_z_m: float
    circle_radius_m: float
    circle_rmse_m: float
    raw_points: int
    clean_points: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only SAM2-mask-guided RGB-D 3-pass reconstruction for "
            "RealSense turntable captures."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--passes",
        nargs="+",
        default=["pass_01_level", "pass_02_high", "pass_03_low"],
        help="Pass folders to reconstruct and merge.",
    )
    parser.add_argument(
        "--mask-dir",
        default="mask_refined",
        help="Preferred mask directory inside each pass folder.",
    )
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames-per-pass", type=int, default=0)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--max-points-per-frame", type=int, default=12000)
    parser.add_argument("--angle-direction", type=float, default=-1.0)
    parser.add_argument("--angle-offset-deg", type=float, default=0.0)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=1.4)
    parser.add_argument("--foreground-low-percentile", type=float, default=0.5)
    parser.add_argument("--foreground-high-percentile", type=float, default=99.5)
    parser.add_argument("--mask-threshold", type=int, default=60)
    parser.add_argument("--mask-morph-kernel", type=int, default=5)
    parser.add_argument("--mask-erode-px", type=int, default=1)
    parser.add_argument("--voxel-m", type=float, default=0.003)
    parser.add_argument("--stat-nb", type=int, default=24)
    parser.add_argument("--stat-std", type=float, default=2.0)
    parser.add_argument("--icp-voxel-m", type=float, default=0.003)
    parser.add_argument("--ml-contamination", type=float, default=0.035)
    parser.add_argument("--ml-dbscan-eps-m", type=float, default=0.012)
    parser.add_argument("--ml-dbscan-min-samples", type=int, default=12)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--poisson-density-quantile", type=float, default=0.04)
    parser.add_argument("--bpa-radius-m", type=float, default=0.006)
    parser.add_argument("--component-min-triangles", type=int, default=250)
    parser.add_argument("--reference-pass", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def imread(path: Path, flags: int) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    return (int(match.group(1)) if match else 10**12, path.stem)


def read_intrinsics(dataset: Path) -> Intrinsics:
    path = dataset / "camera_intrinsics.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        color = data.get("color_intrinsics") or data.get("intrinsics") or {}
        return Intrinsics(
            width=int(color.get("width", 1280)),
            height=int(color.get("height", 720)),
            fx=float(color["fx"]),
            fy=float(color["fy"]),
            cx=float(color.get("ppx", color.get("cx"))),
            cy=float(color.get("ppy", color.get("cy"))),
            depth_scale_m=float(data.get("depth_scale_meter_per_unit", 0.001)),
        )

    path = dataset / "cam_K.txt"
    if path.exists():
        nums = [
            float(x)
            for x in re.findall(
                r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?",
                path.read_text(encoding="utf-8", errors="ignore"),
            )
        ]
        if len(nums) < 9:
            raise ValueError(f"cam_K.txt must contain at least 9 numbers: {path}")
        k = np.asarray(nums[:9], dtype=np.float64).reshape(3, 3)
        return Intrinsics(
            width=0,
            height=0,
            fx=float(k[0, 0]),
            fy=float(k[1, 1]),
            cx=float(k[0, 2]),
            cy=float(k[1, 2]),
            depth_scale_m=0.001,
        )

    raise FileNotFoundError(
        f"Missing camera_intrinsics.json or cam_K.txt under dataset: {dataset}"
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def choose_angle(row: dict[str, str]) -> float | None:
    lower = {key.lower(): key for key in row.keys()}
    for column in ANGLE_COLUMNS:
        real = lower.get(column.lower())
        if real is None:
            continue
        try:
            return float(row[real])
        except (TypeError, ValueError):
            return None
    return None


def find_mask(pass_dir: Path, stem: str, preferred_mask_dir: str) -> Path | None:
    for folder in (preferred_mask_dir, "mask_refined", "masks", "mask"):
        path = pass_dir / folder / f"{stem}.png"
        if path.exists():
            return path
    return None


def discover_frames(
    dataset: Path,
    pass_name: str,
    preferred_mask_dir: str,
    frame_step: int,
    max_frames: int,
    angle_direction: float,
    angle_offset_deg: float,
) -> list[Frame]:
    pass_dir = dataset / pass_name
    rgb_dir = pass_dir / "rgb"
    depth_dir = pass_dir / "depth"
    if not rgb_dir.is_dir() or not depth_dir.is_dir():
        raise FileNotFoundError(f"Missing rgb/depth folders in pass: {pass_dir}")

    rgb_files = sorted(
        [p for p in rgb_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )
    rows = read_rows(pass_dir / "metadata.csv")

    frames: list[Frame] = []
    for index, rgb_path in enumerate(rgb_files):
        depth_path = depth_dir / f"{rgb_path.stem}.png"
        mask_path = find_mask(pass_dir, rgb_path.stem, preferred_mask_dir)
        if not depth_path.exists() or mask_path is None:
            continue

        angle = choose_angle(rows[index]) if index < len(rows) else None
        if angle is None:
            angle = 360.0 * index / max(len(rgb_files), 1)
        angle = angle_direction * angle + angle_offset_deg
        frames.append(Frame(pass_name, rgb_path.stem, rgb_path, depth_path, mask_path, angle))

    frames = frames[:: max(1, frame_step)]
    if max_frames > 0:
        frames = frames[:max_frames]
    return frames


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = 1 + int(np.argmax(areas))
    return labels == label


def load_mask(mask_path: Path, shape: tuple[int, int], args: argparse.Namespace) -> np.ndarray:
    mask = imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")
    h, w = shape
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = mask > args.mask_threshold

    kernel_size = int(args.mask_morph_kernel)
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0

    mask = largest_component(mask)

    erode_px = int(args.mask_erode_px)
    if erode_px > 0:
        kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
        if eroded.sum() > 0.25 * max(mask.sum(), 1):
            mask = eroded
    return mask


def backproject_frame(
    frame: Frame,
    intr: Intrinsics,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    color_bgr = imread(frame.rgb_path, cv2.IMREAD_COLOR)
    depth = imread(frame.depth_path, cv2.IMREAD_UNCHANGED)
    if color_bgr is None or depth is None:
        return None
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.shape[:2] != color_bgr.shape[:2]:
        depth = cv2.resize(
            depth,
            (color_bgr.shape[1], color_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    mask = load_mask(frame.mask_path, depth.shape[:2], args)

    depth_m = depth.astype(np.float32) * intr.depth_scale_m
    valid = (
        mask
        & np.isfinite(depth_m)
        & (depth_m > args.min_depth_m)
        & (depth_m < args.max_depth_m)
    )
    values = depth_m[valid]
    if values.size < 100:
        return None

    low, high = np.percentile(
        values,
        [args.foreground_low_percentile, args.foreground_high_percentile],
    )
    margin = max(0.01, 0.02 * float(np.median(values)))
    valid &= (depth_m >= low - margin) & (depth_m <= high + margin)

    if args.pixel_stride > 1:
        stride = np.zeros_like(valid, dtype=bool)
        stride[:: args.pixel_stride, :: args.pixel_stride] = True
        valid &= stride

    ys, xs = np.where(valid)
    if xs.size < 100:
        return None

    max_points = int(args.max_points_per_frame)
    if max_points > 0 and xs.size > max_points:
        keep = rng.choice(xs.size, size=max_points, replace=False)
        xs = xs[keep]
        ys = ys[keep]

    z = depth_m[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - intr.cx) * z / intr.fx
    y = (ys.astype(np.float64) - intr.cy) * z / intr.fy
    points = np.column_stack([x, y, z]).astype(np.float32)
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    colors = (rgb[ys, xs].astype(np.float32) / 255.0).clip(0, 1)
    return points, colors


def fit_circle_xz(centroids: np.ndarray) -> tuple[float, float, float, float]:
    x = centroids[:, 0].astype(np.float64)
    z = centroids[:, 2].astype(np.float64)
    finite = np.isfinite(x) & np.isfinite(z)
    x = x[finite]
    z = z[finite]
    if x.size < 3:
        return float(np.nanmedian(x)), float(np.nanmedian(z)), 0.0, float("nan")
    a = np.column_stack([2 * x, 2 * z, np.ones_like(x)])
    b = x * x + z * z
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cz, c = sol
    radius = math.sqrt(max(float(c + cx * cx + cz * cz), 0.0))
    residual = np.sqrt((x - cx) ** 2 + (z - cz) ** 2) - radius
    rmse = float(np.sqrt(np.mean(residual**2)))
    return float(cx), float(cz), float(radius), rmse


def rot_y(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def make_cloud(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and len(colors) == len(points):
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64).clip(0, 1))
    return cloud


def clean_cloud(cloud: o3d.geometry.PointCloud, args: argparse.Namespace) -> o3d.geometry.PointCloud:
    if args.voxel_m > 0:
        cloud = cloud.voxel_down_sample(float(args.voxel_m))
    if len(cloud.points) > args.stat_nb > 0:
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=int(args.stat_nb),
            std_ratio=float(args.stat_std),
        )
    return cloud


def reconstruct_pass(
    frames: list[Frame],
    intr: Intrinsics,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[o3d.geometry.PointCloud, PassReport]:
    rng = np.random.default_rng(42)
    raw_points: list[np.ndarray] = []
    raw_colors: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    used_records: list[tuple[Frame, np.ndarray, np.ndarray]] = []
    skipped = 0

    for frame in frames:
        result = backproject_frame(frame, intr, args, rng)
        if result is None:
            skipped += 1
            continue
        points, colors = result
        centroids.append(np.median(points, axis=0))
        used_records.append((frame, points, colors))

    if not used_records:
        raise RuntimeError(f"No usable frames for pass: {frames[0].pass_name if frames else '?'}")

    centroid_np = np.asarray(centroids, dtype=np.float64)
    center_x, center_z, radius, rmse = fit_circle_xz(centroid_np)
    center = np.asarray([center_x, 0.0, center_z], dtype=np.float64)

    for frame, points, colors in used_records:
        aligned = (points.astype(np.float64) - center) @ rot_y(-frame.angle_deg).T
        # camera coordinates: x right, y down, z forward
        # output coordinates: x right, y forward, z up
        aligned = np.column_stack([aligned[:, 0], aligned[:, 2], -aligned[:, 1]])
        raw_points.append(aligned.astype(np.float32))
        raw_colors.append(colors)

    points = np.concatenate(raw_points, axis=0)
    colors = np.concatenate(raw_colors, axis=0)
    raw_cloud = make_cloud(points, colors)
    clean = clean_cloud(raw_cloud, args)

    pass_name = used_records[0][0].pass_name
    o3d.io.write_point_cloud(str(out_dir / f"{pass_name}_raw.ply"), raw_cloud)
    o3d.io.write_point_cloud(str(out_dir / f"{pass_name}_clean.ply"), clean)

    return clean, PassReport(
        pass_name=pass_name,
        frames_total=len(frames),
        frames_used=len(used_records),
        frames_skipped=skipped,
        axis_center_x_m=float(center_x),
        axis_center_z_m=float(center_z),
        circle_radius_m=float(radius),
        circle_rmse_m=float(rmse),
        raw_points=int(len(raw_cloud.points)),
        clean_points=int(len(clean.points)),
    )


def transform_cloud(cloud: o3d.geometry.PointCloud, transform: np.ndarray) -> o3d.geometry.PointCloud:
    out = o3d.geometry.PointCloud(cloud)
    out.transform(transform)
    return out


def transform_center(transform: np.ndarray, center: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ center) + transform[:3, 3]


def translation(vector: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = vector
    return matrix


def rot_x4(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    out = np.eye(4)
    out[:3, :3] = np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]])
    return out


def rot_y4(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    out = np.eye(4)
    out[:3, :3] = np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return out


def rot_z4(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    out = np.eye(4)
    out[:3, :3] = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return out


def icp_ready(cloud: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    out = cloud.voxel_down_sample(voxel)
    if len(out.points) > 0:
        out.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 8, max_nn=40)
        )
    return out


def robust_volume(cloud: o3d.geometry.PointCloud) -> float:
    points = np.asarray(cloud.points)
    if points.size == 0:
        return float("inf")
    low = np.percentile(points, 2, axis=0)
    high = np.percentile(points, 98, axis=0)
    return float(np.prod(np.maximum(high - low, 1e-6)))


def initial_transforms(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud):
    source_center = np.asarray(source.points).mean(axis=0)
    target_center = np.asarray(target.points).mean(axis=0)
    for rx in (0, 30, -30, 60, -60, 90, -90, 180):
        for rz in (0, 90, -90, 180):
            transform = rot_z4(rz) @ rot_x4(rx)
            yield f"rx={rx},rz={rz}", translation(target_center - transform_center(transform, source_center)) @ transform
    for ry in (90, -90, 180):
        transform = rot_y4(ry)
        yield f"ry={ry}", translation(target_center - transform_center(transform, source_center)) @ transform


def align_cloud_to_target(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    args: argparse.Namespace,
) -> tuple[o3d.geometry.PointCloud, dict]:
    voxel = float(args.icp_voxel_m)
    source_small = icp_ready(source, voxel)
    target_small = icp_ready(target, voxel)
    best: dict | None = None

    for name, initial in initial_transforms(source_small, target_small):
        coarse = o3d.pipelines.registration.registration_icp(
            source_small,
            target_small,
            voxel * 14,
            initial,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
        )
        fine = o3d.pipelines.registration.registration_icp(
            source_small,
            target_small,
            voxel * 5,
            coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
        )
        merged_test = o3d.geometry.PointCloud(target_small)
        merged_test += transform_cloud(source_small, fine.transformation)
        score = (-float(fine.fitness), float(fine.inlier_rmse), robust_volume(merged_test))
        if best is None or score < best["score"]:
            best = {
                "candidate": name,
                "fitness": float(fine.fitness),
                "rmse": float(fine.inlier_rmse),
                "transformation": fine.transformation,
                "score": score,
            }

    if best is None:
        raise RuntimeError("ICP alignment failed.")
    aligned = transform_cloud(source, best["transformation"])
    return aligned, {
        "candidate": best["candidate"],
        "fitness": best["fitness"],
        "rmse": best["rmse"],
        "transformation": best["transformation"].tolist(),
    }


def merge_passes(
    pass_clouds: dict[str, o3d.geometry.PointCloud],
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[o3d.geometry.PointCloud, dict]:
    if args.reference_pass and args.reference_pass in pass_clouds:
        reference_name = args.reference_pass
    else:
        reference_name = max(pass_clouds, key=lambda name: len(pass_clouds[name].points))

    merged = o3d.geometry.PointCloud(pass_clouds[reference_name])
    target = clean_cloud(merged, args)
    align_report = {
        reference_name: {
            "fitness": 1.0,
            "rmse": 0.0,
            "transformation": np.eye(4).tolist(),
        }
    }

    for name, cloud in pass_clouds.items():
        if name == reference_name:
            continue
        aligned, report = align_cloud_to_target(cloud, target, args)
        align_report[name] = report
        o3d.io.write_point_cloud(str(out_dir / f"{name}_aligned.ply"), aligned)
        merged += aligned
        target = clean_cloud(merged, args)

    merged_clean = clean_cloud(merged, args)
    o3d.io.write_point_cloud(str(out_dir / "merged_passes_raw.ply"), merged)
    o3d.io.write_point_cloud(str(out_dir / "merged_passes_clean.ply"), merged_clean)
    return merged_clean, {"reference_pass": reference_name, "alignments": align_report}


def ml_refine(
    cloud: o3d.geometry.PointCloud,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[o3d.geometry.PointCloud, dict]:
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors) if cloud.has_colors() else np.zeros_like(points)
    features = np.column_stack([points * 4.0, colors])
    features = StandardScaler().fit_transform(features)

    iso = IsolationForest(
        n_estimators=300,
        contamination=float(args.ml_contamination),
        random_state=42,
        n_jobs=-1,
    )
    inlier_mask = iso.fit_predict(features) == 1
    inlier_cloud = cloud.select_by_index(np.flatnonzero(inlier_mask).astype(np.int64))
    inlier_points = np.asarray(inlier_cloud.points)

    labels = DBSCAN(
        eps=float(args.ml_dbscan_eps_m),
        min_samples=int(args.ml_dbscan_min_samples),
        n_jobs=-1,
    ).fit_predict(inlier_points)
    valid = labels >= 0
    if np.any(valid):
        unique, counts = np.unique(labels[valid], return_counts=True)
        best_label = int(unique[np.argmax(counts)])
        keep = labels == best_label
        refined = inlier_cloud.select_by_index(np.flatnonzero(keep).astype(np.int64))
        clusters = int(len(unique))
        noise = int(np.sum(labels < 0))
    else:
        refined = inlier_cloud
        best_label = None
        clusters = 0
        noise = int(len(labels))

    o3d.io.write_point_cloud(str(out_dir / "points_ml_refined.ply"), refined)
    report = {
        "tools": ["scikit-learn IsolationForest", "scikit-learn DBSCAN"],
        "input_points": int(len(cloud.points)),
        "isolation_forest": {
            "n_estimators": 300,
            "contamination": float(args.ml_contamination),
            "inlier_points": int(inlier_mask.sum()),
            "outlier_points": int((~inlier_mask).sum()),
        },
        "dbscan": {
            "eps_m": float(args.ml_dbscan_eps_m),
            "min_samples": int(args.ml_dbscan_min_samples),
            "clusters": clusters,
            "kept_label": best_label,
            "noise_points": noise,
        },
        "output_points": int(len(refined.points)),
    }
    return refined, report


def estimate_normals(cloud: o3d.geometry.PointCloud, radius: float = 0.025) -> None:
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50)
    )
    try:
        cloud.orient_normals_consistent_tangent_plane(60)
    except RuntimeError:
        pass


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def keep_large_components(
    mesh: o3d.geometry.TriangleMesh,
    min_triangles: int,
) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0 or min_triangles <= 0:
        return mesh
    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    counts = np.asarray(counts)
    if counts.size == 0:
        return mesh
    keep = np.where(counts >= min_triangles)[0]
    if keep.size == 0:
        keep = np.asarray([int(np.argmax(counts))])
    mesh.remove_triangles_by_mask(~np.isin(clusters, keep))
    mesh.remove_unreferenced_vertices()
    return clean_mesh(mesh)


def transfer_colors(mesh: o3d.geometry.TriangleMesh, cloud: o3d.geometry.PointCloud) -> None:
    if len(mesh.vertices) == 0 or not cloud.has_colors():
        return
    tree = o3d.geometry.KDTreeFlann(cloud)
    colors = np.asarray(cloud.colors)
    vertex_colors = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
    for index, vertex in enumerate(np.asarray(mesh.vertices)):
        _, nearest, _ = tree.search_knn_vector_3d(vertex, 1)
        vertex_colors[index] = colors[nearest[0]]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)


def write_mesh(mesh: o3d.geometry.TriangleMesh, stem: Path) -> dict:
    outputs: dict[str, str] = {}
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return outputs
    ply = stem.with_suffix(".ply")
    stl_m = stem.with_name(stem.name + "_meter").with_suffix(".stl")
    stl_mm = stem.with_name(stem.name + "_mm").with_suffix(".stl")
    glb = stem.with_suffix(".glb")

    o3d.io.write_triangle_mesh(str(ply), mesh, write_vertex_colors=True)
    o3d.io.write_triangle_mesh(str(stl_m), mesh)
    scaled = o3d.geometry.TriangleMesh(mesh)
    scaled.scale(1000.0, center=(0, 0, 0))
    o3d.io.write_triangle_mesh(str(stl_mm), scaled)
    try:
        o3d.io.write_triangle_mesh(str(glb), mesh, write_vertex_colors=True)
    except RuntimeError:
        glb = None

    outputs["ply"] = str(ply)
    outputs["stl_meter"] = str(stl_m)
    outputs["stl_mm"] = str(stl_mm)
    if glb is not None:
        outputs["glb"] = str(glb)
    return outputs


def mesh_from_cloud(
    cloud: o3d.geometry.PointCloud,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    cloud = clean_cloud(cloud, args)
    estimate_normals(cloud)
    o3d.io.write_point_cloud(str(out_dir / "points_clean.ply"), cloud)

    poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud,
        depth=int(args.poisson_depth),
    )
    densities = np.asarray(densities)
    if densities.size and args.poisson_density_quantile > 0:
        threshold = np.quantile(densities, float(args.poisson_density_quantile))
        poisson.remove_vertices_by_mask(densities < threshold)
    bbox = cloud.get_axis_aligned_bounding_box()
    margin = max(float(np.linalg.norm(bbox.get_extent())) * 0.03, 0.01)
    poisson = poisson.crop(
        o3d.geometry.AxisAlignedBoundingBox(
            min_bound=bbox.min_bound - margin,
            max_bound=bbox.max_bound + margin,
        )
    )
    poisson = keep_large_components(clean_mesh(poisson), int(args.component_min_triangles))
    transfer_colors(poisson, cloud)

    dists = np.asarray(cloud.compute_nearest_neighbor_distance())
    median_dist = float(np.median(dists)) if dists.size else float(args.bpa_radius_m)
    radii = sorted(
        {
            max(median_dist * 1.5, 1e-5),
            max(median_dist * 2.5, 1e-5),
            max(median_dist * 4.0, 1e-5),
            max(float(args.bpa_radius_m), 1e-5),
            max(float(args.bpa_radius_m) * 1.5, 1e-5),
        }
    )
    bpa = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        cloud,
        o3d.utility.DoubleVector(radii),
    )
    bpa = keep_large_components(clean_mesh(bpa), max(60, int(args.component_min_triangles) // 3))
    transfer_colors(bpa, cloud)

    return {
        "clean_point_count": int(len(cloud.points)),
        "clean_extent_m": [float(x) for x in cloud.get_axis_aligned_bounding_box().get_extent()],
        "poisson": {
            "vertices": int(len(poisson.vertices)),
            "triangles": int(len(poisson.triangles)),
            "outputs": write_mesh(poisson, out_dir / "mesh_poisson"),
        },
        "bpa": {
            "vertices": int(len(bpa.vertices)),
            "triangles": int(len(bpa.triangles)),
            "outputs": write_mesh(bpa, out_dir / "mesh_bpa"),
        },
    }


def equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.55)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_preview(cloud: o3d.geometry.PointCloud, path: Path) -> None:
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors) if cloud.has_colors() else None
    if len(points) > 12000:
        rng = np.random.default_rng(7)
        sample = rng.choice(len(points), size=12000, replace=False)
        points = points[sample]
        colors = colors[sample] if colors is not None else None

    views = [("iso", 25, 45), ("front", 0, 0), ("side", 0, 90), ("top", 90, -90)]
    fig = plt.figure(figsize=(12, 10), dpi=160)
    for index, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=0.6,
            c=colors if colors is not None else "#4a90e2",
            linewidths=0,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        equal_axes(ax, points)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    ensure_dir(output)

    intr = read_intrinsics(dataset)
    all_frames = {
        pass_name: discover_frames(
            dataset,
            pass_name,
            args.mask_dir,
            args.frame_step,
            args.max_frames_per_pass,
            args.angle_direction,
            args.angle_offset_deg,
        )
        for pass_name in args.passes
    }

    if args.dry_run:
        print(json.dumps({name: len(frames) for name, frames in all_frames.items()}, indent=2))
        return

    pass_dir = output / "passes"
    ensure_dir(pass_dir)
    pass_clouds: dict[str, o3d.geometry.PointCloud] = {}
    pass_reports: list[PassReport] = []
    for pass_name, frames in all_frames.items():
        cloud, report = reconstruct_pass(frames, intr, args, pass_dir)
        pass_clouds[pass_name] = cloud
        pass_reports.append(report)

    merged, merge_report = merge_passes(pass_clouds, args, output)
    refined, ml_report = ml_refine(merged, args, output)
    mesh_report = mesh_from_cloud(refined, args, output)
    save_preview(refined, output / "preview.png")

    report = {
        "dataset": str(dataset),
        "output": str(output),
        "intrinsics": asdict(intr),
        "passes": [asdict(item) for item in pass_reports],
        "merge": merge_report,
        "ml_refinement": ml_report,
        "mesh": mesh_report,
        "outputs": {
            "preview": str(output / "preview.png"),
            "points_clean": str(output / "points_clean.ply"),
            "points_ml_refined": str(output / "points_ml_refined.ply"),
            "report": str(output / "reconstruction_report.json"),
        },
    }
    write_json(output / "reconstruction_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
