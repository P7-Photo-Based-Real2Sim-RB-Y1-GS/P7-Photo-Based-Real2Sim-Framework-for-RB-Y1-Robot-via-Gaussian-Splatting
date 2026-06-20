#!/usr/bin/env python3
"""Pure NumPy helpers for RB-Y1 webcam 6D task-space teleoperation.

The module intentionally has no Isaac Sim or ROS dependency so its geometry,
filtering, and IK routines can be unit-tested with ordinary Python.
"""
from __future__ import annotations

import json
import math
import socket
import time
from dataclasses import dataclass

import numpy as np


def normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm > 1.0e-9:
        return value / norm
    if fallback is None:
        return np.zeros_like(value)
    fallback_value = np.asarray(fallback, dtype=np.float64)
    fallback_norm = float(np.linalg.norm(fallback_value))
    if fallback_norm <= 1.0e-9:
        return np.zeros_like(value)
    return fallback_value / fallback_norm


def smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def minimum_jerk(value: float) -> float:
    """Quintic interpolation with zero velocity/acceleration at both ends."""
    x = float(np.clip(value, 0.0, 1.0))
    return 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5


def wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize(np.asarray(q_wxyz, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0]))
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""
    R = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    # Project small landmark noise back onto SO(3).
    u, _, vh = np.linalg.svd(R)
    R = u @ vh
    if np.linalg.det(R) < 0.0:
        u[:, -1] *= -1.0
        R = u @ vh

    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(R)))
        if index == 0:
            s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1.0e-12)) * 2.0
            q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
        elif index == 1:
            s = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1.0e-12)) * 2.0
            q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
        else:
            s = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1.0e-12)) * 2.0
            q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q = normalize(q, np.array([1.0, 0.0, 0.0, 0.0]))
    if q[0] < 0.0:
        q *= -1.0
    return q


def quat_error_rotvec(target_wxyz: np.ndarray, current_wxyz: np.ndarray) -> np.ndarray:
    """Shortest target * inverse(current) quaternion error as rotation vector."""
    target = normalize(np.asarray(target_wxyz, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0]))
    current = normalize(np.asarray(current_wxyz, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0]))
    error = quat_multiply(target, quat_conjugate(current))
    if error[0] < 0.0:
        error *= -1.0
    scalar = float(np.clip(error[0], -1.0, 1.0))
    vector = error[1:]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm < 1.0e-9:
        return 2.0 * vector
    angle = 2.0 * math.atan2(vector_norm, scalar)
    return vector / vector_norm * angle


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to an axis-angle rotation vector."""
    R = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1.0e-9:
        return np.zeros(3, dtype=np.float64)
    if math.pi - angle < 1.0e-5:
        # Stable near pi: recover an axis from the diagonal terms.
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) * 0.5, 0.0))
        if R[2, 1] - R[1, 2] < 0.0:
            axis[0] *= -1.0
        if R[0, 2] - R[2, 0] < 0.0:
            axis[1] *= -1.0
        if R[1, 0] - R[0, 1] < 0.0:
            axis[2] *= -1.0
        axis = normalize(axis, np.array([1.0, 0.0, 0.0]))
        return axis * angle
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    ) / (2.0 * math.sin(angle))
    return axis * angle


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Convert an axis-angle rotation vector to a 3x3 rotation matrix."""
    vector = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-9:
        return np.eye(3, dtype=np.float64)
    axis = vector / angle
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def quat_slerp(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    qa = normalize(np.asarray(a, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0]))
    qb = normalize(np.asarray(b, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0]))
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    t = float(np.clip(fraction, 0.0, 1.0))
    if dot > 0.9995:
        return normalize(qa + t * (qb - qa), qa)
    theta = math.acos(float(np.clip(dot, -1.0, 1.0)))
    sin_theta = math.sin(theta)
    return normalize(
        math.sin((1.0 - t) * theta) / sin_theta * qa + math.sin(t * theta) / sin_theta * qb,
        qa,
    )


def limit_quat_step(current: np.ndarray, target: np.ndarray, maximum_angle: float) -> np.ndarray:
    error = quat_error_rotvec(target, current)
    angle = float(np.linalg.norm(error))
    if angle <= maximum_angle or angle < 1.0e-9:
        return normalize(target, current)
    fraction = maximum_angle / angle
    return quat_slerp(current, target, fraction)


def clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= maximum or norm < 1.0e-12:
        return value
    return value * (maximum / norm)


class UdpPoseReceiver:
    def __init__(self, bind_ip: str, port: int):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((bind_ip, port))
        self.socket.setblocking(False)
        self.last_packet: dict | None = None
        self.last_receive_monotonic = 0.0

    def poll(self) -> dict | None:
        newest = None
        while True:
            try:
                payload, _ = self.socket.recvfrom(1_000_000)
            except BlockingIOError:
                break
            try:
                candidate = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(candidate, dict):
                newest = candidate
        if newest is not None:
            self.last_packet = newest
            self.last_receive_monotonic = time.monotonic()
        return newest

    def age(self) -> float:
        if self.last_receive_monotonic <= 0.0:
            return float("inf")
        return time.monotonic() - self.last_receive_monotonic

    def close(self) -> None:
        self.socket.close()


def _pose_dictionary(packet: dict, visibility: float) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    output: dict[str, np.ndarray] = {}
    visibilities: dict[str, float] = {}
    for item in packet.get("pose_landmarks", []):
        name = str(item.get("landmark_name", ""))
        if not name:
            continue
        vis = float(item.get("visibility", 1.0))
        visibilities[name] = vis
        if vis < visibility:
            continue
        position = item.get("world_position")
        if not isinstance(position, dict):
            position = item.get("image_landmark")
        if not isinstance(position, dict):
            continue
        output[name] = np.array(
            [float(position.get("x", 0.0)), float(position.get("y", 0.0)), float(position.get("z", 0.0))],
            dtype=np.float64,
        )
    return output, visibilities


def _pose_image_dictionary(packet: dict) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Return image-space pose landmarks regardless of world-landmark availability.

    MediaPipe world depth can jitter when the arm approaches the camera.  The
    image y coordinate is nevertheless very reliable for the operator's wrist
    up/down motion, so v12 uses it as a vertical-reference signal.
    """
    output: dict[str, np.ndarray] = {}
    visibilities: dict[str, float] = {}
    for item in packet.get("pose_landmarks", []):
        name = str(item.get("landmark_name", ""))
        if not name:
            continue
        vis = float(item.get("visibility", 1.0))
        visibilities[name] = vis
        position = item.get("image_landmark")
        if not isinstance(position, dict):
            continue
        output[name] = np.array(
            [float(position.get("x", 0.0)), float(position.get("y", 0.0)), float(position.get("z", 0.0))],
            dtype=np.float64,
        )
    return output, visibilities


def _hand_dictionary(packet: dict, side: str) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for item in packet.get(f"{side}_hand_landmarks", []):
        try:
            index = int(item.get("landmark_id"))
        except (TypeError, ValueError):
            continue
        position = item.get("image_landmark")
        if not isinstance(position, dict):
            continue
        output[index] = np.array(
            [float(position.get("x", 0.0)), float(position.get("y", 0.0)), float(position.get("z", 0.0))],
            dtype=np.float64,
        )
    return output


def _camera_to_robot_matrix(human_forward_sign: float) -> np.ndarray:
    # MediaPipe image/world convention used by this project:
    # camera +x right, +y down, z depth. Robot task: +x forward, +y left, +z up.
    return np.array(
        [[0.0, 0.0, float(human_forward_sign)], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )


def palm_rotation_from_packet(packet: dict, side: str, human_forward_sign: float) -> np.ndarray | None:
    """Return a palm frame expressed in robot task axes.

    Columns are: across-palm (pinky->index), wrist->fingers, palm normal.  Only
    relative motion after calibration is used, so a fixed left/right frame offset
    is harmless while wrist roll/pitch/yaw changes remain observable.
    """
    hand = _hand_dictionary(packet, side)
    if not all(index in hand for index in (0, 5, 9, 17)):
        return None
    wrist = hand[0]
    index_mcp = hand[5]
    middle_mcp = hand[9]
    pinky_mcp = hand[17]
    x_camera = normalize(index_mcp - pinky_mcp)
    y_camera = normalize(middle_mcp - wrist)
    z_camera = normalize(np.cross(x_camera, y_camera))
    if np.linalg.norm(z_camera) < 1.0e-8:
        return None
    y_camera = normalize(np.cross(z_camera, x_camera))
    C = _camera_to_robot_matrix(human_forward_sign)
    x_robot = normalize(C @ x_camera)
    y_robot = normalize(C @ y_camera)
    z_robot = normalize(np.cross(x_robot, y_robot))
    y_robot = normalize(np.cross(z_robot, x_robot))
    R = np.column_stack((x_robot, y_robot, z_robot))
    if np.linalg.det(R) < 0.0:
        R[:, 2] *= -1.0
    return R


@dataclass
class MapperConfig:
    visibility: float = 0.45
    forward_scale: float = 0.85
    lateral_scale: float = 0.75
    vertical_scale: float = 1.05
    wrist_up_image_scale: float = 0.70
    wrist_up_blend: float = 0.85
    wrist_up_deadzone: float = 0.012
    elbow_forward_scale: float = 0.55
    elbow_lateral_scale: float = 0.55
    elbow_vertical_scale: float = 0.70
    extension_forward_scale: float = 0.42
    human_forward_sign: float = -1.0
    filter_alpha: float = 0.28
    palm_filter_alpha: float = 0.18
    left_wrist_signs: tuple[float, float, float] = (1.0, 1.0, 1.0)
    right_wrist_signs: tuple[float, float, float] = (1.0, 1.0, 1.0)
    pinch_deadzone: float = 0.05
    pinch_full_close: float = 0.62
    pinch_gamma: float = 0.65
    human_elbow_bent_deg: float = 55.0
    human_elbow_straight_deg: float = 168.0


class WebcamTaskMapper:
    """Map calibrated human wrist, elbow, shoulder, and palm motion to robot tasks."""

    def __init__(self, config: MapperConfig):
        self.config = config
        self.neutral: dict[str, np.ndarray] | None = None
        self.filtered: dict[str, np.ndarray] = {}
        self.neutral_palms: dict[str, np.ndarray] = {}
        self.filtered_palm_targets: dict[str, np.ndarray] = {}
        self.neutral_shoulder_features: dict[str, np.ndarray] = {}
        self.neutral_elbow_extensions: dict[str, float] = {}
        self.neutral_wrist_heights: dict[str, float] = {}
        self.filtered_wrist_heights: dict[str, float] = {}

    def _body_frame(self, pose: dict[str, np.ndarray]):
        needed = ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP")
        if not all(name in pose for name in needed):
            return None
        shoulder_mid = 0.5 * (pose["LEFT_SHOULDER"] + pose["RIGHT_SHOULDER"])
        hip_mid = 0.5 * (pose["LEFT_HIP"] + pose["RIGHT_HIP"])
        left_axis = normalize(pose["LEFT_SHOULDER"] - pose["RIGHT_SHOULDER"], np.array([-1.0, 0.0, 0.0]))
        up_axis = normalize(shoulder_mid - hip_mid, np.array([0.0, -1.0, 0.0]))
        front_axis = normalize(np.cross(left_axis, up_axis), np.array([0.0, 0.0, 1.0]))
        if float(np.dot(front_axis, np.array([0.0, 0.0, 1.0]))) < 0.0:
            front_axis *= -1.0
        up_axis = normalize(np.cross(front_axis, left_axis), up_axis)
        return shoulder_mid, left_axis, up_axis, front_axis

    def _project(self, relative: np.ndarray, frame, shoulder_available: bool) -> np.ndarray:
        if frame is not None and shoulder_available:
            _, left_axis, up_axis, front_axis = frame
            return np.array(
                [
                    self.config.human_forward_sign * float(np.dot(relative, front_axis)),
                    float(np.dot(relative, left_axis)),
                    float(np.dot(relative, up_axis)),
                ],
                dtype=np.float64,
            )
        return np.array(
            [
                self.config.human_forward_sign * float(relative[2]),
                float(relative[0]),
                -float(relative[1]),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _shoulder_features(upper_arm: np.ndarray, forearm: np.ndarray) -> np.ndarray:
        u = normalize(upper_arm, np.array([1.0, 0.0, 0.0]))
        yaw = math.atan2(float(u[1]), float(u[0]))
        elevation = math.atan2(float(u[2]), math.hypot(float(u[0]), float(u[1])))
        plane_normal = normalize(np.cross(u, normalize(forearm, np.array([1.0, 0.0, 0.0]))))
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        reference = normalize(world_up - float(np.dot(world_up, u)) * u, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(plane_normal) < 1.0e-8:
            twist = 0.0
        else:
            twist = math.atan2(float(np.dot(np.cross(reference, plane_normal), u)), float(np.dot(reference, plane_normal)))
        return np.array([elevation, yaw, twist], dtype=np.float64)

    def _observations(
        self, packet: dict, required_sides: tuple[str, ...]
    ) -> tuple[
        dict[str, np.ndarray] | None,
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, float],
        str,
    ]:
        strict = float(np.clip(self.config.visibility, 0.0, 1.0))
        thresholds = (strict, max(0.12, strict * 0.45))
        last_reason = "no pose landmarks"

        image_pose, image_vis = _pose_image_dictionary(packet)
        for threshold in thresholds:
            pose, vis = _pose_dictionary(packet, threshold)
            frame = self._body_frame(pose)
            coordinates: dict[str, np.ndarray] = {}
            palms: dict[str, np.ndarray] = {}
            shoulder_features: dict[str, np.ndarray] = {}
            elbow_extensions: dict[str, float] = {}
            wrist_heights: dict[str, float] = {}
            missing: list[str] = []
            elbow_count = 0
            palm_count = 0

            for side_lower in required_sides:
                side = side_lower.upper()
                shoulder_name = f"{side}_SHOULDER"
                elbow_name = f"{side}_ELBOW"
                wrist_name = f"{side}_WRIST"
                shoulder = pose.get(shoulder_name)
                elbow = pose.get(elbow_name)
                wrist = pose.get(wrist_name)
                if wrist is None:
                    missing.append(f"{wrist_name}(vis={vis.get(wrist_name, -1.0):.2f})")
                    continue

                wrist_relative = wrist - shoulder if shoulder is not None else wrist
                coordinates[side_lower] = self._project(wrist_relative, frame, shoulder is not None)

                image_shoulder = image_pose.get(shoulder_name)
                image_wrist = image_pose.get(wrist_name)
                image_visibility_ok = (
                    image_shoulder is not None
                    and image_wrist is not None
                    and image_vis.get(shoulder_name, 1.0) >= max(0.08, threshold * 0.35)
                    and image_vis.get(wrist_name, 1.0) >= max(0.08, threshold * 0.35)
                )
                if image_visibility_ok:
                    # Image +y points downward.  shoulder_y - wrist_y therefore
                    # grows when the operator raises the wrist.
                    wrist_heights[side_lower] = float(image_shoulder[1] - image_wrist[1])

                if shoulder is not None and elbow is not None:
                    upper = self._project(elbow - shoulder, frame, True)
                    coordinates[f"{side_lower}_elbow"] = upper
                    elbow_count += 1
                    if wrist is not None:
                        forearm = self._project(wrist - elbow, frame, True)
                        shoulder_features[side_lower] = self._shoulder_features(upper, forearm)
                        to_shoulder = normalize(shoulder - elbow)
                        to_wrist = normalize(wrist - elbow)
                        angle = math.acos(float(np.clip(np.dot(to_shoulder, to_wrist), -1.0, 1.0)))
                        bent = math.radians(self.config.human_elbow_bent_deg)
                        straight = math.radians(self.config.human_elbow_straight_deg)
                        elbow_extensions[side_lower] = float(
                            np.clip((angle - bent) / max(straight - bent, 1.0e-4), 0.0, 1.0)
                        )

                palm = palm_rotation_from_packet(packet, side_lower, self.config.human_forward_sign)
                if palm is not None:
                    palms[side_lower] = palm
                    palm_count += 1

            if all(side in coordinates for side in required_sides):
                mode = "torso-frame" if frame is not None else "upper-body-fallback"
                mode += "+elbow" if elbow_count == len(required_sides) else "+wrist-only"
                mode += "+palm" if palm_count == len(required_sides) else "+no-palm"
                if threshold < strict:
                    mode += f"@vis{threshold:.2f}"
                return coordinates, palms, shoulder_features, elbow_extensions, wrist_heights, mode
            last_reason = "missing " + ", ".join(missing or ["required controlled-arm landmarks"])

        return None, {}, {}, {}, {}, last_reason

    def shape_pinch(self, raw: float | None) -> float:
        value = float(np.clip(0.0 if raw is None else raw, 0.0, 1.0))
        low = float(np.clip(self.config.pinch_deadzone, 0.0, 0.95))
        high = float(np.clip(self.config.pinch_full_close, low + 1.0e-3, 1.0))
        shaped = smoothstep((value - low) / (high - low))
        return float(np.clip(shaped ** max(self.config.pinch_gamma, 0.1), 0.0, 1.0))

    def packet_pinch(self, packet: dict, side: str) -> float:
        override = packet.get(f"{side}_gripper_override")
        if override is not None:
            return float(np.clip(float(override), 0.0, 1.0))
        return self.shape_pinch(packet.get(f"{side}_pinch"))

    def update(
        self,
        packet: dict,
        anchors: dict[str, np.ndarray],
        orientation_anchors: dict[str, np.ndarray],
        shoulder_anchors: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict]:
        required_sides = tuple(side for side in ("left", "right") if side in anchors)
        coordinates, palms, shoulder_features, elbow_extensions, wrist_heights, tracking_mode = self._observations(
            packet, required_sides
        )
        if coordinates is None:
            return {}, {
                "pose": False,
                "reason": tracking_mode,
                "left_pinch": self.packet_pinch(packet, "left"),
                "right_pinch": self.packet_pinch(packet, "right"),
            }

        calibrate_request = bool(packet.get("calibrate", False))
        if self.neutral is None and not calibrate_request:
            return {}, {
                "pose": False,
                "reason": "press C to calibrate wrist, elbow, shoulder, and palm",
                "left_pinch": self.packet_pinch(packet, "left"),
                "right_pinch": self.packet_pinch(packet, "right"),
            }

        if calibrate_request:
            self.neutral = {key: value.copy() for key, value in coordinates.items()}
            self.filtered = {key: value.copy() for key, value in coordinates.items()}
            self.neutral_palms = {side: value.copy() for side, value in palms.items()}
            self.filtered_palm_targets = {
                side: normalize(np.asarray(orientation_anchors[side], dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0]))
                for side in palms
                if side in orientation_anchors
            }
            self.neutral_shoulder_features = {side: value.copy() for side, value in shoulder_features.items()}
            self.neutral_elbow_extensions = dict(elbow_extensions)
            self.neutral_wrist_heights = dict(wrist_heights)
            self.filtered_wrist_heights = dict(wrist_heights)

        assert self.neutral is not None
        alpha = float(np.clip(self.config.filter_alpha, 0.01, 1.0))
        palm_alpha = float(np.clip(self.config.palm_filter_alpha, 0.01, 1.0))
        targets: dict[str, np.ndarray] = {}
        wrist_scale = np.array([self.config.forward_scale, self.config.lateral_scale, self.config.vertical_scale])
        elbow_scale = np.array(
            [self.config.elbow_forward_scale, self.config.elbow_lateral_scale, self.config.elbow_vertical_scale]
        )

        for key, value in coordinates.items():
            if key not in self.neutral:
                self.neutral[key] = value.copy()
                self.filtered[key] = value.copy()
            previous = self.filtered.get(key, value)
            filtered = previous + alpha * (value - previous)
            self.filtered[key] = filtered
            if key not in anchors:
                continue
            scale = elbow_scale if key.endswith("_elbow") else wrist_scale
            targets[key] = np.asarray(anchors[key], dtype=np.float64) + scale * (filtered - self.neutral[key])

        # Robust vertical mapping: fuse the world/body-frame wrist z delta with
        # image-space wrist height.  This prevents MediaPipe depth noise from
        # cancelling an obvious upward hand motion.
        for side, current_height in wrist_heights.items():
            if side not in targets or side not in anchors:
                continue
            if side not in self.neutral_wrist_heights:
                self.neutral_wrist_heights[side] = float(current_height)
                self.filtered_wrist_heights[side] = float(current_height)
            previous_height = self.filtered_wrist_heights.get(side, float(current_height))
            filtered_height = previous_height + alpha * (float(current_height) - previous_height)
            self.filtered_wrist_heights[side] = filtered_height
            image_delta = filtered_height - self.neutral_wrist_heights[side]
            deadzone = max(float(self.config.wrist_up_deadzone), 0.0)
            if abs(image_delta) < deadzone:
                image_delta = 0.0
            world_delta = float(targets[side][2] - np.asarray(anchors[side], dtype=np.float64)[2])
            image_delta_m = float(self.config.wrist_up_image_scale) * image_delta
            blend = float(np.clip(self.config.wrist_up_blend, 0.0, 1.0))
            fused_delta = (1.0 - blend) * world_delta + blend * image_delta_m
            targets[side][2] = float(np.asarray(anchors[side], dtype=np.float64)[2] + fused_delta)
            targets[f"{side}_wrist_up_delta"] = np.array([fused_delta], dtype=np.float64)

        for side, current_palm in palms.items():
            if side not in orientation_anchors:
                continue
            if side not in self.neutral_palms:
                self.neutral_palms[side] = current_palm.copy()
            # Geometric calibrated hand-frame retargeting.
            #
            # Previous revisions multiplied individual rotation-vector components
            # by ad-hoc signs.  That does not preserve the physical relationship
            # between the human palm plane and the robot tool frame and caused the
            # wrist to bend while the human hand dorsum stayed flat.
            #
            # R_rel is the real SO(3) change of the human palm since calibration.
            # It is applied directly in the calibrated robot EE local frame:
            #     R_robot_des = R_robot_at_C @ R_human_at_C^T @ R_human_now
            relative = self.neutral_palms[side].T @ current_palm

            # Optional signs remain available for diagnostics, but the default is
            # identity.  Sign edits are applied to the small local rotation vector
            # only when the user explicitly changes them.
            wrist_signs = np.asarray(
                self.config.left_wrist_signs if side == "left" else self.config.right_wrist_signs,
                dtype=np.float64,
            ).reshape(3)
            relative_rotvec = matrix_to_rotvec(relative)
            if not np.allclose(wrist_signs, np.ones(3), atol=1.0e-9):
                relative_rotvec = wrist_signs * relative_rotvec
                relative = rotvec_to_matrix(relative_rotvec)

            # Also expose the calibrated human wrist rotation as a 3-vector for
            # direct RB-Y1 arm_4~6 joint retargeting.  The normal teleoperation
            # mode uses this joint target; 6-D IK is reserved for cup picking.
            targets[f"{side}_wrist_joint_delta"] = relative_rotvec.copy()

            desired_matrix = quat_to_matrix(orientation_anchors[side]) @ relative
            desired_quat = matrix_to_quat(desired_matrix)
            previous = self.filtered_palm_targets.get(side, np.asarray(orientation_anchors[side], dtype=np.float64))
            filtered_quat = quat_slerp(previous, desired_quat, palm_alpha)
            self.filtered_palm_targets[side] = filtered_quat
            targets[f"{side}_orientation"] = filtered_quat.copy()

        for side, feature in shoulder_features.items():
            if side not in shoulder_anchors:
                continue
            if side not in self.neutral_shoulder_features:
                self.neutral_shoulder_features[side] = feature.copy()
            neutral_feature = self.neutral_shoulder_features[side]
            # Elevation is physically bounded to [-pi/2, +pi/2].  Do not wrap
            # its difference: moving from arm-down (-90 deg) to overhead
            # (+90 deg) is a real +180 deg change.  wrap_angle() converted that
            # transition to -180 deg, which the RB-Y1 mapping interpreted as a
            # backward shoulder command and then clipped to the backward limit.
            elevation_delta = float(feature[0] - neutral_feature[0])
            yaw_delta = wrap_angle(float(feature[1] - neutral_feature[1]))
            twist_delta = wrap_angle(float(feature[2] - neutral_feature[2]))
            delta = np.array([elevation_delta, yaw_delta, twist_delta], dtype=np.float64)
            targets[f"{side}_shoulder_delta"] = delta

        for side, extension in elbow_extensions.items():
            if side not in self.neutral_elbow_extensions:
                self.neutral_elbow_extensions[side] = float(extension)
            extension_delta = float(extension - self.neutral_elbow_extensions[side])
            targets[f"{side}_elbow_extension_delta"] = extension_delta
            targets[f"{side}_elbow_extension"] = float(extension)
            # MediaPipe depth is noisy, so a visibly straightening elbow also
            # advances the robot hand target in +x. This makes reaching a cup
            # in front of the robot possible even when webcam depth barely moves.
            if side in targets:
                targets[side][0] += float(self.config.extension_forward_scale) * extension_delta
            targets[f"{side}_extension_forward_delta"] = float(
                self.config.extension_forward_scale * extension_delta
            )

        metrics = {
            "pose": True,
            "tracking_mode": tracking_mode,
            "calibrated": calibrate_request,
            "left_elbow": "left_elbow" in coordinates,
            "right_elbow": "right_elbow" in coordinates,
            "left_palm": "left" in palms,
            "right_palm": "right" in palms,
            "left_pinch": self.packet_pinch(packet, "left"),
            "right_pinch": self.packet_pinch(packet, "right"),
            "left_elbow_extension": float(elbow_extensions.get("left", -1.0)),
            "right_elbow_extension": float(elbow_extensions.get("right", -1.0)),
            "left_wrist_up_delta": float(targets.get("left_wrist_up_delta", np.array([0.0]))[0]),
            "right_wrist_up_delta": float(targets.get("right_wrist_up_delta", np.array([0.0]))[0]),
            "left_extension_forward_delta": float(targets.get("left_extension_forward_delta", 0.0)),
            "right_extension_forward_delta": float(targets.get("right_extension_forward_delta", 0.0)),
        }
        return targets, metrics


@dataclass
class DlsConfig:
    damping: float = 0.08
    position_gain: float = 1.0
    orientation_weight: float = 0.65
    null_gain: float = 0.01
    joint_limit_gain: float = 0.025
    max_position_error: float = 0.08
    max_orientation_error: float = 0.35
    max_joint_step: float = 0.08


def solve_dls_pose_step(
    jacobian_6xn: np.ndarray,
    position_error: np.ndarray,
    orientation_error: np.ndarray,
    q: np.ndarray,
    q_home: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: DlsConfig,
    joint_weights: np.ndarray | None = None,
    secondary_velocity: np.ndarray | None = None,
) -> np.ndarray:
    orientation_weight = float(max(config.orientation_weight, 0.0))
    return solve_stacked_dls_step(
        [(np.asarray(jacobian_6xn)[:3], position_error, 1.0)],
        q,
        q_home,
        lower,
        upper,
        config,
        joint_weights=joint_weights,
        secondary_velocity=secondary_velocity,
        orientation_task=(np.asarray(jacobian_6xn)[3:], orientation_error, orientation_weight),
    )



def apply_shoulder_reference_correction(
    dq: np.ndarray,
    shoulder_error: np.ndarray,
    ee_jacobian: np.ndarray,
    *,
    reference_gain: float = 0.42,
    blend: float = 0.58,
    max_shoulder_step: float = math.radians(1.65),
    position_weight: float = 1.0,
    orientation_weight: float = 0.8,
    damping: float = 0.08,
    distal_compensation: float = 0.78,
) -> np.ndarray:
    """Strengthen arm_0~2 tracking while preserving the hand pose.

    The stacked IK solution can under-use the shoulder because several joint
    configurations produce a similar end-effector pose.  This correction blends
    the first three joint steps toward the calibrated human shoulder reference.
    The task-space disturbance introduced by that blend is then compensated by
    arm_3~6 using a damped least-squares solve.  It therefore improves shoulder
    imitation without simply dragging the robot hand away from its trajectory.
    """
    result = np.asarray(dq, dtype=np.float64).reshape(-1).copy()
    if result.size < 7:
        raise ValueError(f"Expected a 7-DoF arm step, got {result.size}")

    error = np.asarray(shoulder_error, dtype=np.float64).reshape(3)
    jacobian = np.asarray(ee_jacobian, dtype=np.float64).reshape(6, result.size)
    gain = float(max(reference_gain, 0.0))
    mix = float(np.clip(blend, 0.0, 1.0))
    step_limit = float(max(max_shoulder_step, 0.0))
    if gain <= 0.0 or mix <= 0.0 or step_limit <= 0.0:
        return result

    requested = np.clip(gain * error, -step_limit, step_limit)
    original_shoulder = result[:3].copy()
    corrected_shoulder = (1.0 - mix) * original_shoulder + mix * requested
    shoulder_change = corrected_shoulder - original_shoulder
    result[:3] = corrected_shoulder

    rows: list[np.ndarray] = []
    if position_weight > 0.0:
        rows.append(math.sqrt(float(position_weight)) * jacobian[:3, :])
    if orientation_weight > 0.0:
        rows.append(math.sqrt(float(orientation_weight)) * jacobian[3:, :])
    if not rows or float(np.linalg.norm(shoulder_change)) <= 1.0e-10:
        return result

    task_jacobian = np.vstack(rows)
    shoulder_effect = task_jacobian[:, :3] @ shoulder_change
    distal_jacobian = task_jacobian[:, 3:]
    if distal_jacobian.size == 0 or float(np.linalg.norm(distal_jacobian)) <= 1.0e-10:
        return result

    damping_sq = float(max(damping, 1.0e-5)) ** 2
    normal = distal_jacobian.T @ distal_jacobian + damping_sq * np.eye(distal_jacobian.shape[1])
    rhs = -distal_jacobian.T @ shoulder_effect
    try:
        distal_step = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        distal_step = np.linalg.lstsq(normal, rhs, rcond=None)[0]
    result[3:] += float(np.clip(distal_compensation, 0.0, 1.5)) * distal_step
    return result

def solve_stacked_dls_step(
    position_tasks: list[tuple[np.ndarray, np.ndarray, float]],
    q: np.ndarray,
    q_home: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: DlsConfig,
    *,
    joint_weights: np.ndarray | None = None,
    secondary_velocity: np.ndarray | None = None,
    orientation_task: tuple[np.ndarray, np.ndarray, float] | None = None,
    joint_tasks: list[tuple[np.ndarray, np.ndarray, float]] | None = None,
) -> np.ndarray:
    """Solve Cartesian and explicit joint-posture tasks in one weighted DLS step.

    ``joint_tasks`` entries are ``(joint_indices_within_arm, q_error, weight)``.
    This is deliberately a primary least-squares term rather than a null-space
    suggestion, so human shoulder motion remains represented even when the 6D
    end-effector task consumes most of the seven arm degrees of freedom.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    rows: list[np.ndarray] = []
    errors: list[np.ndarray] = []

    for jacobian, error, task_weight in position_tasks:
        weight = float(max(task_weight, 0.0))
        if weight <= 0.0:
            continue
        J = np.asarray(jacobian, dtype=np.float64).reshape(3, q.size)
        e = clip_norm(np.asarray(error, dtype=np.float64).reshape(3), config.max_position_error)
        scale = math.sqrt(weight)
        rows.append(scale * J)
        errors.append(scale * config.position_gain * e)

    if orientation_task is not None:
        jacobian_angular, orientation_error, task_weight = orientation_task
        weight = float(max(task_weight, 0.0))
        if weight > 0.0:
            Jw = np.asarray(jacobian_angular, dtype=np.float64).reshape(3, q.size)
            ew = clip_norm(np.asarray(orientation_error, dtype=np.float64).reshape(3), config.max_orientation_error)
            scale = math.sqrt(weight)
            rows.append(scale * Jw)
            errors.append(scale * ew)

    for indices, error, task_weight in joint_tasks or []:
        weight = float(max(task_weight, 0.0))
        if weight <= 0.0:
            continue
        indices_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        error_array = np.asarray(error, dtype=np.float64).reshape(indices_array.size)
        selector = np.zeros((indices_array.size, q.size), dtype=np.float64)
        selector[np.arange(indices_array.size), indices_array] = 1.0
        scale = math.sqrt(weight)
        rows.append(scale * selector)
        errors.append(scale * np.clip(error_array, -config.max_joint_step, config.max_joint_step))

    if not rows:
        return np.zeros_like(q)

    J_stack = np.vstack(rows)
    task_error = np.concatenate(errors)
    if joint_weights is None:
        weights = np.ones(q.size, dtype=np.float64)
    else:
        weights = np.clip(np.asarray(joint_weights, dtype=np.float64).reshape(q.size), 1.0e-3, 1.0e3)
    weight_inverse = np.diag(1.0 / weights)

    damping_sq = float(config.damping) ** 2
    regularized = J_stack @ weight_inverse @ J_stack.T + damping_sq * np.eye(J_stack.shape[0])
    try:
        solved = np.linalg.solve(regularized, task_error)
    except np.linalg.LinAlgError:
        solved = np.linalg.lstsq(regularized, task_error, rcond=None)[0]
    pseudo = weight_inverse @ J_stack.T @ np.linalg.pinv(regularized)
    dq_task = weight_inverse @ J_stack.T @ solved
    null_projector = np.eye(q.size) - pseudo @ J_stack

    posture = config.null_gain * (np.asarray(q_home, dtype=np.float64) - q)
    finite = np.isfinite(lower) & np.isfinite(upper) & ((upper - lower) > 1.0e-4)
    midpoint = np.where(finite, 0.5 * (lower + upper), q)
    half_range = np.where(finite, 0.5 * (upper - lower), 1.0)
    limit_push = config.joint_limit_gain * (midpoint - q) / np.maximum(half_range * half_range, 1.0e-4)
    secondary = np.zeros_like(q) if secondary_velocity is None else np.asarray(secondary_velocity, dtype=np.float64)
    dq = dq_task + null_projector @ (posture + limit_push + secondary)
    return np.clip(dq, -config.max_joint_step, config.max_joint_step)
