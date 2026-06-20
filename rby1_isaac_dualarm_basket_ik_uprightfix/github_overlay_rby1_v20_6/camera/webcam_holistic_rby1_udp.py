#!/usr/bin/env python3
"""Webcam + MediaPipe Holistic full-body sender for RB-Y1 Isaac Sim teleoperation.

The script keeps the webcam preview visible while streaming pose/hand landmarks
as JSON over UDP. It does not import ROS 2, so it avoids Python 3.11/3.12 rclpy
compatibility problems between Isaac Sim and ROS Jazzy.

Keys
----
q / ESC : quit
c       : request receiver re-calibration
g       : cycle right gripper AUTO -> CLOSE -> OPEN
f       : cycle left gripper AUTO -> CLOSE -> OPEN
SPACE   : pause/resume command transmission
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe full-body UDP sender")
    parser.add_argument("--udp-ip", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5005)
    parser.add_argument("--camera-index", type=int, default=0,
                        help="Preferred Linux /dev/videoN number. Existing nodes are auto-scanned.")
    parser.add_argument("--camera-device", type=str, default="",
                        help="Explicit V4L2 device path, e.g. /dev/video1.")
    parser.add_argument("--list-cameras", action="store_true",
                        help="List visible /dev/video* nodes and exit.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.55)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.55)
    parser.add_argument("--no-mirror-display", action="store_true")
    parser.add_argument("--window-x", type=int, default=20)
    parser.add_argument("--window-y", type=int, default=40)
    parser.add_argument("--no-topmost", action="store_true")
    parser.add_argument(
        "--save-last-frame",
        type=str,
        default="",
        help="Optional path written whenever C is pressed.",
    )
    return parser.parse_args()


def _video_index(device: str) -> int:
    name = Path(device).name
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else -1


def available_camera_nodes() -> list[str]:
    """Return real Linux V4L2 device nodes without inventing invalid indices."""
    nodes: list[str] = []

    # Stable USB-camera symlinks first when available.
    by_id = Path("/dev/v4l/by-id")
    if by_id.exists():
        for item in sorted(by_id.glob("*-video-index0")):
            try:
                nodes.append(str(item.resolve()))
            except OSError:
                pass

    nodes.extend(str(path) for path in sorted(Path("/dev").glob("video*")))

    unique: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        try:
            real = str(Path(node).resolve())
        except OSError:
            real = node
        if real not in seen and Path(real).exists():
            seen.add(real)
            unique.append(real)
    return unique


def print_camera_nodes() -> None:
    nodes = available_camera_nodes()
    print("[CAM] Visible V4L2 nodes:", flush=True)
    if not nodes:
        print("  (none)", flush=True)
        return
    for node in nodes:
        readable = os.access(node, os.R_OK | os.W_OK)
        print(f"  {node}  access={'yes' if readable else 'no'}", flush=True)


def _busy_processes(device: str) -> str:
    if shutil.which("fuser") is None:
        return ""
    proc = subprocess.run(
        ["fuser", "-v", device],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.stdout.strip()


def _open_attempt(
    device: str,
    backend: int,
    width: int,
    height: int,
    fps: int,
    fourcc: str | None = None,
):
    camera = cv2.VideoCapture(device, backend)
    if not camera.isOpened():
        camera.release()
        return None

    if fourcc:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(35):
        ok, frame = camera.read()
        if ok and frame is not None and frame.size > 0:
            print(
                f"[CAM] Opened {device}: "
                f"{int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
                f"{camera.get(cv2.CAP_PROP_FPS):.1f} FPS",
                flush=True,
            )
            return camera
        time.sleep(0.04)

    camera.release()
    return None


def _try_open_device(device: str, width: int, height: int, fps: int):
    print(f"[CAM] Trying {device}", flush=True)
    busy = _busy_processes(device)
    if busy:
        print(f"[CAM] Device may be busy:\n{busy}", flush=True)

    attempts = [
        (cv2.CAP_V4L2, width, height, fps, None),
        (cv2.CAP_V4L2, width, height, fps, "MJPG"),
        (cv2.CAP_V4L2, width, height, fps, "YUYV"),
        (cv2.CAP_V4L2, 640, 480, min(fps, 30), None),
        (cv2.CAP_V4L2, 1280, 720, min(fps, 30), None),
        (cv2.CAP_ANY, 640, 480, min(fps, 30), None),
        (cv2.CAP_ANY, 320, 240, min(fps, 30), None),
    ]

    for backend, w, h, f, fourcc in attempts:
        camera = _open_attempt(device, backend, w, h, f, fourcc)
        if camera is not None:
            return camera
    return None


def open_camera(
    preferred_index: int,
    preferred_device: str,
    width: int,
    height: int,
    fps: int,
):
    nodes = available_camera_nodes()
    candidates: list[str] = []

    if preferred_device:
        candidates.append(str(Path(preferred_device).expanduser()))

    preferred_node = f"/dev/video{preferred_index}"
    if Path(preferred_node).exists():
        candidates.append(preferred_node)

    candidates.extend(nodes)

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(Path(candidate).resolve())
        except OSError:
            key = candidate
        if key not in seen and Path(candidate).exists():
            seen.add(key)
            unique.append(candidate)

    for device in unique:
        camera = _try_open_device(device, width, height, fps)
        if camera is not None:
            return camera, device, _video_index(device)

    visible = ", ".join(nodes) if nodes else "(none)"
    raise RuntimeError(
        "웹캠 영상 노드를 열지 못했습니다.\n"
        f"보이는 노드: {visible}\n"
        "다음을 실행해 실제 capture 노드를 찾으세요:\n"
        "  python ~/rby1_ros2_hand_teleop/tools/probe_rby1_camera.py"
    )


def point_dict(landmark) -> dict[str, float]:
    return {
        "x": float(landmark.x),
        "y": float(landmark.y),
        "z": float(landmark.z),
    }


def euclidean_2d(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def pinch_strength(hand_landmarks) -> float | None:
    """Return a robust 0=open, 1=pinched signal from image-plane geometry.

    The previous 3-D landmark distance was dominated by MediaPipe depth noise and
    often saturated around 0.2~0.4 even when the fingertips touched.  Normalizing
    the 2-D thumb/index distance by palm width is much more stable for a webcam.
    """
    if hand_landmarks is None:
        return None
    lm = hand_landmarks.landmark
    thumb_tip = lm[4]
    index_tip = lm[8]
    index_mcp = lm[5]
    pinky_mcp = lm[17]
    wrist = lm[0]
    middle_mcp = lm[9]
    palm_width = euclidean_2d(index_mcp, pinky_mcp)
    palm_length = euclidean_2d(wrist, middle_mcp)
    palm_scale = max(0.65 * palm_width + 0.35 * palm_length, 1.0e-5)
    ratio = euclidean_2d(thumb_tip, index_tip) / palm_scale
    # Typical webcam values: touching ~= 0.05~0.25, open ~= 0.9~1.6.
    linear = float(np.clip((0.95 - ratio) / (0.95 - 0.20), 0.0, 1.0))
    return linear * linear * (3.0 - 2.0 * linear)


def pose_packet(result, pose_enum) -> list[dict]:
    if result.pose_landmarks is None:
        return []

    image_landmarks = result.pose_landmarks.landmark
    world_landmarks = (
        result.pose_world_landmarks.landmark
        if result.pose_world_landmarks is not None
        else None
    )

    output = []
    for enum_item in pose_enum:
        index = int(enum_item.value)
        image_lm = image_landmarks[index]
        entry = {
            "landmark_id": index,
            "landmark_name": enum_item.name,
            "image_landmark": point_dict(image_lm),
            "visibility": float(getattr(image_lm, "visibility", 1.0)),
            "presence": float(getattr(image_lm, "presence", 1.0)),
        }
        if world_landmarks is not None:
            entry["world_position"] = point_dict(world_landmarks[index])
        output.append(entry)
    return output


def hand_packet(hand_landmarks, handedness: str) -> list[dict]:
    if hand_landmarks is None:
        return []
    output = []
    for index, landmark in enumerate(hand_landmarks.landmark):
        output.append(
            {
                "landmark_id": index,
                "landmark_name": str(index),
                "handedness": handedness,
                "image_landmark": point_dict(landmark),
            }
        )
    return output


def draw_status(
    frame,
    *,
    camera_index: int,
    udp_ip: str,
    udp_port: int,
    pose_ok: bool,
    left_pinch: float | None,
    right_pinch: float | None,
    enabled: bool,
    fps_value: float,
    left_override: float | None,
    right_override: float | None,
    pick_pending: bool,
    reset_pending: bool,
) -> None:
    def grip_label(value: float | None) -> str:
        if value is None:
            return "AUTO"
        return "CLOSE" if value > 0.5 else "OPEN"

    lines = [
        f"CAM /dev/video{camera_index}   UDP {udp_ip}:{udp_port}",
        f"POSE {'OK' if pose_ok else 'NOT DETECTED'}   TX {'ON' if enabled else 'PAUSED'}   {fps_value:4.1f} FPS",
        f"PINCH L={left_pinch if left_pinch is not None else -1.0:.2f}  R={right_pinch if right_pinch is not None else -1.0:.2f}",
        f"GRIP L={grip_label(left_override)} R={grip_label(right_override)}",
        f"MOVE ARM=AUTO PICK | P MANUAL{' (queued)' if pick_pending else ''} | R RESET{' (queued)' if reset_pending else ''}",
        "C CALIBRATE | G right grip | F left grip | Q/ESC quit | SPACE pause",
    ]
    y = 28
    for line_index, text in enumerate(lines):
        color = (80, 255, 80) if line_index < 2 and pose_ok else (0, 220, 255)
        cv2.putText(
            frame,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 25


def main() -> None:
    args = parse_args()
    if args.list_cameras:
        print_camera_nodes()
        return
    camera, actual_device, actual_index = open_camera(
        args.camera_index, args.camera_device, args.width, args.height, args.fps
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    window_name = "RB-Y1 Webcam Skeleton Teleoperation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)
    cv2.moveWindow(window_name, args.window_x, args.window_y)
    if not args.no_topmost and hasattr(cv2, "WND_PROP_TOPMOST"):
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass

    print(f"[CAM] Using {actual_device}", flush=True)
    print(f"[UDP] Sending full-body pose to udp://{args.udp_ip}:{args.udp_port}", flush=True)
    print("[KEY] C calibrate, move controlled arm=auto pick, P manual fallback, R reset, G/F gripper", flush=True)

    enabled = True
    # None=AUTO, 1.0=forced CLOSE, 0.0=forced OPEN.
    left_gripper_override = None
    right_gripper_override = None
    calibrate_request = False
    pick_request = False
    reset_request = False
    sequence = 0
    fps_filtered = 0.0
    previous_time = time.perf_counter()
    last_display_frame = None

    try:
        while True:
            ok, raw_frame = camera.read()
            if not ok or raw_frame is None:
                time.sleep(0.02)
                continue

            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = holistic.process(rgb)
            rgb.flags.writeable = True

            annotated = raw_frame.copy()
            if result.pose_landmarks is not None:
                mp_draw.draw_landmarks(
                    annotated,
                    result.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )
            if result.left_hand_landmarks is not None:
                mp_draw.draw_landmarks(
                    annotated,
                    result.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                )
            if result.right_hand_landmarks is not None:
                mp_draw.draw_landmarks(
                    annotated,
                    result.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                )

            pose_landmarks = pose_packet(result, mp_holistic.PoseLandmark)
            left_hand = hand_packet(result.left_hand_landmarks, "Left")
            right_hand = hand_packet(result.right_hand_landmarks, "Right")
            left_pinch = pinch_strength(result.left_hand_landmarks)
            right_pinch = pinch_strength(result.right_hand_landmarks)

            now = time.time()
            packet = {
                "timestamp": now,
                "sequence": sequence,
                "source": "webcam_mediapipe_holistic_fullbody",
                "camera_index": actual_index,
                "camera_device": actual_device,
                "mirror_display_only": not args.no_mirror_display,
                "enabled": enabled,
                "calibrate": calibrate_request,
                "pose_landmarks": pose_landmarks,
                "left_hand_landmarks": left_hand,
                "right_hand_landmarks": right_hand,
                "left_pinch": left_pinch,
                "right_pinch": right_pinch,
                "left_gripper_override": left_gripper_override,
                "right_gripper_override": right_gripper_override,
                "pick_request": pick_request,
                "reset_request": reset_request,
            }
            sequence += 1

            # Send control keys and gripper commands even during a temporary
            # pose dropout.  Joint retargeting itself still requires pose data.
            if enabled:
                encoded = json.dumps(packet, separators=(",", ":")).encode("utf-8")
                sock.sendto(encoded, (args.udp_ip, args.udp_port))
            calibrate_request = False
            pick_request = False
            reset_request = False

            current_perf = time.perf_counter()
            instantaneous_fps = 1.0 / max(current_perf - previous_time, 1.0e-6)
            previous_time = current_perf
            fps_filtered = 0.92 * fps_filtered + 0.08 * instantaneous_fps if fps_filtered else instantaneous_fps

            display = cv2.flip(annotated, 1) if not args.no_mirror_display else annotated
            draw_status(
                display,
                camera_index=actual_index,
                udp_ip=args.udp_ip,
                udp_port=args.udp_port,
                pose_ok=bool(pose_landmarks),
                left_pinch=left_pinch,
                right_pinch=right_pinch,
                enabled=enabled,
                fps_value=fps_filtered,
                left_override=left_gripper_override,
                right_override=right_gripper_override,
                pick_pending=pick_request,
                reset_pending=reset_request,
            )
            cv2.imshow(window_name, display)
            last_display_frame = display

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                calibrate_request = True
                print("[CAM] Calibration requested.", flush=True)
                if args.save_last_frame and last_display_frame is not None:
                    output = Path(args.save_last_frame).expanduser()
                    output.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(output), last_display_frame)
            elif key == ord("g"):
                right_gripper_override = (
                    1.0 if right_gripper_override is None
                    else (0.0 if right_gripper_override > 0.5 else None)
                )
                state = "AUTO" if right_gripper_override is None else ("CLOSE" if right_gripper_override > 0.5 else "OPEN")
                print(f"[CAM] Right gripper override: {state}", flush=True)
            elif key == ord("f"):
                left_gripper_override = (
                    1.0 if left_gripper_override is None
                    else (0.0 if left_gripper_override > 0.5 else None)
                )
                state = "AUTO" if left_gripper_override is None else ("CLOSE" if left_gripper_override > 0.5 else "OPEN")
                print(f"[CAM] Left gripper override: {state}", flush=True)
            elif key == ord("p"):
                pick_request = True
                print("[CAM] Manual cup-pick fallback requested.", flush=True)
            elif key == ord("r"):
                reset_request = True
                print("[CAM] Reset/release requested.", flush=True)
            elif key == 32:
                enabled = not enabled
                print(f"[CAM] Transmission {'enabled' if enabled else 'paused'}.", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        holistic.close()
        camera.release()
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
