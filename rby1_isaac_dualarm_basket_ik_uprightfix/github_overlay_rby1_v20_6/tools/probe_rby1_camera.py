#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2


def list_nodes() -> list[str]:
    nodes = [str(p) for p in sorted(Path("/dev").glob("video*"))]
    by_id = Path("/dev/v4l/by-id")
    if by_id.exists():
        for p in sorted(by_id.glob("*-video-index0")):
            nodes.insert(0, str(p))
    result: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        try:
            key = str(Path(node).resolve())
        except OSError:
            key = node
        if key not in seen and Path(node).exists():
            seen.add(key)
            result.append(node)
    return result


def busy_processes(device: str) -> str:
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


def v4l2_info(device: str) -> str:
    if shutil.which("v4l2-ctl") is None:
        return ""
    proc = subprocess.run(
        ["v4l2-ctl", "-d", device, "--all"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    lines = []
    for line in proc.stdout.splitlines():
        if any(key in line for key in (
            "Driver name", "Card type", "Bus info",
            "Video Capture", "Metadata Capture",
            "Device Caps", "Capabilities",
        )):
            lines.append(line.strip())
    return " | ".join(lines)


def open_attempt(device: str, backend: int, fourcc: str | None, width: int, height: int):
    cap = cv2.VideoCapture(device, backend)
    if not cap.isOpened():
        cap.release()
        return False, None, "open failed"

    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(35):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            info = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                float(cap.get(cv2.CAP_PROP_FPS)),
            )
            return True, cap, info
        time.sleep(0.04)

    cap.release()
    return False, None, "no valid frame"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    nodes = list_nodes()
    if args.device:
        nodes = [args.device] + [n for n in nodes if str(Path(n).resolve()) != str(Path(args.device).resolve())]

    print("[PROBE] Visible nodes:", nodes or "(none)", flush=True)

    attempts = [
        (cv2.CAP_V4L2, None, 640, 480),
        (cv2.CAP_V4L2, "MJPG", 640, 480),
        (cv2.CAP_V4L2, "YUYV", 640, 480),
        (cv2.CAP_V4L2, None, 1280, 720),
        (cv2.CAP_ANY, None, 640, 480),
        (cv2.CAP_ANY, None, 320, 240),
    ]

    for device in nodes:
        print(f"\n[PROBE] {device}", flush=True)
        print(f"  access={'yes' if os.access(device, os.R_OK | os.W_OK) else 'no'}", flush=True)

        info = v4l2_info(device)
        if info:
            print(f"  v4l2={info}", flush=True)

        busy = busy_processes(device)
        if busy:
            print(f"  busy:\n{busy}", flush=True)

        for backend, fourcc, width, height in attempts:
            label = f"{'V4L2' if backend == cv2.CAP_V4L2 else 'ANY'} fourcc={fourcc or 'default'} {width}x{height}"
            ok, cap, result = open_attempt(device, backend, fourcc, width, height)
            print(f"  {label}: {'OK' if ok else 'fail'} {result}", flush=True)
            if ok:
                cap.release()
                print(f"\n[RESULT] WORKING_CAMERA={device}", flush=True)
                return

    print("\n[RESULT] No node returned a valid image frame.", flush=True)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
