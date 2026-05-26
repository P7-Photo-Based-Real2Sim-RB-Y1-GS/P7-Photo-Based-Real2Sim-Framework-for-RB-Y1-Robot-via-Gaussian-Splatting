#!/usr/bin/env python3
"""ROS2 HumanSkeleton -> UDP JSON bridge.

Run this in the ROS2 + human_skeleton environment. It subscribes to
/human/skeleton/arm_hand_stitched and sends compact landmark dictionaries to
Isaac Sim. Isaac does not need rclpy; it only receives UDP JSON.
"""
from __future__ import annotations

import argparse
import json
import socket
import time

import rclpy
from rclpy.node import Node
from human_skeleton_pipeline.msg import HumanSkeleton


class SkeletonUdpBridge(Node):
    def __init__(self, topic: str, host: str, port: int, min_visibility: float):
        super().__init__("skeleton_udp_bridge")
        self.host = host
        self.port = int(port)
        self.min_visibility = float(min_visibility)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sub = self.create_subscription(HumanSkeleton, topic, self.cb, 10)
        self.count = 0
        self.last_print = time.time()
        self.get_logger().info(f"Listening: {topic}")
        self.get_logger().info(f"Sending UDP JSON to {host}:{port}")

    def cb(self, msg: HumanSkeleton):
        landmarks = {}
        valid_count = 0
        for lm in msg.landmarks:
            visible = float(lm.visibility) >= self.min_visibility
            valid = bool(lm.valid) and visible
            if valid:
                valid_count += 1
            landmarks[str(lm.name)] = {
                "id": int(lm.id),
                "x": float(lm.position.x),
                "y": float(lm.position.y),
                "z": float(lm.position.z),
                "visibility": float(lm.visibility),
                "presence": float(lm.presence),
                "valid": bool(valid),
            }

        packet = {
            "stamp_sec": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
            "frame_id": str(msg.header.frame_id),
            "source": str(msg.source),
            "coordinate_type": str(msg.coordinate_type),
            "tracking_ok": bool(msg.tracking_ok) and valid_count >= 6,
            "mean_visibility": float(msg.mean_visibility),
            "scale_reference": float(msg.scale_reference),
            "scale_reference_name": str(msg.scale_reference_name),
            "landmarks": landmarks,
        }
        data = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        self.sock.sendto(data, (self.host, self.port))
        self.count += 1
        now = time.time()
        if now - self.last_print > 2.0:
            self.get_logger().info(
                f"sent={self.count}, tracking_ok={packet['tracking_ok']}, valid={valid_count}, bytes={len(data)}"
            )
            self.last_print = now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/human/skeleton/arm_hand_stitched")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50555)
    ap.add_argument("--min-visibility", type=float, default=0.25)
    args = ap.parse_args()

    rclpy.init()
    node = SkeletonUdpBridge(args.topic, args.host, args.port, args.min_visibility)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
