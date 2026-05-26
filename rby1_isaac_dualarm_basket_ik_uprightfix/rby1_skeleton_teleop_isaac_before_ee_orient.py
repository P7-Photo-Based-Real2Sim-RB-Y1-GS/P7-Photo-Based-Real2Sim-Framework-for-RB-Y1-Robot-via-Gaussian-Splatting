"""RBY1 Isaac Sim skeleton teleoperation from UDP JSON.

Pipeline:
  webcam -> MediaPipe ROS2 /human/skeleton/arm_hand_stitched
  -> scripts/ros2_skeleton_udp_bridge.py
  -> this Isaac script
  -> RBY1 left/right arm position IK

Run with Isaac Lab, not plain python:
  cd ~/IsaacLab
  ./isaaclab.sh -p /path/to/rby1_skeleton_teleop_isaac.py --asset v1_1

This script is intended to be placed inside the existing
rby1_isaac_dualarm_basket_ik_uprightfix project root, so it can import
rby1_isaac_connected.common and use the prepared RBY1 USD assets.
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import time
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--asset", choices=["v1_0", "v1_1"], default="v1_1")
parser.add_argument("--udp-host", default="127.0.0.1")
parser.add_argument("--udp-port", type=int, default=50555)
parser.add_argument("--calib-frames", type=int, default=45, help="Number of valid skeleton frames used for neutral calibration.")
parser.add_argument("--human-gain", type=float, default=0.75, help="Global scale from human wrist displacement to robot EE displacement.")
parser.add_argument("--gain-x", type=float, default=0.35, help="Forward/backward EE gain.")
parser.add_argument("--gain-y", type=float, default=0.55, help="Left/right EE gain.")
parser.add_argument("--gain-z", type=float, default=0.45, help="Up/down EE gain.")
parser.add_argument("--deadband", type=float, default=0.015, help="Ignore small human wrist motion [m] in normalized feature space.")
parser.add_argument("--posture-pull", type=float, default=0.085, help="Null posture pull strength to avoid ugly elbow folding.")
parser.add_argument("--swap-arms", action="store_true", help="Swap left/right human arms if the camera view is mirrored.")
parser.add_argument("--invert-x", action="store_true", help="Invert robot forward/backward mapping.")
parser.add_argument("--invert-y", action="store_true", help="Invert robot left/right mapping.")
parser.add_argument("--invert-z", action="store_true", help="Invert robot up/down mapping.")
parser.add_argument("--filter-alpha", type=float, default=0.25, help="Low-pass target filter. 1=no filtering, lower=smoother.")
parser.add_argument("--lambda-dls", type=float, default=0.10)
parser.add_argument("--ik-gain", type=float, default=0.55)
parser.add_argument("--max-joint-step", type=float, default=0.025)
parser.add_argument("--max-target-step", type=float, default=0.040, help="Max EE target change per sim step [m].")
parser.add_argument("--workspace-x", nargs=2, type=float, default=[-0.25, 0.45], help="Target x limits around robot world origin.")
parser.add_argument("--workspace-y-left", nargs=2, type=float, default=[0.05, 0.75])
parser.add_argument("--workspace-y-right", nargs=2, type=float, default=[-0.75, -0.05])
parser.add_argument("--workspace-z", nargs=2, type=float, default=[0.35, 1.45])
parser.add_argument("--left-ee-body", default="ee_finger_l1")
parser.add_argument("--right-ee-body", default="ee_finger_r1")
parser.add_argument("--hold-timeout", type=float, default=0.7, help="Hold last target for this many seconds when skeleton drops.")
parser.add_argument("--print-every", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import UsdPhysics
import isaaclab.sim as sim_utils
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, XFormPrim
from isaacsim.core.utils.types import ArticulationActions

from rby1_isaac_connected.common import (
    DUAL_ARM_PRESET,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    USD_MAP,
    apply_articulation_root_to_best_candidate,
    candidate_root_paths,
    find_articulation_roots_under,
    merge_targets,
    resolve_joint_indices,
)


def _as_np(x):
    return np.asarray(x, dtype=np.float64)


def _load_world(usd_path: Path, scene_root: str) -> World:
    world = World(stage_units_in_meters=1.0)
    usd_cfg = sim_utils.UsdFileCfg(usd_path=str(usd_path))
    usd_cfg.func(scene_root, usd_cfg)
    return world


def _resolve_robot(scene_root: str, world: World) -> tuple[Articulation, str]:
    roots = find_articulation_roots_under(scene_root)
    if not roots:
        patched = apply_articulation_root_to_best_candidate(scene_root)
        print("[INFO] Applied ArticulationRootAPI candidate:", patched)
    world.reset()
    roots = find_articulation_roots_under(scene_root)
    trial_paths = roots if roots else candidate_root_paths(scene_root)
    last_exc = None
    for path in trial_paths:
        try:
            robot = Articulation(prim_paths_expr=path, name="rby1_skeleton_teleop")
            robot.initialize()
            return robot, path
        except Exception as exc:
            last_exc = exc
            print(f"[WARN] Failed to initialize articulation at {path}: {exc}")
    raise RuntimeError(f"Could not initialize articulation. Tried={trial_paths}, last={last_exc}")


def _resolve_body_name(body_names: list[str], candidates: list[str]) -> str:
    names = list(body_names)
    upper_to_name = {n.upper(): n for n in names}
    for c in candidates:
        if c.upper() in upper_to_name:
            return upper_to_name[c.upper()]
    for c in candidates:
        hits = [n for n in names if n.upper().endswith(c.upper())]
        if len(hits) == 1:
            return hits[0]
    for c in candidates:
        hits = [n for n in names if c.upper() in n.upper()]
        if hits:
            return sorted(hits, key=len)[0]
    raise RuntimeError(f"Could not resolve body from candidates={candidates}. Available={names}")


def _find_link_prim_under(prefixes: list[str], candidates: list[str]) -> str:
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    wanted = [c.upper() for c in candidates]
    hits = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prefixes and not any(path.startswith(p) for p in prefixes):
            continue
        base = path.split("/")[-1].upper()
        if base in wanted or any(w in path.upper() for w in wanted):
            hits.append(path)
    if not hits:
        raise RuntimeError(f"Could not find link prim for {candidates}")
    return sorted(set(hits), key=lambda p: (p.count('/'), p))[0]


def _get_world_pos(view: XFormPrim) -> np.ndarray:
    pos, _quat = view.get_world_poses()
    return _as_np(pos[0])


def _dls_step(jacobian_xyz: np.ndarray, pos_err: np.ndarray, damping: float, gain: float) -> np.ndarray:
    j = jacobian_xyz
    a = j @ j.T + (damping ** 2) * np.eye(3)
    return j.T @ np.linalg.solve(a, gain * pos_err)


def _clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= max_norm or n < 1e-12:
        return v
    return v * (max_norm / n)


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    return v / n


class UdpSkeletonReceiver:
    def __init__(self, host: str, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, int(port)))
        self.sock.setblocking(False)
        self.latest = None
        self.latest_time = 0.0
        self.count = 0
        print(f"[UDP] listening on {host}:{port}")

    def poll(self):
        while True:
            try:
                data, _addr = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                self.latest = json.loads(data.decode("utf-8"))
                self.latest_time = time.time()
                self.count += 1
            except Exception as exc:
                print("[UDP] bad packet:", repr(exc))
        return self.latest

    def age(self) -> float:
        return time.time() - self.latest_time if self.latest_time > 0 else 1e9


def _point(pkt: dict, name: str) -> np.ndarray | None:
    lm = pkt.get("landmarks", {}).get(name)
    if not lm or not lm.get("valid", False):
        return None
    return np.array([float(lm["x"]), float(lm["y"]), float(lm["z"])], dtype=np.float64)


def _human_basis(pkt: dict) -> tuple[np.ndarray, np.ndarray] | None:
    ls, rs = _point(pkt, "left_shoulder"), _point(pkt, "right_shoulder")
    lh, rh = _point(pkt, "left_hip"), _point(pkt, "right_hip")
    if any(p is None for p in (ls, rs, lh, rh)):
        return None
    shoulder = 0.5 * (ls + rs)
    pelvis = 0.5 * (lh + rh)
    x_axis = _normalize(rs - ls)  # human right
    z_axis = _normalize(shoulder - pelvis)  # human up
    if x_axis is None or z_axis is None:
        return None
    y_axis = _normalize(np.cross(z_axis, x_axis))  # human front
    if y_axis is None:
        return None
    x_axis = _normalize(np.cross(y_axis, z_axis))
    basis = np.stack([x_axis, y_axis, z_axis], axis=0)  # world -> human rows
    return basis, shoulder


def _human_arm_feature(pkt: dict, side: str) -> np.ndarray | None:
    basis_pack = _human_basis(pkt)
    if basis_pack is None:
        return None
    basis, _shoulder_center = basis_pack
    shoulder = _point(pkt, f"{side}_shoulder")
    wrist = _point(pkt, f"{side}_wrist")
    if shoulder is None or wrist is None:
        return None
    rel = wrist - shoulder
    human = basis @ rel  # [human_right, human_front, human_up]
    # Map human axes to Isaac/RBY1 world-style EE delta: x=front, y=left, z=up.
    # human +right -> robot -left, therefore y = -human_right.
    return np.array([human[1], -human[0], human[2]], dtype=np.float64)


class ArmTargetMapper:
    def __init__(self, calib_frames: int, gain: float, alpha: float):
        self.calib_frames = int(calib_frames)
        self.gain = float(gain)
        self.alpha = float(alpha)
        self.samples = {"left": [], "right": []}
        self.human_neutral = {"left": None, "right": None}
        self.robot_neutral = {"left": None, "right": None}
        self.filtered_target = {"left": None, "right": None}
        self.calibrated = False

    def update_calibration(self, pkt: dict, left_ee: np.ndarray, right_ee: np.ndarray):
        if self.calibrated:
            return
        for side in ("left", "right"):
            feat = _human_arm_feature(pkt, side)
            if feat is not None and np.all(np.isfinite(feat)):
                self.samples[side].append(feat)
        n = min(len(self.samples["left"]), len(self.samples["right"]))
        if n >= self.calib_frames:
            self.human_neutral["left"] = np.mean(self.samples["left"][-self.calib_frames:], axis=0)
            self.human_neutral["right"] = np.mean(self.samples["right"][-self.calib_frames:], axis=0)
            self.robot_neutral["left"] = left_ee.copy()
            self.robot_neutral["right"] = right_ee.copy()
            self.filtered_target["left"] = left_ee.copy()
            self.filtered_target["right"] = right_ee.copy()
            self.calibrated = True
            print("[CALIB] done. Move your wrists slowly. Neutral robot EE positions captured.")
        else:
            if n % 15 == 0:
                print(f"[CALIB] collecting neutral skeleton frames: {n}/{self.calib_frames}")

    def target(self, pkt: dict, side: str, workspace_min: np.ndarray, workspace_max: np.ndarray) -> np.ndarray | None:
        if not self.calibrated:
            return None
        feat = _human_arm_feature(pkt, side)
        if feat is None:
            return self.filtered_target[side]
        delta = feat - self.human_neutral[side]

        # Axis-wise deadband removes hand jitter near neutral pose.
        for k in range(3):
            if abs(delta[k]) < args_cli.deadband:
                delta[k] = 0.0

        # Axis-wise gain. This is much easier to tune than one global gain.
        axis_gain = np.array([args_cli.gain_x, args_cli.gain_y, args_cli.gain_z], dtype=np.float64)
        delta = self.gain * axis_gain * delta

        if args_cli.invert_x:
            delta[0] *= -1.0
        if args_cli.invert_y:
            delta[1] *= -1.0
        if args_cli.invert_z:
            delta[2] *= -1.0

        raw = self.robot_neutral[side] + delta
        raw = np.minimum(np.maximum(raw, workspace_min), workspace_max)
        prev = self.filtered_target[side]
        if prev is None:
            filt = raw
        else:
            step = _clip_norm(raw - prev, args_cli.max_target_step)
            limited = prev + step
            filt = (1.0 - self.alpha) * prev + self.alpha * limited
        self.filtered_target[side] = filt
        return filt


def _joint_target_from_map(names: list[str], preset: dict[str, float]) -> np.ndarray:
    return np.array([preset.get(n.split("/")[-1], 0.0) for n in names], dtype=np.float64)


def main():
    scene_root = "/World/RobotScene"
    usd_path = USD_MAP[args_cli.asset]
    if not Path(usd_path).exists():
        raise FileNotFoundError(usd_path)

    world = _load_world(usd_path, scene_root)
    robot, root_path = _resolve_robot(scene_root, world)

    print("=" * 100)
    print("[INFO] Loaded USD:", usd_path)
    print("[INFO] Articulation root:", root_path)
    print("[INFO] DOF names:", list(robot.dof_names))
    print("[INFO] Body names:", list(robot.body_names))
    print("=" * 100)

    all_arm_req = RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS
    arm_ids, arm_names, missing = resolve_joint_indices(robot.dof_names, all_arm_req)
    if missing:
        raise RuntimeError(f"Could not resolve arm joints: {missing}")
    right_ids, left_ids = arm_ids[:7], arm_ids[7:]
    right_names, left_names = arm_names[:7], arm_names[7:]

    left_ee_name = _resolve_body_name(list(robot.body_names), [args_cli.left_ee_body, "ee_finger_l1", "ee_finger_l2", "link_left_arm_6"])
    right_ee_name = _resolve_body_name(list(robot.body_names), [args_cli.right_ee_body, "ee_finger_r1", "ee_finger_r2", "link_right_arm_6"])
    prefixes = [scene_root, root_path]
    left_ee_path = _find_link_prim_under(prefixes, [left_ee_name, args_cli.left_ee_body])
    right_ee_path = _find_link_prim_under(prefixes, [right_ee_name, args_cli.right_ee_body])
    left_ee_view = XFormPrim(prim_paths_expr=left_ee_path, name="left_ee_skel")
    right_ee_view = XFormPrim(prim_paths_expr=right_ee_path, name="right_ee_skel")
    left_ee_view.initialize(); right_ee_view.initialize()

    left_link_idx = robot.get_link_index(left_ee_name)
    right_link_idx = robot.get_link_index(right_ee_name)

    jac_shape = tuple(int(v) for v in np.array(robot.get_jacobian_shape()).flatten())
    q_all = _as_np(robot.get_joint_positions())
    num_dof = int(q_all.shape[1])
    floating_base = jac_shape[-1] == num_dof + 6
    left_body_idx = left_link_idx if floating_base else left_link_idx - 1
    right_body_idx = right_link_idx if floating_base else right_link_idx - 1
    left_cols = np.asarray(left_ids, dtype=np.int32) + (6 if floating_base else 0)
    right_cols = np.asarray(right_ids, dtype=np.int32) + (6 if floating_base else 0)

    print(f"[INFO] Left EE : {left_ee_name} @ {left_ee_path}, body_idx={left_body_idx}")
    print(f"[INFO] Right EE: {right_ee_name} @ {right_ee_path}, body_idx={right_body_idx}")
    print(f"[INFO] Floating base={floating_base}, jac_shape={jac_shape}")

    # Put both arms in a visible, safe teleop-ready posture.
    ready = merge_targets(DUAL_ARM_PRESET["right_assist"], DUAL_ARM_PRESET["left_pregrasp_high"])
    q_right_ready = _joint_target_from_map(right_names, ready)
    q_left_ready = _joint_target_from_map(left_names, ready)

    rx_min = np.array([args_cli.workspace_x[0], args_cli.workspace_y_right[0], args_cli.workspace_z[0]])
    rx_max = np.array([args_cli.workspace_x[1], args_cli.workspace_y_right[1], args_cli.workspace_z[1]])
    lx_min = np.array([args_cli.workspace_x[0], args_cli.workspace_y_left[0], args_cli.workspace_z[0]])
    lx_max = np.array([args_cli.workspace_x[1], args_cli.workspace_y_left[1], args_cli.workspace_z[1]])

    receiver = UdpSkeletonReceiver(args_cli.udp_host, args_cli.udp_port)
    mapper = ArmTargetMapper(args_cli.calib_frames, args_cli.human_gain, args_cli.filter_alpha)

    world.play()
    print("[INFO] Skeleton teleop started.")
    print("[INFO] Stand/sit in view, hold a neutral pose until calibration finishes.")
    print("[INFO] Close Isaac Sim to stop.")

    t_last_print = 0.0
    initialized_posture = False
    while simulation_app.is_running():
        world.step(render=True)
        q = _as_np(robot.get_joint_positions())
        if not initialized_posture:
            act_ids = np.asarray(right_ids + left_ids, dtype=np.int32)
            act_pos = np.concatenate([q_right_ready, q_left_ready], axis=0)
            robot.apply_action(ArticulationActions(joint_positions=act_pos.reshape(1, -1), joint_indices=act_ids))
            # Let the robot settle for a few frames before calibration captures neutral EE positions.
            if world.current_time > 1.2:
                initialized_posture = True
            continue

        pkt = receiver.poll()
        left_pos = _get_world_pos(left_ee_view)
        right_pos = _get_world_pos(right_ee_view)

        tracking = bool(pkt and pkt.get("tracking_ok", False) and receiver.age() < args_cli.hold_timeout)
        if tracking:
            mapper.update_calibration(pkt, left_pos, right_pos)
        if not mapper.calibrated:
            # Hold safe arm posture while waiting for skeleton calibration.
            act_ids = np.asarray(right_ids + left_ids, dtype=np.int32)
            act_pos = np.concatenate([q_right_ready, q_left_ready], axis=0)
            robot.apply_action(ArticulationActions(joint_positions=act_pos.reshape(1, -1), joint_indices=act_ids))
            continue

        if tracking:
            if args_cli.swap_arms:
                target_left = mapper.target(pkt, "right", lx_min, lx_max)
                target_right = mapper.target(pkt, "left", rx_min, rx_max)
            else:
                target_left = mapper.target(pkt, "left", lx_min, lx_max)
                target_right = mapper.target(pkt, "right", rx_min, rx_max)
        else:
            target_left = mapper.filtered_target["left"]
            target_right = mapper.filtered_target["right"]

        jac = _as_np(robot.get_jacobians())
        q_left = q[0, left_ids].copy()
        q_right = q[0, right_ids].copy()
        dq_left = np.zeros(7)
        dq_right = np.zeros(7)
        err_l = np.zeros(3)
        err_r = np.zeros(3)
        if target_left is not None:
            err_l = _clip_norm(target_left - left_pos, 0.08)
            j_l = jac[0, left_body_idx, 0:3, :][:, left_cols]
            dq_left = _dls_step(j_l, err_l, args_cli.lambda_dls, args_cli.ik_gain)
        if target_right is not None:
            err_r = _clip_norm(target_right - right_pos, 0.08)
            j_r = jac[0, right_body_idx, 0:3, :][:, right_cols]
            dq_right = _dls_step(j_r, err_r, args_cli.lambda_dls, args_cli.ik_gain)

        # mild posture pull keeps elbows from folding strangely
        dq_left += args_cli.posture_pull * (q_left_ready - q_left)
        dq_right += args_cli.posture_pull * (q_right_ready - q_right)
        dq_left = np.clip(dq_left, -args_cli.max_joint_step, args_cli.max_joint_step)
        dq_right = np.clip(dq_right, -args_cli.max_joint_step, args_cli.max_joint_step)

        act_ids = np.asarray(right_ids + left_ids, dtype=np.int32)
        act_pos = np.concatenate([q_right + dq_right, q_left + dq_left], axis=0)
        robot.apply_action(ArticulationActions(joint_positions=act_pos.reshape(1, -1), joint_indices=act_ids))

        now = time.time()
        if now - t_last_print > args_cli.print_every:
            print(
                f"[TELEOP] udp={receiver.count} age={receiver.age():.2f}s tracking={tracking} "
                f"calib={mapper.calibrated} |eL|={np.linalg.norm(err_l):.3f} |eR|={np.linalg.norm(err_r):.3f}",
                flush=True,
            )
            t_last_print = now

    simulation_app.close()


if __name__ == "__main__":
    main()
