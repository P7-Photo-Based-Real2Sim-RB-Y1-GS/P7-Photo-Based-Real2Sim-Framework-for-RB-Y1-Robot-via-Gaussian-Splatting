"""Robust RBY1 effort-PID demo using the prepared USD files.

This does not rely on Isaac Lab gym registration. It launches Isaac Sim through Isaac Lab,
loads the prepared USD scene, auto-discovers or patches the articulation root, and then applies
manual joint-effort PID to the right arm.

Run with Isaac Lab:
    ./isaaclab.sh -p /absolute/path/to/run_effort_pid_demo.py --asset v1_1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--asset", choices=["v1_0", "v1_1"], default="v1_1")
parser.add_argument("--kp", type=float, default=80.0)
parser.add_argument("--ki", type=float, default=2.0)
parser.add_argument("--kd", type=float, default=10.0)
parser.add_argument("--effort_limit", type=float, default=40.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation

from rby1_isaac_connected.common import (
    USD_MAP,
    DEFAULT_RIGHT_ARM_TARGET_RAD,
    RIGHT_ARM_JOINTS,
    apply_articulation_root_to_best_candidate,
    candidate_root_paths,
    find_articulation_roots_under,
    resolve_joint_indices,
)


class JointPID:
    def __init__(self, kp: float, ki: float, kd: float, effort_limit: float, integral_limit: float = 0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.effort_limit = effort_limit
        self.integral_limit = integral_limit
        self.target = None
        self.integral = None

    def set_target(self, target: np.ndarray):
        self.target = np.array(target, dtype=np.float64)
        self.integral = np.zeros_like(self.target)

    def compute(self, q: np.ndarray, qd: np.ndarray, dt: float) -> np.ndarray:
        dt = max(float(dt), 1e-4)
        err = self.target - q
        self.integral += err * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        tau = self.kp * err + self.ki * self.integral - self.kd * qd
        return np.clip(tau, -self.effort_limit, self.effort_limit)


def _load_world(usd_path: Path, scene_root: str) -> World:
    world = World(stage_units_in_meters=1.0)
    usd_cfg = sim_utils.UsdFileCfg(usd_path=str(usd_path))
    usd_cfg.func(scene_root, usd_cfg)
    return world


def _resolve_robot(scene_root: str, world: World) -> tuple[SingleArticulation, str]:
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
            robot = SingleArticulation(prim_path=path, name="rby1")
            robot.initialize()
            return robot, path
        except Exception as exc:
            last_exc = exc
            print(f"[WARN] Failed to initialize articulation at {path}: {exc}")

    raise RuntimeError(
        "Could not initialize any articulation candidate. "
        f"Tried: {trial_paths}. Last error: {last_exc}"
    )


def main():
    usd_path = USD_MAP[args_cli.asset]
    if not Path(usd_path).exists():
        raise FileNotFoundError(usd_path)

    scene_root = "/World/RobotScene"
    world = _load_world(usd_path, scene_root)
    robot, root_path = _resolve_robot(scene_root, world)

    print("=" * 100)
    print("[INFO] Loaded USD:", usd_path)
    print("[INFO] Resolved articulation root:", root_path)
    print("=" * 100)
    print("[INFO] Available DOF names:")
    for i, name in enumerate(robot.dof_names):
        print(f"  [{i:02d}] {name}")

    ctrl_idx, matched_names, missing = resolve_joint_indices(robot.dof_names, RIGHT_ARM_JOINTS)
    if missing:
        raise RuntimeError(
            f"Could not resolve these joints: {missing}. Available DOFs were: {robot.dof_names}"
        )

    ctrl_idx = np.array(ctrl_idx, dtype=np.int32)
    q0 = np.array(robot.get_joint_positions(joint_indices=ctrl_idx), dtype=np.float64)
    target = q0.copy()
    for i, name in enumerate(matched_names):
        base_name = name.split("/")[-1]
        key = base_name if base_name in DEFAULT_RIGHT_ARM_TARGET_RAD else RIGHT_ARM_JOINTS[i]
        target[i] = DEFAULT_RIGHT_ARM_TARGET_RAD.get(key, q0[i])

    try:
        robot.switch_control_mode("effort", joint_indices=ctrl_idx)
    except Exception as exc:
        print("[WARN] switch_control_mode('effort') failed or is unavailable:", exc)

    try:
        robot.set_effort_modes("force", joint_indices=ctrl_idx)
    except Exception as exc:
        print("[WARN] set_effort_modes('force') failed or is unavailable:", exc)

    pid = JointPID(args_cli.kp, args_cli.ki, args_cli.kd, args_cli.effort_limit)
    pid.set_target(target)

    print("[INFO] Controlled joints:", matched_names)
    print("[INFO] Initial q:", q0)
    print("[INFO] Target  q:", target)

    def on_physics_step(dt: float):
        q = np.array(robot.get_joint_positions(joint_indices=ctrl_idx), dtype=np.float64)
        qd = np.array(robot.get_joint_velocities(joint_indices=ctrl_idx), dtype=np.float64)
        tau = pid.compute(q, qd, dt)
        robot.set_joint_efforts(tau, joint_indices=ctrl_idx)

    try:
        world.remove_physics_callback("rby1_effort_pid")
    except Exception:
        pass
    world.add_physics_callback("rby1_effort_pid", on_physics_step)
    world.play()

    print("[INFO] Effort PID demo started. Close Isaac Sim window to stop.")
    while simulation_app.is_running():
        world.step(render=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
