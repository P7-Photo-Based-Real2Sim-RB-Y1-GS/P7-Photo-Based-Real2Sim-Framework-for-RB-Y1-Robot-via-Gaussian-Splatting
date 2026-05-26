"""Dual-arm basket demo for RBY1 prepared USD scenes.

Sequence:
1. Both arms move to ready pose.
2. Left arm approaches object.
3. Left gripper closes.
4. Left arm lifts.
5. Left arm moves above basket.
6. Left gripper opens to drop into basket.
7. Both arms return to ready pose.

This is a scripted joint-space demo intended to verify Isaac Sim + Isaac Lab connection
with the uploaded USD assets. It does not solve IK or grasp planning.

Run:
    ./isaaclab.sh -p /absolute/path/to/run_dual_arm_basket_demo.py --asset v1_1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--asset", choices=["v1_0", "v1_1"], default="v1_1")
parser.add_argument("--kp", type=float, default=90.0)
parser.add_argument("--ki", type=float, default=2.5)
parser.add_argument("--kd", type=float, default=12.0)
parser.add_argument("--effort_limit", type=float, default=55.0)
parser.add_argument("--loop", action="store_true", help="Loop the basket sequence.")
parser.add_argument("--mug-dx", type=float, default=-0.18, help="Shift mug in X to bring it closer to the robot.")
parser.add_argument("--mug-dy", type=float, default=0.0, help="Shift mug in Y.")
parser.add_argument("--mug-dz", type=float, default=0.0, help="Shift mug in Z.")
parser.add_argument("--basket-dx", type=float, default=-0.08, help="Shift basket in X to make placement easier.")
parser.add_argument("--basket-dy", type=float, default=0.10, help="Shift basket in Y.")
parser.add_argument("--basket-dz", type=float, default=0.0, help="Shift basket in Z.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation

from rby1_isaac_connected.common import (
    DUAL_ARM_PRESET,
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_CLOSED,
    LEFT_GRIPPER_JOINTS,
    LEFT_GRIPPER_OPEN,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_JOINTS,
    RIGHT_GRIPPER_OPEN,
    USD_MAP,
    apply_articulation_root_to_best_candidate,
    candidate_root_paths,
    find_articulation_roots_under,
    find_first_prim_containing,
    merge_targets,
    offset_prim_translation,
    resolve_joint_indices,
    resolve_optional_joint_indices,
)


@dataclass
class Phase:
    name: str
    duration: float
    targets: dict[str, float]


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


class PhasePlayer:
    def __init__(self, phases: list[Phase], loop: bool = False):
        self.phases = phases
        self.loop = loop
        self.index = 0
        self.elapsed = 0.0
        self.last_reported = None

    @property
    def current(self) -> Phase:
        return self.phases[self.index]

    def step(self, dt: float) -> Phase:
        self.elapsed += dt
        while self.elapsed >= self.current.duration:
            self.elapsed -= self.current.duration
            if self.index < len(self.phases) - 1:
                self.index += 1
            elif self.loop:
                self.index = 0
            else:
                self.elapsed = min(self.elapsed, self.current.duration)
                break
        return self.current

    def maybe_report(self):
        if self.last_reported != self.index:
            self.last_reported = self.index
            print(f"[PHASE] {self.index + 1}/{len(self.phases)} : {self.current.name}")


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


def _build_control_targets(matched_joint_names: list[str], q0: np.ndarray, target_map: dict[str, float]) -> np.ndarray:
    target = np.array(q0, dtype=np.float64).copy()
    for i, full_name in enumerate(matched_joint_names):
        base_name = full_name.split("/")[-1]
        if base_name in target_map:
            target[i] = float(target_map[base_name])
    return target


def _print_joint_summary(title: str, names: list[str]):
    print(title)
    if names:
        for n in names:
            print("  -", n)
    else:
        print("  (none)")




def _apply_scene_offsets(scene_root: str):
    mug = find_first_prim_containing(scene_root, ["MUG"], exclude_tokens=["MATERIAL", "LOOK", "SHADER"])
    basket = find_first_prim_containing(scene_root, ["BASKET"], exclude_tokens=["MATERIAL", "LOOK", "SHADER"])
    if basket is None:
        basket = find_first_prim_containing(scene_root, ["CRATE", "TABLE"], exclude_tokens=["MATERIAL", "LOOK", "SHADER"])
    if mug:
        new_t = offset_prim_translation(mug, (args_cli.mug_dx, args_cli.mug_dy, args_cli.mug_dz))
        print(f"[INFO] Mug shifted: {mug} -> {new_t}")
    else:
        print("[WARN] Mug prim not found. Skipping mug offset.")
    if basket:
        new_t = offset_prim_translation(basket, (args_cli.basket_dx, args_cli.basket_dy, args_cli.basket_dz))
        print(f"[INFO] Basket shifted: {basket} -> {new_t}")
    else:
        print("[WARN] Basket prim not found. Skipping basket offset.")

def main():
    usd_path = USD_MAP[args_cli.asset]
    if not Path(usd_path).exists():
        raise FileNotFoundError(usd_path)

    scene_root = "/World/RobotScene"
    world = _load_world(usd_path, scene_root)
    _apply_scene_offsets(scene_root)
    robot, root_path = _resolve_robot(scene_root, world)

    print("=" * 100)
    print("[INFO] Loaded USD:", usd_path)
    print("[INFO] Resolved articulation root:", root_path)
    print("=" * 100)
    print("[INFO] Available DOF names:")
    for i, name in enumerate(robot.dof_names):
        print(f"  [{i:02d}] {name}")

    arm_joint_request = RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS
    arm_indices, arm_names, missing_arms = resolve_joint_indices(robot.dof_names, arm_joint_request)
    if missing_arms:
        raise RuntimeError(
            f"Could not resolve these arm joints: {missing_arms}. Available DOFs were: {robot.dof_names}"
        )

    left_gripper_idx, left_gripper_names = resolve_optional_joint_indices(robot.dof_names, LEFT_GRIPPER_JOINTS)
    right_gripper_idx, right_gripper_names = resolve_optional_joint_indices(robot.dof_names, RIGHT_GRIPPER_JOINTS)

    all_indices = np.array(arm_indices + left_gripper_idx + right_gripper_idx, dtype=np.int32)
    all_names = arm_names + left_gripper_names + right_gripper_names
    q0 = np.array(robot.get_joint_positions(joint_indices=all_indices), dtype=np.float64)

    _print_joint_summary("[INFO] Controlled arm joints:", arm_names)
    _print_joint_summary("[INFO] Left gripper joints:", left_gripper_names)
    _print_joint_summary("[INFO] Right gripper joints:", right_gripper_names)

    ready_targets = merge_targets(
        DUAL_ARM_PRESET["right_home"],
        DUAL_ARM_PRESET["left_home"],
        LEFT_GRIPPER_OPEN if left_gripper_names else {},
        RIGHT_GRIPPER_OPEN if right_gripper_names else {},
    )
    pregrasp_targets = merge_targets(
        DUAL_ARM_PRESET["right_assist"],
        DUAL_ARM_PRESET["left_pregrasp_high"],
        LEFT_GRIPPER_OPEN if left_gripper_names else {},
        RIGHT_GRIPPER_OPEN if right_gripper_names else {},
    )
    grasp_targets = merge_targets(
        DUAL_ARM_PRESET["right_assist"],
        DUAL_ARM_PRESET["left_grasp_down"],
        LEFT_GRIPPER_CLOSED if left_gripper_names else {},
        RIGHT_GRIPPER_OPEN if right_gripper_names else {},
    )
    lift_targets = merge_targets(
        DUAL_ARM_PRESET["right_assist"],
        DUAL_ARM_PRESET["left_lift_high"],
        LEFT_GRIPPER_CLOSED if left_gripper_names else {},
        RIGHT_GRIPPER_OPEN if right_gripper_names else {},
    )
    over_basket_targets = merge_targets(
        DUAL_ARM_PRESET["right_assist"],
        DUAL_ARM_PRESET["left_over_basket_high"],
        LEFT_GRIPPER_CLOSED if left_gripper_names else {},
        RIGHT_GRIPPER_OPEN if right_gripper_names else {},
    )
    release_targets = merge_targets(
        DUAL_ARM_PRESET["right_assist"],
        DUAL_ARM_PRESET["left_release_down"],
        LEFT_GRIPPER_OPEN if left_gripper_names else {},
        RIGHT_GRIPPER_OPEN if right_gripper_names else {},
    )
    return_targets = ready_targets

    phases = [
        Phase("ready", 2.0, ready_targets),
        Phase("left pregrasp high", 2.0, pregrasp_targets),
        Phase("left grasp down + close", 1.5, grasp_targets),
        Phase("lift object high", 2.0, lift_targets),
        Phase("move over basket high", 2.5, over_basket_targets),
        Phase("release down into basket", 1.5, release_targets),
        Phase("return to ready", 2.0, return_targets),
    ]

    try:
        robot.switch_control_mode("effort", joint_indices=all_indices)
    except Exception as exc:
        print("[WARN] switch_control_mode('effort') failed or is unavailable:", exc)

    try:
        robot.set_effort_modes("force", joint_indices=all_indices)
    except Exception as exc:
        print("[WARN] set_effort_modes('force') failed or is unavailable:", exc)

    pid = JointPID(args_cli.kp, args_cli.ki, args_cli.kd, args_cli.effort_limit)
    pid.set_target(_build_control_targets(all_names, q0, phases[0].targets))
    player = PhasePlayer(phases, loop=args_cli.loop)
    player.maybe_report()

    def on_physics_step(dt: float):
        phase = player.step(dt)
        player.maybe_report()
        pid.set_target(_build_control_targets(all_names, q0, phase.targets))

        q = np.array(robot.get_joint_positions(joint_indices=all_indices), dtype=np.float64)
        qd = np.array(robot.get_joint_velocities(joint_indices=all_indices), dtype=np.float64)
        tau = pid.compute(q, qd, dt)
        robot.set_joint_efforts(tau, joint_indices=all_indices)

    try:
        world.remove_physics_callback("rby1_dual_arm_basket_demo")
    except Exception:
        pass
    world.add_physics_callback("rby1_dual_arm_basket_demo", on_physics_step)
    world.play()

    print("[INFO] Dual-arm basket demo started. Close Isaac Sim window to stop.")
    while simulation_app.is_running():
        world.step(render=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
