"""Dual-arm basket demo with posture-safe position IK.

Main fixes in this version:
- use the same EE body for Jacobian and world-pose tracking
- keep the torso mostly upright by default (torso_mode=one)
- apply stronger torso regularization and smaller torso steps

Drop-target improvements over the previous version:
- basket placement uses the basket world-space AABB top center instead of the basket root prim origin
- release no longer teleports the mug to the basket root, so the mug will stay at the carried drop pose

Stability improvements over the previous version:
- phase targets are reached through a ramped Cartesian reference instead of a sudden jump
- lower default IK gain and smaller joint step
- task-space error clamp to avoid huge Jacobian updates near singularities
- simple settle-count hysteresis before switching phases
- carry offset is measured at attach time instead of being hard-coded

This is still a waypoint IK demo, not a full collision-avoiding planner.

Run with Isaac Lab, for example:
    ./isaaclab.sh -p /absolute/path/to/run_dual_arm_basket_ik_demo.py --asset v1_1
Do not run this file with plain `python`, because Omniverse/Isaac Sim modules (including `pxr`)
are made available after Isaac Lab launches the SimulationApp.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--asset", choices=["v1_0", "v1_1"], default="v1_1")
parser.add_argument("--loop", action="store_true", help="Loop the whole pick-and-place sequence.")
parser.add_argument("--ready-time", type=float, default=2.5)
parser.add_argument("--right-assist-time", type=float, default=2.0, help="Time reserved for visibly moving the right arm into the assist pose.")
parser.add_argument("--lambda-dls", type=float, default=0.12, help="Damping term for DLS IK.")
parser.add_argument("--ik-gain", type=float, default=0.55, help="Position error gain for IK updates.")
parser.add_argument("--max-joint-step", type=float, default=0.025, help="Max joint delta [rad] per simulation step.")
parser.add_argument("--goal-tol", type=float, default=0.020, help="Distance tolerance [m] to switch phases.")
parser.add_argument("--phase-timeout", type=float, default=5.0, help="Timeout [s] for each IK phase.")
parser.add_argument("--settle-steps", type=int, default=12, help="How many consecutive in-tolerance steps are needed.")
parser.add_argument("--phase-ramp", type=float, default=0.70, help="Fraction of each phase used to move the Cartesian reference.")
parser.add_argument("--max-task-error", type=float, default=0.08, help="Clamp on Cartesian position error norm [m].")
parser.add_argument("--approach-height", type=float, default=0.22)
parser.add_argument("--grasp-height", type=float, default=0.11)
parser.add_argument("--lift-height", type=float, default=0.28)
parser.add_argument("--basket-height", type=float, default=0.30)
parser.add_argument("--drop-height", type=float, default=0.06, help="Drop goal height above basket top center [m].")
parser.add_argument("--basket-inside-offset", type=float, default=0.03, help="Vertical offset above basket top used as the internal drop anchor [m].")
parser.add_argument("--basket-target-dx", type=float, default=0.0, help="Extra X offset applied to basket drop target center [m].")
parser.add_argument("--basket-target-dy", type=float, default=0.0, help="Extra Y offset applied to basket drop target center [m].")
parser.add_argument("--basket-pre-dx", type=float, default=-0.16, help="Pre-drop anchor X offset toward the robot [m].")
parser.add_argument("--basket-pre-dy", type=float, default=0.12, help="Pre-drop anchor Y offset toward the left arm [m].")
parser.add_argument("--release-tol", type=float, default=0.065, help="Required distance [m] from drop goal before release.")
parser.add_argument("--critical-phase-timeout", type=float, default=8.0, help="Timeout used for basket approach/drop phases.")
parser.add_argument("--mug-dx", type=float, default=-0.15)
parser.add_argument("--left-ee-body", type=str, default="ee_finger_l1")
parser.add_argument("--right-ee-body", type=str, default="ee_finger_r1")
parser.add_argument("--torso-mode", choices=["none", "one", "two", "all"], default="one", help="How many torso joints may participate in IK.")
parser.add_argument("--arm-seed-pull", type=float, default=0.06, help="Posture pull toward the arm seed pose.")
parser.add_argument("--torso-seed-pull", type=float, default=0.20, help="Stronger posture pull that keeps the torso upright.")
parser.add_argument("--torso-step-scale", type=float, default=0.20, help="Scale applied to torso IK steps to prevent folding.")
parser.add_argument("--torso-limit-rad", type=float, default=0.30, help="Clamp torso joints around the torso seed posture.")
parser.add_argument("--mug-dy", type=float, default=0.0)
parser.add_argument("--mug-dz", type=float, default=0.0)
parser.add_argument("--basket-dx", type=float, default=-0.08)
parser.add_argument("--basket-dy", type=float, default=0.10)
parser.add_argument("--basket-dz", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom
import isaaclab.sim as sim_utils
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, XFormPrim
from isaacsim.core.utils.types import ArticulationActions

from rby1_isaac_connected.common import (
    DUAL_ARM_PRESET,
    RIGHT_ARM_JOINTS,
    USD_MAP,
    apply_articulation_root_to_best_candidate,
    candidate_root_paths,
    find_articulation_roots_under,
    find_first_prim_containing,
    merge_targets,
    offset_prim_translation,
    resolve_joint_indices,
)


@dataclass
class IKPhase:
    name: str
    mode: str  # "joint_hold", "ik_track", "attach", "release"
    min_time: float = 0.4
    timeout: float = 4.0


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
            robot = Articulation(prim_paths_expr=path, name="rby1_view")
            robot.initialize()
            return robot, path
        except Exception as exc:
            last_exc = exc
            print(f"[WARN] Failed to initialize articulation at {path}: {exc}")

    raise RuntimeError(
        "Could not initialize any articulation candidate. "
        f"Tried: {trial_paths}. Last error: {last_exc}"
    )


def _print_names(title: str, items: list[str]):
    print(title)
    for i, name in enumerate(items):
        print(f"  [{i:02d}] {name}")


def _resolve_body_name(body_names: list[str], name_candidates: list[str]) -> str | None:
    names = list(body_names)
    upper_to_name = {n.upper(): n for n in names}
    for cand in name_candidates:
        if cand.upper() in upper_to_name:
            return upper_to_name[cand.upper()]
    for cand in name_candidates:
        suffix_hits = [n for n in names if n.upper().endswith(cand.upper())]
        if len(suffix_hits) == 1:
            return suffix_hits[0]
    for cand in name_candidates:
        sub_hits = [n for n in names if cand.upper() in n.upper()]
        if sub_hits:
            return sorted(sub_hits, key=len)[0]
    return None


def _find_link_prim_under(prefixes: list[str], name_candidates: list[str]) -> str | None:
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None
    prefixes = [p for p in prefixes if p]
    wanted = [x.upper() for x in name_candidates]
    hits: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prefixes and not any(path.startswith(pref) for pref in prefixes):
            continue
        base = path.split("/")[-1].upper()
        if base in wanted:
            hits.append(path)
    if hits:
        return sorted(set(hits), key=lambda p: (p.count("/"), p))[0]
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prefixes and not any(path.startswith(pref) for pref in prefixes):
            continue
        up = path.upper()
        if any((f"/{w}" in up) or up.endswith(w) or (w in up) for w in wanted):
            hits.append(path)
    return sorted(set(hits), key=lambda p: (p.count("/"), p))[0] if hits else None


def _apply_scene_offsets(scene_root: str):
    mug = find_first_prim_containing(scene_root, ["MUG"], exclude_tokens=["MATERIAL", "LOOK", "SHADER"])
    basket = find_first_prim_containing(scene_root, ["BASKET"], exclude_tokens=["MATERIAL", "LOOK", "SHADER"])
    if basket is None:
        basket = find_first_prim_containing(scene_root, ["CRATE"], exclude_tokens=["MATERIAL", "LOOK", "SHADER"])
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
    return mug, basket


def _as_np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _dls_position_step(jacobian_xyz: np.ndarray, pos_err: np.ndarray, damping: float, gain: float) -> np.ndarray:
    j = jacobian_xyz
    a = j @ j.T + (damping ** 2) * np.eye(3)
    dq = j.T @ np.linalg.solve(a, gain * pos_err)
    return dq


def _clip_joint_step(dq: np.ndarray, max_step: float) -> np.ndarray:
    return np.clip(dq, -max_step, max_step)


def _set_joint_targets(robot: Articulation, joint_ids: list[int], joint_pos: np.ndarray):
    act = ArticulationActions(joint_positions=joint_pos.reshape(1, -1), joint_indices=np.asarray(joint_ids, dtype=np.int32))
    robot.apply_action(act)


def _set_dual_arm_targets(
    robot: Articulation,
    right_joint_ids: list[int],
    right_joint_pos: np.ndarray,
    left_joint_ids: list[int],
    left_joint_pos: np.ndarray,
):
    joint_ids = np.asarray(list(right_joint_ids) + list(left_joint_ids), dtype=np.int32)
    joint_pos = np.concatenate([right_joint_pos.reshape(-1), left_joint_pos.reshape(-1)], axis=0)
    act = ArticulationActions(joint_positions=joint_pos.reshape(1, -1), joint_indices=joint_ids)
    robot.apply_action(act)


def _get_world_pose(view: XFormPrim) -> tuple[np.ndarray, np.ndarray]:
    pos, quat = view.get_world_poses()
    return _as_np(pos[0]), _as_np(quat[0])


def _set_world_pose(view: XFormPrim, pos: np.ndarray, quat: np.ndarray):
    view.set_world_poses(_as_np(pos).reshape(1, 3), _as_np(quat).reshape(1, 4))




def _compute_world_aabb(prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available for AABB query.")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid prim for AABB query: {prim_path}")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)
    bound = bbox_cache.ComputeWorldBound(prim)
    box = bound.ComputeAlignedRange()
    mn = np.array([box.GetMin()[0], box.GetMin()[1], box.GetMin()[2]], dtype=np.float64)
    mx = np.array([box.GetMax()[0], box.GetMax()[1], box.GetMax()[2]], dtype=np.float64)
    return mn, mx


def _basket_drop_anchor(basket_path: str) -> np.ndarray:
    mn, mx = _compute_world_aabb(basket_path)
    center = 0.5 * (mn + mx)
    anchor = np.array([center[0], center[1], mx[2] + args_cli.basket_inside_offset], dtype=np.float64)
    anchor[0] += args_cli.basket_target_dx
    anchor[1] += args_cli.basket_target_dy
    return anchor

def _basket_pre_anchor(basket_anchor: np.ndarray) -> np.ndarray:
    pre = basket_anchor.copy()
    pre[0] += args_cli.basket_pre_dx
    pre[1] += args_cli.basket_pre_dy
    return pre


def _debug_goal_error(label: str, actual: np.ndarray, goal: np.ndarray):
    err = np.linalg.norm(goal - actual)
    print(f"[DEBUG] {label}: actual={actual}, goal={goal}, err={err:.4f} m")


def _phase_target_pos(phase_name: str, mug_pos: np.ndarray, basket_anchor: np.ndarray) -> np.ndarray:
    basket_pre = _basket_pre_anchor(basket_anchor)
    if phase_name == "above_mug":
        return mug_pos + np.array([0.00, 0.00, args_cli.approach_height])
    if phase_name == "down_to_mug":
        return mug_pos + np.array([0.00, 0.00, args_cli.grasp_height])
    if phase_name == "lift_mug":
        return mug_pos + np.array([0.00, 0.00, args_cli.lift_height])
    if phase_name == "pre_basket":
        return basket_pre + np.array([0.00, 0.00, args_cli.basket_height + 0.08])
    if phase_name == "above_basket":
        return basket_anchor + np.array([0.00, 0.00, args_cli.basket_height])
    if phase_name == "down_to_drop":
        return basket_anchor + np.array([0.00, 0.00, args_cli.drop_height])
    if phase_name == "retreat":
        return basket_anchor + np.array([0.00, 0.00, args_cli.basket_height + 0.10])
    raise KeyError(phase_name)


def _clamp_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n <= max_norm or n < 1e-12:
        return vec
    return vec * (max_norm / n)


def _smooth_phase_target(start: np.ndarray, goal: np.ndarray, elapsed: float, timeout: float, ramp_fraction: float) -> np.ndarray:
    ramp_time = max(0.2, timeout * max(0.1, min(ramp_fraction, 1.0)))
    alpha = min(1.0, elapsed / ramp_time)
    # cubic smoothstep to avoid abrupt start/stop
    s = alpha * alpha * (3.0 - 2.0 * alpha)
    return start + s * (goal - start)


def main():
    usd_path = USD_MAP[args_cli.asset]
    if not Path(usd_path).exists():
        raise FileNotFoundError(usd_path)

    scene_root = "/World/RobotScene"
    world = _load_world(usd_path, scene_root)
    mug_hint_path, basket_hint_path = _apply_scene_offsets(scene_root)
    robot, root_path = _resolve_robot(scene_root, world)

    print("=" * 100)
    print("[INFO] Loaded USD:", usd_path)
    print("[INFO] Resolved articulation root:", root_path)
    print("=" * 100)
    _print_names("[INFO] Robot DOF names:", list(robot.dof_names))
    _print_names("[INFO] Robot body names:", list(robot.body_names))

    all_arm_request = RIGHT_ARM_JOINTS + [f"left_arm_{i}" for i in range(7)]
    arm_ids, arm_names, missing = resolve_joint_indices(robot.dof_names, all_arm_request)
    torso_request = [f"torso_{i}" for i in range(6)]
    torso_all_ids, torso_all_names, _ = resolve_joint_indices(robot.dof_names, torso_request)
    if missing:
        raise RuntimeError(f"Could not resolve arm joints: {missing}")

    right_joint_ids = arm_ids[:7]
    left_joint_ids = arm_ids[7:]
    right_joint_names = arm_names[:7]
    left_joint_names = arm_names[7:]

    if args_cli.torso_mode == "none":
        torso_ids = []
        torso_names = []
    elif args_cli.torso_mode == "one":
        torso_ids = torso_all_ids[:1]
        torso_names = torso_all_names[:1]
    elif args_cli.torso_mode == "two":
        torso_ids = torso_all_ids[:2]
        torso_names = torso_all_names[:2]
    else:
        torso_ids = torso_all_ids
        torso_names = torso_all_names

    ik_joint_ids = torso_ids + left_joint_ids
    ik_joint_names = torso_names + left_joint_names

    left_body_candidates = [args_cli.left_ee_body, "ee_finger_l1", "ee_left", "link_left_arm_6", "ee_finger_l2"]
    right_body_candidates = [args_cli.right_ee_body, "ee_finger_r1", "ee_right", "link_right_arm_6", "ee_finger_r2"]

    left_ee_name = _resolve_body_name(list(robot.body_names), left_body_candidates)
    right_ee_name = _resolve_body_name(list(robot.body_names), right_body_candidates)

    if left_ee_name is None:
        raise RuntimeError(
            f"Could not resolve a left EE body from robot.body_names. Tried: {left_body_candidates}. "
            f"Available bodies: {list(robot.body_names)}"
        )

    search_prefixes = [scene_root, root_path]
    left_ee_path = _find_link_prim_under(search_prefixes, [left_ee_name] + left_body_candidates)
    right_ee_path = _find_link_prim_under(search_prefixes, ([right_ee_name] if right_ee_name else []) + right_body_candidates)

    if left_ee_path is None:
        raise RuntimeError(
            f"Resolved left EE body '{left_ee_name}' but could not find a prim path under {search_prefixes}."
        )
    if right_ee_path is None:
        print("[WARN] Right EE prim not found; continuing without right EE tracking.")

    left_ee_link_idx = robot.get_link_index(left_ee_name)

    left_ee_view = XFormPrim(prim_paths_expr=left_ee_path, name="left_ee")
    left_ee_view.initialize()
    right_ee_view = None
    if right_ee_path:
        right_ee_view = XFormPrim(prim_paths_expr=right_ee_path, name="right_ee")
        right_ee_view.initialize()

    mug_path = mug_hint_path or find_first_prim_containing(scene_root, ["MUG"], exclude_tokens=["MATERIAL", "LOOK", "SHADER"])
    basket_path = basket_hint_path or find_first_prim_containing(scene_root, ["BASKET"], exclude_tokens=["MATERIAL", "LOOK", "SHADER"])
    if basket_path is None:
        basket_path = find_first_prim_containing(scene_root, ["CRATE"], exclude_tokens=["MATERIAL", "LOOK", "SHADER"])
    if mug_path is None or basket_path is None:
        raise RuntimeError(f"Could not resolve mug/basket paths. mug={mug_path}, basket={basket_path}")

    mug_view = XFormPrim(prim_paths_expr=mug_path, name="mug")
    mug_view.initialize()
    basket_view = XFormPrim(prim_paths_expr=basket_path, name="basket")
    basket_view.initialize()

    print(f"[INFO] Left EE body   : {left_ee_name}")
    print(f"[INFO] Left EE path   : {left_ee_path}")
    print(f"[INFO] Right EE body  : {right_ee_name}")
    print(f"[INFO] Right EE path  : {right_ee_path}")
    basket_aabb_min, basket_aabb_max = _compute_world_aabb(basket_path)
    basket_drop_anchor = _basket_drop_anchor(basket_path)

    print(f"[INFO] Mug path       : {mug_path}")
    print(f"[INFO] Basket path    : {basket_path}")
    print(f"[INFO] Basket AABB min: {basket_aabb_min}")
    print(f"[INFO] Basket AABB max: {basket_aabb_max}")
    print(f"[INFO] Basket drop anchor: {basket_drop_anchor}")

    jac_shape = tuple(int(v) for v in np.array(robot.get_jacobian_shape()).flatten())
    q_all = _as_np(robot.get_joint_positions())
    num_dof = int(q_all.shape[1])
    floating_base = jac_shape[-1] == num_dof + 6
    jac_body_idx = left_ee_link_idx if floating_base else (left_ee_link_idx - 1)
    # Jacobian columns must match every joint used by IK (torso + left arm).
    # The previous version only selected left-arm columns, which made dq shape (7,)
    # while q_ik/ik_seed had shape (13,), causing a broadcast error during seed pull.
    jac_joint_cols = np.asarray(ik_joint_ids, dtype=np.int32) + (6 if floating_base else 0)

    print(f"[INFO] Jacobian shape : {jac_shape}")
    print(f"[INFO] Floating base  : {floating_base}")
    print(f"[INFO] Left EE link idx: {left_ee_link_idx}, Jacobian body idx: {jac_body_idx}")
    print(f"[INFO] Left arm joints: {list(zip(left_joint_names, left_joint_ids))}")
    print(f"[INFO] Right arm joints: {list(zip(right_joint_names, right_joint_ids))}")
    print(f"[INFO] Torso mode     : {args_cli.torso_mode}")
    print(f"[INFO] Torso joints used for IK: {list(zip(torso_names, torso_ids))}")
    print(f"[INFO] IK joints (torso+left): {list(zip(ik_joint_names, ik_joint_ids))}")

    ready_map = merge_targets(DUAL_ARM_PRESET["right_assist"], DUAL_ARM_PRESET["left_pregrasp_high"])
    right_ready = np.array([ready_map.get(name.split("/")[-1], 0.0) for name in right_joint_names], dtype=np.float64)
    left_seed = np.array([ready_map.get(name.split("/")[-1], 0.0) for name in left_joint_names], dtype=np.float64)
    torso_seed = np.zeros(len(torso_ids), dtype=np.float64)
    right_home_map = merge_targets(DUAL_ARM_PRESET["right_home"], DUAL_ARM_PRESET["left_pregrasp_high"])
    right_home = np.array([right_home_map.get(name.split("/")[-1], 0.0) for name in right_joint_names], dtype=np.float64)
    ik_seed = np.concatenate([torso_seed, left_seed], axis=0)

    phases = [
        IKPhase("ready", mode="joint_hold", min_time=args_cli.ready_time, timeout=args_cli.ready_time),
        IKPhase("right_assist_move", mode="joint_hold", min_time=args_cli.right_assist_time, timeout=args_cli.right_assist_time),
        IKPhase("above_mug", mode="ik_track", min_time=0.8, timeout=args_cli.phase_timeout),
        IKPhase("down_to_mug", mode="ik_track", min_time=0.6, timeout=args_cli.phase_timeout),
        IKPhase("attach", mode="attach", min_time=0.4, timeout=0.6),
        IKPhase("lift_mug", mode="ik_track", min_time=0.8, timeout=args_cli.phase_timeout),
        IKPhase("pre_basket", mode="ik_track", min_time=0.8, timeout=args_cli.critical_phase_timeout),
        IKPhase("above_basket", mode="ik_track", min_time=1.0, timeout=args_cli.critical_phase_timeout),
        IKPhase("down_to_drop", mode="ik_track", min_time=0.8, timeout=args_cli.critical_phase_timeout),
        IKPhase("release", mode="release", min_time=0.5, timeout=0.8),
        IKPhase("retreat", mode="ik_track", min_time=0.8, timeout=args_cli.phase_timeout),
    ]

    phase_idx = 0
    phase_elapsed = 0.0
    phase_start_left_pos = None
    phase_goal_left_pos = None
    right_phase_start = None
    phase_settle = 0
    carrying = False
    carry_offset = np.array([0.00, 0.00, -0.08], dtype=np.float64)
    released_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def current_phase() -> IKPhase:
        return phases[phase_idx]

    def begin_phase(left_pos_w: np.ndarray, mug_pos_w: np.ndarray, basket_pos_w: np.ndarray, right_q: np.ndarray | None = None):
        nonlocal phase_elapsed, phase_start_left_pos, phase_goal_left_pos, right_phase_start, phase_settle
        phase_elapsed = 0.0
        phase_settle = 0
        phase = current_phase()
        phase_start_left_pos = left_pos_w.copy()
        right_phase_start = None if right_q is None else right_q.copy()
        if phase.mode == "ik_track":
            basket_anchor_w = _basket_drop_anchor(basket_path)
            phase_goal_left_pos = _phase_target_pos(phase.name, mug_pos_w, basket_anchor_w)
        else:
            phase_goal_left_pos = None
        print(f"[PHASE] {phase_idx + 1}/{len(phases)} {phase.name}")
        if phase_goal_left_pos is not None:
            print(f"        goal = {phase_goal_left_pos}")

    def advance_phase(left_pos_w: np.ndarray, mug_pos_w: np.ndarray, basket_pos_w: np.ndarray, right_q: np.ndarray):
        nonlocal phase_idx
        phase_idx += 1
        if phase_idx >= len(phases):
            return False
        begin_phase(left_pos_w, mug_pos_w, basket_pos_w, right_q)
        return True

    def reset_sequence(left_pos_w: np.ndarray, mug_pos_w: np.ndarray, basket_pos_w: np.ndarray, right_q: np.ndarray):
        nonlocal phase_idx, carrying, released_quat, carry_offset
        phase_idx = 0
        carrying = False
        released_quat = _get_world_pose(mug_view)[1].copy()
        carry_offset = np.array([0.00, 0.00, -0.08], dtype=np.float64)
        begin_phase(left_pos_w, mug_pos_w, basket_pos_w, right_q)
        print("[INFO] Sequence reset")

    world.play()
    print("[INFO] Upright-safe IK basket demo started. Close Isaac Sim window to stop.")

    initialized = False
    while simulation_app.is_running():
        dt = float(world.get_physics_dt())
        world.step(render=True)

        q_all = _as_np(robot.get_joint_positions())
        q_left = q_all[0, left_joint_ids].copy()
        q_right = q_all[0, right_joint_ids].copy()
        left_pos_w, left_quat_w = _get_world_pose(left_ee_view)
        mug_pos_w, mug_quat_w = _get_world_pose(mug_view)
        basket_pos_w, basket_quat_w = _get_world_pose(basket_view)

        if not initialized:
            reset_sequence(left_pos_w, mug_pos_w, basket_pos_w, q_right)
            initialized = True

        phase = current_phase()
        reached = False
        q_des_right = right_ready.copy()
        q_des_left = q_left.copy()
        q_des_torso = q_all[0, torso_ids].copy() if torso_ids else np.zeros(0, dtype=np.float64)

        if phase.mode == "joint_hold":
            if phase.name == "ready":
                q_des_right = right_home.copy()
                q_des_left = left_seed.copy()
                q_des_torso = torso_seed.copy() if torso_ids else q_des_torso
                right_err = np.linalg.norm(q_right - q_des_right)
                left_err = max(np.linalg.norm(q_left - q_des_left), np.linalg.norm(q_all[0, torso_ids] - q_des_torso) if torso_ids else 0.0)
                reached = max(right_err, left_err) < 0.08
            elif phase.name == "right_assist_move":
                start_right = q_right if right_phase_start is None else right_phase_start
                q_des_right = _smooth_phase_target(start_right, right_ready, phase_elapsed, phase.timeout, args_cli.phase_ramp)
                q_des_left = left_seed.copy()
                q_des_torso = torso_seed.copy() if torso_ids else q_des_torso
                right_err = np.linalg.norm(right_ready - q_right)
                left_err = max(np.linalg.norm(q_left - q_des_left), np.linalg.norm(q_all[0, torso_ids] - q_des_torso) if torso_ids else 0.0)
                reached = max(right_err, left_err) < 0.06
            else:
                q_des_left = left_seed.copy()
                q_des_torso = torso_seed.copy() if torso_ids else q_des_torso
                reached = max(np.linalg.norm(q_left - q_des_left), np.linalg.norm(q_all[0, torso_ids] - q_des_torso) if torso_ids else 0.0) < 0.05
            if torso_ids:
                hold_ids = list(right_joint_ids) + list(torso_ids) + list(left_joint_ids)
                hold_pos = np.concatenate([q_des_right, q_des_torso, q_des_left], axis=0)
                robot.apply_action(ArticulationActions(joint_positions=hold_pos.reshape(1, -1), joint_indices=np.asarray(hold_ids, dtype=np.int32)))
            else:
                _set_dual_arm_targets(robot, right_joint_ids, q_des_right, left_joint_ids, q_des_left)
        elif phase.mode == "ik_track":
            if phase_goal_left_pos is None or phase_start_left_pos is None:
                begin_phase(left_pos_w, mug_pos_w, basket_pos_w, q_right)
            assert phase_goal_left_pos is not None and phase_start_left_pos is not None
            ref_pos_w = _smooth_phase_target(
                phase_start_left_pos,
                phase_goal_left_pos,
                phase_elapsed,
                phase.timeout,
                args_cli.phase_ramp,
            )
            pos_err = ref_pos_w - left_pos_w
            pos_err = _clamp_norm(pos_err, args_cli.max_task_error)

            jac_all = _as_np(robot.get_jacobians())
            jac_xyz = jac_all[0, jac_body_idx, 0:3, :]
            q_ik = q_all[0, ik_joint_ids].copy()
            jac_arm = jac_xyz[:, jac_joint_cols]

            dq = _dls_position_step(jac_arm, pos_err, damping=args_cli.lambda_dls, gain=args_cli.ik_gain)
            torso_count = len(torso_ids)
            if torso_count:
                dq[:torso_count] += args_cli.torso_seed_pull * (ik_seed[:torso_count] - q_ik[:torso_count])
                dq[:torso_count] *= args_cli.torso_step_scale
            dq[torso_count:] += args_cli.arm_seed_pull * (ik_seed[torso_count:] - q_ik[torso_count:])
            dq = _clip_joint_step(dq, args_cli.max_joint_step)
            q_des_ik = q_ik + dq
            if torso_count:
                q_des_ik[:torso_count] = np.clip(
                    q_des_ik[:torso_count],
                    ik_seed[:torso_count] - args_cli.torso_limit_rad,
                    ik_seed[:torso_count] + args_cli.torso_limit_rad,
                )
            q_des_torso = q_des_ik[: len(torso_ids)] if torso_ids else np.zeros(0, dtype=np.float64)
            q_des_left = q_des_ik[len(torso_ids):]
            if torso_ids:
                follow_ids = list(right_joint_ids) + list(torso_ids) + list(left_joint_ids)
                follow_pos = np.concatenate([q_des_right, q_des_torso, q_des_left], axis=0)
                robot.apply_action(ArticulationActions(joint_positions=follow_pos.reshape(1, -1), joint_indices=np.asarray(follow_ids, dtype=np.int32)))
            else:
                _set_dual_arm_targets(robot, right_joint_ids, q_des_right, left_joint_ids, q_des_left)

            goal_err = np.linalg.norm(phase_goal_left_pos - left_pos_w)
            right_err = np.linalg.norm(right_ready - q_right)
            reached = goal_err < args_cli.goal_tol and right_err < 0.10
            if carrying:
                mug_target = left_pos_w + carry_offset
                _set_world_pose(mug_view, mug_target, mug_quat_w)
        elif phase.mode == "attach":
            carrying = True
            released_quat = mug_quat_w.copy()
            carry_offset = mug_pos_w - left_pos_w
            mug_target = left_pos_w + carry_offset
            _set_world_pose(mug_view, mug_target, mug_quat_w)
            if torso_ids:
                attach_ids = list(right_joint_ids) + list(torso_ids) + list(left_joint_ids)
                attach_pos = np.concatenate([q_des_right, q_all[0, torso_ids], q_left], axis=0)
                robot.apply_action(ArticulationActions(joint_positions=attach_pos.reshape(1, -1), joint_indices=np.asarray(attach_ids, dtype=np.int32)))
            else:
                _set_dual_arm_targets(robot, right_joint_ids, q_des_right, left_joint_ids, q_left)
            reached = True
        elif phase.mode == "release":
            # Only release if the end-effector is genuinely close to the planned drop pose.
            # Otherwise keep carrying and let the timeout/phase logic retry or stop advancing.
            drop_goal = _phase_target_pos("down_to_drop", mug_pos_w, _basket_drop_anchor(basket_path))
            dist_to_drop = np.linalg.norm(drop_goal - left_pos_w)
            if torso_ids:
                release_ids = list(right_joint_ids) + list(torso_ids) + list(left_joint_ids)
                release_pos = np.concatenate([q_des_right, q_all[0, torso_ids], q_left], axis=0)
                robot.apply_action(ArticulationActions(joint_positions=release_pos.reshape(1, -1), joint_indices=np.asarray(release_ids, dtype=np.int32)))
            else:
                _set_dual_arm_targets(robot, right_joint_ids, q_des_right, left_joint_ids, q_left)
            if dist_to_drop > args_cli.release_tol:
                print(f"[WARN] Release blocked: end-effector is still {dist_to_drop:.3f} m away from drop goal.")
                carrying = True
                reached = False
            else:
                # Do not teleport the mug to the basket root. The mug has already been carried
                # to the drop pose during the previous IK phase. We simply stop kinematically
                # attaching it to the end-effector here so it stays in place or falls under physics.
                carrying = False
                released_quat = mug_quat_w.copy()
                reached = True
        else:
            raise RuntimeError(f"Unknown phase mode: {phase.mode}")

        phase_elapsed += dt

        if reached:
            phase_settle += 1
        else:
            phase_settle = 0

        if phase.mode == "ik_track" and phase_elapsed < dt * 1.5:
            print(f"        left_ee = {left_pos_w}, mug = {mug_pos_w}, basket = {basket_pos_w}, drop_anchor = {_basket_drop_anchor(basket_path)}")

        should_advance = False
        if phase.mode in ("attach", "release"):
            should_advance = phase_elapsed >= phase.min_time and reached
        else:
            timeout_advance = phase_elapsed >= phase.timeout
            critical = phase.name in ("pre_basket", "above_basket", "down_to_drop")
            goal_far = (phase_goal_left_pos is not None) and (np.linalg.norm(phase_goal_left_pos - left_pos_w) > args_cli.release_tol)
            if critical and timeout_advance and goal_far:
                timeout_advance = False
            should_advance = (phase_elapsed >= phase.min_time and phase_settle >= args_cli.settle_steps) or timeout_advance

        if phase.mode == "ik_track" and phase_elapsed >= phase.timeout and phase_settle < args_cli.settle_steps:
            _debug_goal_error(f"timeout in {phase.name}", left_pos_w, phase_goal_left_pos)
        if should_advance:
            if phase_idx + 1 >= len(phases):
                if args_cli.loop:
                    reset_sequence(left_pos_w, mug_pos_w, basket_pos_w, q_right)
                else:
                    print("[INFO] Upright-safe IK basket demo finished.")
                    break
            else:
                advance_phase(left_pos_w, mug_pos_w, basket_pos_w, q_right)

    simulation_app.close()


if __name__ == "__main__":
    main()
