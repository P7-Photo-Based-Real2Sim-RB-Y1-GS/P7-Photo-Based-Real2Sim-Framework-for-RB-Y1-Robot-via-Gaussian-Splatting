from __future__ import annotations

from pathlib import Path
from typing import Iterable

import omni.usd
from pxr import UsdPhysics

ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
USD_MAP = {
    "v1_0": ASSET_ROOT / "RBY1_A_v1_0.usd",
    "v1_1": ASSET_ROOT / "RBY1_A_v1_1.usd",
}

RIGHT_ARM_JOINTS = [f"right_arm_{i}" for i in range(7)]
LEFT_ARM_JOINTS = [f"left_arm_{i}" for i in range(7)]
LEFT_GRIPPER_JOINTS = ["gripper_finger_l1", "gripper_finger_l2"]
RIGHT_GRIPPER_JOINTS = ["gripper_finger_r1", "gripper_finger_r2"]

DEFAULT_RIGHT_ARM_TARGET_RAD = {
    "right_arm_0": 0.0,
    "right_arm_1": -0.35,
    "right_arm_2": 0.0,
    "right_arm_3": -0.85,
    "right_arm_4": 0.0,
    "right_arm_5": 0.65,
    "right_arm_6": 0.0,
}

DUAL_ARM_PRESET = {
    "right_home": {
        "right_arm_0": -0.15,
        "right_arm_1": -0.70,
        "right_arm_2": 0.00,
        "right_arm_3": -1.20,
        "right_arm_4": 0.00,
        "right_arm_5": 1.10,
        "right_arm_6": 0.00,
    },
    "right_assist": {
        "right_arm_0": -0.45,
        "right_arm_1": -1.00,
        "right_arm_2": 0.15,
        "right_arm_3": -1.55,
        "right_arm_4": 0.00,
        "right_arm_5": 1.25,
        "right_arm_6": -0.10,
    },
    "left_home": {
        "left_arm_0": 0.15,
        "left_arm_1": 0.70,
        "left_arm_2": 0.00,
        "left_arm_3": -1.20,
        "left_arm_4": 0.00,
        "left_arm_5": 1.10,
        "left_arm_6": 0.00,
    },
    "left_pregrasp": {
        "left_arm_0": 0.45,
        "left_arm_1": 1.05,
        "left_arm_2": 0.10,
        "left_arm_3": -1.55,
        "left_arm_4": 0.05,
        "left_arm_5": 1.25,
        "left_arm_6": 0.10,
    },
    "left_grasp": {
        "left_arm_0": 0.55,
        "left_arm_1": 1.15,
        "left_arm_2": 0.15,
        "left_arm_3": -1.65,
        "left_arm_4": 0.05,
        "left_arm_5": 1.30,
        "left_arm_6": 0.10,
    },
    "left_lift": {
        "left_arm_0": 0.35,
        "left_arm_1": 0.90,
        "left_arm_2": 0.05,
        "left_arm_3": -1.35,
        "left_arm_4": 0.00,
        "left_arm_5": 1.15,
        "left_arm_6": 0.05,
    },
    "left_over_basket": {
        "left_arm_0": -0.10,
        "left_arm_1": 1.00,
        "left_arm_2": -0.15,
        "left_arm_3": -1.45,
        "left_arm_4": 0.10,
        "left_arm_5": 1.20,
        "left_arm_6": -0.10,
    },
    "left_release": {
        "left_arm_0": -0.15,
        "left_arm_1": 0.95,
        "left_arm_2": -0.15,
        "left_arm_3": -1.35,
        "left_arm_4": 0.10,
        "left_arm_5": 1.10,
        "left_arm_6": -0.10,
    },
}

LEFT_GRIPPER_OPEN = {
    "gripper_finger_l1": -0.045,
    "gripper_finger_l2": 0.045,
}
LEFT_GRIPPER_CLOSED = {
    "gripper_finger_l1": -0.002,
    "gripper_finger_l2": 0.002,
}
RIGHT_GRIPPER_OPEN = {
    "gripper_finger_r1": -0.045,
    "gripper_finger_r2": 0.045,
}
RIGHT_GRIPPER_CLOSED = {
    "gripper_finger_r1": -0.002,
    "gripper_finger_r2": 0.002,
}


def get_stage():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available.")
    return stage


def find_articulation_roots_under(prefix: str) -> list[str]:
    stage = get_stage()
    roots = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(prefix):
            continue
        try:
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                roots.append(path)
        except Exception:
            continue
    return sorted(set(roots), key=lambda p: (0 if "RBY1" in p.upper() else 1, p.count("/"), p))


def find_rby1_like_prims(prefix: str) -> list[str]:
    stage = get_stage()
    hits = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(prefix) and "RBY1" in path.upper():
            hits.append(path)
    return sorted(set(hits), key=lambda p: (p.count("/"), p))


def candidate_root_paths(prefix: str) -> list[str]:
    common_suffixes = [
        "/RBY1_A_v1_0/RBY1_A_v1_0",
        "/RBY1_A_v1_0",
        "/RBY1_A_v1_1/RBY1_A_v1_1",
        "/RBY1_A_v1_1",
    ]
    paths = [prefix + s for s in common_suffixes]
    paths.extend(find_rby1_like_prims(prefix))
    return sorted(set(paths), key=lambda p: (0 if p.endswith("/RBY1_A_v1_0/RBY1_A_v1_0") else 1, p.count("/"), p))


def apply_articulation_root_to_best_candidate(prefix: str) -> str | None:
    stage = get_stage()
    for path in candidate_root_paths(prefix):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            try:
                UsdPhysics.ArticulationRootAPI.Apply(prim)
                return path
            except Exception:
                continue
    return None


def resolve_joint_indices(dof_names: Iterable[str], requested_joint_names: list[str]) -> tuple[list[int], list[str], list[str]]:
    dof_names = list(dof_names)
    indices: list[int] = []
    matched: list[str] = []
    missing: list[str] = []
    for req in requested_joint_names:
        if req in dof_names:
            idx = dof_names.index(req)
            indices.append(idx)
            matched.append(dof_names[idx])
            continue
        suffix_matches = [i for i, name in enumerate(dof_names) if name.endswith(req)]
        if len(suffix_matches) == 1:
            idx = suffix_matches[0]
            indices.append(idx)
            matched.append(dof_names[idx])
        else:
            missing.append(req)
    return indices, matched, missing


def resolve_optional_joint_indices(dof_names: Iterable[str], requested_joint_names: list[str]) -> tuple[list[int], list[str]]:
    indices, matched, _ = resolve_joint_indices(dof_names, requested_joint_names)
    return indices, matched


def merge_targets(*target_dicts: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for d in target_dicts:
        out.update(d)
    return out


# Safer top-down preset to avoid table collision in scripted demos.
DUAL_ARM_PRESET.update({
    "left_pregrasp_high": {
        "left_arm_0": 0.30,
        "left_arm_1": 0.90,
        "left_arm_2": 0.00,
        "left_arm_3": -1.10,
        "left_arm_4": 0.00,
        "left_arm_5": 0.85,
        "left_arm_6": 0.00,
    },
    "left_grasp_down": {
        "left_arm_0": 0.42,
        "left_arm_1": 1.05,
        "left_arm_2": 0.05,
        "left_arm_3": -1.42,
        "left_arm_4": 0.00,
        "left_arm_5": 1.02,
        "left_arm_6": 0.00,
    },
    "left_lift_high": {
        "left_arm_0": 0.22,
        "left_arm_1": 0.82,
        "left_arm_2": -0.02,
        "left_arm_3": -1.05,
        "left_arm_4": 0.00,
        "left_arm_5": 0.92,
        "left_arm_6": 0.00,
    },
    "left_over_basket_high": {
        "left_arm_0": -0.05,
        "left_arm_1": 0.88,
        "left_arm_2": -0.10,
        "left_arm_3": -1.12,
        "left_arm_4": 0.05,
        "left_arm_5": 0.98,
        "left_arm_6": -0.05,
    },
    "left_release_down": {
        "left_arm_0": -0.08,
        "left_arm_1": 1.00,
        "left_arm_2": -0.08,
        "left_arm_3": -1.30,
        "left_arm_4": 0.05,
        "left_arm_5": 1.05,
        "left_arm_6": -0.05,
    },
})


def find_first_prim_containing(prefix: str, include_tokens: list[str], exclude_tokens: list[str] | None = None) -> str | None:
    stage = get_stage()
    hits = []
    exclude_tokens = exclude_tokens or []
    include_tokens_upper = [t.upper() for t in include_tokens]
    exclude_tokens_upper = [t.upper() for t in exclude_tokens]
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(prefix):
            continue
        up = path.upper()
        if all(tok in up for tok in include_tokens_upper) and not any(tok in up for tok in exclude_tokens_upper):
            hits.append(path)
    hits = sorted(set(hits), key=lambda p: (p.count('/'), p))
    return hits[0] if hits else None


def add_translate_op_if_needed(prim):
    from pxr import UsdGeom
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return xformable.AddTranslateOp()


def set_prim_translation(path: str, xyz: tuple[float, float, float]):
    from pxr import Gf
    stage = get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid prim path: {path}")
    op = add_translate_op_if_needed(prim)
    op.Set(Gf.Vec3d(*xyz))


def offset_prim_translation(path: str, delta_xyz: tuple[float, float, float]):
    from pxr import Gf, UsdGeom
    stage = get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid prim path: {path}")
    xformable = UsdGeom.Xformable(prim)
    current = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            current = op.Get()
            if current is None:
                current = Gf.Vec3d(0.0, 0.0, 0.0)
            new_v = Gf.Vec3d(float(current[0]) + delta_xyz[0], float(current[1]) + delta_xyz[1], float(current[2]) + delta_xyz[2])
            op.Set(new_v)
            return tuple(new_v)
    op = xformable.AddTranslateOp()
    new_v = Gf.Vec3d(*delta_xyz)
    op.Set(new_v)
    return tuple(new_v)
