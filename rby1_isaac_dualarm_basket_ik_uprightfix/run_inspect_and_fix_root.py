"""Load the prepared USD, inspect articulation roots, and optionally patch the best RBY1 prim with ArticulationRootAPI.

Run with Isaac Lab:
    ./isaaclab.sh -p /absolute/path/to/run_inspect_and_fix_root.py --asset v1_1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--asset", choices=["v1_0", "v1_1"], default="v1_1")
parser.add_argument("--auto_fix_root", action="store_true", default=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaacsim.core.api import World

from rby1_isaac_connected.common import (
    USD_MAP,
    apply_articulation_root_to_best_candidate,
    candidate_root_paths,
    find_articulation_roots_under,
    find_rby1_like_prims,
)


def main():
    usd_path = USD_MAP[args_cli.asset]
    if not Path(usd_path).exists():
        raise FileNotFoundError(usd_path)

    world = World(stage_units_in_meters=1.0)
    scene_root = "/World/RobotScene"
    usd_cfg = sim_utils.UsdFileCfg(usd_path=str(usd_path))
    usd_cfg.func(scene_root, usd_cfg)

    print("=" * 100)
    print("[INFO] Loaded USD:", usd_path)
    print("[INFO] Scene root:", scene_root)
    print("=" * 100)

    print("\n[INFO] RBY1-like prims:")
    for path in find_rby1_like_prims(scene_root)[:80]:
        print("  ", path)

    roots = find_articulation_roots_under(scene_root)
    print("\n[INFO] Articulation roots before patch:")
    if roots:
        for path in roots:
            print("  ", path)
    else:
        print("  (none)")

    if not roots and args_cli.auto_fix_root:
        patched = apply_articulation_root_to_best_candidate(scene_root)
        print("\n[INFO] Applied ArticulationRootAPI to:", patched)

    world.reset()
    roots = find_articulation_roots_under(scene_root)
    print("\n[INFO] Articulation roots after reset:")
    if roots:
        for path in roots:
            print("  ", path)
    else:
        print("  (none)")

    print("\n[INFO] Candidate root paths to try manually:")
    for path in candidate_root_paths(scene_root)[:20]:
        print("  ", path)


if __name__ == "__main__":
    main()
    simulation_app.close()
