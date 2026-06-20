#!/usr/bin/env python3
"""RB-Y1 webcam control v20.6: corrected XYZ placement with robust 6x7 Jacobian selection.

Pipeline
--------
Normal teleoperation maps the human torso/shoulder/elbow/palm directly to
RB-Y1 arm_0~6 joint targets.  The task controller temporarily switches to Cartesian DLS IK for yellow-cup pickup, lift, basket transfer, release, and retreat.

ROS/rclpy is not used.  The RB-Y1 two-jaw gripper is driven only by the human
thumb-index pinch value.  Grasp-assist keeps the detected yellow cup attached
while pinched, which makes the webcam pick-up demo reliable even when imported
finger collision geometry/friction is imperfect.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RB-Y1 webcam Cartesian IK table-pick teleoperation")
parser.add_argument("--asset", choices=["v1_0", "v1_1"], default="v1_1")
parser.add_argument("--udp-bind", default="127.0.0.1")
parser.add_argument("--udp-port", type=int, default=5005)
parser.add_argument("--watchdog", type=float, default=0.8)
parser.add_argument("--floating-base", action="store_true")
parser.add_argument("--enable-robot-gravity", action="store_true")
parser.add_argument("--control-side", choices=["left", "right", "both"], default="right")
parser.add_argument("--teleop-mode", choices=["joint", "ik"], default="joint", help="joint directly retargets arm_0~6; ik keeps the legacy Cartesian follower")
parser.add_argument("--wrist-joint-gains", default="0.72,0.62,0.70", help="Human palm relative rotation gains for RB-Y1 arm_4~6")
parser.add_argument("--left-wrist-joint-signs", default="1,-1,1", help="Direct arm_4~6 signs for the left wrist")
parser.add_argument("--right-wrist-joint-signs", default="1,-1,1", help="Direct arm_4~6 signs for the right wrist")
parser.add_argument("--joint-retarget-deadband-deg", type=float, default=0.9, help="Joint target deadband for stationary holding")
parser.add_argument("--visibility", type=float, default=0.30)
parser.add_argument("--filter-alpha", type=float, default=0.42, help="Human wrist/elbow landmark response; larger is faster")
parser.add_argument("--palm-filter-alpha", type=float, default=0.30, help="Palm orientation response; larger is faster")
parser.add_argument("--target-deadband-xy", type=float, default=0.012, help="Hold wrist XY target until webcam motion exceeds this many meters")
parser.add_argument("--target-deadband-z", type=float, default=0.014, help="Hold wrist Z target until webcam motion exceeds this many meters")
parser.add_argument("--elbow-target-deadband", type=float, default=0.015, help="Hold elbow Cartesian target inside this deadband")
parser.add_argument("--orientation-deadband-deg", type=float, default=2.5, help="Ignore smaller palm-orientation changes")
parser.add_argument("--shoulder-target-deadband-deg", type=float, default=1.2, help="Ignore smaller shoulder-reference changes")
parser.add_argument("--elbow-joint-deadband-deg", type=float, default=1.0, help="Ignore smaller arm_3 reference changes")
parser.add_argument("--ik-position-hold-deadband", type=float, default=0.007, help="Stop IK correction when hand position error is below this")
parser.add_argument("--ik-orientation-hold-deadband-deg", type=float, default=1.8, help="Stop wrist orientation correction below this error")
parser.add_argument("--ik-elbow-hold-deadband", type=float, default=0.010, help="Stop elbow Cartesian correction below this error")
parser.add_argument("--ik-joint-hold-deadband-deg", type=float, default=0.65, help="Stop shoulder/elbow joint correction below this error")
parser.add_argument("--forward-scale", type=float, default=1.15)
parser.add_argument("--lateral-scale", type=float, default=0.75)
parser.add_argument("--vertical-scale", type=float, default=1.05)
parser.add_argument("--wrist-up-image-scale", type=float, default=0.92, help="Meters of robot wrist rise per unit image-height rise; expanded for overhead reach")
parser.add_argument("--wrist-up-blend", type=float, default=0.85, help="Blend of robust image-space vertical signal over noisy MediaPipe world-z")
parser.add_argument("--wrist-up-deadzone", type=float, default=0.012, help="Image-height deadzone for wrist up/down")
parser.add_argument("--wrist-up-speed", type=float, default=1.10, help="Independent vertical target slew limit in m/s")
parser.add_argument("--wrist-up-task-weight", type=float, default=4.0, help="Extra DLS weight for end-effector vertical tracking")
parser.add_argument("--wrist-up-direct-gain", type=float, default=1.15, help="Direct Jacobian-z correction gain when the wrist target is above the EE")
parser.add_argument("--wrist-up-direct-max-step", type=float, default=0.045, help="Maximum direct vertical IK correction norm per step")
parser.add_argument("--elbow-forward-scale", type=float, default=0.55)
parser.add_argument("--elbow-lateral-scale", type=float, default=0.55)
parser.add_argument("--elbow-vertical-scale", type=float, default=0.72)
parser.add_argument("--human-forward-sign", type=float, default=-1.0)
parser.add_argument("--ee-speed", type=float, default=0.85, help="Cartesian wrist-target slew limit in m/s")
parser.add_argument("--elbow-speed", type=float, default=0.70, help="Cartesian elbow-target slew limit in m/s")
parser.add_argument("--hand-task-weight", type=float, default=1.0)
parser.add_argument("--elbow-task-weight", type=float, default=0.80, help="Elbow Cartesian task; explicit arm_3 extension is handled separately")
parser.add_argument("--elbow-hand-backoff", type=float, default=0.05, help="Keep elbow this far behind the hand in robot +x direction")
parser.add_argument("--elbow-angle-weight", type=float, default=2.40, help="Explicit arm_3 task weight driven by human elbow extension")
parser.add_argument("--elbow-extension-gain", type=float, default=1.35, help="Human elbow-extension delta to robot arm_3 range gain")
parser.add_argument("--extension-forward-scale", type=float, default=0.42, help="Extra robot +x reach generated when the human straightens the elbow")
parser.add_argument("--elbow-direct-blend", type=float, default=0.85, help="Blend DLS arm_3 step toward the explicit extension target")
parser.add_argument("--elbow-direct-max-step-deg", type=float, default=3.2, help="Maximum explicit arm_3 correction per simulation step")
parser.add_argument("--robot-elbow-bend-fraction", type=float, default=0.78, help="Fraction of usable arm_3 range treated as a strongly bent elbow")
parser.add_argument("--human-elbow-bent-deg", type=float, default=55.0)
parser.add_argument("--human-elbow-straight-deg", type=float, default=168.0)
parser.add_argument("--workspace-x", type=float, default=0.70)
parser.add_argument("--workspace-y", type=float, default=0.34)
parser.add_argument("--workspace-z", type=float, default=0.78, help="Vertical EE workspace; expanded for hand-above-shoulder motion")
parser.add_argument("--dls-damping", type=float, default=0.08)
parser.add_argument("--ik-position-gain", type=float, default=1.45)
parser.add_argument("--orientation-weight", type=float, default=0.80, help="Palm orientation task weight; controls arm_4~6 wrist joints")
parser.add_argument("--orientation-speed", type=float, default=1.80, help="Maximum target palm angular speed in rad/s")
parser.add_argument("--max-palm-angle-deg", type=float, default=75.0, help="Maximum palm rotation from the calibrated robot wrist pose")
parser.add_argument("--left-wrist-signs", default="1,1,1", help="Optional local palm rotation signs; geometric identity is the v15 default")
parser.add_argument("--right-wrist-signs", default="1,1,1", help="Optional local palm rotation signs; geometric identity is the v15 default")
parser.add_argument("--wrist-posture-weight", type=float, default=0.55, help="Keep arm_4~6 near their calibrated straight posture while following palm orientation")
parser.add_argument("--wrist-joint-window-deg", default="42,35,48", help="Maximum arm_4~6 deviation from startup pose")
parser.add_argument("--wrist-posture-deadband-deg", type=float, default=1.0, help="Ignore tiny arm_4~6 posture errors")
parser.add_argument("--shoulder-posture-weight", type=float, default=1.20, help="Explicit human shoulder posture task weight; higher for overhead tracking")
parser.add_argument("--shoulder-feature-gains", default="1.05,0.90,0.28", help="elevation,yaw,arm-plane twist gains mapped to arm_0~2; v9 suppresses unreliable twist")
parser.add_argument("--max-shoulder-delta-deg", type=float, default=132.0, help="Human shoulder elevation range, including arm-down to overhead transitions")
parser.add_argument("--shoulder-axis-limit-deg", default="118,68,32", help="arm_0/1/2 relative limits; arm_0 expanded for overhead reach")
parser.add_argument("--shoulder-backward-limit-deg", type=float, default=12.0, help="Maximum arm_0 motion toward the backward shoulder-pitch direction after calibration")
parser.add_argument("--shoulder-deadzone-deg", type=float, default=0.8, help="Ignore tiny human shoulder feature noise")
parser.add_argument("--shoulder-reference-gain", type=float, default=0.72, help="Direct arm_0~2 reference correction gain per IK step")
parser.add_argument("--shoulder-direct-blend", type=float, default=0.65, help="Blend DLS shoulder step toward the explicit shoulder reference")
parser.add_argument("--shoulder-direct-max-step-deg", type=float, default=3.20, help="Maximum direct arm_0~2 correction per simulation step")
parser.add_argument("--shoulder-distal-compensation", type=float, default=0.78, help="arm_3~6 compensation that preserves the EE pose after shoulder correction")
parser.add_argument("--left-shoulder-signs", default="-1,1,1", help="v9 flips arm_0 so raising the human arm moves the RB-Y1 arm forward/up")
parser.add_argument("--right-shoulder-signs", default="-1,1,1", help="v9 flips arm_0 so raising the human arm moves the RB-Y1 arm forward/up")
parser.add_argument("--left-shoulder-offset-deg", default="0,0,0", help="Static arm_0~2 trim in degrees after calibration")
parser.add_argument("--right-shoulder-offset-deg", default="0,0,0", help="Static arm_0~2 trim in degrees after calibration")
parser.add_argument("--null-gain", type=float, default=0.005)
parser.add_argument("--joint-limit-gain", type=float, default=0.025)
parser.add_argument("--ik-max-joint-step", type=float, default=0.085)
parser.add_argument("--arm-rate-limit", type=float, default=1.75)
parser.add_argument("--shoulder-rate-limit", type=float, default=2.50)
parser.add_argument("--wrist-rate-limit", type=float, default=1.35)
parser.add_argument("--arm-kp", type=float, default=220.0)
parser.add_argument("--shoulder-kp", type=float, default=360.0)
parser.add_argument("--arm-kd", type=float, default=26.0)
parser.add_argument("--arm-effort-limit", type=float, default=80.0)
parser.add_argument("--shoulder-effort-limit", type=float, default=140.0)
parser.add_argument("--support-kp", type=float, default=80.0)
parser.add_argument("--support-kd", type=float, default=10.0)
parser.add_argument("--support-effort-limit", type=float, default=30.0)
parser.add_argument("--gripper-kp", type=float, default=220.0)
parser.add_argument("--gripper-kd", type=float, default=12.0)
parser.add_argument("--gripper-effort-limit", type=float, default=12.0)
parser.add_argument("--gripper-rate-limit", type=float, default=1.00)
parser.add_argument("--pinch-deadzone", type=float, default=0.05)
parser.add_argument("--pinch-full-close", type=float, default=0.62)
parser.add_argument("--pinch-gamma", type=float, default=0.65, help="Less than 1 boosts mid-range webcam pinch values")
parser.add_argument("--gripper-fallback-travel", type=float, default=0.035, help="Fallback jaw travel when USD limits are unusable")
parser.add_argument("--left-gripper-open-targets", default="", help="Optional explicit comma-separated open targets")
parser.add_argument("--left-gripper-close-targets", default="", help="Optional explicit comma-separated close targets")
parser.add_argument("--right-gripper-open-targets", default="", help="Optional explicit comma-separated open targets")
parser.add_argument("--right-gripper-close-targets", default="", help="Optional explicit comma-separated close targets")
parser.add_argument("--left-gripper-invert", action="store_true", help="Swap inferred left gripper open/close targets")
parser.add_argument("--right-gripper-invert", action="store_true", help="Swap inferred right gripper open/close targets")
parser.add_argument("--left-precision-joints", default="")
parser.add_argument("--right-precision-joints", default="")
parser.add_argument(
    "--left-ee-body", default="auto",
    help="Articulation body used as the left IK endpoint. 'auto' selects the terminal left-arm/gripper body.",
)
parser.add_argument(
    "--right-ee-body", default="auto",
    help="Articulation body used as the right IK endpoint. 'auto' selects the terminal right-arm/gripper body.",
)
parser.add_argument("--no-demo-table", action="store_true", help="Disable the demo cup and all table-pick task objects")
parser.add_argument("--add-demo-table", action="store_true", help="Explicitly create /World/WebcamPickTable. Off by default because the RBY1 USD scene already contains a table.")
parser.add_argument("--existing-table-prim", default="/World/Environment", help="USD subtree searched for the existing tabletop mesh")
parser.add_argument("--disable-table-surface-detect", action="store_true", help="Use --table-top-z and object coordinates without mesh surface detection")
parser.add_argument("--table-surface-search-radius", type=float, default=0.30, help="XY radius searched around the requested cup location")
parser.add_argument("--table-surface-search-step", type=float, default=0.04, help="XY spacing used while searching for the nearest tabletop point")
parser.add_argument("--table-surface-min-z", type=float, default=0.30, help="Minimum accepted horizontal surface height")
parser.add_argument("--table-surface-max-z", type=float, default=1.30, help="Maximum accepted horizontal surface height")
parser.add_argument("--table-surface-normal-z", type=float, default=0.70, help="Minimum absolute vertical normal component for a tabletop triangle")
parser.add_argument("--disable-existing-cup-anchor", action="store_true", help="Do not search for the yellow cup; use requested object coordinates")
parser.add_argument("--existing-cup-prim", default="auto", help="Existing yellow cup prim path, or auto for name/color detection")
parser.add_argument("--cup-neighbor-gap", type=float, default=0.035, help="Free gap between the existing yellow cup and the generated cup")
parser.add_argument("--cup-neighbor-direction", choices=["toward-robot", "away-from-robot", "x+", "x-", "y+", "y-"], default="toward-robot", help="Preferred side of the existing yellow cup")
parser.add_argument("--cup-neighbor-offset-x", type=float, default=0.0, help="Final manual X adjustment after yellow-cup anchoring")
parser.add_argument("--cup-neighbor-offset-y", type=float, default=0.0, help="Final manual Y adjustment after yellow-cup anchoring")
parser.add_argument("--reference-cup-min-z", type=float, default=0.35, help="Minimum center height for an automatically detected existing cup")
parser.add_argument("--reference-cup-max-z", type=float, default=1.30, help="Maximum center height for an automatically detected existing cup")
parser.add_argument("--basket-prim", default="auto", help="Existing basket/bin/crate prim path, or auto for scene detection")
parser.add_argument("--basket-center-offset-x", type=float, default=0.0, help="Manual X adjustment of the basket drop center")
parser.add_argument("--basket-center-offset-y", type=float, default=0.0, help="Manual Y adjustment of the basket drop center")
parser.add_argument("--basket-drop-clearance", type=float, default=0.018, help="Cup-bottom clearance above the detected basket bottom")
parser.add_argument("--basket-transfer-clearance", type=float, default=0.16, help="Cup-bottom clearance above the basket rim during horizontal transfer")
parser.add_argument("--basket-transfer-duration", type=float, default=1.40, help="Minimum-jerk duration from lifted cup to above basket")
parser.add_argument("--basket-descend-duration", type=float, default=0.90, help="Minimum-jerk duration lowering the cup into the basket")
parser.add_argument("--basket-release-hold", type=float, default=0.32, help="Time to keep the gripper open before retreating")
parser.add_argument("--basket-retreat-height", type=float, default=0.18, help="Vertical gripper retreat after releasing the cup")
parser.add_argument("--basket-retreat-duration", type=float, default=0.75, help="Minimum-jerk retreat duration")
parser.add_argument("--basket-min-center-z", type=float, default=0.30, help="Minimum accepted basket world center height")
parser.add_argument("--basket-max-center-z", type=float, default=1.40, help="Maximum accepted basket world center height")
parser.add_argument("--target-stl-mesh", default="", help="NPZ target mesh path. Empty uses project/assets/target_stl_mesh.npz")
parser.add_argument("--target-x", type=float, default=0.32, help="Target STL center X")
parser.add_argument("--target-y", type=float, default=0.16, help="Target STL center Y")
parser.add_argument("--target-table-z", type=float, default=0.74, help="Table surface Z under the target STL")
parser.add_argument("--target-yaw-deg", type=float, default=0.0, help="Target STL visual yaw")
parser.add_argument("--target-mass", type=float, default=0.12, help="Target STL rigid-body proxy mass")
parser.add_argument("--target-placement", choices=["auto", "manual"], default="auto", help="Auto-detect the main tabletop or use manual coordinates")
parser.add_argument("--target-reach-distance", type=float, default=0.42, help="Preferred robot-to-target horizontal distance")
parser.add_argument("--target-table-margin", type=float, default=0.085, help="Required tabletop margin around target")
parser.add_argument("--target-table-z-bin", type=float, default=0.025, help="Height bin for horizontal tabletop triangles")
parser.add_argument("--target-table-min-area", type=float, default=0.10, help="Minimum horizontal area for tabletop candidate")
parser.add_argument("--target-pointcloud-z-bin", type=float, default=0.03, help="Z histogram bin used when scan triangle normals are unreliable")
parser.add_argument("--target-pointcloud-min-points", type=int, default=60, help="Minimum vertices in a scan-mesh tabletop height cluster")
parser.add_argument("--target-fallback-height", type=float, default=0.78, help="Last-resort target height above robot-scene minimum Z")
parser.add_argument("--target-fallback-direction", choices=["environment","x+","x-","y+","y-"], default="x-", help="Last-resort target direction from the robot")
parser.add_argument("--target-direction-sectors", type=int, default=36, help="Angular sectors used to find the elevated table side around the robot")
parser.add_argument("--target-direction-min-height", type=float, default=0.42, help="Minimum elevated scene height above robot minimum Z for table-direction detection")
parser.add_argument("--target-direction-max-height", type=float, default=1.12, help="Maximum elevated scene height above robot minimum Z for table-direction detection")
parser.add_argument("--target-probe-radius", type=float, default=0.16, help="Local XY radius used to estimate the tabletop Z")
parser.add_argument("--target-probe-z-bin", type=float, default=0.018, help="Local Z histogram bin used to estimate tabletop height")
parser.add_argument("--target-direction-offset-deg", type=float, default=0.0, help="Manual rotation of the detected table direction")
parser.add_argument("--target-lateral-offset", type=float, default=0.0, help="Manual lateral offset perpendicular to the detected table direction")
parser.add_argument("--target-table-side", choices=["x+","x-","y+","y-"], default="x+", help="Known table direction from the robot. The current scene uses x+.")
parser.add_argument("--target-side-search-min", type=float, default=0.30, help="Minimum distance searched along the known table direction")
parser.add_argument("--target-side-search-max", type=float, default=0.62, help="Maximum distance searched along the known table direction")
parser.add_argument("--target-side-search-step", type=float, default=0.035, help="Distance step along the known table direction")
parser.add_argument("--target-side-lateral-range", type=float, default=0.16, help="Lateral range searched across the table edge")
parser.add_argument("--target-side-lateral-step", type=float, default=0.04, help="Lateral search step")
parser.add_argument("--target-side-fallback-height", type=float, default=0.82, help="Fallback tabletop height above the robot-scene minimum Z")
parser.add_argument("--target-offset-x", type=float, default=0.18, help="Final world-X correction after automatic placement; positive moves toward the yellow-cup side in this scene")
parser.add_argument("--target-offset-y", type=float, default=-0.10, help="Final world-Y correction after automatic placement")
parser.add_argument("--target-offset-z", type=float, default=0.18, help="Final upward correction applied to the target support and STL")
parser.add_argument("--disable-auto-grasp", action="store_true", help="Disable startup auto grasp")
parser.add_argument("--auto-grasp-delay", type=float, default=1.0, help="Startup delay before automatic grasp")
parser.add_argument("--autonomous-orientation-weight", type=float, default=0.0, help="Orientation weight during autonomous grasp")
parser.add_argument("--task-cup-x", type=float, default=0.32, help="Movable STL cup center X")
parser.add_argument("--task-cup-y", type=float, default=0.16, help="Movable STL cup center Y")
parser.add_argument("--task-table-z", type=float, default=0.74, help="Existing tabletop world Z beneath the cup")
parser.add_argument("--task-basket-x", type=float, default=0.54, help="Basket interior center X")
parser.add_argument("--task-basket-y", type=float, default=0.15, help="Basket interior center Y")
parser.add_argument("--task-basket-floor-z", type=float, default=0.755, help="Basket interior floor world Z")
parser.add_argument("--task-basket-rim-z", type=float, default=0.865, help="Basket upper rim world Z")
parser.add_argument("--task-cup-yaw-deg", type=float, default=0.0, help="Visual STL cup yaw rotation")
parser.add_argument("--table-x", type=float, default=0.50)
parser.add_argument("--table-y", type=float, default=0.0)
parser.add_argument("--table-top-z", type=float, default=0.74)
parser.add_argument("--table-size-x", type=float, default=0.55)
parser.add_argument("--table-size-y", type=float, default=0.82)
parser.add_argument("--table-thickness", type=float, default=0.08)
parser.add_argument("--object-x", type=float, default=0.32, help="Cup/object X position; default is near the table front edge")
parser.add_argument("--object-forward-adjust", type=float, default=0.0, help="Positive value pulls the cup further toward the robot")
parser.add_argument("--object-left-y", type=float, default=0.16)
parser.add_argument("--object-right-y", type=float, default=-0.14)
parser.add_argument("--object-size", type=float, default=0.075, help="Deprecated cube-size option retained for command compatibility")
parser.add_argument("--cup-radius", type=float, default=0.045, help="Physical cup body radius")
parser.add_argument("--cup-height", type=float, default=0.110, help="Physical cup body height")
parser.add_argument("--cup-wall-thickness", type=float, default=0.0045, help="Visual cup rim/wall thickness")
parser.add_argument("--cup-handle-radius", type=float, default=0.032, help="Visual cup handle major radius")
parser.add_argument("--cup-handle-thickness", type=float, default=0.007, help="Visual cup handle thickness")
parser.add_argument("--cup-mass", type=float, default=0.10, help="Cup rigid-body mass")
parser.add_argument("--approach-height", type=float, default=0.15)
parser.add_argument("--pick-pregrasp-height", type=float, default=0.145, help="Automatic pick hover height above the cup center")
parser.add_argument("--pick-grasp-z-offset", type=float, default=0.010, help="Automatic gripper height above the cup center for side pinch")
parser.add_argument("--pick-position-tolerance", type=float, default=0.035)
parser.add_argument("--pick-orientation-tolerance-deg", type=float, default=14.0)
parser.add_argument("--pick-close-duration", type=float, default=0.55)
parser.add_argument("--pick-stage-timeout", type=float, default=8.0)
parser.add_argument("--pick-attach-distance", type=float, default=0.20, help="Maximum EE-to-cup-center distance for grasp assist")
parser.add_argument("--disable-any-motion-auto-pick", action="store_true", help="Disable automatic cup pick on small arm motion")
parser.add_argument("--motion-trigger-position", type=float, default=0.003, help="Wrist Cartesian displacement threshold in meters")
parser.add_argument("--motion-trigger-shoulder-deg", type=float, default=1.0, help="Shoulder joint-retarget displacement threshold in degrees")
parser.add_argument("--motion-trigger-elbow", type=float, default=0.005, help="Normalized human elbow-extension change threshold")
parser.add_argument("--motion-trigger-wrist-deg", type=float, default=1.0, help="Human palm/wrist relative rotation threshold in degrees")
parser.add_argument("--motion-trigger-hold", type=float, default=0.02, help="Seconds movement must remain above a threshold")
parser.add_argument("--motion-trigger-guard", type=float, default=0.35, help="Ignore motion briefly after C calibration")
parser.add_argument("--motion-trigger-follow-time", type=float, default=0.30, help="Continue direct arm_0~6 following for this long after detecting movement before starting cup pick")
parser.add_argument("--autonomous-posture-weight-scale", type=float, default=0.0, help="Scale shoulder/elbow/wrist posture tasks during cup-pick IK so they do not block reaching")
parser.add_argument("--table-clearance", type=float, default=0.045)
parser.add_argument("--safe-hover-height", type=float, default=0.24, help="Height above table used before horizontal motion")
parser.add_argument("--auto-safe-route", action="store_true", help="Use autonomous raise/translate/descend routing before teleoperation. Off by default for direct hand following.")
parser.add_argument("--anchor-mode", choices=["current", "object-hover"], default="current", help="Map calibration pose to current robot hand pose (direct teleop) or to a point above the demo object.")
parser.add_argument("--min-ee-clearance", type=float, default=0.055, help="Minimum controlled end-effector height above the tabletop during direct teleoperation")
parser.add_argument("--descent-radius", type=float, default=0.065, help="Allow descent only when hand is horizontally over the target")
parser.add_argument("--elbow-clearance", type=float, default=0.16, help="Minimum elbow-link height above the table")
parser.add_argument("--elbow-avoid-gain", type=float, default=0.75)
parser.add_argument("--shoulder-boost", type=float, default=0.0, help="Deprecated fallback; elbow Cartesian task is used instead")
parser.add_argument("--physical-table", action="store_true", help="Give the optional --add-demo-table full collision. Has no effect on the table already included in the imported USD.")
parser.add_argument("--no-grasp-assist", action="store_true")
parser.add_argument("--grasp-distance", type=float, default=0.13)
parser.add_argument("--grasp-close-threshold", type=float, default=0.72)
parser.add_argument("--grasp-release-threshold", type=float, default=0.25)
parser.add_argument("--no-object-reach-assist", action="store_true", help="Disable gentle XY attraction toward the demo object while the human arm is extended")
parser.add_argument("--reach-assist-extension", type=float, default=0.58, help="Human elbow-extension fraction where object reach assist begins")
parser.add_argument("--reach-assist-gain", type=float, default=0.68, help="Maximum XY blend toward the object")
parser.add_argument("--reach-assist-radius", type=float, default=0.60, help="Only assist when the requested hand target is within this horizontal distance of the object")
parser.add_argument("--no-auto-lift", action="store_true", help="Disable minimum-jerk vertical lift after a successful pinch grasp")
parser.add_argument("--grasp-straighten", type=float, default=0.45, help="Blend grasp/lift wrist orientation toward calibrated straight pose")
parser.add_argument("--lift-height", type=float, default=0.0, help="Unused in grasp-only v20; retained for compatibility")
parser.add_argument("--lift-duration", type=float, default=0.85)
parser.add_argument("--record-demo", default="", help="Optional JSONL path for imitation-learning demonstrations")
parser.add_argument("--print-hz", type=float, default=2.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCylinder, FixedCuboid
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.types import ArticulationAction

from rby1_isaac_connected.common import (
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_JOINTS,
    USD_MAP,
    apply_articulation_root_to_best_candidate,
    candidate_root_paths,
    find_articulation_roots_under,
    resolve_joint_indices,
    resolve_optional_joint_indices,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
TASK_SCENE_INFO: dict[str, object] = {}

from rby1_taskspace_ik import (
    DlsConfig,
    MapperConfig,
    UdpPoseReceiver,
    WebcamTaskMapper,
    clip_norm,
    limit_quat_step,
    minimum_jerk,
    quat_error_rotvec,
    quat_slerp,
    quat_to_matrix,
    solve_dls_pose_step,
    solve_stacked_dls_step,
    apply_shoulder_reference_correction,
)


def joint_number(name: str) -> int | None:
    try:
        return int(name.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return None


def parse_joint_name_list(text: str) -> list[str]:
    return [item.strip().split("/")[-1] for item in text.split(",") if item.strip()]


def parse_float_triplet(text: str, label: str) -> np.ndarray:
    try:
        values = np.array([float(item.strip()) for item in text.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{label} must contain three comma-separated numbers: {text!r}") from exc
    if values.shape != (3,):
        raise ValueError(f"{label} must contain exactly three values: {text!r}")
    return values


def parse_optional_float_list(text: str, expected: int, label: str) -> np.ndarray | None:
    if not text.strip():
        return None
    try:
        values = np.array([float(item.strip()) for item in text.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{label} must contain comma-separated numbers: {text!r}") from exc
    if values.size != expected:
        raise ValueError(f"{label} expected {expected} values for the selected gripper joints, got {values.size}")
    return values


def natural_key(name: str):
    import re
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name.lower()))


def discover_gripper_names(dof_names: list[str], resolved: list[str], side: str) -> list[str]:
    output = [name.split("/")[-1] for name in resolved]
    for full_name in dof_names:
        base = full_name.split("/")[-1]
        lowered = base.lower()
        if side in lowered and any(token in lowered for token in ("gripper", "finger", "thumb", "index", "jaw")):
            output.append(base)
    return sorted(dict.fromkeys(output), key=natural_key)


def select_precision_joints(available: list[str], override: str, side: str) -> list[str]:
    available = sorted(dict.fromkeys(name.split("/")[-1] for name in available), key=natural_key)
    explicit = parse_joint_name_list(override)
    if explicit:
        missing = [name for name in explicit if name not in available]
        if missing:
            raise RuntimeError(f"{side} gripper override missing {missing}; available={available}")
        return explicit
    canonical = [f"gripper_finger_{side[0]}1", f"gripper_finger_{side[0]}2"]
    selected = [name for name in canonical if name in available]
    if selected:
        return selected
    named = [name for name in available if any(token in name.lower() for token in ("thumb", "index", "jaw"))]
    if named:
        return named[:2]
    return available[:2]


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def sanitize_limits(lower: np.ndarray, upper: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(lower, dtype=np.float64).reshape(-1).copy()
    upper = np.asarray(upper, dtype=np.float64).reshape(-1).copy()
    if lower.size != count or upper.size != count:
        raise ValueError(f"limit size mismatch lower={lower.size} upper={upper.size} count={count}")
    invalid = (~np.isfinite(lower)) | (~np.isfinite(upper)) | (np.abs(lower) > 100.0) | (np.abs(upper) > 100.0) | (upper < lower)
    lower[invalid] = -2.0 * math.pi
    upper[invalid] = 2.0 * math.pi
    return lower, upper


def read_limits(robot: SingleArticulation, count: int) -> tuple[np.ndarray, np.ndarray]:
    errors: list[str] = []
    try:
        props = robot.dof_properties
        names = getattr(getattr(props, "dtype", None), "names", None)
        if names and "lower" in names and "upper" in names:
            return sanitize_limits(props["lower"], props["upper"], count)
        errors.append(f"unexpected dof_properties fields {names}")
    except Exception as exc:
        errors.append(f"dof_properties={exc}")
    try:
        result = robot.get_articulation_controller().get_joint_limits()
        if isinstance(result, tuple):
            return sanitize_limits(result[0], result[1], count)
        limits = as_numpy(result)
        if limits.ndim == 3:
            limits = limits[0]
        return sanitize_limits(limits[:, 0], limits[:, 1], count)
    except Exception as exc:
        errors.append(f"controller limits={exc}")
    print("[WARN] DOF limits unavailable; broad fallback. " + " | ".join(errors))
    return np.full(count, -2.0 * math.pi), np.full(count, 2.0 * math.pi)


def load_world(usd_path: Path, scene_root: str) -> World:
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    usd_cfg = sim_utils.UsdFileCfg(
        usd_path=str(usd_path),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=not args_cli.enable_robot_gravity,
            linear_damping=0.05,
            angular_damping=0.05,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=not args_cli.floating_base,
            enabled_self_collisions=False,
            solver_position_iteration_count=24,
            solver_velocity_iteration_count=6,
        ),
    )
    usd_cfg.func(scene_root, usd_cfg)
    return world


def resolve_robot(scene_root: str, world: World) -> tuple[SingleArticulation, str]:
    roots = find_articulation_roots_under(scene_root)
    if not roots:
        patched = apply_articulation_root_to_best_candidate(scene_root)
        print("[INFO] Applied ArticulationRootAPI candidate:", patched)
    world.reset()
    roots = find_articulation_roots_under(scene_root)
    candidates = roots if roots else candidate_root_paths(scene_root)
    last_error = None
    for path in candidates:
        try:
            robot = SingleArticulation(prim_path=path, name="rby1_webcam_ik")
            robot.initialize()
            return robot, path
        except Exception as exc:
            last_error = exc
            print(f"[WARN] Articulation init failed at {path}: {exc}")
    raise RuntimeError(f"Could not initialize RB-Y1 at {candidates}; last={last_error}")


def _base_name(name: str) -> str:
    return str(name).rstrip("/").split("/")[-1]


def _side_name_score(name: str, side: str) -> float:
    """Semantic preference for a terminal body on one side of RB-Y1."""
    import re

    text = _base_name(name).lower()
    short = "l" if side == "left" else "r"
    score = 0.0
    if side in text:
        score += 220.0
    # Common imported-name conventions: arm_l_6, link_r_arm_7, gripper_l.
    if re.search(rf"(^|_){short}($|_)", text):
        score += 180.0
    if re.search(rf"(^|_){short}[0-9]+($|_)", text):
        score += 80.0
    if "end_effector" in text or text.startswith("ee_") or text.endswith("_ee"):
        score += 210.0
    if "tool" in text or "tcp" in text:
        score += 200.0
    if "gripper" in text:
        score += 170.0
    if "hand" in text:
        score += 155.0
    if "wrist" in text:
        score += 135.0
    if "arm" in text:
        score += 95.0
    if "finger" in text or "jaw" in text:
        score -= 85.0
    numbers = [int(value) for value in re.findall(r"[0-9]+", text)]
    if numbers:
        score += min(max(numbers), 20) * 2.0
    return score


def _jacobian_index_for_body(body_index: int, body_count: int, jacobian_body_count: int) -> int | None:
    # PhysX omits the root-body Jacobian for fixed-base articulations.
    if jacobian_body_count == body_count - 1:
        return None if body_index == 0 else body_index - 1
    if jacobian_body_count == body_count:
        return body_index
    return None


def select_arm_spatial_jacobian(
    jacobians: np.ndarray,
    jacobian_index: int,
    arm_joint_indices: np.ndarray,
) -> np.ndarray:
    """Return one body's spatial Jacobian in canonical shape ``(6, arm_dofs)``.

    PhysX normally returns ``(body, 6, dof)``.  Two problems are handled here:

    1. NumPy advanced indexing such as
       ``jacobians[body, :3, joint_indices]`` moves the indexed DOF axis to the
       front and can produce ``(7, 3)`` instead of ``(3, 7)``.
    2. Some wrappers expose a transposed per-body Jacobian ``(dof, 6)``.

    Always select the body first, then select joint columns from the resulting
    2-D matrix.  The returned matrix is guaranteed to have one column per
    requested arm joint.
    """
    body_jacobian = np.asarray(
        jacobians[int(jacobian_index)],
        dtype=np.float64,
    )
    joint_indices = np.asarray(
        arm_joint_indices,
        dtype=np.int64,
    ).reshape(-1)

    if body_jacobian.ndim != 2:
        raise RuntimeError(
            "Expected a 2-D body Jacobian, "
            f"got shape {body_jacobian.shape} at index {jacobian_index}"
        )
    if joint_indices.size == 0:
        raise RuntimeError("Arm joint index list is empty")

    maximum_joint_index = int(np.max(joint_indices))

    # Canonical PhysX layout: spatial rows × articulation DOF columns.
    if (
        body_jacobian.shape[0] >= 6
        and body_jacobian.shape[1] > maximum_joint_index
    ):
        selected = body_jacobian[:6, :][:, joint_indices]

    # Defensive support for a transposed wrapper layout: DOF rows × spatial cols.
    elif (
        body_jacobian.shape[1] >= 6
        and body_jacobian.shape[0] > maximum_joint_index
    ):
        selected = body_jacobian[:, :6][joint_indices, :].T

    else:
        raise RuntimeError(
            "Cannot select arm columns from body Jacobian: "
            f"body_shape={body_jacobian.shape}, "
            f"max_joint_index={maximum_joint_index}, "
            f"joint_indices={joint_indices.tolist()}"
        )

    expected_shape = (6, int(joint_indices.size))
    if selected.shape != expected_shape:
        raise RuntimeError(
            "Arm Jacobian shape normalization failed: "
            f"expected={expected_shape}, got={selected.shape}"
        )

    return np.ascontiguousarray(selected, dtype=np.float64)


def select_end_effector_body(
    body_names: list[str],
    jacobians: np.ndarray,
    arm_joint_indices: np.ndarray,
    requested: str,
    side: str,
) -> tuple[str, int]:
    """Select an actual articulation body and its PhysX Jacobian row.

    Selection is based primarily on kinematics: a valid endpoint must have
    non-zero Jacobian columns for the side's seven arm joints. Semantic body
    names only break ties between the final arm link, hand, and gripper links.
    """
    if not body_names:
        raise RuntimeError("Articulation returned no body_names; cannot resolve an IK endpoint.")

    bases = [_base_name(name) for name in body_names]
    requested_base = _base_name(requested)
    exact_index = None
    if requested and requested.lower() != "auto":
        for index, base in enumerate(bases):
            if base == requested_base or base.lower() == requested_base.lower():
                exact_index = index
                break

    ranked: list[tuple[float, int, float, str, int]] = []
    for body_index, full_name in enumerate(body_names):
        jacobian_index = _jacobian_index_for_body(body_index, len(body_names), jacobians.shape[0])
        if jacobian_index is None or jacobian_index < 0 or jacobian_index >= jacobians.shape[0]:
            continue
        side_jacobian = select_arm_spatial_jacobian(
            jacobians,
            jacobian_index,
            arm_joint_indices,
        )
        column_norms = np.linalg.norm(side_jacobian, axis=0)
        active_columns = int(np.count_nonzero(column_norms > 1.0e-7))
        jacobian_norm = float(np.linalg.norm(side_jacobian))
        semantic = _side_name_score(full_name, side)
        exact_bonus = 100000.0 if body_index == exact_index else 0.0
        # Kinematic chain coverage dominates naming. The terminal link normally
        # depends on all seven arm joints.
        score = exact_bonus + active_columns * 10000.0 + semantic + min(jacobian_norm, 100.0)
        ranked.append((score, active_columns, jacobian_norm, full_name, jacobian_index))

    ranked.sort(key=lambda item: item[0], reverse=True)
    viable = [item for item in ranked if item[1] >= max(5, len(arm_joint_indices) - 1)]
    if not viable:
        summary = [f"{_base_name(item[3])}:cols={item[1]}" for item in ranked[:12]]
        raise RuntimeError(
            f"No {side} endpoint body depends on the seven {side} arm joints. "
            f"Top Jacobian candidates: {summary}; all bodies={bases}"
        )

    chosen = viable[0]
    print(f"[INFO] {side} EE auto-ranking (top 8):")
    for item in viable[:8]:
        print(
            f"       {_base_name(item[3]):32s} active_arm_cols={item[1]}/7 "
            f"Jnorm={item[2]:.4f} semantic={_side_name_score(item[3], side):.1f}"
        )
    selected_name = _base_name(chosen[3])
    print(f"[INFO] {side} IK endpoint selected: {selected_name} (Jacobian row {chosen[4]})")
    return selected_name, chosen[4]


def select_elbow_body(
    body_names: list[str],
    jacobians: np.ndarray,
    arm_joint_indices: np.ndarray,
    side: str,
) -> tuple[str, int] | None:
    """Pick an intermediate arm link suitable for elbow/table clearance control."""
    ranked: list[tuple[float, str, int, int, int]] = []
    for body_index, full_name in enumerate(body_names):
        jacobian_index = _jacobian_index_for_body(body_index, len(body_names), jacobians.shape[0])
        if jacobian_index is None or not (0 <= jacobian_index < jacobians.shape[0]):
            continue
        J = select_arm_spatial_jacobian(
            jacobians,
            jacobian_index,
            arm_joint_indices,
        )
        norms = np.linalg.norm(J, axis=0)
        proximal = int(np.count_nonzero(norms[:4] > 1.0e-7))
        wrist = int(np.count_nonzero(norms[4:] > 1.0e-7))
        total = proximal + wrist
        name = _base_name(full_name)
        semantic = _side_name_score(full_name, side)
        lowered = name.lower()
        if "elbow" in lowered:
            semantic += 260.0
        if "arm_3" in lowered or "arm3" in lowered or lowered.endswith("_3"):
            semantic += 180.0
        if "gripper" in lowered or "finger" in lowered or "wrist" in lowered:
            semantic -= 220.0
        score = proximal * 2000.0 - wrist * 900.0 - abs(total - 4) * 350.0 + semantic
        ranked.append((score, full_name, jacobian_index, proximal, wrist))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked or ranked[0][3] < 3:
        return None
    chosen = ranked[0]
    print(
        f"[INFO] {side} elbow-clearance body selected: {_base_name(chosen[1])} "
        f"(Jacobian row {chosen[2]}, proximal={chosen[3]}, wrist={chosen[4]})"
    )
    return _base_name(chosen[1]), chosen[2]


def find_body_prim_path(scene_root: str, articulation_root: str, body_name: str) -> str:
    """Find a rigid-body prim across the RB-Y1 model, not below root_joint only."""
    stage = omni.usd.get_context().get_stage()
    requested = str(body_name)
    if requested.startswith("/"):
        prim = stage.GetPrimAtPath(requested)
        if prim.IsValid():
            return requested

    base = _base_name(requested)
    model_scope = articulation_root.rsplit("/", 1)[0]
    candidates: list[tuple[float, str]] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(scene_root.rstrip("/") + "/"):
            continue
        if prim.GetName() != base and path != requested:
            continue
        score = 0.0
        if path.startswith(model_scope.rstrip("/") + "/"):
            score += 100.0
        try:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                score += 60.0
        except Exception:
            pass
        # Prefer link Xforms over visual/collision descendants with duplicate names.
        lowered = path.lower()
        if "/visual" in lowered or "/collision" in lowered or "/mesh" in lowered:
            score -= 80.0
        score -= 0.01 * path.count("/")
        candidates.append((score, path))

    if not candidates:
        # Include useful model-level names in the diagnostic, regardless of the
        # articulation-root prim chosen by the importer.
        diagnostic = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith(scene_root.rstrip("/") + "/") and any(
                token in prim.GetName().lower() for token in ("arm", "wrist", "hand", "gripper", "finger", "ee", "tool")
            ):
                diagnostic.append(path)
        raise RuntimeError(
            f"Articulation body '{body_name}' exists in body_names but no matching USD prim was found under {scene_root}. "
            f"Candidate model prims: {diagnostic[:80]}"
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def initialize_ee(scene_root: str, root_path: str, body_name: str, side: str) -> SingleRigidPrim:
    candidate = find_body_prim_path(scene_root, root_path, body_name)
    print(f"[INFO] {side} end-effector rigid prim: {candidate}")
    ee = SingleRigidPrim(prim_path=candidate, name=f"rby1_{side}_ee")
    ee.initialize()
    return ee

def rigid_pose(prim: SingleRigidPrim) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(prim, "get_world_pose"):
        position, orientation = prim.get_world_pose()
        return as_numpy(position).reshape(3).astype(np.float64), as_numpy(orientation).reshape(4).astype(np.float64)
    positions, orientations = prim._rigid_prim_view.get_world_poses()
    return as_numpy(positions)[0].astype(np.float64), as_numpy(orientations)[0].astype(np.float64)


def get_jacobian_tensor(robot: SingleArticulation) -> np.ndarray:
    errors = []
    for source in (robot, getattr(robot, "_articulation_view", None)):
        if source is None or not hasattr(source, "get_jacobians"):
            continue
        try:
            jac = as_numpy(source.get_jacobians()).astype(np.float64)
            if jac.ndim == 4:
                jac = jac[0]
            if jac.ndim != 3 or jac.shape[1] != 6:
                raise ValueError(f"unexpected Jacobian shape {jac.shape}")
            return jac
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("PhysX Jacobian unavailable: " + " | ".join(errors))



def _disable_collision_subtree(prim_path: str) -> None:
    """Remove collision APIs below a visual-only demo object."""
    stage = omni.usd.get_context().get_stage()
    prefix = prim_path.rstrip("/")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path == prefix or path.startswith(prefix + "/"):
            try:
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    prim.RemoveAPI(UsdPhysics.CollisionAPI)
            except Exception:
                pass


def _set_display_color(geom, color: np.ndarray) -> None:
    color = np.asarray(color, dtype=np.float64).reshape(3)
    geom.CreateDisplayColorAttr(
        [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
    )


def _define_visual_cube(
    stage,
    prim_path: str,
    *,
    center: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    color: np.ndarray,
) -> None:
    """Create a visual-only bar using the standard UsdGeom.Cube schema."""
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    _set_display_color(cube, color)
    cube.AddTranslateOp().Set(
        Gf.Vec3d(
            float(center[0]),
            float(center[1]),
            float(center[2]),
        )
    )
    cube.AddScaleOp().Set(
        Gf.Vec3f(
            float(dimensions[0]),
            float(dimensions[1]),
            float(dimensions[2]),
        )
    )


def _add_cup_visual_details(
    cup_prim_path: str,
    *,
    side: str,
    radius: float,
    height: float,
    wall: float,
    handle_radius: float,
    handle_thickness: float,
    body_color: np.ndarray,
) -> None:
    """Add a cup opening, rim, and U-shaped handle using Isaac Sim 5.1 schemas.

    The DynamicCylinder at ``cup_prim_path`` is the only physical rigid body.
    All children created here are visual-only and follow the cup root.
    """
    stage = omni.usd.get_context().get_stage()

    radius = float(max(radius, 0.012))
    height = float(max(height, 0.025))
    wall = float(np.clip(wall, 0.0015, 0.25 * radius))
    handle_radius = float(max(handle_radius, 0.010))
    handle_thickness = float(
        np.clip(handle_thickness, 0.002, 0.40 * handle_radius)
    )

    visuals_path = f"{cup_prim_path}/CupVisualDetails"
    UsdGeom.Xform.Define(stage, visuals_path)

    # Thin body-colored disk at the top.
    rim_disk = UsdGeom.Cylinder.Define(
        stage,
        f"{visuals_path}/RimDisk",
    )
    rim_disk.CreateAxisAttr("Z")
    rim_disk.CreateRadiusAttr(radius + 0.20 * wall)
    rim_disk.CreateHeightAttr(max(0.0020, 0.70 * wall))
    _set_display_color(rim_disk, body_color)
    rim_disk.AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, 0.5 * height + 0.0010)
    )

    # Smaller dark disk above it leaves a visible circular rim and gives the
    # impression of a hollow cup.
    opening = UsdGeom.Cylinder.Define(
        stage,
        f"{visuals_path}/Opening",
    )
    opening.CreateAxisAttr("Z")
    opening.CreateRadiusAttr(max(radius - 1.35 * wall, 0.006))
    opening.CreateHeightAttr(max(0.0012, 0.30 * wall))
    _set_display_color(
        opening,
        np.array([0.025, 0.025, 0.035], dtype=np.float64),
    )
    opening.AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, 0.5 * height + 0.0030)
    )

    # U-shaped handle built from three standard cubes. The handle lies in the
    # Y-Z plane and is placed on the outer side of the selected arm.
    outward = 1.0 if side == "left" else -1.0
    handle_half_height = min(handle_radius, 0.34 * height)
    bridge_length = max(1.25 * handle_radius, 0.018)

    bridge_center_y = outward * (
        radius + 0.50 * bridge_length
    )
    outer_y = outward * (
        radius + bridge_length
    )

    bar_x = max(2.0 * handle_thickness, 0.006)
    bar_y = max(handle_thickness, 0.004)
    bar_z = max(handle_thickness, 0.004)

    _define_visual_cube(
        stage,
        f"{visuals_path}/HandleTop",
        center=(
            0.0,
            bridge_center_y,
            handle_half_height,
        ),
        dimensions=(
            bar_x,
            bridge_length,
            bar_z,
        ),
        color=body_color,
    )
    _define_visual_cube(
        stage,
        f"{visuals_path}/HandleBottom",
        center=(
            0.0,
            bridge_center_y,
            -handle_half_height,
        ),
        dimensions=(
            bar_x,
            bridge_length,
            bar_z,
        ),
        color=body_color,
    )
    _define_visual_cube(
        stage,
        f"{visuals_path}/HandleOuter",
        center=(0.0, outer_y, 0.0),
        dimensions=(
            bar_x,
            bar_y,
            2.0 * handle_half_height + bar_z,
        ),
        color=body_color,
    )

    # Decorative children must not create additional contact geometry.
    _disable_collision_subtree(visuals_path)




def _bbox_values(
    prim,
    bbox_cache: UsdGeom.BBoxCache,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return aligned world-space bbox min/max values for a prim."""
    try:
        bbox = bbox_cache.ComputeWorldBound(prim)
        box = bbox.ComputeAlignedBox()
        minimum = box.GetMin()
        maximum = box.GetMax()
        min_values = np.array(
            [float(minimum[0]), float(minimum[1]), float(minimum[2])],
            dtype=np.float64,
        )
        max_values = np.array(
            [float(maximum[0]), float(maximum[1]), float(maximum[2])],
            dtype=np.float64,
        )
    except Exception:
        return None

    if not np.all(np.isfinite(min_values)) or not np.all(np.isfinite(max_values)):
        return None
    if np.any(max_values <= min_values):
        return None
    return min_values, max_values


def _rgb_array(value) -> np.ndarray | None:
    """Convert a USD color value to a finite normalized RGB vector."""
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if array.size < 3:
        return None
    rgb = array[:3].copy()
    if not np.all(np.isfinite(rgb)):
        return None
    if float(np.max(rgb)) > 1.5:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _prim_color_samples(prim) -> list[np.ndarray]:
    """Read displayColor and common material color inputs safely."""
    colors: list[np.ndarray] = []

    try:
        if prim.IsA(UsdGeom.Gprim):
            display_values = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
            if display_values:
                for value in display_values:
                    rgb = _rgb_array(value)
                    if rgb is not None:
                        colors.append(rgb)
    except Exception:
        pass

    try:
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material and material.GetPrim().IsValid():
            source = material.ComputeSurfaceSource()
            shader = source[0] if source else None
            if shader:
                for shader_input in shader.GetInputs():
                    name = shader_input.GetBaseName().lower()
                    if not any(
                        token in name
                        for token in (
                            "base_color",
                            "basecolor",
                            "diffuse",
                            "albedo",
                            "color",
                        )
                    ):
                        continue
                    rgb = _rgb_array(shader_input.Get())
                    if rgb is not None:
                        colors.append(rgb)
    except Exception:
        pass

    return colors


def _yellow_color_score(rgb: np.ndarray) -> float:
    """Return a [0,1]-ish score for saturated yellow/gold colors."""
    r, g, b = [float(value) for value in rgb[:3]]
    yellow_strength = min(r, g) - b
    brightness = 0.5 * (r + g)
    red_green_balance = 1.0 - min(abs(r - g), 1.0)
    score = (
        1.6 * max(yellow_strength - 0.08, 0.0)
        + 0.35 * max(brightness - 0.35, 0.0)
        + 0.15 * red_green_balance
    )
    return float(max(score, 0.0))


def _compact_anchor_ancestor(
    prim,
    bbox_cache: UsdGeom.BBoxCache,
):
    """Ascend while the complete object remains cup-sized."""
    current = prim
    best = prim
    while current and current.IsValid():
        values = _bbox_values(current, bbox_cache)
        if values is None:
            break
        minimum, maximum = values
        dimensions = maximum - minimum

        compact = (
            0.018 <= float(max(dimensions[0], dimensions[1])) <= 0.24
            and 0.035 <= float(dimensions[2]) <= 0.28
        )
        if not compact:
            break

        best = current
        parent = current.GetParent()
        if not parent or parent.IsPseudoRoot():
            break
        current = parent

    return best


def _reference_cup_candidates(stage):
    """Rank compact cup-like prims using names, dimensions, and yellow color."""
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
    )
    root = stage.GetPrimAtPath(args_cli.existing_table_prim)
    if not root or not root.IsValid():
        root = stage.GetPseudoRoot()

    excluded_tokens = (
        "tablecup",
        "cupsupport",
        "webcampicktable",
        "rby1",
        "robot",
        "ground",
        "light",
    )
    cup_name_tokens = (
        "cup",
        "mug",
        "coffee",
        "glass",
        "tumbler",
    )

    ranked: dict[str, dict] = {}

    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Gprim):
            continue

        path = str(prim.GetPath())
        lowered = path.lower()
        if any(token in lowered for token in excluded_tokens):
            continue

        colors = _prim_color_samples(prim)
        yellow = max(
            (_yellow_color_score(color) for color in colors),
            default=0.0,
        )
        name_hit = any(token in lowered for token in cup_name_tokens)

        # Do not expand every scene primitive into a candidate. A prim must
        # either look yellow or have a cup-like name.
        if yellow < 0.10 and not name_hit:
            continue

        anchor = _compact_anchor_ancestor(prim, bbox_cache)
        values = _bbox_values(anchor, bbox_cache)
        if values is None:
            continue
        minimum, maximum = values
        dimensions = maximum - minimum
        center = 0.5 * (minimum + maximum)

        if not (
            args_cli.reference_cup_min_z
            <= float(center[2])
            <= args_cli.reference_cup_max_z
        ):
            continue
        if not (
            0.018 <= float(max(dimensions[0], dimensions[1])) <= 0.24
            and 0.035 <= float(dimensions[2]) <= 0.28
        ):
            continue

        anchor_path = str(anchor.GetPath())
        anchor_lower = anchor_path.lower()
        anchor_name_hit = any(
            token in anchor_lower for token in cup_name_tokens
        )

        upright_bonus = 1.0 if dimensions[2] >= 0.70 * max(
            dimensions[0],
            dimensions[1],
        ) else 0.0
        compact_bonus = 1.0 - min(
            abs(float(dimensions[2]) - 0.11) / 0.18,
            1.0,
        )
        score = (
            7.0 * float(anchor_name_hit or name_hit)
            + 10.0 * yellow
            + 1.5 * upright_bonus
            + compact_bonus
        )

        previous = ranked.get(anchor_path)
        if previous is None or score > previous["score"]:
            ranked[anchor_path] = {
                "score": float(score),
                "path": anchor_path,
                "center": center,
                "minimum": minimum,
                "maximum": maximum,
                "dimensions": dimensions,
                "yellow": float(yellow),
            }

    return sorted(
        ranked.values(),
        key=lambda item: item["score"],
        reverse=True,
    )


def _find_existing_reference_cup(stage):
    """Resolve an explicitly named cup or automatically detect the yellow cup."""
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
    )

    if args_cli.existing_cup_prim != "auto":
        prim = stage.GetPrimAtPath(args_cli.existing_cup_prim)
        if not prim or not prim.IsValid():
            print(
                f"[WARN] Existing cup prim not found: "
                f"{args_cli.existing_cup_prim}"
            )
            return None
        values = _bbox_values(prim, bbox_cache)
        if values is None:
            print(
                f"[WARN] Existing cup prim has no usable world bbox: "
                f"{args_cli.existing_cup_prim}"
            )
            return None
        minimum, maximum = values
        return {
            "score": 999.0,
            "path": str(prim.GetPath()),
            "center": 0.5 * (minimum + maximum),
            "minimum": minimum,
            "maximum": maximum,
            "dimensions": maximum - minimum,
            "yellow": 0.0,
        }

    candidates = _reference_cup_candidates(stage)
    if not candidates:
        print(
            "[WARN] No compact yellow/cup-like reference object was found; "
            "falling back to requested coordinates."
        )
        return None

    print("[INFO] Existing cup auto-detection candidates:")
    for item in candidates[:8]:
        print(
            f"       score={item['score']:.2f} "
            f"yellow={item['yellow']:.2f} "
            f"dims={np.round(item['dimensions'], 3).tolist()} "
            f"path={item['path']}"
        )

    chosen = candidates[0]
    print(
        f"[INFO] Existing yellow cup selected: {chosen['path']} "
        f"center={np.round(chosen['center'], 3).tolist()}"
    )
    return chosen


def _robot_scene_center_xy(
    stage,
    fallback_xy: np.ndarray,
) -> np.ndarray:
    """Estimate the robot position for 'toward-robot' cup placement."""
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
    )

    for path in (
        "/World/RobotScene",
        "/World/RobotScene/RBY1_A_v1_0",
    ):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        values = _bbox_values(prim, bbox_cache)
        if values is None:
            continue
        minimum, maximum = values
        dimensions = maximum - minimum
        if float(np.max(dimensions[:2])) > 5.0:
            continue
        return 0.5 * (minimum[:2] + maximum[:2])

    return np.asarray(fallback_xy, dtype=np.float64).reshape(2)


def _surface_at_xy_near_height(
    candidates,
    xy: np.ndarray,
    target_z: float,
    tolerance: float = 0.075,
) -> tuple[float, str] | None:
    """Find a horizontal surface at exact XY close to the reference table Z."""
    best = None
    x, y = float(xy[0]), float(xy[1])

    for path, vertices, polygons, minimum, maximum in candidates:
        if (
            x < minimum[0] - 1.0e-4
            or x > maximum[0] + 1.0e-4
            or y < minimum[1] - 1.0e-4
            or y > maximum[1] + 1.0e-4
        ):
            continue

        z = _vertical_surface_hit(
            x,
            y,
            vertices,
            polygons,
            min_z=float(target_z - tolerance),
            max_z=float(target_z + tolerance),
            min_normal_z=float(args_cli.table_surface_normal_z),
        )
        if z is None:
            continue

        score = abs(float(z) - float(target_z))
        if best is None or score < best[0]:
            best = (score, float(z), path)

    if best is None:
        return None
    return best[1], best[2]


def _direction_vector(
    reference_xy: np.ndarray,
    robot_xy: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "x+":
        return np.array([1.0, 0.0], dtype=np.float64)
    if mode == "x-":
        return np.array([-1.0, 0.0], dtype=np.float64)
    if mode == "y+":
        return np.array([0.0, 1.0], dtype=np.float64)
    if mode == "y-":
        return np.array([0.0, -1.0], dtype=np.float64)

    toward = np.asarray(robot_xy, dtype=np.float64) - np.asarray(
        reference_xy,
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(toward))
    if norm < 1.0e-6:
        toward = np.array([0.0, -1.0], dtype=np.float64)
    else:
        toward = toward / norm

    return -toward if mode == "away-from-robot" else toward


def _placement_next_to_reference_cup(
    stage,
    reference,
    generated_radius: float,
) -> tuple[np.ndarray, float, str] | None:
    """Place the new cup beside the yellow cup on the same horizontal surface."""
    reference_xy = np.asarray(reference["center"][:2], dtype=np.float64)
    reference_bottom_z = float(reference["minimum"][2])
    reference_dimensions = np.asarray(
        reference["dimensions"],
        dtype=np.float64,
    )

    robot_xy = _robot_scene_center_xy(stage, reference_xy)
    preferred = _direction_vector(
        reference_xy,
        robot_xy,
        args_cli.cup_neighbor_direction,
    )
    perpendicular = np.array(
        [-preferred[1], preferred[0]],
        dtype=np.float64,
    )

    directions = [
        preferred,
        perpendicular,
        -perpendicular,
        -preferred,
    ]

    reference_radius = 0.5 * float(
        max(reference_dimensions[0], reference_dimensions[1])
    )
    separation = (
        reference_radius
        + float(generated_radius)
        + max(float(args_cli.cup_neighbor_gap), 0.005)
    )

    table_candidates = _table_mesh_candidates(
        stage,
        args_cli.existing_table_prim,
    )
    manual_offset = np.array(
        [
            args_cli.cup_neighbor_offset_x,
            args_cli.cup_neighbor_offset_y,
        ],
        dtype=np.float64,
    )

    for direction in directions:
        candidate_xy = (
            reference_xy
            + separation * direction
            + manual_offset
        )
        hit = _surface_at_xy_near_height(
            table_candidates,
            candidate_xy,
            reference_bottom_z,
        )
        if hit is None:
            continue
        surface_z, surface_path = hit
        return candidate_xy, surface_z, surface_path

    # The reference cup bottom itself is still the best estimate of the tabletop
    # if the imported environment mesh cannot be intersected.
    fallback_xy = (
        reference_xy
        + separation * preferred
        + manual_offset
    )
    return (
        fallback_xy,
        reference_bottom_z,
        f"reference-cup bottom ({reference['path']})",
    )


def _world_mesh_arrays(prim) -> tuple[np.ndarray, list[list[int]]] | None:
    """Return world-space vertices and polygon index lists for one UsdGeom.Mesh."""
    if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        return None

    mesh = UsdGeom.Mesh(prim)
    points_attr = mesh.GetPointsAttr()
    counts_attr = mesh.GetFaceVertexCountsAttr()
    indices_attr = mesh.GetFaceVertexIndicesAttr()

    points = points_attr.Get()
    counts = counts_attr.Get()
    indices = indices_attr.Get()
    if not points or not counts or not indices:
        return None

    xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    world_points = np.empty((len(points), 3), dtype=np.float64)
    for i, point in enumerate(points):
        world = xform.Transform(
            Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
        )
        world_points[i] = (float(world[0]), float(world[1]), float(world[2]))

    polygons: list[list[int]] = []
    cursor = 0
    for count in counts:
        count_i = int(count)
        if count_i >= 3:
            polygons.append(
                [int(value) for value in indices[cursor : cursor + count_i]]
            )
        cursor += count_i

    return world_points, polygons


def _vertical_surface_hit(
    x: float,
    y: float,
    vertices: np.ndarray,
    polygons: list[list[int]],
    *,
    min_z: float,
    max_z: float,
    min_normal_z: float,
) -> float | None:
    """Return the highest horizontal-ish mesh intersection at world XY."""
    best_z: float | None = None
    point_xy = np.array([float(x), float(y)], dtype=np.float64)
    eps = 1.0e-9
    bary_tol = 2.0e-5

    for polygon in polygons:
        first = polygon[0]
        for local_i in range(1, len(polygon) - 1):
            ia, ib, ic = first, polygon[local_i], polygon[local_i + 1]
            a = vertices[ia]
            b = vertices[ib]
            c = vertices[ic]

            normal = np.cross(b - a, c - a)
            norm = float(np.linalg.norm(normal))
            if norm <= eps:
                continue
            if abs(float(normal[2])) / norm < min_normal_z:
                continue

            # Fast XY bounds test.
            min_xy = np.minimum(np.minimum(a[:2], b[:2]), c[:2])
            max_xy = np.maximum(np.maximum(a[:2], b[:2]), c[:2])
            if np.any(point_xy < min_xy - bary_tol) or np.any(
                point_xy > max_xy + bary_tol
            ):
                continue

            v0 = b[:2] - a[:2]
            v1 = c[:2] - a[:2]
            v2 = point_xy - a[:2]
            denom = float(v0[0] * v1[1] - v1[0] * v0[1])
            if abs(denom) <= eps:
                continue

            u = float((v2[0] * v1[1] - v1[0] * v2[1]) / denom)
            v = float((v0[0] * v2[1] - v2[0] * v0[1]) / denom)
            w = 1.0 - u - v
            if (
                u < -bary_tol
                or v < -bary_tol
                or w < -bary_tol
            ):
                continue

            z = float(w * a[2] + u * b[2] + v * c[2])
            if z < min_z or z > max_z:
                continue
            if best_z is None or z > best_z:
                best_z = z

    return best_z


def _table_mesh_candidates(stage, subtree_path: str):
    """Collect non-robot meshes that may contain the existing tabletop."""
    root = stage.GetPrimAtPath(subtree_path)
    roots = [root] if root and root.IsValid() else [stage.GetPseudoRoot()]
    excluded_tokens = (
        "rby1",
        "robot",
        "groundplane",
        "ground_plane",
        "tablecup",
        "cupsupport",
        "webcampicktable",
        "light",
    )

    candidates = []
    seen_paths: set[str] = set()
    for search_root in roots:
        for prim in Usd.PrimRange(search_root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            path = str(prim.GetPath())
            lowered = path.lower()
            if any(token in lowered for token in excluded_tokens):
                continue
            if path in seen_paths:
                continue
            seen_paths.add(path)
            arrays = _world_mesh_arrays(prim)
            if arrays is None:
                continue
            vertices, polygons = arrays
            if vertices.size == 0 or not polygons:
                continue
            min_values = vertices.min(axis=0)
            max_values = vertices.max(axis=0)
            xy_size = max_values[:2] - min_values[:2]
            if float(xy_size[0] * xy_size[1]) < 0.02:
                continue
            candidates.append(
                (
                    path,
                    vertices,
                    polygons,
                    min_values,
                    max_values,
                )
            )
    return candidates


def _search_existing_tabletop(
    stage,
    requested_xy: np.ndarray,
) -> tuple[np.ndarray, float, str] | None:
    """Find the nearest valid horizontal mesh surface to the requested cup XY."""
    if args_cli.disable_table_surface_detect or args_cli.add_demo_table:
        return None

    candidates = _table_mesh_candidates(
        stage,
        args_cli.existing_table_prim,
    )
    if not candidates:
        print(
            f"[WARN] No candidate meshes found below "
            f"{args_cli.existing_table_prim}; using configured table coordinates."
        )
        return None

    radius = max(float(args_cli.table_surface_search_radius), 0.0)
    step = max(float(args_cli.table_surface_search_step), 0.01)
    offsets: list[tuple[float, float]] = [(0.0, 0.0)]

    rings = int(math.ceil(radius / step))
    for ring in range(1, rings + 1):
        r = min(ring * step, radius)
        samples = max(8, 8 * ring)
        for sample in range(samples):
            angle = 2.0 * math.pi * sample / samples
            offsets.append(
                (
                    r * math.cos(angle),
                    r * math.sin(angle),
                )
            )

    best = None
    for dx, dy in offsets:
        query_x = float(requested_xy[0] + dx)
        query_y = float(requested_xy[1] + dy)
        distance = math.hypot(dx, dy)

        for path, vertices, polygons, min_values, max_values in candidates:
            if (
                query_x < min_values[0] - 1.0e-4
                or query_x > max_values[0] + 1.0e-4
                or query_y < min_values[1] - 1.0e-4
                or query_y > max_values[1] + 1.0e-4
            ):
                continue

            z = _vertical_surface_hit(
                query_x,
                query_y,
                vertices,
                polygons,
                min_z=float(args_cli.table_surface_min_z),
                max_z=float(args_cli.table_surface_max_z),
                min_normal_z=float(args_cli.table_surface_normal_z),
            )
            if z is None:
                continue

            # Prefer the nearest point. At equal distance prefer a surface close
            # to the configured table height instead of a taller object.
            score = (
                distance,
                abs(z - float(args_cli.table_top_z)),
                -z,
            )
            if best is None or score < best[0]:
                best = (
                    score,
                    np.array([query_x, query_y], dtype=np.float64),
                    float(z),
                    path,
                )

        # Once an exact or very near surface has been found, no need to search
        # farther rings.
        if best is not None and distance <= step + 1.0e-9:
            break

    if best is None:
        print(
            f"[WARN] No horizontal tabletop hit near "
            f"({requested_xy[0]:+.3f}, {requested_xy[1]:+.3f}); "
            "using configured table coordinates."
        )
        print("[INFO] Candidate mesh paths:")
        for path, *_ in candidates[:12]:
            print(f"       {path}")
        return None

    _, xy, z, path = best
    return xy, z, path



def _set_subtree_visible(prim, visible: bool) -> None:
    """Set visibility at an object root; parent visibility propagates to children."""
    if not prim or not prim.IsValid():
        return
    try:
        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
    except Exception as exc:
        print(f"[WARN] Could not change visibility for {prim.GetPath()}: {exc}")


def _basket_anchor_ancestor(
    prim,
    bbox_cache: UsdGeom.BBoxCache,
):
    """Ascend to a container-sized ancestor without swallowing the whole table."""
    current = prim
    best = prim
    while current and current.IsValid():
        values = _bbox_values(current, bbox_cache)
        if values is None:
            break
        minimum, maximum = values
        dimensions = maximum - minimum
        container_sized = (
            0.14 <= float(max(dimensions[0], dimensions[1])) <= 1.20
            and 0.035 <= float(dimensions[2]) <= 0.55
        )
        if not container_sized:
            break
        best = current
        parent = current.GetParent()
        if not parent or parent.IsPseudoRoot():
            break
        current = parent
    return best


def _basket_candidates(stage):
    """Rank basket/bin/crate/tray-like objects using names and world bounds."""
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
    )
    root = stage.GetPrimAtPath(args_cli.existing_table_prim)
    if not root or not root.IsValid():
        root = stage.GetPseudoRoot()

    strong_tokens = ("basket", "crate", "bin", "tray", "container")
    weak_tokens = ("box", "case", "holder")
    excluded_tokens = (
        "rby1",
        "robot",
        "ground",
        "tablecup",
        "yellowcupproxy",
        "cupsupport",
        "webcampicktable",
        "light",
    )

    ranked: dict[str, dict] = {}

    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        lowered = path.lower()
        if any(token in lowered for token in excluded_tokens):
            continue

        strong = sum(token in lowered for token in strong_tokens)
        weak = sum(token in lowered for token in weak_tokens)

        # Geometry fallback is allowed, but named containers dominate.
        if strong == 0 and weak == 0 and not prim.IsA(UsdGeom.Mesh):
            continue

        anchor = _basket_anchor_ancestor(prim, bbox_cache)
        values = _bbox_values(anchor, bbox_cache)
        if values is None:
            continue
        minimum, maximum = values
        dimensions = maximum - minimum
        center = 0.5 * (minimum + maximum)

        horizontal_max = float(max(dimensions[0], dimensions[1]))
        horizontal_min = float(min(dimensions[0], dimensions[1]))
        height = float(dimensions[2])
        if not (
            0.14 <= horizontal_max <= 1.20
            and 0.10 <= horizontal_min <= 1.00
            and 0.035 <= height <= 0.55
            and args_cli.basket_min_center_z
            <= float(center[2])
            <= args_cli.basket_max_center_z
        ):
            continue

        anchor_path = str(anchor.GetPath())
        anchor_lower = anchor_path.lower()
        strong = max(strong, sum(token in anchor_lower for token in strong_tokens))
        weak = max(weak, sum(token in anchor_lower for token in weak_tokens))

        flat_container_bonus = 1.0 - min(
            abs(height / max(horizontal_max, 1.0e-6) - 0.28),
            1.0,
        )
        area_bonus = min(
            float(dimensions[0] * dimensions[1]) / 0.20,
            2.0,
        )
        score = (
            120.0 * strong
            + 25.0 * weak
            + 8.0 * flat_container_bonus
            + 4.0 * area_bonus
        )

        previous = ranked.get(anchor_path)
        if previous is None or score > previous["score"]:
            ranked[anchor_path] = {
                "score": float(score),
                "path": anchor_path,
                "center": center,
                "minimum": minimum,
                "maximum": maximum,
                "dimensions": dimensions,
            }

    return sorted(
        ranked.values(),
        key=lambda item: item["score"],
        reverse=True,
    )


def _find_existing_basket(stage):
    """Resolve an explicit basket prim or detect a basket-like scene object."""
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
    )

    if args_cli.basket_prim != "auto":
        prim = stage.GetPrimAtPath(args_cli.basket_prim)
        if not prim or not prim.IsValid():
            raise RuntimeError(
                f"Basket prim does not exist: {args_cli.basket_prim}"
            )
        values = _bbox_values(prim, bbox_cache)
        if values is None:
            raise RuntimeError(
                f"Basket prim has no usable world bounds: {args_cli.basket_prim}"
            )
        minimum, maximum = values
        return {
            "score": 9999.0,
            "path": str(prim.GetPath()),
            "center": 0.5 * (minimum + maximum),
            "minimum": minimum,
            "maximum": maximum,
            "dimensions": maximum - minimum,
        }

    candidates = _basket_candidates(stage)
    if not candidates:
        raise RuntimeError(
            "Could not auto-detect the existing basket. "
            "Run once with the basket prim path using --basket-prim /World/..."
        )

    print("[INFO] Basket auto-detection candidates:")
    for item in candidates[:10]:
        print(
            f"       score={item['score']:.1f} "
            f"dims={np.round(item['dimensions'], 3).tolist()} "
            f"path={item['path']}"
        )

    chosen = candidates[0]
    print(
        f"[INFO] Existing basket selected: {chosen['path']} "
        f"center={np.round(chosen['center'], 3).tolist()} "
        f"dims={np.round(chosen['dimensions'], 3).tolist()}"
    )
    return chosen


def _basket_task_points(
    basket,
    cup_height: float,
) -> dict[str, np.ndarray]:
    """Compute cup-center targets above and inside the basket."""
    minimum = np.asarray(basket["minimum"], dtype=np.float64)
    maximum = np.asarray(basket["maximum"], dtype=np.float64)
    center = np.asarray(basket["center"], dtype=np.float64).copy()

    center[0] += float(args_cli.basket_center_offset_x)
    center[1] += float(args_cli.basket_center_offset_y)

    # Cup center above the rim for collision-safe horizontal transfer.
    hover_object_center = np.array(
        [
            center[0],
            center[1],
            maximum[2]
            + 0.5 * cup_height
            + max(args_cli.basket_transfer_clearance, 0.03),
        ],
        dtype=np.float64,
    )

    # Put the cup bottom just above the basket bottom.  Clamp so the target
    # remains inside the rim even for a shallow basket.
    desired_drop_z = (
        minimum[2]
        + max(args_cli.basket_drop_clearance, 0.004)
        + 0.5 * cup_height
    )
    inside_limit = maximum[2] - 0.15 * cup_height
    drop_center_z = min(desired_drop_z, inside_limit)
    drop_center_z = max(
        drop_center_z,
        minimum[2] + 0.40 * cup_height,
    )

    drop_object_center = np.array(
        [center[0], center[1], drop_center_z],
        dtype=np.float64,
    )

    return {
        "hover_object_center": hover_object_center,
        "drop_object_center": drop_object_center,
        "basket_center": center,
    }


def _create_movable_yellow_cup_proxy(
    world: World,
    stage,
    reference,
    side: str,
) -> DynamicCylinder:
    """Replace the static scene cup visually with a movable physical proxy."""
    source_prim = stage.GetPrimAtPath(reference["path"])
    if not source_prim or not source_prim.IsValid():
        raise RuntimeError(
            f"Detected yellow cup prim disappeared: {reference['path']}"
        )

    dimensions = np.asarray(reference["dimensions"], dtype=np.float64)
    center = np.asarray(reference["center"], dtype=np.float64)
    minimum = np.asarray(reference["minimum"], dtype=np.float64)

    # The handle usually expands only one horizontal bbox dimension.  The
    # smaller horizontal dimension is a better body-diameter estimate.
    radius = float(
        np.clip(
            0.5 * min(dimensions[0], dimensions[1]),
            0.022,
            0.075,
        )
    )
    height = float(np.clip(dimensions[2], 0.055, 0.18))

    _set_subtree_visible(source_prim, False)
    _disable_collision_subtree(reference["path"])

    support_thickness = 0.010
    world.scene.add(
        FixedCuboid(
            prim_path="/World/YellowCupStartSupport",
            name="yellow_cup_start_support",
            position=np.array(
                [
                    center[0],
                    center[1],
                    minimum[2] - 0.5 * support_thickness,
                ],
                dtype=np.float64,
            ),
            scale=np.array(
                [
                    2.15 * radius,
                    2.15 * radius,
                    support_thickness,
                ],
                dtype=np.float64,
            ),
            color=np.array([0.72, 0.70, 0.50]),
        )
    )

    proxy_path = "/World/YellowCupProxy"
    cup = world.scene.add(
        DynamicCylinder(
            prim_path=proxy_path,
            name="yellow_cup_proxy",
            position=center.copy(),
            radius=radius,
            height=height,
            color=np.array([0.96, 0.78, 0.05]),
            mass=float(max(args_cli.cup_mass, 0.01)),
        )
    )
    _add_cup_visual_details(
        proxy_path,
        side=side,
        radius=radius,
        height=height,
        wall=args_cli.cup_wall_thickness,
        handle_radius=max(args_cli.cup_handle_radius, 0.65 * radius),
        handle_thickness=args_cli.cup_handle_thickness,
        body_color=np.array([0.96, 0.78, 0.05]),
    )

    TASK_SCENE_INFO["yellow_cup_source_path"] = reference["path"]
    TASK_SCENE_INFO["cup_radius"] = radius
    TASK_SCENE_INFO["cup_height"] = height
    TASK_SCENE_INFO["cup_start_center"] = center.copy()
    TASK_SCENE_INFO["table_top_z"] = float(minimum[2])

    print(
        f"[INFO] Existing yellow cup replaced by movable proxy: "
        f"source={reference['path']}, center={np.round(center, 3).tolist()}, "
        f"radius={radius:.3f}, height={height:.3f}"
    )
    return cup



def _default_target_stl_mesh_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "target_stl_mesh.npz"
    )


def _load_target_stl_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh_path = (
        Path(args_cli.target_stl_mesh).expanduser()
        if args_cli.target_stl_mesh
        else _default_target_stl_mesh_path()
    )
    if not mesh_path.exists():
        raise FileNotFoundError(
            f"Target STL mesh asset not found: {mesh_path}"
        )

    data = np.load(mesh_path)
    vertices = np.asarray(data["vertices"], dtype=np.float32)
    faces = np.asarray(data["faces"], dtype=np.int32)
    extents = np.asarray(data["extents"], dtype=np.float64).reshape(3)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError(
            f"Invalid target vertices in {mesh_path}: {vertices.shape}"
        )
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(
            f"Invalid target faces in {mesh_path}: {faces.shape}"
        )

    print(
        f"[INFO] Loaded target STL: {mesh_path} "
        f"vertices={len(vertices)}, faces={len(faces)}, "
        f"extents={np.round(extents, 4).tolist()}m"
    )
    return vertices, faces, extents


def _add_uploaded_stl_visual(
    cup_prim_path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    yaw_degrees: float,
) -> None:
    """Attach the uploaded STL geometry as visual-only mesh to the cup body."""
    stage = omni.usd.get_context().get_stage()
    mesh_path = f"{cup_prim_path}/UploadedYellowCupMesh"
    visual = UsdGeom.Mesh.Define(stage, mesh_path)

    try:
        visual.CreatePointsAttr(
            Vt.Vec3fArray.FromNumpy(
                np.ascontiguousarray(vertices, dtype=np.float32)
            )
        )
    except Exception:
        visual.CreatePointsAttr(
            [
                Gf.Vec3f(float(v[0]), float(v[1]), float(v[2]))
                for v in vertices
            ]
        )

    face_counts = np.full(
        len(faces),
        3,
        dtype=np.int32,
    )
    flat_indices = np.ascontiguousarray(
        faces.reshape(-1),
        dtype=np.int32,
    )
    try:
        visual.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(face_counts)
        )
        visual.CreateFaceVertexIndicesAttr(
            Vt.IntArray.FromNumpy(flat_indices)
        )
    except Exception:
        visual.CreateFaceVertexCountsAttr(
            face_counts.tolist()
        )
        visual.CreateFaceVertexIndicesAttr(
            flat_indices.tolist()
        )

    visual.CreateSubdivisionSchemeAttr(
        UsdGeom.Tokens.none
    )
    visual.CreateDoubleSidedAttr(True)
    visual.CreateDisplayColorAttr(
        [Gf.Vec3f(0.98, 0.78, 0.02)]
    )

    if abs(float(yaw_degrees)) > 1.0e-9:
        visual.AddRotateZOp().Set(
            float(yaw_degrees)
        )

    _disable_collision_subtree(mesh_path)


def _fixed_basket_task_points(
    cup_height: float,
) -> dict[str, np.ndarray]:
    basket_xy = np.array(
        [
            args_cli.task_basket_x,
            args_cli.task_basket_y,
        ],
        dtype=np.float64,
    )

    hover = np.array(
        [
            basket_xy[0],
            basket_xy[1],
            args_cli.task_basket_rim_z
            + 0.5 * cup_height
            + max(args_cli.basket_transfer_clearance, 0.03),
        ],
        dtype=np.float64,
    )
    drop = np.array(
        [
            basket_xy[0],
            basket_xy[1],
            args_cli.task_basket_floor_z
            + 0.5 * cup_height
            + max(args_cli.basket_drop_clearance, 0.004),
        ],
        dtype=np.float64,
    )
    return {
        "hover_object_center": hover,
        "drop_object_center": drop,
        "basket_center": np.array(
            [
                basket_xy[0],
                basket_xy[1],
                0.5
                * (
                    args_cli.task_basket_floor_z
                    + args_cli.task_basket_rim_z
                ),
            ],
            dtype=np.float64,
        ),
    }



def _horizontal_surface_clusters(stage):
    """Cluster horizontal environment triangles by world height and area."""
    candidates = _table_mesh_candidates(stage, args_cli.existing_table_prim)
    z_bin = max(float(args_cli.target_table_z_bin), 0.008)
    min_z = float(args_cli.table_surface_min_z)
    max_z = float(args_cli.table_surface_max_z)
    min_normal_z = max(float(args_cli.table_surface_normal_z), 0.82)
    clusters = {}
    for path, vertices, polygons, _, _ in candidates:
        for polygon in polygons:
            first = polygon[0]
            for i in range(1, len(polygon)-1):
                a,b,c = vertices[first],vertices[polygon[i]],vertices[polygon[i+1]]
                normal=np.cross(b-a,c-a); norm=float(np.linalg.norm(normal))
                if norm<=1e-10 or abs(float(normal[2]))/norm<min_normal_z: continue
                centroid=(a+b+c)/3.0; z=float(centroid[2])
                if not(min_z<=z<=max_z): continue
                area=0.5*norm
                if area<=1e-6: continue
                key=int(round(z/z_bin))
                cl=clusters.setdefault(key,{'area':0.0,'weighted_z':0.0,'samples':[],'paths':{}})
                cl['area']+=area; cl['weighted_z']+=area*z
                cl['samples'].append((centroid[:2].copy(),area,path))
                cl['paths'][path]=cl['paths'].get(path,0.0)+area
    valid=[]; min_area=max(float(args_cli.target_table_min_area),0.01); zspan=max(max_z-min_z,1e-3)
    for cl in clusters.values():
        area=float(cl['area'])
        if area<min_area: continue
        z=float(cl['weighted_z']/max(area,1e-12))
        cl['mean_z']=z; cl['score']=area*(0.8+0.2*np.clip((z-min_z)/zspan,0,1))
        cl['dominant_path']=max(cl['paths'].items(),key=lambda x:x[1])[0]
        valid.append(cl)
    valid.sort(key=lambda x:x['score'],reverse=True)
    return candidates,valid

def _surface_margin_is_valid(candidates,xy,z,margin):
    offsets=[(0,0),(margin,0),(-margin,0),(0,margin),(0,-margin),(.7*margin,.7*margin),(.7*margin,-.7*margin),(-.7*margin,.7*margin),(-.7*margin,-.7*margin)]
    for dx,dy in offsets:
        hit=_surface_at_xy_near_height(candidates,np.array([xy[0]+dx,xy[1]+dy],dtype=np.float64),z,tolerance=max(.035,1.8*float(args_cli.target_table_z_bin)))
        if hit is None: return False
    return True


def _compact_robot_bounds(stage):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    for path in ("/World/RobotScene/RBY1_A_v1_0", "/World/RobotScene"):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        values = _bbox_values(prim, cache)
        if values is None:
            continue
        minimum, maximum = values
        if float(max((maximum - minimum)[:2])) <= 4.0:
            return minimum, maximum
    return None


def _fallback_direction(mode, environment_vector):
    mapping = {
        "x+": np.array([1.0, 0.0], dtype=np.float64),
        "x-": np.array([-1.0, 0.0], dtype=np.float64),
        "y+": np.array([0.0, 1.0], dtype=np.float64),
        "y-": np.array([0.0, -1.0], dtype=np.float64),
    }
    if mode in mapping:
        return mapping[mode]
    vector = np.asarray(environment_vector, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-8:
        return np.array([-1.0, 0.0], dtype=np.float64)
    return vector / norm


def _pointheight_table_pose(stage, target_extents):
    """Detect a scan tabletop from dense vertices at nearly equal world Z."""
    mesh_candidates = _table_mesh_candidates(
        stage,
        args_cli.existing_table_prim,
    )
    if not mesh_candidates:
        return None

    bounds = _compact_robot_bounds(stage)
    if bounds is None:
        robot_min = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        robot_max = np.array([0.0, 0.0, 1.6], dtype=np.float64)
    else:
        robot_min, robot_max = bounds

    robot_xy = 0.5 * (robot_min[:2] + robot_max[:2])
    robot_height = max(float(robot_max[2] - robot_min[2]), 1.0)
    min_z = max(
        float(args_cli.table_surface_min_z),
        float(robot_min[2] + 0.25 * robot_height),
    )
    max_z = min(
        float(args_cli.table_surface_max_z),
        float(robot_min[2] + 0.85 * robot_height),
    )
    if max_z <= min_z + 0.05:
        min_z = float(args_cli.table_surface_min_z)
        max_z = float(args_cli.table_surface_max_z)

    samples = []
    for _, vertices, _, _, _ in mesh_candidates:
        if len(vertices) == 0:
            continue
        stride = max(1, int(len(vertices) / 160000))
        samples.append(vertices[::stride])
    if not samples:
        return None

    points = np.concatenate(samples, axis=0)
    radial = np.linalg.norm(points[:, :2] - robot_xy.reshape(1, 2), axis=1)
    mask = (
        (points[:, 2] >= min_z)
        & (points[:, 2] <= max_z)
        & (radial >= 0.18)
        & (radial <= 1.20)
    )
    points = points[mask]
    if len(points) < max(int(args_cli.target_pointcloud_min_points), 25):
        return None

    z_bin = max(float(args_cli.target_pointcloud_z_bin), 0.01)
    keys = np.round(points[:, 2] / z_bin).astype(np.int64)
    unique_keys, counts = np.unique(keys, return_counts=True)

    clusters = []
    for key, count in zip(unique_keys, counts):
        if count < max(int(args_cli.target_pointcloud_min_points), 25):
            continue
        cluster = points[keys == key]
        z = float(np.median(cluster[:, 2]))
        lo = np.percentile(cluster[:, :2], 5.0, axis=0)
        hi = np.percentile(cluster[:, :2], 95.0, axis=0)
        spread = hi - lo
        area_proxy = float(max(spread[0], 0.0) * max(spread[1], 0.0))
        if area_proxy < 0.03:
            continue
        distances = np.linalg.norm(
            cluster[:, :2] - robot_xy.reshape(1, 2),
            axis=1,
        )
        near_fraction = float(np.mean((distances >= 0.22) & (distances <= 0.85)))
        normalized_height = float(
            np.clip((z - min_z) / max(max_z - min_z, 1.0e-6), 0.0, 1.0)
        )
        score = (
            np.log1p(float(count))
            * np.sqrt(max(area_proxy, 1.0e-6))
            * (0.55 + 0.45 * near_fraction)
            * (0.70 + 0.30 * normalized_height)
        )
        clusters.append((float(score), z, cluster, int(count), area_proxy))

    if not clusters:
        return None
    clusters.sort(key=lambda item: item[0], reverse=True)

    preferred = max(float(args_cli.target_reach_distance), 0.25)
    half_width = 0.5 * float(max(target_extents[0], target_extents[1]))
    local_radius = max(half_width + 0.025, 0.065)

    for _, z, cluster, count, area_proxy in clusters[:8]:
        distances = np.linalg.norm(
            cluster[:, :2] - robot_xy.reshape(1, 2),
            axis=1,
        )
        valid = (distances >= 0.24) & (distances <= 0.78)
        candidate_xy = cluster[valid, :2]
        candidate_distances = distances[valid]
        if len(candidate_xy) == 0:
            continue

        order = np.argsort(np.abs(candidate_distances - preferred))
        for index in order[:700]:
            xy = np.asarray(candidate_xy[index], dtype=np.float64)
            local_distance = np.linalg.norm(
                cluster[:, :2] - xy.reshape(1, 2),
                axis=1,
            )
            local = cluster[local_distance <= local_radius]
            if len(local) < 12:
                continue
            same_height = np.mean(np.abs(local[:, 2] - z) <= 1.8 * z_bin)
            if float(same_height) < 0.55:
                continue

            print(
                f"[INFO] Scan-point tabletop fallback selected: "
                f"z={z:.3f}, points={count}, area_proxy={area_proxy:.3f}m^2"
            )
            print(
                f"[INFO] Scan-point target: "
                f"xy=({xy[0]:+.3f},{xy[1]:+.3f}), "
                f"robot_distance={candidate_distances[index]:.3f}m"
            )
            return xy, z, "scan-point height cluster"
    return None


def _no_abort_robot_relative_pose(stage):
    """Last-resort target placement that always allows startup to continue."""
    bounds = _compact_robot_bounds(stage)
    if bounds is None:
        robot_min = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        robot_max = np.array([0.0, 0.0, 1.6], dtype=np.float64)
    else:
        robot_min, robot_max = bounds

    robot_xy = 0.5 * (robot_min[:2] + robot_max[:2])
    table_z = float(
        robot_min[2] + max(float(args_cli.target_fallback_height), 0.35)
    )

    environment_vector = np.array([-1.0, 0.0], dtype=np.float64)
    mesh_candidates = _table_mesh_candidates(
        stage,
        args_cli.existing_table_prim,
    )
    nearby = []
    for _, vertices, _, _, _ in mesh_candidates:
        if len(vertices) == 0:
            continue
        stride = max(1, int(len(vertices) / 90000))
        sampled = vertices[::stride]
        radial = np.linalg.norm(
            sampled[:, :2] - robot_xy.reshape(1, 2),
            axis=1,
        )
        mask = (
            (np.abs(sampled[:, 2] - table_z) <= 0.25)
            & (radial >= 0.20)
            & (radial <= 1.10)
        )
        if np.any(mask):
            nearby.append(sampled[mask, :2])
    if nearby:
        environment_center = np.median(np.concatenate(nearby, axis=0), axis=0)
        environment_vector = environment_center - robot_xy

    direction = _fallback_direction(
        args_cli.target_fallback_direction,
        environment_vector,
    )
    xy = robot_xy + direction * max(float(args_cli.target_reach_distance), 0.30)

    print(
        "[WARN] No reliable tabletop plane or point-height cluster was found. "
        "Using robot-relative fallback instead of stopping."
    )
    print(
        f"[INFO] Robot-relative target: "
        f"xy=({xy[0]:+.3f},{xy[1]:+.3f}), table_z={table_z:.3f}"
    )
    return xy, table_z, "robot-relative fallback"



def _sample_environment_vertices(stage) -> np.ndarray:
    """Collect a bounded world-space sample from the merged scan mesh."""
    arrays = []
    for _, vertices, _, _, _ in _table_mesh_candidates(
        stage,
        args_cli.existing_table_prim,
    ):
        if len(vertices) == 0:
            continue
        stride = max(1, int(len(vertices) / 180000))
        arrays.append(
            np.asarray(vertices[::stride], dtype=np.float64)
        )
    if not arrays:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(arrays, axis=0)


def _local_surface_height(
    points: np.ndarray,
    xy: np.ndarray,
    *,
    robot_min_z: float,
    robot_height: float,
    target_extents: np.ndarray,
) -> tuple[float, float, int] | None:
    """Estimate a tabletop height near XY from a local scan-point Z histogram."""
    probe_radius = max(
        float(args_cli.target_probe_radius),
        0.5 * float(max(target_extents[0], target_extents[1])) + 0.045,
    )
    radial = np.linalg.norm(
        points[:, :2] - np.asarray(xy, dtype=np.float64).reshape(1, 2),
        axis=1,
    )
    local = points[radial <= probe_radius]
    if len(local) < 18:
        return None

    min_z = robot_min_z + max(
        float(args_cli.target_direction_min_height),
        0.20,
    )
    max_z = robot_min_z + max(
        float(args_cli.target_direction_max_height),
        float(args_cli.target_direction_min_height) + 0.15,
    )
    local = local[
        (local[:, 2] >= min_z)
        & (local[:, 2] <= max_z)
    ]
    if len(local) < 18:
        return None

    z_bin = max(float(args_cli.target_probe_z_bin), 0.008)
    keys = np.round(local[:, 2] / z_bin).astype(np.int64)
    unique, counts = np.unique(keys, return_counts=True)

    best = None
    expected_z = robot_min_z + 0.50 * robot_height
    for key, count in zip(unique, counts):
        if int(count) < 10:
            continue
        band = local[keys == key]
        z = float(np.median(band[:, 2]))

        # A tabletop band should cover a useful part of the probe circle.
        lo = np.percentile(band[:, :2], 5.0, axis=0)
        hi = np.percentile(band[:, :2], 95.0, axis=0)
        spread = hi - lo
        coverage = float(
            max(spread[0], 0.0) * max(spread[1], 0.0)
        )
        if coverage < 0.0018:
            continue

        # Prefer dense/broad bands near waist/table height. A slightly higher
        # band wins only when it has comparable spatial support.
        height_penalty = abs(z - expected_z)
        score = (
            float(count)
            * (1.0 + 12.0 * min(coverage, 0.03))
            / (1.0 + 1.8 * height_penalty)
        )
        if best is None or score > best[0]:
            best = (score, z, coverage, int(count))

    if best is None:
        return None
    _, z, coverage, count = best
    return float(z), float(coverage), int(count)


def _robot_near_table_pose(
    stage,
    target_extents: np.ndarray,
) -> tuple[np.ndarray, float, str] | None:
    """Find the table side from elevated scan-point density around the robot.

    The floor dominates global mesh statistics. This method first finds which
    direction from the robot contains the most elevated scene geometry, then
    searches that direction for a dense local tabletop-height band.
    """
    points = _sample_environment_vertices(stage)
    if len(points) == 0:
        return None

    bounds = _compact_robot_bounds(stage)
    if bounds is None:
        robot_min = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        robot_max = np.array([0.0, 0.0, 1.6], dtype=np.float64)
    else:
        robot_min, robot_max = bounds

    robot_xy = 0.5 * (robot_min[:2] + robot_max[:2])
    robot_height = max(float(robot_max[2] - robot_min[2]), 1.0)
    relative = points[:, :2] - robot_xy.reshape(1, 2)
    radial = np.linalg.norm(relative, axis=1)
    relative_z = points[:, 2] - float(robot_min[2])

    min_height = max(float(args_cli.target_direction_min_height), 0.20)
    max_height = max(
        float(args_cli.target_direction_max_height),
        min_height + 0.15,
    )
    mask = (
        (radial >= 0.22)
        & (radial <= 1.20)
        & (relative_z >= min_height)
        & (relative_z <= max_height)
    )
    elevated = points[mask]
    elevated_relative = relative[mask]
    elevated_radial = radial[mask]
    if len(elevated) < 40:
        return None

    sector_count = max(int(args_cli.target_direction_sectors), 12)
    angles = np.arctan2(
        elevated_relative[:, 1],
        elevated_relative[:, 0],
    )
    sector_width = 2.0 * np.pi / sector_count
    sector_indices = np.floor(
        (angles + np.pi) / sector_width
    ).astype(np.int64)
    sector_indices = np.clip(
        sector_indices,
        0,
        sector_count - 1,
    )

    # Elevated, nearby, dense points are a strong indicator of the table side.
    height_norm = np.clip(
        (elevated[:, 2] - (robot_min[2] + min_height))
        / max(max_height - min_height, 1.0e-6),
        0.0,
        1.0,
    )
    radial_weight = np.exp(
        -0.5
        * (
            (elevated_radial - max(float(args_cli.target_reach_distance), 0.42))
            / 0.34
        )
        ** 2
    )
    point_weights = (
        0.65 + 0.35 * height_norm
    ) * (
        0.40 + 0.60 * radial_weight
    )
    sector_scores = np.bincount(
        sector_indices,
        weights=point_weights,
        minlength=sector_count,
    )

    # Smooth neighboring sectors so a table spanning several angles is favored.
    smooth_scores = (
        sector_scores
        + 0.65 * np.roll(sector_scores, 1)
        + 0.65 * np.roll(sector_scores, -1)
        + 0.25 * np.roll(sector_scores, 2)
        + 0.25 * np.roll(sector_scores, -2)
    )
    ranked_sectors = np.argsort(smooth_scores)[::-1]

    preferred_distance = max(float(args_cli.target_reach_distance), 0.28)
    distance_candidates = [
        preferred_distance,
        preferred_distance + 0.06,
        preferred_distance - 0.06,
        preferred_distance + 0.12,
        preferred_distance - 0.12,
        preferred_distance + 0.18,
    ]
    distance_candidates = [
        value for value in distance_candidates
        if 0.26 <= value <= 0.72
    ]

    angle_offset = np.radians(
        float(args_cli.target_direction_offset_deg)
    )
    lateral_offset = float(args_cli.target_lateral_offset)

    evaluated = []
    for sector in ranked_sectors[:8]:
        center_angle = (
            -np.pi
            + (float(sector) + 0.5) * sector_width
            + angle_offset
        )
        direction = np.array(
            [np.cos(center_angle), np.sin(center_angle)],
            dtype=np.float64,
        )
        lateral = np.array(
            [-direction[1], direction[0]],
            dtype=np.float64,
        )

        # Search the center and nearby angular offsets within the table sector.
        for angular_delta in (0.0, -0.5 * sector_width, 0.5 * sector_width):
            angle = center_angle + angular_delta
            direction_i = np.array(
                [np.cos(angle), np.sin(angle)],
                dtype=np.float64,
            )
            lateral_i = np.array(
                [-direction_i[1], direction_i[0]],
                dtype=np.float64,
            )
            for distance in distance_candidates:
                xy = (
                    robot_xy
                    + distance * direction_i
                    + lateral_offset * lateral_i
                )
                surface = _local_surface_height(
                    points,
                    xy,
                    robot_min_z=float(robot_min[2]),
                    robot_height=robot_height,
                    target_extents=target_extents,
                )
                if surface is None:
                    continue
                z, coverage, count = surface

                # Prefer close-to-requested reach, broad surface support, and
                # the highest-density table-facing sector.
                distance_penalty = abs(distance - preferred_distance)
                candidate_score = (
                    float(smooth_scores[sector])
                    + 1800.0 * coverage
                    + 0.30 * count
                    - 40.0 * distance_penalty
                )
                evaluated.append(
                    (
                        candidate_score,
                        xy,
                        z,
                        coverage,
                        count,
                        center_angle,
                        distance,
                        int(sector),
                    )
                )

    if not evaluated:
        return None

    evaluated.sort(
        key=lambda item: item[0],
        reverse=True,
    )
    (
        score,
        xy,
        z,
        coverage,
        count,
        angle,
        distance,
        sector,
    ) = evaluated[0]

    print(
        "[INFO] Robot-near table direction selected: "
        f"angle={np.degrees(angle):+.1f}deg, "
        f"sector={sector}/{sector_count}, "
        f"direction_score={smooth_scores[sector]:.1f}"
    )
    print(
        "[INFO] Corrected table target: "
        f"xy=({xy[0]:+.3f},{xy[1]:+.3f}), "
        f"table_z={z:.3f}, "
        f"robot_distance={distance:.3f}m, "
        f"local_points={count}, coverage={coverage:.4f}m^2"
    )
    return (
        np.asarray(xy, dtype=np.float64),
        float(z),
        "robot-near elevated-density tabletop",
    )



def _axis_direction(mode: str) -> np.ndarray:
    mapping = {
        "x+": np.array([1.0, 0.0], dtype=np.float64),
        "x-": np.array([-1.0, 0.0], dtype=np.float64),
        "y+": np.array([0.0, 1.0], dtype=np.float64),
        "y-": np.array([0.0, -1.0], dtype=np.float64),
    }
    return mapping[mode].copy()


def _known_table_side_pose(
    stage,
    target_extents: np.ndarray,
) -> tuple[np.ndarray, float, str]:
    """Search only the table side confirmed from the Isaac Sim axis widget.

    In the current scene the table is located in world +X from the robot.
    Previous global/sector detectors repeatedly selected the opposite floor side.
    This method therefore searches a rectangular strip along the selected axis
    and never changes to an unrelated direction.
    """
    points = _sample_environment_vertices(stage)
    bounds = _compact_robot_bounds(stage)

    if bounds is None:
        robot_min = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        robot_max = np.array([0.0, 0.0, 1.6], dtype=np.float64)
    else:
        robot_min, robot_max = bounds

    robot_xy = 0.5 * (robot_min[:2] + robot_max[:2])
    robot_height = max(float(robot_max[2] - robot_min[2]), 1.0)

    direction = _axis_direction(args_cli.target_table_side)
    lateral = np.array(
        [-direction[1], direction[0]],
        dtype=np.float64,
    )

    search_min = max(float(args_cli.target_side_search_min), 0.20)
    search_max = max(
        float(args_cli.target_side_search_max),
        search_min + 0.04,
    )
    search_step = max(float(args_cli.target_side_search_step), 0.02)
    lateral_range = max(float(args_cli.target_side_lateral_range), 0.0)
    lateral_step = max(float(args_cli.target_side_lateral_step), 0.025)

    preferred_distance = float(
        np.clip(
            args_cli.target_reach_distance,
            search_min,
            search_max,
        )
    )

    distances = np.arange(
        search_min,
        search_max + 0.5 * search_step,
        search_step,
        dtype=np.float64,
    )
    lateral_offsets = np.arange(
        -lateral_range,
        lateral_range + 0.5 * lateral_step,
        lateral_step,
        dtype=np.float64,
    )

    candidates = []
    if len(points) > 0:
        for distance in distances:
            for lateral_value in lateral_offsets:
                xy = (
                    robot_xy
                    + distance * direction
                    + (
                        float(args_cli.target_lateral_offset)
                        + lateral_value
                    )
                    * lateral
                )
                surface = _local_surface_height(
                    points,
                    xy,
                    robot_min_z=float(robot_min[2]),
                    robot_height=robot_height,
                    target_extents=target_extents,
                )
                if surface is None:
                    continue

                z, coverage, count = surface

                # Prefer requested reach, broad local support, and a table-height
                # band around the robot waist rather than low shelves or floor.
                expected_z = float(
                    robot_min[2]
                    + max(
                        args_cli.target_side_fallback_height,
                        0.40,
                    )
                )
                distance_penalty = abs(float(distance) - preferred_distance)
                lateral_penalty = abs(float(lateral_value))
                height_penalty = abs(float(z) - expected_z)

                score = (
                    2200.0 * float(coverage)
                    + 0.40 * float(count)
                    - 45.0 * distance_penalty
                    - 16.0 * lateral_penalty
                    - 35.0 * height_penalty
                )
                candidates.append(
                    (
                        score,
                        np.asarray(xy, dtype=np.float64),
                        float(z),
                        float(distance),
                        float(lateral_value),
                        float(coverage),
                        int(count),
                    )
                )

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        (
            score,
            xy,
            z,
            distance,
            lateral_value,
            coverage,
            count,
        ) = candidates[0]

        print(
            f"[INFO] Forced table side selected: "
            f"side={args_cli.target_table_side}"
        )
        print(
            f"[INFO] Forced-side table target: "
            f"xy=({xy[0]:+.3f},{xy[1]:+.3f}), "
            f"table_z={z:.3f}, distance={distance:.3f}m, "
            f"lateral={lateral_value:+.3f}m, "
            f"points={count}, coverage={coverage:.4f}m^2"
        )
        return (
            xy,
            z,
            f"forced {args_cli.target_table_side} table-side scan",
        )

    # Do not jump to another direction. Keep the object on the confirmed table
    # side even when the scan mesh is too noisy for local height estimation.
    distance = preferred_distance
    xy = (
        robot_xy
        + distance * direction
        + float(args_cli.target_lateral_offset) * lateral
    )
    table_z = float(
        robot_min[2]
        + max(
            args_cli.target_side_fallback_height,
            0.40,
        )
    )

    print(
        f"[WARN] No reliable local surface found on {args_cli.target_table_side}; "
        "using deterministic table-side coordinates."
    )
    print(
        f"[INFO] Deterministic table-side target: "
        f"xy=({xy[0]:+.3f},{xy[1]:+.3f}), "
        f"table_z={table_z:.3f}, distance={distance:.3f}m"
    )
    return (
        xy,
        table_z,
        f"deterministic {args_cli.target_table_side} fallback",
    )


def _auto_target_table_pose(stage,target_extents):
    return _known_table_side_pose(stage, target_extents)

def _remove_old_generated_task_prims(stage) -> None:
    """Remove cups, basket helpers, and targets generated by older patches."""
    paths = [
        "/World/WebcamPickTable",
        "/World/TableCup_left",
        "/World/TableCup_right",
        "/World/YellowCupProxy",
        "/World/TargetSTLObject",
        "/World/YellowCupStartSupport",
        "/World/BasketCupSupport",
        "/World/CupSupport_left",
        "/World/CupSupport_right",
    ]
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            stage.RemovePrim(path)
            print(f"[INFO] Removed old generated prim: {path}")


def create_demo_scene(world: World) -> dict[str, DynamicCylinder]:
    """Create exactly one uploaded STL target and no generated cups/basket."""
    if args_cli.no_demo_table:
        return {}

    TASK_SCENE_INFO.clear()
    stage = omni.usd.get_context().get_stage()
    _remove_old_generated_task_prims(stage)

    vertices, faces, extents = _load_target_stl_mesh()

    target_height = float(max(extents[2], 0.025))
    # Use a compact cylinder entirely inside the mesh as stable physical proxy.
    proxy_radius = float(
        np.clip(
            0.38 * min(extents[0], extents[1]),
            0.018,
            0.055,
        )
    )
    proxy_height = float(
        np.clip(
            0.86 * target_height,
            0.025,
            0.16,
        )
    )

    if args_cli.target_placement == "auto":
        target_xy, detected_table_z, table_source = _auto_target_table_pose(
            stage,
            extents,
        )
        base_target_xy = np.asarray(
            target_xy,
            dtype=np.float64,
        ).copy()
        base_table_z = float(detected_table_z)
    else:
        base_target_xy = np.array(
            [args_cli.target_x, args_cli.target_y],
            dtype=np.float64,
        )
        base_table_z = float(args_cli.target_table_z)
        table_source = "manual coordinates"

    # v20.5 scene correction taken from the user's Isaac Sim screenshot.
    # The previous placement was too low and too close to the robot.  Apply a
    # deterministic world-frame correction toward the existing yellow cup and
    # upward to tabletop height.
    target_xy = (
        base_target_xy
        + np.array(
            [
                args_cli.target_offset_x,
                args_cli.target_offset_y,
            ],
            dtype=np.float64,
        )
    )
    corrected_table_z = (
        base_table_z
        + float(args_cli.target_offset_z)
    )

    args_cli.target_x = float(target_xy[0])
    args_cli.target_y = float(target_xy[1])
    args_cli.target_table_z = float(corrected_table_z)

    print(
        "[INFO] Target coordinate correction:\n"
        f"       base_xy={np.round(base_target_xy, 3).tolist()}, "
        f"base_z={base_table_z:.3f}\n"
        f"       offset="
        f"({args_cli.target_offset_x:+.3f},"
        f"{args_cli.target_offset_y:+.3f},"
        f"{args_cli.target_offset_z:+.3f})m\n"
        f"       corrected_xy={np.round(target_xy, 3).tolist()}, "
        f"corrected_z={corrected_table_z:.3f}"
    )

    target_center = np.array(
        [
            target_xy[0],
            target_xy[1],
            corrected_table_z + 0.5 * target_height,
        ],
        dtype=np.float64,
    )
    robot_xy=_robot_scene_center_xy(stage,target_center[:2])
    toward=robot_xy-target_center[:2]; norm=float(np.linalg.norm(toward))
    toward=np.array([0.0,-1.0]) if norm<1e-6 else toward/norm
    clearance=.5*float(max(extents[0],extents[1]))+.045
    TASK_SCENE_INFO["grasp_offset"]=np.array([toward[0]*clearance,toward[1]*clearance,0.0],dtype=np.float64)

    # Tiny support only beneath this STL. No second table or basket is created.
    support_thickness = 0.010
    world.scene.add(
        FixedCuboid(
            prim_path="/World/TargetSTLSupport",
            name="target_stl_support",
            position=np.array(
                [
                    target_center[0],
                    target_center[1],
                    corrected_table_z
                    - 0.5 * support_thickness,
                ],
                dtype=np.float64,
            ),
            scale=np.array(
                [
                    max(1.20 * extents[0], 0.07),
                    max(1.20 * extents[1], 0.07),
                    support_thickness,
                ],
                dtype=np.float64,
            ),
            color=np.array([0.48, 0.48, 0.48]),
        )
    )

    proxy_path = "/World/TargetSTLObject"
    target_object = world.scene.add(
        DynamicCylinder(
            prim_path=proxy_path,
            name="target_stl_object",
            position=target_center,
            radius=proxy_radius,
            height=proxy_height,
            color=np.array([0.98, 0.78, 0.02]),
            mass=float(max(args_cli.target_mass, 0.01)),
        )
    )

    _add_uploaded_stl_visual(
        proxy_path,
        vertices,
        faces,
        yaw_degrees=args_cli.target_yaw_deg,
    )

    TASK_SCENE_INFO["target_center"] = target_center.copy()
    TASK_SCENE_INFO["target_extents"] = extents.copy()
    TASK_SCENE_INFO["table_top_z"] = float(corrected_table_z)
    TASK_SCENE_INFO["table_source"] = table_source

    args_cli.table_top_z = float(corrected_table_z)

    active_side = (
        "right"
        if args_cli.control_side == "both"
        else args_cli.control_side
    )

    print(
        "[INFO] GRASP-ONLY TARGET:\n"
        f"       center={np.round(target_center, 3).tolist()}\n"
        f"       extents={np.round(extents, 3).tolist()}\n"
        f"       physical_proxy_radius={proxy_radius:.3f}m\n"
        f"       physical_proxy_height={proxy_height:.3f}m\n"
        f"       world_offset="
        f"({args_cli.target_offset_x:+.3f},"
        f"{args_cli.target_offset_y:+.3f},"
        f"{args_cli.target_offset_z:+.3f})m"
    )
    print(
        "[INFO] No generated cups, basket, transfer, lift, or release task."
    )
    return {active_side: target_object}


def object_pose(obj: DynamicCylinder) -> tuple[np.ndarray, np.ndarray]:
    position, orientation = obj.get_world_pose()
    return as_numpy(position).reshape(3).astype(np.float64), as_numpy(orientation).reshape(4).astype(np.float64)


def set_object_pose(obj: DynamicCylinder, position: np.ndarray, orientation: np.ndarray) -> None:
    obj.set_world_pose(position=np.asarray(position, dtype=np.float64), orientation=np.asarray(orientation, dtype=np.float64))
    for method_name in ("set_linear_velocity", "set_angular_velocity"):
        method = getattr(obj, method_name, None)
        if method is not None:
            try:
                method(np.zeros(3, dtype=np.float64))
            except Exception:
                pass


def gripper_open_close_targets(
    names: list[str],
    name_to_index: dict[str, int],
    lower: np.ndarray,
    upper: np.ndarray,
    q_initial: np.ndarray,
    *,
    explicit_open: np.ndarray | None = None,
    explicit_close: np.ndarray | None = None,
    invert: bool = False,
    fallback_travel: float = 0.035,
) -> tuple[dict[str, float], dict[str, float]]:
    """Infer stable RB-Y1 parallel-jaw open/close position targets.

    The imported RB-Y1 model may expose two finger joints although each physical
    gripper has one opening DOF.  At startup the model is normally already open,
    so q_initial is the most trustworthy open target.  Closing is inferred toward
    zero when zero lies inside the joint range.  If the imported limits are broad
    or unusable, a small mirrored travel is used rather than commanding ±2π.
    """
    open_targets: dict[str, float] = {}
    close_targets: dict[str, float] = {}
    travel = float(max(abs(fallback_travel), 1.0e-4))

    for local_i, name in enumerate(names):
        index = name_to_index[name]
        q0 = float(q_initial[index])
        lo, hi = float(lower[index]), float(upper[index])
        valid_limits = (
            math.isfinite(lo)
            and math.isfinite(hi)
            and hi > lo + 1.0e-6
            and max(abs(lo), abs(hi)) < 1.0
        )

        if explicit_open is not None:
            opened = float(explicit_open[local_i])
        else:
            opened = q0

        if explicit_close is not None:
            closed = float(explicit_close[local_i])
        elif valid_limits and lo <= 0.0 <= hi:
            closed = 0.0
            # If startup is also essentially zero, infer an open pose away from
            # zero using the mirrored jaw convention.
            if abs(opened - closed) < 1.0e-4:
                # RB-Y1 finger_1 and finger_2 are mirrored jaws.  Using the
                # same range endpoint for both fingers makes the whole gripper
                # translate instead of opening.  Select opposite endpoints.
                lowered_name = name.lower()
                if lowered_name.endswith("1"):
                    endpoint = hi
                elif lowered_name.endswith("2"):
                    endpoint = lo
                else:
                    endpoint = max((lo, hi), key=lambda value: abs(value - closed))
                opened = closed + math.copysign(min(abs(endpoint - closed), travel), endpoint - closed)
        elif valid_limits:
            # For a one-sided prismatic range, the endpoint closest to zero is
            # treated as closed and startup remains the preferred open pose.
            closed = lo if abs(lo) <= abs(hi) else hi
            if abs(opened - closed) < 1.0e-4:
                endpoint = hi if closed == lo else lo
                opened = closed + math.copysign(min(abs(endpoint - closed), travel), endpoint - closed)
        else:
            # Canonical RB-Y1 finger pairs are mirrored.  Keep startup as closed
            # only when it is near zero, otherwise regard startup as open.
            if abs(q0) > 1.0e-4:
                opened = q0
                closed = 0.0
            else:
                direction = 1.0 if name.lower().endswith("1") else -1.0
                opened = q0 + direction * travel
                closed = q0

        if valid_limits:
            opened = float(np.clip(opened, lo, hi))
            closed = float(np.clip(closed, lo, hi))

        if invert:
            opened, closed = closed, opened

        open_targets[name] = opened
        close_targets[name] = closed

    return open_targets, close_targets


def robot_elbow_range_targets(
    q0: float, lower: float, upper: float, bend_fraction: float
) -> tuple[float, float]:
    """Return (bent, extended) targets for arm_3 without assuming its sign."""
    if not (math.isfinite(lower) and math.isfinite(upper) and upper - lower > 1.0e-4):
        extended = q0 * 0.15
        direction = -1.0 if q0 < 0.0 else 1.0
        return q0 + direction * 0.8, extended
    extended = float(np.clip(0.0, lower, upper))
    if abs(q0 - extended) > 0.10:
        endpoint = lower if q0 < extended else upper
    else:
        endpoint = lower if abs(lower - extended) >= abs(upper - extended) else upper
    bent = extended + float(np.clip(bend_fraction, 0.2, 0.95)) * (endpoint - extended)
    return float(np.clip(bent, lower, upper)), extended



@dataclass
class ArmRuntime:
    side: str
    joint_indices: np.ndarray
    joint_names: list[str]
    ee_name: str
    ee: SingleRigidPrim
    jacobian_index: int
    q_home: np.ndarray
    anchor: np.ndarray
    target_raw: np.ndarray
    target_filtered: np.ndarray
    orientation_anchor: np.ndarray
    orientation_target: np.ndarray
    shoulder_anchor: np.ndarray
    shoulder_target: np.ndarray
    wrist_joint_target: np.ndarray
    object_handle: DynamicCylinder | None = None
    elbow_name: str | None = None
    elbow: SingleRigidPrim | None = None
    elbow_jacobian_index: int | None = None
    elbow_anchor: np.ndarray | None = None
    elbow_target_raw: np.ndarray | None = None
    elbow_target_filtered: np.ndarray | None = None
    elbow_tracking: bool = False
    elbow_bent_target: float = 0.0
    elbow_extended_target: float = 0.0
    elbow_joint_target: float = 0.0
    human_elbow_extension: float = -1.0
    reach_assist_blend: float = 0.0
    motion_stage: str = "teleop"
    stage_xy: np.ndarray | None = None
    last_dq: np.ndarray | None = None
    last_shoulder_correction: np.ndarray | None = None
    attached: bool = False
    object_local_offset: np.ndarray | None = None
    object_orientation: np.ndarray | None = None
    grasp_orientation_locked: np.ndarray | None = None
    lift_start_position: np.ndarray | None = None
    lift_goal_position: np.ndarray | None = None
    lift_start_elbow: np.ndarray | None = None
    lift_goal_elbow: np.ndarray | None = None
    phase_elapsed: float = 0.0
    arm_raise_elapsed: float = 0.0
    arm_raise_armed: bool = True
    last_wrist_up_delta: float = 0.0
    last_shoulder_elevation_delta: float = 0.0
    motion_guard_remaining: float = 0.0
    last_motion_score: float = 0.0
    last_motion_source: str = "none"
    pick_pending: bool = False
    pick_pending_elapsed: float = 0.0
    pick_posture_reference: np.ndarray | None = None
    stage_start_position: np.ndarray | None = None
    stage_goal_position: np.ndarray | None = None
    basket_drop_object_position: np.ndarray | None = None
    grasp_hold_position: np.ndarray | None = None
    auto_grasp_elapsed: float = 0.0
    auto_grasp_started: bool = False


def main() -> None:
    usd_path = USD_MAP[args_cli.asset]
    if not usd_path.exists():
        raise FileNotFoundError(usd_path)

    scene_root = "/World/RobotScene"
    world = load_world(usd_path, scene_root)
    demo_objects = create_demo_scene(world)
    robot, root_path = resolve_robot(scene_root, world)

    # A world reset is required after all dynamic demo objects have been added.
    world.reset()
    robot.initialize()

    dof_names = list(robot.dof_names)
    base_names = [name.split("/")[-1] for name in dof_names]
    name_to_index = {name: index for index, name in enumerate(base_names)}
    all_indices = np.arange(len(dof_names), dtype=np.int32)

    requested_arms = RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS
    _, resolved_arm_names, missing = resolve_joint_indices(dof_names, requested_arms)
    if missing:
        raise RuntimeError(f"Missing arm joints {missing}; DOFs={dof_names}")
    left_arm_names = sorted(
        [name.split("/")[-1] for name in resolved_arm_names if name.split("/")[-1].startswith("left_arm_")],
        key=lambda name: joint_number(name) or 0,
    )
    right_arm_names = sorted(
        [name.split("/")[-1] for name in resolved_arm_names if name.split("/")[-1].startswith("right_arm_")],
        key=lambda name: joint_number(name) or 0,
    )
    if len(left_arm_names) != 7 or len(right_arm_names) != 7:
        raise RuntimeError(f"Expected 7 joints per arm; left={left_arm_names}, right={right_arm_names}")

    _, left_resolved = resolve_optional_joint_indices(dof_names, LEFT_GRIPPER_JOINTS)
    _, right_resolved = resolve_optional_joint_indices(dof_names, RIGHT_GRIPPER_JOINTS)
    left_gripper_names = select_precision_joints(
        discover_gripper_names(dof_names, left_resolved, "left"), args_cli.left_precision_joints, "left"
    )
    right_gripper_names = select_precision_joints(
        discover_gripper_names(dof_names, right_resolved, "right"), args_cli.right_precision_joints, "right"
    )

    q_initial = as_numpy(robot.get_joint_positions(joint_indices=all_indices)).astype(np.float64)
    lower, upper = read_limits(robot, len(dof_names))
    lower_safe = lower.copy()
    upper_safe = upper.copy()
    for index, name in enumerate(base_names):
        margin = 0.0005 if "gripper" in name else 0.015
        if math.isfinite(lower_safe[index]):
            lower_safe[index] += margin
        if math.isfinite(upper_safe[index]):
            upper_safe[index] -= margin
        if lower_safe[index] > upper_safe[index]:
            lower_safe[index] = lower[index]
            upper_safe[index] = upper[index]

    left_indices = np.array([name_to_index[name] for name in left_arm_names], dtype=np.int32)
    right_indices = np.array([name_to_index[name] for name in right_arm_names], dtype=np.int32)
    wrist_windows = np.radians(
        np.abs(parse_float_triplet(args_cli.wrist_joint_window_deg, "--wrist-joint-window-deg"))
    )
    for arm_names in (left_arm_names, right_arm_names):
        for local_i, name in enumerate(arm_names[4:7]):
            index = name_to_index[name]
            window = float(wrist_windows[local_i])
            lower_safe[index] = max(lower_safe[index], float(q_initial[index]) - window)
            upper_safe[index] = min(upper_safe[index], float(q_initial[index]) + window)

    side_joint_indices = {"left": left_indices, "right": right_indices}
    side_joint_names = {"left": left_arm_names, "right": right_arm_names}
    requested_ee = {"left": args_cli.left_ee_body, "right": args_cli.right_ee_body}
    active_sides = {"left", "right"} if args_cli.control_side == "both" else {args_cli.control_side}

    jacobian_tensor = get_jacobian_tensor(robot)
    body_names = list(getattr(robot, "body_names", []) or [])
    if not body_names:
        body_names = list(getattr(getattr(robot, "_articulation_view", None), "body_names", []) or [])
    print("[INFO] Articulation body names:", [_base_name(name) for name in body_names])
    print("[INFO] Jacobian tensor shape:", jacobian_tensor.shape)

    def anchor_for(side: str, current_position: np.ndarray) -> np.ndarray:
        # Direct teleoperation begins from the robot's current hand pose.
        # Object-hover mode is optional for users who want calibration to map
        # immediately above the demo object.
        if args_cli.anchor_mode == "current":
            return current_position.copy()
        obj = demo_objects.get(side)
        if obj is None:
            return current_position.copy()
        pos, _ = object_pose(obj)
        return pos + np.array([0.0, 0.0, args_cli.approach_height], dtype=np.float64)

    # Initialize only the side(s) requested by --control-side. The previous
    # version always tried ee_left first even during right-arm-only operation.
    runtimes: dict[str, ArmRuntime] = {}
    for side in sorted(active_sides):
        selected_body, jacobian_index = select_end_effector_body(
            body_names,
            jacobian_tensor,
            side_joint_indices[side],
            requested_ee[side],
            side,
        )
        ee = initialize_ee(scene_root, root_path, selected_body, side)
        position, orientation = rigid_pose(ee)
        indices = side_joint_indices[side]
        elbow_name = None
        elbow = None
        elbow_jacobian_index = None
        elbow_selection = select_elbow_body(body_names, jacobian_tensor, indices, side)
        if elbow_selection is not None:
            elbow_name, elbow_jacobian_index = elbow_selection
            try:
                elbow = initialize_ee(scene_root, root_path, elbow_name, f"{side}_elbow")
            except Exception as exc:
                print(f"[WARN] {side} elbow-link monitor unavailable: {exc}")
                elbow_name = None
                elbow_jacobian_index = None
                elbow = None
        elbow_position = None
        if elbow is not None:
            elbow_position, _ = rigid_pose(elbow)
        elbow_joint_index = int(indices[3])
        elbow_bent_target, elbow_extended_target = robot_elbow_range_targets(
            float(q_initial[elbow_joint_index]),
            float(lower_safe[elbow_joint_index]),
            float(upper_safe[elbow_joint_index]),
            args_cli.robot_elbow_bend_fraction,
        )
        runtimes[side] = ArmRuntime(
            side=side,
            joint_indices=indices,
            joint_names=side_joint_names[side],
            ee_name=selected_body,
            ee=ee,
            jacobian_index=jacobian_index,
            q_home=q_initial[indices].copy(),
            anchor=anchor_for(side, position),
            target_raw=position.copy(),
            target_filtered=position.copy(),
            orientation_anchor=orientation.copy(),
            orientation_target=orientation.copy(),
            shoulder_anchor=q_initial[indices[:3]].copy(),
            shoulder_target=q_initial[indices[:3]].copy(),
            wrist_joint_target=q_initial[indices[4:7]].copy(),
            object_handle=demo_objects.get(side),
            elbow_name=elbow_name,
            elbow=elbow,
            elbow_jacobian_index=elbow_jacobian_index,
            elbow_anchor=None if elbow_position is None else elbow_position.copy(),
            elbow_target_raw=None if elbow_position is None else elbow_position.copy(),
            elbow_target_filtered=None if elbow_position is None else elbow_position.copy(),
            elbow_bent_target=elbow_bent_target,
            elbow_extended_target=elbow_extended_target,
            elbow_joint_target=float(q_initial[elbow_joint_index]),
            motion_stage="raise" if args_cli.auto_safe_route else "teleop",
            stage_xy=position[:2].copy() if args_cli.auto_safe_route else None,
            last_dq=np.zeros(7, dtype=np.float64),
        )

    # Controller setup.
    kp = np.full(len(dof_names), args_cli.support_kp, dtype=np.float64)
    kd = np.full(len(dof_names), args_cli.support_kd, dtype=np.float64)
    efforts = np.full(len(dof_names), args_cli.support_effort_limit, dtype=np.float64)
    rate_limit = np.zeros(len(dof_names), dtype=np.float64)
    for index, name in enumerate(base_names):
        lowered = name.lower()
        if "wheel" in lowered or "caster" in lowered:
            kp[index] = 0.0
            kd[index] = 4.0
            efforts[index] = 10.0
    for runtime in runtimes.values():
        for index, name in zip(runtime.joint_indices, runtime.joint_names):
            number = joint_number(name)
            wrist = number is not None and number >= 4
            elbow_joint = number == 3
            shoulder = number is not None and number <= 2
            if shoulder:
                kp[index] = args_cli.shoulder_kp
                kd[index] = args_cli.arm_kd
                efforts[index] = args_cli.shoulder_effort_limit
                rate_limit[index] = args_cli.shoulder_rate_limit
            else:
                kp[index] = args_cli.arm_kp * (0.60 if wrist else (0.90 if elbow_joint else 1.0))
                kd[index] = args_cli.arm_kd * (0.65 if wrist else 1.0)
                efforts[index] = args_cli.arm_effort_limit * (0.60 if wrist else (0.90 if elbow_joint else 1.0))
                rate_limit[index] = args_cli.wrist_rate_limit if wrist else args_cli.arm_rate_limit
    all_gripper_names = left_gripper_names + right_gripper_names
    for name in all_gripper_names:
        index = name_to_index[name]
        kp[index] = args_cli.gripper_kp
        kd[index] = args_cli.gripper_kd
        efforts[index] = args_cli.gripper_effort_limit
        rate_limit[index] = args_cli.gripper_rate_limit

    controller = robot.get_articulation_controller()
    controller.switch_control_mode("position")
    controller.set_gains(kps=kp, kds=kd)
    controller.set_max_efforts(efforts, joint_indices=all_indices)

    left_open_override = parse_optional_float_list(
        args_cli.left_gripper_open_targets, len(left_gripper_names), "--left-gripper-open-targets"
    )
    left_close_override = parse_optional_float_list(
        args_cli.left_gripper_close_targets, len(left_gripper_names), "--left-gripper-close-targets"
    )
    right_open_override = parse_optional_float_list(
        args_cli.right_gripper_open_targets, len(right_gripper_names), "--right-gripper-open-targets"
    )
    right_close_override = parse_optional_float_list(
        args_cli.right_gripper_close_targets, len(right_gripper_names), "--right-gripper-close-targets"
    )
    left_open, left_close = gripper_open_close_targets(
        left_gripper_names,
        name_to_index,
        lower_safe,
        upper_safe,
        q_initial,
        explicit_open=left_open_override,
        explicit_close=left_close_override,
        invert=args_cli.left_gripper_invert,
        fallback_travel=args_cli.gripper_fallback_travel,
    )
    right_open, right_close = gripper_open_close_targets(
        right_gripper_names,
        name_to_index,
        lower_safe,
        upper_safe,
        q_initial,
        explicit_open=right_open_override,
        explicit_close=right_close_override,
        invert=args_cli.right_gripper_invert,
        fallback_travel=args_cli.gripper_fallback_travel,
    )
    gripper_targets = {
        "left": (left_gripper_names, left_open, left_close),
        "right": (right_gripper_names, right_open, right_close),
    }

    mapper = WebcamTaskMapper(
        MapperConfig(
            visibility=args_cli.visibility,
            forward_scale=args_cli.forward_scale,
            lateral_scale=args_cli.lateral_scale,
            vertical_scale=args_cli.vertical_scale,
            wrist_up_image_scale=args_cli.wrist_up_image_scale,
            wrist_up_blend=args_cli.wrist_up_blend,
            wrist_up_deadzone=args_cli.wrist_up_deadzone,
            elbow_forward_scale=args_cli.elbow_forward_scale,
            elbow_lateral_scale=args_cli.elbow_lateral_scale,
            elbow_vertical_scale=args_cli.elbow_vertical_scale,
            human_forward_sign=args_cli.human_forward_sign,
            extension_forward_scale=args_cli.extension_forward_scale,
            filter_alpha=args_cli.filter_alpha,
            palm_filter_alpha=args_cli.palm_filter_alpha,
            left_wrist_signs=tuple(parse_float_triplet(args_cli.left_wrist_signs, "--left-wrist-signs")),
            right_wrist_signs=tuple(parse_float_triplet(args_cli.right_wrist_signs, "--right-wrist-signs")),
            pinch_deadzone=args_cli.pinch_deadzone,
            pinch_full_close=args_cli.pinch_full_close,
            pinch_gamma=args_cli.pinch_gamma,
            human_elbow_bent_deg=args_cli.human_elbow_bent_deg,
            human_elbow_straight_deg=args_cli.human_elbow_straight_deg,
        )
    )
    dls_config = DlsConfig(
        damping=args_cli.dls_damping,
        position_gain=args_cli.ik_position_gain,
        orientation_weight=args_cli.orientation_weight,
        null_gain=args_cli.null_gain,
        joint_limit_gain=args_cli.joint_limit_gain,
        max_joint_step=args_cli.ik_max_joint_step,
    )
    receiver = UdpPoseReceiver(args_cli.udp_bind, args_cli.udp_port)

    desired = q_initial.copy()
    commanded = q_initial.copy()
    # Start with the grippers open rather than trusting the imported default.
    for side, (names, open_map, _) in gripper_targets.items():
        for name in names:
            desired[name_to_index[name]] = open_map[name]
            commanded[name_to_index[name]] = open_map[name]

    robot.apply_action(ArticulationAction(joint_positions=commanded.copy(), joint_indices=all_indices))

    # The hand orientation task must be allowed to use arm_4~6.  Shoulder motion
    # is supplied explicitly by a human upper-arm posture task below.
    shoulder_joint_weights = np.array([0.42, 0.48, 0.62, 0.92, 1.28, 1.42, 1.58], dtype=np.float64)
    shoulder_feature_gains = parse_float_triplet(args_cli.shoulder_feature_gains, "--shoulder-feature-gains")
    shoulder_signs = {
        "left": parse_float_triplet(args_cli.left_shoulder_signs, "--left-shoulder-signs"),
        "right": parse_float_triplet(args_cli.right_shoulder_signs, "--right-shoulder-signs"),
    }
    shoulder_offsets = {
        "left": np.radians(parse_float_triplet(args_cli.left_shoulder_offset_deg, "--left-shoulder-offset-deg")),
        "right": np.radians(parse_float_triplet(args_cli.right_shoulder_offset_deg, "--right-shoulder-offset-deg")),
    }
    wrist_joint_gains = parse_float_triplet(args_cli.wrist_joint_gains, "--wrist-joint-gains")
    wrist_joint_signs = {
        "left": parse_float_triplet(args_cli.left_wrist_joint_signs, "--left-wrist-joint-signs"),
        "right": parse_float_triplet(args_cli.right_wrist_joint_signs, "--right-wrist-joint-signs"),
    }
    shoulder_axis_limits = np.radians(
        np.abs(parse_float_triplet(args_cli.shoulder_axis_limit_deg, "--shoulder-axis-limit-deg"))
    )
    shoulder_backward_limit = math.radians(max(args_cli.shoulder_backward_limit_deg, 0.0))

    print("=" * 108)
    print(f"[INFO] v17.3 control: teleop={args_cli.teleop_mode}; small motion follows arm first, then starts cup pick")
    print("[INFO] v20.6 Jacobian convention: spatial_rows x selected_arm_joints = 6x7")
    print(
        f"[INFO] any-motion trigger: enabled={not args_cli.disable_any_motion_auto_pick}, "
        f"position={args_cli.motion_trigger_position:.4f}m, "
        f"shoulder={args_cli.motion_trigger_shoulder_deg:.1f}deg, "
        f"elbow={args_cli.motion_trigger_elbow:.4f}, "
        f"wrist={args_cli.motion_trigger_wrist_deg:.1f}deg, "
        f"guard={args_cli.motion_trigger_guard:.2f}s, "
        f"follow_before_pick={args_cli.motion_trigger_follow_time:.2f}s"
    )
    print("[INFO] USD:", usd_path)
    print("[INFO] Articulation root:", root_path)
    print("[INFO] Body names:", [_base_name(name) for name in body_names])
    for side, runtime in runtimes.items():
        print(
            f"[INFO] {side} EE={runtime.ee_name}, Jacobian row={runtime.jacobian_index}, "
            f"joints={runtime.joint_names}"
        )
    print("[INFO] left gripper jaws:", left_gripper_names)
    print("[INFO] right gripper jaws:", right_gripper_names)
    for side, (names, open_map, close_map) in gripper_targets.items():
        print(f"[GRIPPER] {side} targets:", {name: (open_map[name], close_map[name]) for name in names})
    for side, runtime in runtimes.items():
        print(
            f"[ELBOW] {side} arm_3 home={runtime.q_home[3]:+.3f}, "
            f"bent={runtime.elbow_bent_target:+.3f}, extended={runtime.elbow_extended_target:+.3f}"
        )
    print(
        "[INFO] scene furniture:",
        "existing USD table" if not args_cli.add_demo_table else "generated demo table",
        "| cups:", list(demo_objects),
    )
    effective_object_x = args_cli.object_x - max(args_cli.object_forward_adjust, 0.0)
    print(
        f"[INFO] REAL CUP requested position: x={effective_object_x:.3f}, "
        f"right_y={args_cli.object_right_y:.3f}, "
        f"left_y={args_cli.object_left_y:.3f}"
    )
    print(
        f"[INFO] Existing tabletop search: "
        f"enabled={not args_cli.disable_table_surface_detect and not args_cli.add_demo_table}, "
        f"subtree={args_cli.existing_table_prim}, "
        f"final_table_top_z={args_cli.table_top_z:.3f}"
    )
    print(
        f"[INFO] Yellow-cup neighbor placement: "
        f"enabled={not args_cli.disable_existing_cup_anchor}, "
        f"reference={args_cli.existing_cup_prim}, "
        f"direction={args_cli.cup_neighbor_direction}, "
        f"gap={args_cli.cup_neighbor_gap:.3f}m"
    )
    print("[INFO] joint DLS weights:", shoulder_joint_weights.tolist())
    print(f"[INFO] task weights: hand={args_cli.hand_task_weight:.2f}, palm={args_cli.orientation_weight:.2f}, elbow={args_cli.elbow_task_weight:.2f}, shoulder={args_cli.shoulder_posture_weight:.2f}")
    print(f"[INFO] v15 geometric palm mapping: left={args_cli.left_wrist_signs}, right={args_cli.right_wrist_signs}")
    print(f"[INFO] v15 wrist posture: weight={args_cli.wrist_posture_weight:.2f}, window={args_cli.wrist_joint_window_deg} deg")
    print(f"[INFO] v14 stationary hold: target_db=({args_cli.target_deadband_xy:.3f},{args_cli.target_deadband_z:.3f})m, IK_hold={args_cli.ik_position_hold_deadband:.3f}m/{args_cli.ik_orientation_hold_deadband_deg:.1f}deg")
    print(f"[INFO] v14 forward reach: extension_scale={args_cli.extension_forward_scale:.2f}m, object_assist={not args_cli.no_object_reach_assist}, gain={args_cli.reach_assist_gain:.2f}")
    print(f"[INFO] shoulder feature gains={shoulder_feature_gains.tolist()} signs={ {k: v.tolist() for k, v in shoulder_signs.items()} }")
    print(f"[INFO] direct wrist arm_4~6 gains={wrist_joint_gains.tolist()} signs={ {k: v.tolist() for k, v in wrist_joint_signs.items()} }")
    print(f"[INFO] shoulder offsets deg={ {k: np.degrees(v).round(2).tolist() for k, v in shoulder_offsets.items()} }")
    print(
        f"[INFO] v9 shoulder-only fix: axis limits={np.degrees(shoulder_axis_limits).round(1).tolist()}deg, "
        f"backward arm_0 limit={args_cli.shoulder_backward_limit_deg:.1f}deg"
    )
    print(
        f"[INFO] shoulder correction: ref_gain={args_cli.shoulder_reference_gain:.2f}, "
        f"blend={args_cli.shoulder_direct_blend:.2f}, max={args_cli.shoulder_direct_max_step_deg:.2f}deg/step, "
        f"distal_comp={args_cli.shoulder_distal_compensation:.2f}"
    )
    print(f"[INFO] auto minimum-jerk lift={not args_cli.no_auto_lift}, height={args_cli.lift_height:.2f}m, duration={args_cli.lift_duration:.2f}s, wrist-straighten={args_cli.grasp_straighten:.2f}")
    print(f"[INFO] v15 extension+gripper response: filter={args_cli.filter_alpha:.2f}, ee={args_cli.ee_speed:.2f}m/s, shoulder={args_cli.shoulder_rate_limit:.2f}rad/s, arm={args_cli.arm_rate_limit:.2f}rad/s, wrist={args_cli.wrist_rate_limit:.2f}rad/s")
    print(f"[INFO] v12 wrist-up priority: image_scale={args_cli.wrist_up_image_scale:.2f}, blend={args_cli.wrist_up_blend:.2f}, z_speed={args_cli.wrist_up_speed:.2f}m/s, z_weight={args_cli.wrist_up_task_weight:.2f}")
    print(
        f"[INFO] v15.2 overhead shoulder: workspace_z={args_cli.workspace_z:.2f}m, "
        f"axis_limits={args_cli.shoulder_axis_limit_deg}deg, "
        f"posture_weight={args_cli.shoulder_posture_weight:.2f}"
    )
    print("[INFO] v20.1 auto tabletop placement + automatic STL grasp; P is manual fallback")
    print(f"[INFO] Mapping mode: anchor={args_cli.anchor_mode}, auto_safe_route={args_cli.auto_safe_route}")
    print("[INFO] Default is direct teleoperation. Only the controlled arm follows the matching human wrist.")
    print("=" * 108)

    last_packet = None
    last_print = 0.0
    active_previous = False
    metrics: dict = {"pose": False, "left_pinch": 0.0, "right_pinch": 0.0}
    last_pinch = {"left": 0.0, "right": 0.0}
    demo_stream = None
    demo_accumulator = 0.0
    if args_cli.record_demo:
        demo_path = Path(args_cli.record_demo).expanduser()
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        demo_stream = demo_path.open("a", encoding="utf-8")
        print(f"[IMITATION] Recording JSONL demonstrations to {demo_path}")

    def update_grasp_assist(
        runtime: ArmRuntime,
        pinch: float,
        ee_position: np.ndarray,
        ee_orientation: np.ndarray,
        q_arm: np.ndarray,
    ) -> None:
        """Latch a nearby object and start a smooth orientation-locked lift."""
        obj = runtime.object_handle
        if obj is None or args_cli.no_grasp_assist:
            return
        obj_position, obj_orientation = object_pose(obj)
        if not runtime.attached:
            distance = float(np.linalg.norm(obj_position - ee_position))
            distance_limit = args_cli.pick_attach_distance if runtime.motion_stage == "pick_close" else args_cli.grasp_distance
            close_ready = runtime.motion_stage != "pick_close" or runtime.phase_elapsed >= 0.35 * args_cli.pick_close_duration
            if pinch >= args_cli.grasp_close_threshold and close_ready and distance <= distance_limit:
                rotation = quat_to_matrix(ee_orientation)
                runtime.object_local_offset = rotation.T @ (obj_position - ee_position)
                runtime.object_orientation = obj_orientation.copy()
                runtime.attached = True
                # Lock the commanded palm orientation, not the possibly lagging
                # measured quaternion. This prevents a bent wrist from continuing
                # to rotate while the object is lifted.
                runtime.grasp_orientation_locked = quat_slerp(
                    runtime.orientation_target,
                    runtime.orientation_anchor,
                    float(np.clip(args_cli.grasp_straighten, 0.0, 1.0)),
                )
                runtime.lift_start_position = None
                runtime.lift_goal_position = None
                runtime.lift_start_elbow = None
                runtime.lift_goal_elbow = None
                runtime.grasp_hold_position = ee_position.copy()
                runtime.phase_elapsed = 0.0
                runtime.motion_stage = "grasp_hold"
                print(
                    f"[GRASP] {runtime.side} attached target STL at "
                    f"{distance:.3f}m; holding grasp pose"
                )
        elif runtime.motion_stage == "teleop" and pinch <= args_cli.grasp_release_threshold:
            runtime.attached = False
            runtime.object_local_offset = None
            runtime.object_orientation = None
            runtime.grasp_orientation_locked = None
            runtime.lift_start_position = None
            runtime.lift_goal_position = None
            runtime.lift_start_elbow = None
            runtime.lift_goal_elbow = None
            runtime.phase_elapsed = 0.0
            runtime.motion_stage = "teleop"
            runtime.target_raw = ee_position.copy()
            runtime.target_filtered = ee_position.copy()
            runtime.orientation_target = ee_orientation.copy()
            runtime.shoulder_target = q_arm[:3].copy()
            print(f"[GRASP] {runtime.side} released object; press C if a new neutral pose is needed")
        if runtime.attached and runtime.object_local_offset is not None:
            rotation = quat_to_matrix(ee_orientation)
            position = ee_position + rotation @ runtime.object_local_offset
            orientation = runtime.object_orientation if runtime.object_orientation is not None else obj_orientation
            set_object_pose(obj, position, orientation)

    def reset_safe_route(runtime: ArmRuntime, ee_position: np.ndarray) -> None:
        if args_cli.auto_safe_route:
            runtime.motion_stage = "raise"
            runtime.stage_xy = ee_position[:2].copy()
        else:
            runtime.motion_stage = "teleop"
            runtime.stage_xy = None

    def collision_safe_target(runtime: ArmRuntime, requested: np.ndarray, ee_position: np.ndarray) -> np.ndarray:
        """Apply a tabletop floor; autonomous routing is opt-in."""
        target = np.asarray(requested, dtype=np.float64).copy()
        # The grasp/lift state machine owns the target until release.
        if runtime.motion_stage == "grasp_hold":
            return target
        if args_cli.no_demo_table or runtime.object_handle is None:
            runtime.motion_stage = "teleop"
            return target

        # Direct mode preserves webcam motion instead of replacing it with a
        # hidden raise/translate waypoint. Only the EE target is kept above the
        # tabletop; elbow clearance remains a null-space task below.
        if not args_cli.auto_safe_route:
            runtime.motion_stage = "teleop"
            target[2] = max(target[2], args_cli.table_top_z + args_cli.min_ee_clearance)
            return target

        object_position, _ = object_pose(runtime.object_handle)
        grasp_floor = args_cli.table_top_z + 0.5 * args_cli.cup_height + args_cli.table_clearance
        safe_z = args_cli.table_top_z + max(args_cli.safe_hover_height, args_cli.approach_height)
        target[2] = max(target[2], grasp_floor)

        if runtime.attached:
            runtime.motion_stage = "carry"
            target[2] = max(target[2], grasp_floor + 0.06)
            return target

        if runtime.stage_xy is None:
            runtime.stage_xy = ee_position[:2].copy()

        if runtime.motion_stage == "raise":
            waypoint = np.array([runtime.stage_xy[0], runtime.stage_xy[1], safe_z], dtype=np.float64)
            if ee_position[2] >= safe_z - 0.025:
                runtime.motion_stage = "translate"
            return waypoint

        if runtime.motion_stage == "translate":
            waypoint = np.array([target[0], target[1], safe_z], dtype=np.float64)
            horizontal_error = float(np.linalg.norm(ee_position[:2] - target[:2]))
            if horizontal_error <= args_cli.descent_radius and ee_position[2] >= safe_z - 0.04:
                runtime.motion_stage = "teleop"
            return waypoint

        horizontal_motion = float(np.linalg.norm(target[:2] - ee_position[:2]))
        if runtime.motion_stage == "teleop" and horizontal_motion > args_cli.descent_radius * 1.8 and ee_position[2] < safe_z - 0.05:
            reset_safe_route(runtime, ee_position)
            return np.array([ee_position[0], ee_position[1], safe_z], dtype=np.float64)
        return target

    def elbow_clearance_velocity(runtime: ArmRuntime, jacobians: np.ndarray) -> tuple[np.ndarray, float | None]:
        secondary = np.zeros(7, dtype=np.float64)
        if runtime.elbow is None or runtime.elbow_jacobian_index is None or args_cli.no_demo_table:
            return secondary, None
        elbow_position, _ = rigid_pose(runtime.elbow)
        minimum_z = args_cli.table_top_z + args_cli.elbow_clearance
        error = max(0.0, minimum_z - float(elbow_position[2]))
        if error <= 0.0:
            return secondary, float(elbow_position[2])
        J_elbow_spatial = select_arm_spatial_jacobian(
            jacobians,
            runtime.elbow_jacobian_index,
            runtime.joint_indices,
        )
        J_elbow_linear = J_elbow_spatial[:3, :]
        z_row = np.asarray(
            J_elbow_linear[2, :],
            dtype=np.float64,
        ).reshape(-1)

        if z_row.shape != secondary.shape:
            raise RuntimeError(
                "Elbow-clearance Jacobian row has the wrong size: "
                f"z_row={z_row.shape}, secondary={secondary.shape}, "
                f"J_elbow={J_elbow_linear.shape}"
            )

        denominator = float(z_row @ z_row + 0.02)
        secondary = (
            secondary
            + args_cli.elbow_avoid_gain
            * error
            * z_row
            / denominator
        )
        secondary = np.clip(secondary, -0.18, 0.18)
        return secondary, float(elbow_position[2])

    def shoulder_task_boost(J_arm: np.ndarray, position_error: np.ndarray) -> np.ndarray:
        correction = np.zeros(7, dtype=np.float64)
        J_shoulder = np.asarray(J_arm[:3, :3], dtype=np.float64)
        if np.linalg.norm(J_shoulder) < 1.0e-8:
            return correction
        error = clip_norm(position_error, 0.06)
        regularized = J_shoulder @ J_shoulder.T + 0.035 ** 2 * np.eye(3)
        try:
            shoulder_step = J_shoulder.T @ np.linalg.solve(regularized, error)
        except np.linalg.LinAlgError:
            shoulder_step = J_shoulder.T @ np.linalg.lstsq(regularized, error, rcond=None)[0]
        correction[:3] = np.clip(args_cli.shoulder_boost * shoulder_step, -0.035, 0.035)
        return correction

    def _smoothstep01(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    def _deadband_cartesian(candidate: np.ndarray, held: np.ndarray) -> np.ndarray:
        """Schmitt-style target hold that rejects frame-to-frame MediaPipe jitter."""
        output = np.asarray(candidate, dtype=np.float64).copy()
        reference = np.asarray(held, dtype=np.float64)
        if float(np.linalg.norm(output[:2] - reference[:2])) < max(args_cli.target_deadband_xy, 0.0):
            output[:2] = reference[:2]
        if abs(float(output[2] - reference[2])) < max(args_cli.target_deadband_z, 0.0):
            output[2] = reference[2]
        return output

    def _object_reach_assist(
        runtime: ArmRuntime,
        requested: np.ndarray,
        elbow_extension: float,
        pinch: float,
    ) -> tuple[np.ndarray, float]:
        """Gently align the hand target with the cup when the arm is intentionally extended."""
        target = np.asarray(requested, dtype=np.float64).copy()
        if args_cli.no_object_reach_assist or runtime.object_handle is None or runtime.attached:
            return target, 0.0
        object_position, _ = object_pose(runtime.object_handle)
        horizontal_distance = float(np.linalg.norm(target[:2] - object_position[:2]))
        if horizontal_distance > max(args_cli.reach_assist_radius, 1.0e-3):
            return target, 0.0
        threshold = float(np.clip(args_cli.reach_assist_extension, 0.0, 0.98))
        extension_gate = _smoothstep01((float(elbow_extension) - threshold) / max(1.0 - threshold, 1.0e-3))
        pinch_gate = 0.30 * _smoothstep01((float(pinch) - 0.20) / 0.55)
        proximity_gate = 1.0 - _smoothstep01(horizontal_distance / max(args_cli.reach_assist_radius, 1.0e-3))
        blend = float(np.clip(args_cli.reach_assist_gain * max(extension_gate, pinch_gate) * (0.35 + 0.65 * proximity_gate), 0.0, 0.92))
        if blend > 0.0:
            target[:2] = (1.0 - blend) * target[:2] + blend * object_position[:2]
        return target, blend

    def release_and_reset(runtime: ArmRuntime, ee_position: np.ndarray, ee_orientation: np.ndarray, q_arm: np.ndarray) -> None:
        runtime.attached = False
        runtime.object_local_offset = None
        runtime.object_orientation = None
        runtime.grasp_orientation_locked = None
        runtime.lift_start_position = None
        runtime.lift_goal_position = None
        runtime.lift_start_elbow = None
        runtime.lift_goal_elbow = None
        runtime.phase_elapsed = 0.0
        runtime.arm_raise_elapsed = 0.0
        runtime.arm_raise_armed = True
        runtime.motion_guard_remaining = max(args_cli.motion_trigger_guard, 0.0)
        runtime.last_motion_score = 0.0
        runtime.last_motion_source = "none"
        runtime.pick_pending = False
        runtime.pick_pending_elapsed = 0.0
        runtime.pick_posture_reference = None
        runtime.stage_start_position = None
        runtime.stage_goal_position = None
        runtime.basket_drop_object_position = None
        runtime.grasp_hold_position = None
        runtime.auto_grasp_elapsed = 0.0
        runtime.auto_grasp_started = False
        runtime.motion_stage = "teleop"
        runtime.target_raw = ee_position.copy()
        runtime.target_filtered = ee_position.copy()
        runtime.orientation_target = ee_orientation.copy()
        runtime.shoulder_target = q_arm[:3].copy()
        runtime.elbow_joint_target = float(q_arm[3])
        runtime.wrist_joint_target = q_arm[4:7].copy()
        print(f"[RESET] {runtime.side} released/reset to direct joint teleoperation")

    def start_cup_pick(
        runtime: ArmRuntime,
        ee_position: np.ndarray,
        q_arm_now: np.ndarray | None = None,
    ) -> None:
        if runtime.object_handle is None:
            print(f"[PICK] {runtime.side}: no demo cup is available")
            return
        if runtime.motion_stage != "teleop" or runtime.attached:
            return
        runtime.motion_stage = "pick_hover"
        runtime.phase_elapsed = 0.0
        runtime.arm_raise_elapsed = 0.0
        runtime.arm_raise_armed = False
        runtime.motion_guard_remaining = 0.0
        runtime.pick_pending = False
        runtime.pick_pending_elapsed = 0.0
        runtime.attached = False

        # Preserve the actual arm posture at the handoff.  Previous versions
        # forced autonomous IK toward q_home, which pulled the shoulder/elbow
        # backward while the Cartesian task tried to reach the cup.
        if q_arm_now is not None:
            runtime.pick_posture_reference = np.asarray(q_arm_now, dtype=np.float64).copy()
        else:
            runtime.pick_posture_reference = runtime.q_home.copy()

        runtime.grasp_orientation_locked = runtime.orientation_target.copy()
        runtime.elbow_tracking = False
        runtime.target_raw = ee_position.copy()
        runtime.target_filtered = ee_position.copy()
        print(
            f"[PICK] {runtime.side}: direct following handoff complete; "
            "hover -> descend -> close -> hold"
        )

    def update_cup_pick_target(runtime: ArmRuntime, dt: float, ee_position: np.ndarray, ee_orientation: np.ndarray) -> None:
        if runtime.motion_stage not in ("pick_hover", "pick_descend", "pick_close"):
            return
        if runtime.object_handle is None:
            runtime.motion_stage = "teleop"
            return
        object_position, _ = object_pose(runtime.object_handle)
        grasp_offset=np.asarray(TASK_SCENE_INFO.get("grasp_offset",np.zeros(3)),dtype=np.float64).reshape(3)
        grasp=object_position+grasp_offset
        grasp[2]=object_position[2]+args_cli.pick_grasp_z_offset
        hover=grasp+np.array([0.0,0.0,args_cli.pick_pregrasp_height],dtype=np.float64)
        runtime.phase_elapsed += dt
        runtime.orientation_target = runtime.orientation_anchor.copy()
        if runtime.motion_stage == "pick_hover":
            goal = hover
        else:
            goal = grasp
        delta = goal - runtime.target_filtered
        runtime.target_raw = goal.copy()
        runtime.target_filtered += clip_norm(delta, args_cli.ee_speed * dt)
        pos_error = float(np.linalg.norm(goal - ee_position))
        ori_error = math.degrees(float(np.linalg.norm(quat_error_rotvec(runtime.orientation_target, ee_orientation))))
        reached = pos_error <= args_cli.pick_position_tolerance and ori_error <= args_cli.pick_orientation_tolerance_deg
        timed_out = runtime.phase_elapsed >= args_cli.pick_stage_timeout
        if runtime.motion_stage == "pick_hover" and (reached or timed_out):
            runtime.motion_stage = "pick_descend"
            runtime.phase_elapsed = 0.0
            print(f"[PICK] {runtime.side}: descend (hover error={pos_error:.3f}m)")
        elif runtime.motion_stage == "pick_descend" and (reached or timed_out):
            runtime.motion_stage = "pick_close"
            runtime.phase_elapsed = 0.0
            print(f"[PICK] {runtime.side}: close gripper (grasp error={pos_error:.3f}m)")


    def _object_offset_world(runtime: ArmRuntime) -> np.ndarray:
        if runtime.object_local_offset is None:
            return np.zeros(3, dtype=np.float64)
        orientation = (
            runtime.grasp_orientation_locked
            if runtime.grasp_orientation_locked is not None
            else runtime.orientation_target
        )
        return quat_to_matrix(orientation) @ runtime.object_local_offset

    def _begin_cartesian_stage(
        runtime: ArmRuntime,
        stage: str,
        start_position: np.ndarray,
        goal_position: np.ndarray,
    ) -> None:
        runtime.motion_stage = stage
        runtime.phase_elapsed = 0.0
        runtime.stage_start_position = np.asarray(
            start_position,
            dtype=np.float64,
        ).copy()
        runtime.stage_goal_position = np.asarray(
            goal_position,
            dtype=np.float64,
        ).copy()

    def _advance_minimum_jerk_stage(
        runtime: ArmRuntime,
        dt: float,
        duration: float,
    ) -> bool:
        if (
            runtime.stage_start_position is None
            or runtime.stage_goal_position is None
        ):
            return True
        runtime.phase_elapsed += dt
        tau = runtime.phase_elapsed / max(duration, 1.0e-3)
        blend = minimum_jerk(tau)
        runtime.target_raw = (
            (1.0 - blend) * runtime.stage_start_position
            + blend * runtime.stage_goal_position
        )
        runtime.target_filtered = runtime.target_raw.copy()
        if runtime.grasp_orientation_locked is not None:
            runtime.orientation_target = (
                runtime.grasp_orientation_locked.copy()
            )
        return tau >= 1.0

    def physics_step(dt: float):
        nonlocal last_packet, last_print, active_previous, metrics, desired, commanded, last_pinch, demo_accumulator
        dt = max(float(dt), 1.0e-4)
        newest = receiver.poll()
        if newest is not None:
            last_packet = newest
        active = last_packet is not None and receiver.age() <= args_cli.watchdog and bool(last_packet.get("enabled", True))

        q = as_numpy(robot.get_joint_positions(joint_indices=all_indices)).astype(np.float64)
        qd = as_numpy(robot.get_joint_velocities(joint_indices=all_indices)).astype(np.float64)
        jacobians = get_jacobian_tensor(robot)
        current_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {
            side: rigid_pose(runtime.ee) for side, runtime in runtimes.items()
        }

        if newest is not None and bool(newest.get("reset_request", False)):
            for side, runtime in runtimes.items():
                ee_position, ee_orientation = current_poses[side]
                release_and_reset(runtime, ee_position, ee_orientation, q[runtime.joint_indices])
                last_pinch[side] = 0.0
        if newest is not None and bool(newest.get("pick_request", False)):
            for side, runtime in runtimes.items():
                if side in active_sides:
                    start_cup_pick(
                        runtime,
                        current_poses[side][0],
                        q[runtime.joint_indices],
                    )
        if not args_cli.disable_auto_grasp:
            for side, runtime in runtimes.items():
                if side not in active_sides or runtime.auto_grasp_started or runtime.object_handle is None:
                    continue
                if runtime.motion_stage != "teleop":
                    continue
                runtime.auto_grasp_elapsed += dt
                if runtime.auto_grasp_elapsed >= max(args_cli.auto_grasp_delay,0.0):
                    runtime.auto_grasp_started=True
                    print(f"[AUTO GRASP] {side}: starting after {runtime.auto_grasp_elapsed:.2f}s")
                    start_cup_pick(runtime,current_poses[side][0],q[runtime.joint_indices])

        if active:
            anchors: dict[str, np.ndarray] = {}
            orientation_anchors: dict[str, np.ndarray] = {}
            shoulder_anchors: dict[str, np.ndarray] = {}
            for side, runtime in runtimes.items():
                anchors[side] = runtime.anchor
                orientation_anchors[side] = runtime.orientation_anchor
                shoulder_anchors[side] = runtime.shoulder_anchor
                if runtime.elbow_anchor is not None:
                    anchors[f"{side}_elbow"] = runtime.elbow_anchor
            cartesian_targets, metrics = mapper.update(
                last_packet, anchors, orientation_anchors, shoulder_anchors
            )
            if bool(metrics.get("calibrated", False)):
                for runtime in runtimes.values():
                    runtime.arm_raise_elapsed = 0.0
                    runtime.arm_raise_armed = True
                    runtime.last_wrist_up_delta = 0.0
                    runtime.last_shoulder_elevation_delta = 0.0
                    runtime.motion_guard_remaining = max(args_cli.motion_trigger_guard, 0.0)
                    runtime.last_motion_score = 0.0
                    runtime.last_motion_source = "none"
                    runtime.pick_pending = False
                    runtime.pick_pending_elapsed = 0.0
                    runtime.pick_posture_reference = None
                print(
                    "[AUTO PICK] calibrated; after the short guard, "
                    "any small controlled-arm motion starts cup pick"
                )
            # Pinch and keyboard gripper overrides are independent of body-pose
            # validity.  Previous code updated these only when pose=True, so the
            # gripper could remain frozen whenever a shoulder/wrist landmark was
            # briefly missing.
            for side in ("left", "right"):
                key = f"{side}_pinch"
                if key in metrics:
                    last_pinch[side] = float(np.clip(metrics[key], 0.0, 1.0))

            # v17.2: any intentional movement triggers cup pick.
            #
            # No "arm raised" classification is used.  After C calibration and a
            # short guard interval, any one of the following starts the fixed
            # cup-pick state machine:
            #   - wrist Cartesian displacement
            #   - shoulder retarget angle displacement
            #   - elbow extension/flexion change
            #   - palm/wrist rotation
            #
            # Each signal is normalized by a very small threshold.  The maximum
            # normalized value is the motion score.
            if (args_cli.disable_auto_grasp and not args_cli.disable_any_motion_auto_pick and metrics.get("pose", False)):
                position_threshold = max(args_cli.motion_trigger_position, 1.0e-6)
                shoulder_threshold = math.radians(
                    max(args_cli.motion_trigger_shoulder_deg, 1.0e-3)
                )
                elbow_threshold = max(args_cli.motion_trigger_elbow, 1.0e-6)
                wrist_threshold = math.radians(
                    max(args_cli.motion_trigger_wrist_deg, 1.0e-3)
                )
                hold_time = max(args_cli.motion_trigger_hold, 0.0)

                for side, runtime in runtimes.items():
                    if side not in active_sides:
                        continue

                    runtime.motion_guard_remaining = max(
                        0.0, runtime.motion_guard_remaining - dt
                    )

                    if runtime.motion_stage != "teleop" or runtime.attached:
                        runtime.arm_raise_elapsed = 0.0
                        continue
                    if not runtime.arm_raise_armed:
                        continue
                    if runtime.motion_guard_remaining > 0.0:
                        runtime.arm_raise_elapsed = 0.0
                        continue

                    wrist_target = np.asarray(
                        cartesian_targets.get(side, runtime.anchor),
                        dtype=np.float64,
                    ).reshape(3)
                    wrist_displacement = float(
                        np.linalg.norm(wrist_target - runtime.anchor)
                    )

                    shoulder_delta = np.asarray(
                        cartesian_targets.get(
                            f"{side}_shoulder_delta",
                            np.zeros(3, dtype=np.float64),
                        ),
                        dtype=np.float64,
                    ).reshape(3)
                    shoulder_displacement = float(np.linalg.norm(shoulder_delta))

                    elbow_displacement = abs(
                        float(
                            cartesian_targets.get(
                                f"{side}_elbow_extension_delta", 0.0
                            )
                        )
                    )

                    wrist_rotation = np.asarray(
                        cartesian_targets.get(
                            f"{side}_wrist_joint_delta",
                            np.zeros(3, dtype=np.float64),
                        ),
                        dtype=np.float64,
                    ).reshape(3)
                    wrist_rotation_norm = float(np.linalg.norm(wrist_rotation))

                    normalized = {
                        "wrist_pos": wrist_displacement / position_threshold,
                        "shoulder": shoulder_displacement / shoulder_threshold,
                        "elbow": elbow_displacement / elbow_threshold,
                        "wrist_rot": wrist_rotation_norm / wrist_threshold,
                    }
                    source, score = max(normalized.items(), key=lambda item: item[1])
                    runtime.last_motion_score = float(score)
                    runtime.last_motion_source = source
                    runtime.last_wrist_up_delta = float(
                        metrics.get(f"{side}_wrist_up_delta", 0.0)
                    )
                    runtime.last_shoulder_elevation_delta = float(shoulder_delta[0])

                    if runtime.pick_pending:
                        # Keep normal direct joint retargeting active so the
                        # operator can actually see arm_0~6 respond before the
                        # deterministic cup-pick controller takes ownership.
                        runtime.pick_pending_elapsed += dt
                        if runtime.pick_pending_elapsed >= max(
                            args_cli.motion_trigger_follow_time, 0.0
                        ):
                            print(
                                f"[AUTO PICK] {side} direct-follow window complete "
                                f"({runtime.pick_pending_elapsed:.2f}s); starting cup pick"
                            )
                            start_cup_pick(
                                runtime,
                                current_poses[side][0],
                                q[runtime.joint_indices],
                            )
                    elif score >= 1.0:
                        runtime.arm_raise_elapsed += dt
                        if runtime.arm_raise_elapsed >= hold_time:
                            runtime.pick_pending = True
                            runtime.pick_pending_elapsed = 0.0
                            runtime.arm_raise_elapsed = 0.0
                            print(
                                f"[AUTO PICK] {side} movement detected; "
                                f"following arm joints first: source={source}, score={score:.2f}, "
                                f"wrist={wrist_displacement:.4f}m, "
                                f"shoulder={math.degrees(shoulder_displacement):.2f}deg, "
                                f"elbow={elbow_displacement:.4f}, "
                                f"wrist_rot={math.degrees(wrist_rotation_norm):.2f}deg"
                            )
                    else:
                        runtime.arm_raise_elapsed = 0.0
            else:
                for runtime in runtimes.values():
                    runtime.arm_raise_elapsed = 0.0
            if metrics.get("calibrated", False):
                print(
                    "[CALIBRATION] Human wrist, elbow, shoulder plane, and palm captured; "
                    "robot position/orientation anchors reset."
                )
                for side, runtime in runtimes.items():
                    current_position, current_orientation = current_poses[side]
                    runtime.anchor = current_position.copy()
                    runtime.target_raw = current_position.copy()
                    runtime.target_filtered = current_position.copy()
                    runtime.orientation_anchor = current_orientation.copy()
                    runtime.orientation_target = current_orientation.copy()
                    runtime.q_home = q[runtime.joint_indices].copy()
                    runtime.shoulder_anchor = runtime.q_home[:3].copy()
                    runtime.shoulder_target = runtime.shoulder_anchor.copy()
                    runtime.elbow_joint_target = float(runtime.q_home[3])
                    runtime.wrist_joint_target = runtime.q_home[4:7].copy()
                    runtime.human_elbow_extension = float(metrics.get(f"{side}_elbow_extension", -1.0))
                    cartesian_targets[side] = runtime.anchor.copy()
                    cartesian_targets[f"{side}_orientation"] = runtime.orientation_anchor.copy()
                    cartesian_targets[f"{side}_shoulder_delta"] = np.zeros(3, dtype=np.float64)
                    cartesian_targets[f"{side}_wrist_joint_delta"] = np.zeros(3, dtype=np.float64)
                    if runtime.elbow is not None:
                        elbow_position, _ = rigid_pose(runtime.elbow)
                        runtime.elbow_anchor = elbow_position.copy()
                        runtime.elbow_target_raw = elbow_position.copy()
                        runtime.elbow_target_filtered = elbow_position.copy()
                        cartesian_targets[f"{side}_elbow"] = runtime.elbow_anchor.copy()
                    reset_safe_route(runtime, current_position)

            if metrics.get("pose", False):
                for side, runtime in runtimes.items():
                    if side not in active_sides or side not in cartesian_targets:
                        continue
                    # Once grasped, the state machine owns position and orientation
                    # until the object is released.
                    if runtime.motion_stage != "teleop":
                        continue

                    requested = cartesian_targets[side].copy()
                    current_extension = float(
                        cartesian_targets.get(f"{side}_elbow_extension", runtime.human_elbow_extension)
                    )
                    requested, runtime.reach_assist_blend = _object_reach_assist(
                        runtime, requested, current_extension, last_pinch[side]
                    )
                    low = runtime.anchor - np.array([args_cli.workspace_x, args_cli.workspace_y, args_cli.workspace_z])
                    high = runtime.anchor + np.array([args_cli.workspace_x, args_cli.workspace_y, args_cli.workspace_z])
                    if not args_cli.no_demo_table:
                        low[2] = max(low[2], args_cli.table_top_z + args_cli.table_clearance)
                    requested = np.clip(requested, low, high)
                    requested = collision_safe_target(runtime, requested, current_poses[side][0])
                    runtime.target_raw = _deadband_cartesian(requested, runtime.target_raw)
                    target_delta = runtime.target_raw - runtime.target_filtered
                    xy_step = clip_norm(target_delta[:2], args_cli.ee_speed * dt)
                    z_step = float(np.clip(target_delta[2], -args_cli.wrist_up_speed * dt, args_cli.wrist_up_speed * dt))
                    runtime.target_filtered[:2] += xy_step
                    runtime.target_filtered[2] += z_step

                    orientation_key = f"{side}_orientation"
                    if orientation_key in cartesian_targets:
                        requested_orientation = limit_quat_step(
                            runtime.orientation_anchor,
                            np.asarray(cartesian_targets[orientation_key], dtype=np.float64),
                            math.radians(args_cli.max_palm_angle_deg),
                        )
                        orientation_change = float(
                            np.linalg.norm(quat_error_rotvec(requested_orientation, runtime.orientation_target))
                        )
                        if orientation_change >= math.radians(max(args_cli.orientation_deadband_deg, 0.0)):
                            runtime.orientation_target = limit_quat_step(
                                runtime.orientation_target,
                                requested_orientation,
                                args_cli.orientation_speed * dt,
                            )

                    shoulder_key = f"{side}_shoulder_delta"
                    if shoulder_key in cartesian_targets:
                        shoulder_delta = np.asarray(cartesian_targets[shoulder_key], dtype=np.float64)
                        deadzone = math.radians(max(args_cli.shoulder_deadzone_deg, 0.0))
                        shoulder_delta = np.where(np.abs(shoulder_delta) < deadzone, 0.0, shoulder_delta)
                        shoulder_delta = np.clip(
                            shoulder_delta,
                            -math.radians(args_cli.max_shoulder_delta_deg),
                            math.radians(args_cli.max_shoulder_delta_deg),
                        )
                        # v9 shoulder-only correction.  The RB-Y1 arm_0 axis is opposite
                        # to the MediaPipe shoulder-elevation sign in this USD.  The default
                        # shoulder sign therefore flips arm_0.  We also use asymmetric arm_0
                        # limits so a noisy pose cannot fold the upper arm behind the torso,
                        # while arm_1/2 retain symmetric side/twist motion.
                        mapped_shoulder_delta = shoulder_signs[side] * shoulder_feature_gains * shoulder_delta
                        mapped_shoulder_delta[0] = np.clip(
                            mapped_shoulder_delta[0],
                            -shoulder_axis_limits[0],
                            min(shoulder_axis_limits[0], shoulder_backward_limit),
                        )
                        mapped_shoulder_delta[1] = np.clip(
                            mapped_shoulder_delta[1], -shoulder_axis_limits[1], shoulder_axis_limits[1]
                        )
                        mapped_shoulder_delta[2] = np.clip(
                            mapped_shoulder_delta[2], -shoulder_axis_limits[2], shoulder_axis_limits[2]
                        )
                        requested_shoulder = runtime.shoulder_anchor + shoulder_offsets[side] + mapped_shoulder_delta
                        requested_shoulder = np.clip(
                            requested_shoulder,
                            lower_safe[runtime.joint_indices[:3]],
                            upper_safe[runtime.joint_indices[:3]],
                        )
                        shoulder_request_error = requested_shoulder - runtime.shoulder_target
                        shoulder_target_deadband = math.radians(max(args_cli.shoulder_target_deadband_deg, 0.0))
                        shoulder_request_error = np.where(
                            np.abs(shoulder_request_error) < shoulder_target_deadband,
                            0.0,
                            shoulder_request_error,
                        )
                        max_shoulder_step = args_cli.shoulder_rate_limit * dt
                        runtime.shoulder_target += np.clip(
                            shoulder_request_error,
                            -max_shoulder_step,
                            max_shoulder_step,
                        )

                    wrist_joint_key = f"{side}_wrist_joint_delta"
                    if wrist_joint_key in cartesian_targets:
                        wrist_delta = np.asarray(cartesian_targets[wrist_joint_key], dtype=np.float64)
                        wrist_delta = wrist_joint_signs[side] * wrist_joint_gains * wrist_delta
                        wrist_delta = np.clip(wrist_delta, -wrist_windows, wrist_windows)
                        requested_wrist = runtime.q_home[4:7] + wrist_delta
                        requested_wrist = np.clip(
                            requested_wrist,
                            lower_safe[runtime.joint_indices[4:7]],
                            upper_safe[runtime.joint_indices[4:7]],
                        )
                        wrist_error = requested_wrist - runtime.wrist_joint_target
                        joint_db = math.radians(max(args_cli.joint_retarget_deadband_deg, 0.0))
                        wrist_error = np.where(np.abs(wrist_error) < joint_db, 0.0, wrist_error)
                        runtime.wrist_joint_target += np.clip(
                            wrist_error,
                            -args_cli.wrist_rate_limit * dt,
                            args_cli.wrist_rate_limit * dt,
                        )

                    elbow_extension_key = f"{side}_elbow_extension_delta"
                    if elbow_extension_key in cartesian_targets:
                        extension_delta = float(cartesian_targets[elbow_extension_key])
                        runtime.human_elbow_extension = float(
                            cartesian_targets.get(f"{side}_elbow_extension", runtime.human_elbow_extension)
                        )
                        full_range = runtime.elbow_extended_target - runtime.elbow_bent_target
                        requested_elbow_joint = runtime.q_home[3] + (
                            args_cli.elbow_extension_gain * extension_delta * full_range
                        )
                        requested_elbow_joint = float(
                            np.clip(
                                requested_elbow_joint,
                                min(runtime.elbow_bent_target, runtime.elbow_extended_target),
                                max(runtime.elbow_bent_target, runtime.elbow_extended_target),
                            )
                        )
                        if abs(requested_elbow_joint - runtime.elbow_joint_target) >= math.radians(
                            max(args_cli.elbow_joint_deadband_deg, 0.0)
                        ):
                            runtime.elbow_joint_target = requested_elbow_joint

                    elbow_key = f"{side}_elbow"
                    runtime.elbow_tracking = False
                    if (
                        runtime.elbow is not None
                        and runtime.elbow_anchor is not None
                        and runtime.elbow_target_filtered is not None
                        and elbow_key in cartesian_targets
                    ):
                        elbow_requested = np.asarray(cartesian_targets[elbow_key], dtype=np.float64).copy()
                        elbow_low = runtime.elbow_anchor - np.array([0.28, 0.28, 0.30], dtype=np.float64)
                        elbow_high = runtime.elbow_anchor + np.array([0.28, 0.28, 0.34], dtype=np.float64)
                        elbow_requested = np.clip(elbow_requested, elbow_low, elbow_high)
                        if not args_cli.no_demo_table:
                            elbow_requested[2] = max(
                                elbow_requested[2], args_cli.table_top_z + args_cli.elbow_clearance
                            )
                        elbow_requested[0] = min(
                            elbow_requested[0], runtime.target_filtered[0] - args_cli.elbow_hand_backoff
                        )
                        if float(np.linalg.norm(elbow_requested - runtime.elbow_target_raw)) >= max(
                            args_cli.elbow_target_deadband, 0.0
                        ):
                            runtime.elbow_target_raw = elbow_requested
                        elbow_step = clip_norm(
                            runtime.elbow_target_raw - runtime.elbow_target_filtered,
                            args_cli.elbow_speed * dt,
                        )
                        runtime.elbow_target_filtered += elbow_step
                        runtime.elbow_tracking = True
        elif active_previous:
            print("[TELEOP] Webcam timeout/pause: holding current Cartesian and palm targets.")

        # Automatic cup-pick phases own the Cartesian target and temporarily
        # suspend direct joint retargeting.
        for side, runtime in runtimes.items():
            ee_position, ee_orientation = current_poses[side]
            update_cup_pick_target(runtime, dt, ee_position, ee_orientation)

        # Grasp-only v20: once attached, freeze the Cartesian pose and keep
        # the gripper closed. There is no lift, transfer, release, or basket.
        for runtime in runtimes.values():
            if runtime.attached and runtime.motion_stage == "grasp_hold":
                if runtime.grasp_hold_position is not None:
                    runtime.target_raw = runtime.grasp_hold_position.copy()
                    runtime.target_filtered = (
                        runtime.grasp_hold_position.copy()
                    )
                if runtime.grasp_orientation_locked is not None:
                    runtime.orientation_target = (
                        runtime.grasp_orientation_locked.copy()
                    )

        desired[:] = q_initial
        for side, runtime in runtimes.items():
            ee_position, ee_orientation = current_poses[side]
            q_arm = q[runtime.joint_indices]
            autonomous_active = runtime.motion_stage in ("pick_hover", "pick_descend", "pick_close", "grasp_hold")
            webcam_active = side in active_sides and active and metrics.get("pose", False)
            joint_retarget_active = webcam_active and args_cli.teleop_mode == "joint" and runtime.motion_stage == "teleop"
            ik_control_active = autonomous_active or (webcam_active and args_cli.teleop_mode == "ik")
            if joint_retarget_active:
                joint_target = np.concatenate(
                    (
                        runtime.shoulder_target,
                        np.array([runtime.elbow_joint_target], dtype=np.float64),
                        runtime.wrist_joint_target,
                    )
                )
                joint_db = math.radians(max(args_cli.joint_retarget_deadband_deg, 0.0))
                joint_error = joint_target - commanded[runtime.joint_indices]
                joint_target = np.where(
                    np.abs(joint_error) < joint_db,
                    commanded[runtime.joint_indices],
                    joint_target,
                )
                desired[runtime.joint_indices] = np.clip(
                    joint_target,
                    lower_safe[runtime.joint_indices],
                    upper_safe[runtime.joint_indices],
                )
                runtime.last_dq = desired[runtime.joint_indices] - q_arm
                runtime.last_shoulder_correction = np.zeros(3, dtype=np.float64)
            elif ik_control_active:
                J_arm = select_arm_spatial_jacobian(
                    jacobians,
                    runtime.jacobian_index,
                    runtime.joint_indices,
                )
                pos_error = runtime.target_filtered - ee_position
                ori_error = quat_error_rotvec(runtime.orientation_target, ee_orientation)
                if not autonomous_active and float(np.linalg.norm(pos_error)) < max(args_cli.ik_position_hold_deadband, 0.0):
                    pos_error[:] = 0.0
                if not autonomous_active and float(np.linalg.norm(ori_error)) < math.radians(
                    max(args_cli.ik_orientation_hold_deadband_deg, 0.0)
                ):
                    ori_error[:] = 0.0
                secondary, elbow_z = elbow_clearance_velocity(runtime, jacobians)
                upward_error = max(0.0, float(pos_error[2]))
                vertical_priority = float(np.clip(upward_error / 0.12, 0.0, 1.0))
                J_vertical = np.zeros_like(J_arm[:3, :])
                J_vertical[2, :] = J_arm[2, :]
                vertical_error = np.array([0.0, 0.0, pos_error[2]], dtype=np.float64)
                position_tasks: list[tuple[np.ndarray, np.ndarray, float]] = [
                    (J_arm[:3, :], pos_error, args_cli.hand_task_weight),
                    (J_vertical, vertical_error, args_cli.wrist_up_task_weight),
                ]
                if (
                    runtime.elbow_tracking
                    and runtime.elbow is not None
                    and runtime.elbow_jacobian_index is not None
                    and runtime.elbow_target_filtered is not None
                ):
                    elbow_position, _ = rigid_pose(runtime.elbow)
                    J_elbow = select_arm_spatial_jacobian(
                        jacobians,
                        runtime.elbow_jacobian_index,
                        runtime.joint_indices,
                    )[:3, :]
                    elbow_error = runtime.elbow_target_filtered - elbow_position
                    if not autonomous_active and float(np.linalg.norm(elbow_error)) < max(
                        args_cli.ik_elbow_hold_deadband, 0.0
                    ):
                        elbow_error[:] = 0.0
                    position_tasks.append((J_elbow, elbow_error, args_cli.elbow_task_weight))

                orientation_task = None
                orientation_weight_effective = (args_cli.autonomous_orientation_weight if autonomous_active else args_cli.orientation_weight * (1.0 - 0.45 * vertical_priority))
                if orientation_weight_effective > 0.0:
                    orientation_task = (J_arm[3:, :], ori_error, orientation_weight_effective)
                autonomous_reference = (
                    runtime.pick_posture_reference
                    if runtime.pick_posture_reference is not None
                    else runtime.q_home
                )
                shoulder_reference = (
                    autonomous_reference[:3]
                    if autonomous_active
                    else runtime.shoulder_target
                )
                shoulder_error = shoulder_reference - q_arm[:3]
                joint_hold_deadband = math.radians(max(args_cli.ik_joint_hold_deadband_deg, 0.0))
                if not autonomous_active:
                    shoulder_error = np.where(np.abs(shoulder_error) < joint_hold_deadband, 0.0, shoulder_error)
                shoulder_task_weight = args_cli.shoulder_posture_weight * (
                    args_cli.autonomous_posture_weight_scale if autonomous_active else 1.0
                )
                # When the hand target rises above the shoulder, use arm_0~2 more
                # strongly instead of asking elbow/wrist joints to fake the lift.
                shoulder_task_weight *= (1.0 + 0.55 * vertical_priority)
                elbow_reference = (
                    float(autonomous_reference[3])
                    if autonomous_active
                    else runtime.elbow_joint_target
                )
                elbow_joint_error = np.array([elbow_reference - q_arm[3]], dtype=np.float64)
                if not autonomous_active and abs(float(elbow_joint_error[0])) < joint_hold_deadband:
                    elbow_joint_error[0] = 0.0
                elbow_angle_weight_effective = args_cli.elbow_angle_weight * (1.0 - 0.45 * vertical_priority)
                if autonomous_active:
                    elbow_angle_weight_effective *= args_cli.autonomous_posture_weight_scale
                wrist_reference = (
                    autonomous_reference[4:7]
                    if autonomous_active
                    else runtime.q_home[4:7]
                )
                wrist_error = wrist_reference - q_arm[4:7]
                wrist_deadband = math.radians(max(args_cli.wrist_posture_deadband_deg, 0.0))
                if not autonomous_active:
                    wrist_error = np.where(np.abs(wrist_error) < wrist_deadband, 0.0, wrist_error)
                wrist_posture_weight_effective = args_cli.wrist_posture_weight * (
                    args_cli.autonomous_posture_weight_scale
                    if autonomous_active
                    else 1.0
                )
                joint_tasks = [
                    (np.array([0, 1, 2], dtype=np.int64), shoulder_error, shoulder_task_weight),
                    (np.array([3], dtype=np.int64), elbow_joint_error, elbow_angle_weight_effective),
                    (np.array([4, 5, 6], dtype=np.int64), wrist_error, wrist_posture_weight_effective),
                ]
                elbow_task_norm = 0.0
                if runtime.elbow_tracking and runtime.elbow is not None and runtime.elbow_target_filtered is not None:
                    elbow_position_for_hold, _ = rigid_pose(runtime.elbow)
                    elbow_task_norm = float(np.linalg.norm(runtime.elbow_target_filtered - elbow_position_for_hold))
                settled = (
                    not autonomous_active
                    and float(np.linalg.norm(pos_error)) <= 1.0e-12
                    and float(np.linalg.norm(ori_error)) <= 1.0e-12
                    and float(np.linalg.norm(shoulder_error)) <= 1.0e-12
                    and abs(float(elbow_joint_error[0])) <= 1.0e-12
                    and float(np.linalg.norm(wrist_error)) <= 1.0e-12
                    and elbow_task_norm < max(args_cli.ik_elbow_hold_deadband, 0.0)
                    and float(np.linalg.norm(secondary)) < 1.0e-5
                )
                if settled:
                    runtime.last_dq = np.zeros(7, dtype=np.float64)
                    runtime.last_shoulder_correction = np.zeros(3, dtype=np.float64)
                    desired[runtime.joint_indices] = commanded[runtime.joint_indices]
                    dq = None
                else:
                    dq = solve_stacked_dls_step(
                        position_tasks,
                        q_arm,
                        runtime.q_home,
                        lower_safe[runtime.joint_indices],
                        upper_safe[runtime.joint_indices],
                        dls_config,
                        joint_weights=shoulder_joint_weights,
                        secondary_velocity=secondary,
                        orientation_task=orientation_task,
                        joint_tasks=joint_tasks,
                    )
                    shoulder_mode_scale = (0.65 if autonomous_active else 1.0) * (1.0 + 0.40 * vertical_priority)
                    dq_before_shoulder = dq.copy()
                    dq = apply_shoulder_reference_correction(
                        dq,
                        shoulder_error,
                        J_arm,
                        reference_gain=args_cli.shoulder_reference_gain * shoulder_mode_scale,
                        blend=args_cli.shoulder_direct_blend * shoulder_mode_scale,
                        max_shoulder_step=math.radians(args_cli.shoulder_direct_max_step_deg),
                        position_weight=args_cli.hand_task_weight,
                        orientation_weight=(orientation_weight_effective if orientation_task is not None else 0.0),
                        damping=args_cli.dls_damping,
                        distal_compensation=args_cli.shoulder_distal_compensation,
                    )
                    runtime.last_shoulder_correction = dq[:3] - dq_before_shoulder[:3]
                    # Guarantee that an upward human wrist command produces an
                    # upward robot end-effector velocity even when shoulder, elbow,
                    # and palm-posture tasks compete for the 7-DoF solution.
                    if upward_error > 1.0e-4:
                        jz = np.asarray(J_arm[2, :], dtype=np.float64)
                        denom = float(jz @ jz + args_cli.dls_damping ** 2)
                        dq_up = (args_cli.wrist_up_direct_gain * upward_error / max(denom, 1.0e-8)) * jz
                        dq_up = clip_norm(dq_up, args_cli.wrist_up_direct_max_step)
                        dq += dq_up
                    elbow_requested_step = np.clip(
                        args_cli.elbow_extension_gain * elbow_joint_error[0],
                        -math.radians(args_cli.elbow_direct_max_step_deg),
                        math.radians(args_cli.elbow_direct_max_step_deg),
                    )
                    elbow_direct_blend = (
                        args_cli.elbow_direct_blend
                        * args_cli.autonomous_posture_weight_scale
                        if autonomous_active
                        else args_cli.elbow_direct_blend
                    )
                    dq[3] = (
                        (1.0 - elbow_direct_blend) * dq[3]
                        + elbow_direct_blend * elbow_requested_step
                    )
                    if args_cli.shoulder_boost > 0.0:
                        dq += shoulder_task_boost(J_arm, pos_error)
                    dq = np.clip(dq, -args_cli.ik_max_joint_step, args_cli.ik_max_joint_step)
                    runtime.last_dq = dq.copy()
                    desired[runtime.joint_indices] = np.clip(
                        q_arm + dq,
                        lower_safe[runtime.joint_indices],
                        upper_safe[runtime.joint_indices],
                    )
            else:
                desired[runtime.joint_indices] = commanded[runtime.joint_indices]

            # On a short webcam dropout, hold the last pinch instead of dropping the object.
            pinch = last_pinch[side]
            if runtime.motion_stage in ("pick_hover", "pick_descend"):
                pinch = 0.0
            elif runtime.motion_stage in ("pick_close", "grasp_hold"):
                pinch = 1.0
            names, open_map, close_map = gripper_targets[side]
            for name in names:
                index = name_to_index[name]
                desired[index] = open_map[name] + pinch * (close_map[name] - open_map[name])
            update_grasp_assist(runtime, pinch, ee_position, ee_orientation, q_arm)

        # Keep support joints at startup pose, but apply slew limits to all commanded joints.
        moving_mask = rate_limit > 0.0
        max_step = rate_limit * dt
        delta = desired - commanded
        commanded[moving_mask] += np.clip(delta[moving_mask], -max_step[moving_mask], max_step[moving_mask])
        commanded[~moving_mask] = desired[~moving_mask]
        commanded = np.clip(commanded, lower_safe, upper_safe)
        too_fast = np.abs(qd) > 7.0
        commanded[too_fast] = q[too_fast]

        robot.apply_action(ArticulationAction(joint_positions=commanded.copy(), joint_indices=all_indices))

        if demo_stream is not None:
            demo_accumulator += dt
            if demo_accumulator >= 1.0 / 30.0:
                demo_accumulator = 0.0
                for side, runtime in runtimes.items():
                    ee_position, ee_orientation = current_poses[side]
                    indices = runtime.joint_indices
                    record = {
                        "timestamp": time.time(),
                        "side": side,
                        "phase": runtime.motion_stage,
                        "pose_valid": bool(metrics.get("pose", False)),
                        "tracking_mode": metrics.get("tracking_mode", ""),
                        "joint_pos": q[indices].tolist(),
                        "joint_vel": qd[indices].tolist(),
                        "joint_command": commanded[indices].tolist(),
                        "shoulder_target": runtime.shoulder_target.tolist(),
                        "ee_pos": ee_position.tolist(),
                        "ee_quat_wxyz": ee_orientation.tolist(),
                        "target_pos": runtime.target_filtered.tolist(),
                        "target_quat_wxyz": runtime.orientation_target.tolist(),
                        "pinch": float(last_pinch[side]),
                        "attached": bool(runtime.attached),
                    }
                    demo_stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                demo_stream.flush()

        now = time.monotonic()
        if now - last_print >= 1.0 / max(args_cli.print_hz, 0.1):
            pieces = [f"active={active}", f"age={receiver.age():.3f}s", f"pose={bool(metrics.get('pose', False))}"]
            for side, runtime in runtimes.items():
                if side in active_sides:
                    pieces.append(
                        f"{side[0].upper()}motion="
                        f"{runtime.last_motion_score:.2f}"
                        f"({runtime.last_motion_source})/"
                        f"{runtime.arm_raise_elapsed:.2f}s "
                        f"guard={runtime.motion_guard_remaining:.2f}s "
                        f"armed={runtime.arm_raise_armed} "
                        f"pending={runtime.pick_pending},"
                        f"{runtime.pick_pending_elapsed:.2f}s"
                    )
            if metrics.get("pose", False):
                pieces.append(f"track={metrics.get('tracking_mode', 'unknown')}")
            else:
                pieces.append(f"reason={metrics.get('reason', 'no valid controlled-arm wrist')}" )
            for side in sorted(active_sides):
                runtime = runtimes[side]
                ee_position, _ = current_poses[side]
                error = float(np.linalg.norm(runtime.target_filtered - ee_position))
                z_error = float(runtime.target_filtered[2] - ee_position[2])
                human_wrist_up = float(metrics.get(f"{side}_wrist_up_delta", 0.0))
                shoulder_deg = np.degrees(runtime.last_dq[:3]) if runtime.last_dq is not None else np.zeros(3)
                shoulder_q = np.degrees(q[runtime.joint_indices[:3]])
                shoulder_cmd = np.degrees(commanded[runtime.joint_indices[:3]])
                shoulder_ref = np.degrees(runtime.shoulder_target)
                shoulder_err = np.degrees(runtime.shoulder_target - q[runtime.joint_indices[:3]])
                shoulder_fix = np.degrees(runtime.last_shoulder_correction) if runtime.last_shoulder_correction is not None else np.zeros(3)
                wrist_q = np.degrees(q[runtime.joint_indices[4:7]])
                wrist_cmd = np.degrees(commanded[runtime.joint_indices[4:7]])
                _, measured_orientation = current_poses[side]
                orientation_error_deg = math.degrees(
                    float(np.linalg.norm(quat_error_rotvec(runtime.orientation_target, measured_orientation)))
                )
                elbow_text = " elbowTrack=False"
                if runtime.elbow is not None:
                    elbow_position, _ = rigid_pose(runtime.elbow)
                    if runtime.elbow_target_filtered is not None:
                        elbow_error_value = float(np.linalg.norm(runtime.elbow_target_filtered - elbow_position))
                        elbow_text = (
                            f" Eerr={elbow_error_value:.3f}m elbowZ={elbow_position[2]:.3f} "
                            f"elbowTrack={runtime.elbow_tracking}"
                        )
                # Gripper diagnostics must be computed per side in this print loop.
                # v11 referenced grip_q/grip_cmd without defining them, which caused
                # a NameError on the first status print and stopped the simulation.
                grip_names, _, _ = gripper_targets[side]
                grip_indices = [name_to_index[name] for name in grip_names]
                grip_q = [float(q[index]) for index in grip_indices]
                grip_cmd = [float(commanded[index]) for index in grip_indices]

                pieces.append(
                    f"{side[0].upper()}err={error:.3f}m stage={runtime.motion_stage} "
                    f"EE=({ee_position[0]:+.2f},{ee_position[1]:+.2f},{ee_position[2]:+.2f}) "
                    f"TGT=({runtime.target_filtered[0]:+.2f},{runtime.target_filtered[1]:+.2f},{runtime.target_filtered[2]:+.2f}) "
                    f"Zerr={z_error:+.3f}m Hup={human_wrist_up:+.3f}m "
                    f"Sdq=[{shoulder_deg[0]:+.2f},{shoulder_deg[1]:+.2f},{shoulder_deg[2]:+.2f}]deg "
                    f"Sq=[{shoulder_q[0]:+.1f},{shoulder_q[1]:+.1f},{shoulder_q[2]:+.1f}] "
                    f"Scmd=[{shoulder_cmd[0]:+.1f},{shoulder_cmd[1]:+.1f},{shoulder_cmd[2]:+.1f}] "
                    f"Sref=[{shoulder_ref[0]:+.1f},{shoulder_ref[1]:+.1f},{shoulder_ref[2]:+.1f}] "
                    f"Serr=[{shoulder_err[0]:+.1f},{shoulder_err[1]:+.1f},{shoulder_err[2]:+.1f}] "
                    f"Sfix=[{shoulder_fix[0]:+.2f},{shoulder_fix[1]:+.2f},{shoulder_fix[2]:+.2f}] "
                    f"Wq=[{wrist_q[0]:+.1f},{wrist_q[1]:+.1f},{wrist_q[2]:+.1f}] "
                    f"Wcmd=[{wrist_cmd[0]:+.1f},{wrist_cmd[1]:+.1f},{wrist_cmd[2]:+.1f}] "
                    f"Oerr={orientation_error_deg:.1f}deg "
                    f"mode={args_cli.teleop_mode} {elbow_text} Hext={runtime.human_elbow_extension:.2f} Reach={runtime.reach_assist_blend:.2f} "
                    f"Ecmd={math.degrees(runtime.elbow_joint_target):+.1f}deg "
                    f"pinch={last_pinch[side]:.2f} Gq={[round(v,3) for v in grip_q]} "
                    f"Gcmd={[round(v,3) for v in grip_cmd]} attached={runtime.attached}"
                )
            print("[RBY1 v16] " + " | ".join(pieces))
            last_print = now
        active_previous = active

    try:
        world.remove_physics_callback("rby1_webcam_dls_ik")
    except Exception:
        pass
    world.add_physics_callback("rby1_webcam_dls_ik", physics_step)
    world.play()
    try:
        while simulation_app.is_running():
            world.step(render=True)
    finally:
        receiver.close()
        if demo_stream is not None:
            demo_stream.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FATAL] RB-Y1 v20.6 Jacobian-safe STL grasp task failed: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
