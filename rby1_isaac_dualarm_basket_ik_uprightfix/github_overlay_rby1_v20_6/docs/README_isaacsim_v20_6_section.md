### Isaac Sim Webcam STL Grasp Test

This branch includes a webcam-based RB-Y1 manipulation test in Isaac Sim.
The current test loads a reconstructed STL object, places it in the robot workspace, and performs an automatic approach-and-grasp motion with a Jacobian-based arm controller.

Main files:

```text
isaac/run_webcam_rby1_ik_table_pick.py   # Isaac Sim task and grasp controller
isaac/rby1_taskspace_ik.py               # RB-Y1 arm/task-space IK utilities
camera/webcam_holistic_rby1_udp.py       # Webcam hand/arm tracker and UDP sender
scripts/07_run_webcam_rby1_ik_pick.sh    # One-command launcher
tools/probe_rby1_camera.py               # Camera device probe
assets/target_stl_mesh.npz               # Processed target mesh
assets/mesh_poisson_clean_mm(1).stl      # Original uploaded STL target
```

Run example:

```bash
cd ~/rby1_ros2_hand_teleop
conda activate env_isaaclab
export ISAACLAB_DIR=$HOME/IsaacLab

./scripts/07_run_webcam_rby1_ik_pick.sh \
  --camera-device "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0" \
  --control-side right \
  --teleop-mode joint \
  --visibility 0.15
```

The v20.6 update fixes NumPy advanced-indexing of the PhysX Jacobian by normalizing all arm Jacobians to a consistent `6 x 7` spatial-Jacobian format before applying IK and elbow-clearance tasks.
