# P7 Photo-Based Real-to-Sim Framework for RB-Y1 Robot

This repository presents a photo-based real-to-sim framework for the RB-Y1 robot.  
The goal of this project is to reconstruct real-world objects and scenes from RGB/RGB-D images, convert them into simulation-ready assets, and use them inside Isaac Sim for robot interaction and data collection.

The final objective is to build a digital twin environment where the RB-Y1 robot can interact with reconstructed real-world objects and generate demonstration data for downstream robot learning tasks.
<p align="center">
<img width="911" height="506" alt="Image" src="https://github.com/user-attachments/assets/e96515cd-02fe-4796-940a-c3707d7bc246" />
</p>
---

## 1. Research Overview

Traditional robot simulators often suffer from a significant sim-to-real gap because manually modeled simulation assets do not fully reflect the geometry and appearance of real-world objects.  
To reduce this gap, this project investigates a photo-based real-to-sim pipeline that converts real-world images into usable simulation assets.

The research focuses on the following goals:

- Capture real-world RGB/RGB-D images of objects and workspace.
- Reconstruct object geometry from captured images.
- Convert reconstructed assets into a simulation-compatible representation.
- Import the generated assets into Isaac Sim.
- Use the RB-Y1 robot for interaction, manipulation, and demonstration data collection.

---

## 2. Research Content

### 2.1 Real-World Object Capture

Real-world objects are captured using multi-view RGB or RGB-D images.  
The captured data are used as input for 3D reconstruction and asset generation.

Example objects include:

- Cup
- Bottle
- Mouse
- Tabletop objects
- Workspace components

<p align="center">
  <img width="955" height="295" alt="Image" src="https://github.com/user-attachments/assets/03d5b4f3-e9a7-4f53-b895-e52eabf7b19f" />
</p>

---

### 2.2 Reconstruction and Asset Generation

The captured images are processed to generate 3D object representations.  
The initial direction considered Gaussian Splatting for photorealistic reconstruction, but the reconstruction quality and image-processing pipeline were not sufficiently clean for direct simulation use. Therefore, the current pipeline focuses on converting reconstructed object geometry into simulation-ready assets.

The generated assets are then verified before importing them into Isaac Sim.

<p align="center">
  <img width="974" height="153" alt="Image" src="https://github.com/user-attachments/assets/a60176b8-f85a-44a6-9b40-955e039fcbbe" />
</p>

---

### 2.3 Isaac Sim Environment Construction

The reconstructed objects are imported into Isaac Sim and arranged inside the robot workspace.  
The RB-Y1 robot is loaded into the environment, and object interaction is tested inside the simulator.

<p align="center">
  <img width="842" height="283" alt="Image" src="https://github.com/user-attachments/assets/4162864e-a297-4399-b6a7-6522a60723a8" />
</p>

---

### 2.4 Robot Interaction and Data Collection

The simulated environment is designed to support robot interaction and data collection.  
In the current stage, the RB-Y1 robot is tested in Isaac Sim with reconstructed objects placed on the table.

Future demonstration data can be collected using:
- RGBcamera-based teleoperation

<p align="center">
  <img width="1260" height="647" alt="Image" src="https://github.com/user-attachments/assets/6bde0744-3961-427f-a48a-64af657ce194" />
</p>

---

## 3. Overall Pipeline

The proposed real-to-sim pipeline is organized as follows:

```text
Real-world objects / workspace
        ↓
Multi-view RGB or RGB-D image capture
        ↓
Image preprocessing and scene reconstruction
        ↓
3D asset generation
        ↓
Asset conversion for simulation
        ↓
Import to Isaac Sim
        ↓
RB-Y1 robot interaction
        ↓
Demonstration data collection

## 4. Isaac Sim + Skeleton Teleoperation Pipeline

In addition to the real-to-sim asset pipeline, this repository also includes a skeleton-based teleoperation test for moving the RB-Y1 robot inside Isaac Sim.

The teleoperation system connects a webcam-based human skeleton tracker to the RB-Y1 robot in Isaac Sim through ROS2 and UDP communication.

```text
Webcam
  ↓
MediaPipe Skeleton Tracking
  ↓
ROS2 Skeleton Topic
  ↓
UDP JSON Bridge
  ↓
Isaac Sim
  ↓
RB-Y1 Robot Motion
```

Detailed data flow:

```text
Webcam
  ↓
MediaPipe ROS2 Node
  ↓
/human/skeleton/arm_hand_stitched
  ↓
ros2_skeleton_udp_bridge.py
  ↓ UDP JSON, 127.0.0.1:50555
rby1_skeleton_teleop_isaac.py
  ↓
RB-Y1 robot arms in Isaac Sim
```

<p align="center">
  <img src="assets/skeleton_teleop_pipeline.png" width="800">
</p>

---

## 5. Four-Terminal Execution Guide

For real-time testing, four terminals are recommended.

```text
┌────────────────────────────┬────────────────────────────┐
│ Terminal 1                 │ Terminal 2                 │
│ MediaPipe Skeleton Node    │ Camera / Skeleton Monitor  │
├────────────────────────────┼────────────────────────────┤
│ Terminal 3                 │ Terminal 4                 │
│ ROS2 → UDP Bridge          │ Isaac Sim RB-Y1 Teleop     │
└────────────────────────────┴────────────────────────────┘
```

---

### Terminal 1: Run MediaPipe Skeleton Node

Before running the skeleton node, make sure that no other camera node is using the webcam.

```bash
pkill -f v4l2_camera_node || true
pkill -f pose_hand_stitched_node.py || true
```

Run the MediaPipe skeleton node:

```bash
conda deactivate

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run human_skeleton_pipeline pose_hand_stitched_node.py
```

This node publishes human skeleton data to:

```text
/human/skeleton/arm_hand_stitched
```

Check whether the skeleton topic is being published:

```bash
ros2 topic hz /human/skeleton/arm_hand_stitched
```

---

### Terminal 2: Check Camera or Skeleton Topic

To check the ROS camera stream:

```bash
source /opt/ros/jazzy/setup.bash

ros2 run rqt_image_view rqt_image_view
```

Select the following topic in the GUI:

```text
/image_raw
```

To check one skeleton message:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 topic echo /human/skeleton/arm_hand_stitched --once
```

If the UDP bridge reports `tracking_ok=False`, check whether the MediaPipe node is actually detecting the human body.

The user should stand approximately 1.5–2 m away from the camera so that the face, shoulders, elbows, wrists, and hands are visible.

---

### Terminal 3: Run ROS2 Skeleton to UDP Bridge

```bash
conda deactivate

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

python3 ~/다운로드/rby1_isaac_dualarm_basket_ik_uprightfix/skeleton_bridge_scripts/ros2_skeleton_udp_bridge.py \
  --topic /human/skeleton/arm_hand_stitched \
  --host 127.0.0.1 \
  --port 50555
```

Expected output:

```text
[INFO] Listening: /human/skeleton/arm_hand_stitched
[INFO] Sending UDP JSON to 127.0.0.1:50555
```

This terminal should remain running while Isaac Sim is active.

---

### Terminal 4: Run Isaac Sim RB-Y1 Teleoperation

```bash
conda activate env_isaaclab
cd ~/IsaacLab

./isaaclab.sh -p ~/다운로드/rby1_isaac_dualarm_basket_ik_uprightfix/rby1_skeleton_teleop_isaac.py \
  --asset v1_1 \
  --udp-host 127.0.0.1 \
  --udp-port 50555 \
  --human-gain 1.25 \
  --gain-x 0.45 \
  --gain-y 1.20 \
  --gain-z 1.60 \
  --workspace-z 0.20 1.95 \
  --filter-alpha 0.18 \
  --max-target-step 0.075 \
  --max-joint-step 0.025 \
  --lambda-dls 0.18 \
  --ik-gain 0.38 \
  --posture-pull 0.22 \
  --wrist-pull 0.45 \
  --center-pull 0.45 \
  --center-y 0.0 \
  --orient-gain 0.45 \
  --orient-weight 0.80 \
  --invert-y \
  --ee-orient-lock
```

If the left and right arms are swapped due to camera mirroring, add:

```bash
--swap-arms
```

If the left-right motion is reversed, add or remove:

```bash
--invert-y
```

---

## 6. Fake Skeleton Test

Before using the real camera, the Isaac-side teleoperation controller can be tested using fake skeleton data.

### Terminal 1: Fake Skeleton Sender

```bash
python3 ~/다운로드/rby1_isaac_dualarm_basket_ik_uprightfix/skeleton_bridge_scripts/fake_skeleton_udp_sender.py \
  --host 127.0.0.1 \
  --port 50555
```

### Terminal 2: Isaac Sim

```bash
conda activate env_isaaclab
cd ~/IsaacLab

./isaaclab.sh -p ~/다운로드/rby1_isaac_dualarm_basket_ik_uprightfix/rby1_skeleton_teleop_isaac.py \
  --asset v1_1 \
  --udp-host 127.0.0.1 \
  --udp-port 50555
```

If the RB-Y1 robot moves with fake skeleton data, the Isaac Sim teleoperation controller and UDP communication are working correctly.

---

## 7. Terminal Layout Recommendation

A four-split terminal layout is recommended for real-time testing.

Install Terminator:

```bash
sudo apt update
sudo apt install terminator -y
```

Run Terminator:

```bash
terminator
```

Useful shortcuts:

```text
Ctrl + Shift + E : vertical split
Ctrl + Shift + O : horizontal split
Ctrl + Shift + W : close current terminal
Alt + Arrow Key  : move between terminals
```

Recommended layout:

```text
Top-left     : MediaPipe skeleton node
Top-right    : Camera or skeleton topic monitor
Bottom-left  : ROS2 skeleton to UDP bridge
Bottom-right : Isaac Sim RB-Y1 teleoperation
```

---

## 8. Troubleshooting

### 8.1 Camera device changed

Sometimes the camera device changes from `/dev/video0` to `/dev/video1`.

Check available devices:

```bash
ls /dev/video*
v4l2-ctl --list-devices
```

Example output:

```text
USB Composite Device: HCAM0
  /dev/video1
  /dev/video2
```

If the camera appears as `/dev/video1`, use `/dev/video1` instead of `/dev/video0`.

Example:

```bash
source /opt/ros/jazzy/setup.bash

ros2 run v4l2_camera v4l2_camera_node \
  --ros-args \
  -p video_device:=/dev/video1 \
  -p image_size:="[640,480]" \
  -p pixel_format:=YUYV
```

---

### 8.2 Camera is busy

If the camera is already occupied, the camera node may fail with:

```text
Device or resource busy
```

Check which process is using the camera:

```bash
fuser -v /dev/video0
```

or:

```bash
fuser -v /dev/video1
```

Kill the process:

```bash
kill -9 <PID>
```

Common cleanup command:

```bash
pkill -f v4l2_camera_node || true
pkill -f pose_hand_stitched_node.py || true
pkill -f rqt_image_view || true
```

---

### 8.3 Skeleton topic is published but tracking is false

If the UDP bridge shows:

```text
tracking_ok=False, valid=0
```

then the skeleton message is being received, but valid human landmarks are not detected.

Try the following:

- Stand 1.5–2 m away from the camera.
- Make sure the face, shoulders, elbows, wrists, and hands are visible.
- Avoid showing only the hands.
- Keep both arms in a neutral pose for 1–2 seconds before moving.

Check the skeleton topic:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 topic echo /human/skeleton/arm_hand_stitched --once
```

---

### 8.4 Isaac Sim does not move

First test with fake skeleton data:

```bash
python3 ~/다운로드/rby1_isaac_dualarm_basket_ik_uprightfix/skeleton_bridge_scripts/fake_skeleton_udp_sender.py \
  --host 127.0.0.1 \
  --port 50555
```

Then run Isaac Sim:

```bash
conda activate env_isaaclab
cd ~/IsaacLab

./isaaclab.sh -p ~/다운로드/rby1_isaac_dualarm_basket_ik_uprightfix/rby1_skeleton_teleop_isaac.py \
  --asset v1_1 \
  --udp-host 127.0.0.1 \
  --udp-port 50555
```

If the robot moves with fake skeleton data, the Isaac-side controller is working correctly.

---

### 8.5 Wrist or end-effector motion is unnatural

If the wrist bends too much or the hand does not reach the center area, tune the following parameters:

```bash
--human-gain 1.25
--gain-y 1.20
--gain-z 1.60
--center-pull 0.45
--wrist-pull 0.45
--ee-orient-lock
```

If the robot hand still does not move toward the object, increase:

```bash
--center-pull 0.60
```

If the wrist still bends too much, increase:

```bash
--wrist-pull 0.70
--orient-weight 1.00
```

---

## 9. Current Status

Current implementation status:

- Isaac Sim environment has been constructed.
- RB-Y1 robot has been imported into the simulation environment.
- Photo-based real-to-sim object asset pipeline has been organized.
- Skeleton-based teleoperation bridge has been implemented.
- Fake skeleton data successfully moves the RB-Y1 robot in Isaac Sim.
- ROS2 camera and MediaPipe skeleton pipeline are under testing.
- Current limitation: hand and wrist motion mapping requires further improvement for natural grasping behavior.

---

## 10. Future Work

Future work includes:

- Improve hand–end-effector mapping.
- Add elbow-aware inverse kinematics.
- Add palm orientation-based end-effector control.
- Improve grasping behavior near the target object.
- Collect robot demonstration data in Isaac Sim.
- Extend the real-to-sim pipeline to more complex objects and workspace scenes.

---

## Contributors

- Daewon Kim
- Seungyeon Lee
- Seunghoon Baek
- Junhyun Jeon

---

## Mentors

- Jebeom Chae
- Jongeun Choi
