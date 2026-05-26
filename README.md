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
