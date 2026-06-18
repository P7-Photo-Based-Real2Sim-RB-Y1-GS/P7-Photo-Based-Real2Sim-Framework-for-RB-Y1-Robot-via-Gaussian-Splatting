## Overview

**System**: The code is currently tested only on Linux.

This project uses the Intel RealSense D435i camera and Google’s open-source MediaPipe 0.10.9 version framework at python virtual environment./Python version 3.8 or higher is required.

The provided code detects pose and hand landmarks(MediaPipe Holistic) in real time.
The detected landmark data is then converted into UDP output, which can be transmitted to Isaac Sim for further use.
