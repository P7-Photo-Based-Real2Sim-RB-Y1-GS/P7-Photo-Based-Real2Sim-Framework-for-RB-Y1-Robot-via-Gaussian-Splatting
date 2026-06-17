## Overview

This project uses the Intel RealSense D435i camera and Google’s open-source MediaPipe 0.10.9 version framework at python virtual environment.

The provided code detects pose and hand landmarks in real time.
The detected landmark data is then converted into UDP output, which can be transmitted to Isaac Sim for further use.
