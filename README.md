# HOLOrIG - Gesture-Based Holographic Interaction System

HOLOrIG is a real-time **gesture-controlled interaction system** built using computer vision. It enables users to interact with 3D objects and a robotic arm using **hand gestures captured via a webcam**, without physical controllers.

## Features

- Gesture-driven menu navigation
- Real-time hand tracking via webcam
- Multiple interactive demos:
  - 3D Gyroscopic Rings
  - 3D Orbit Core (nested cubes)
  - Gesture-controlled robotic arm

## Demos

### 1) 3D Gyro Rings
- Three orthogonal rings rendered in 3D
- Gestures:
  - **Pinch + drag** → rotate
  - **Two hands** → zoom
  - **Palm tilt** → spin

### 2) Orbit Core
- Inner and outer cube structures connected dynamically
- Supports rotation, zoom, and spin using gestures
- Demonstrates 3D transformations and depth-based shading

### 3) Robotic Arm
- 2-link robotic arm using inverse kinematics
- Gestures:
  - **Move index finger** → move arm
  - **Pinch** → close gripper
  - **Open palm** → open gripper

## Tech Stack

- Python
- OpenCV
- cvzone (HandTrackingModule)
- NumPy
- (Optional) pygame for sound effects

## Core Concepts

- Real-time computer vision
- Hand landmark detection
- Gesture recognition
- 3D rotation and projection
- Inverse kinematics
- State-based interactive systems

## Python & MediaPipe Compatibility

⚠️ **Important:** This project depends on **MediaPipe**, which is **not compatible with Python 3.13**.

Use:
- **Python 3.10 or 3.11 (recommended)**

If you encounter:
```text
AttributeError: module 'mediapipe' has no attribute 'solutions'
```
it is likely due to an incompatible Python version.

## Installation

### Requirements
- Python 3.10 / 3.11
- Webcam

### Create and activate a virtual environment (Windows)
```bat
python -m venv venv
venv\Scripts\activate
```

### Install dependencies
```bash
pip install mediapipe==0.10.9
pip install opencv-python numpy cvzone pygame
```

## Run

```bash
python Project.py
```

Press **ESC** to exit.

## Gesture Controls

| Action | Gesture |
|---|---|
| Select menu item | Pinch |
| Rotate 3D object | Pinch + drag |
| Zoom | Two hands |
| Spin | Palm tilt |
| Move arm | Move index finger |
| Close gripper | Pinch |
| Open gripper | Open palm |
| Back to menu | Pinch on **BACK** |

## Notes

- Sound effects are optional and only enabled if audio files are present (e.g., `select.wav`, `back.wav`, `grab.wav`).
- Designed for experimentation with gesture-based interaction and HCI concepts.

## Author

Joshua Jacob Thomas
