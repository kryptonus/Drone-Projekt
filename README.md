# Stealth Quadcopter: ROS 2 Humble + Gazebo Classic

![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)
![C++](https://img.shields.io/badge/C++-17-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5-green)
![Gazebo](https://img.shields.io/badge/Gazebo-Classic_11-orange)
![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-purple)

## Proof of Concept
A fully simulated autonomous quadcopter built from scratch in ROS 2 Humble.
The drone uses its onboard camera to detect and track a human face in real time,
using a C++ PID controller to correct position 30 times per second.

> **Active Development:** Currently working on Stage 4 — C++ flight controller
> with velocity commands replacing direct position control.

---

## Demo
![Demo](demo.gif)
> Drone follows face in real time — C++ PID controller + OpenCV Haar Cascade

---

## Features
- **Physics simulation** at 62 FPS, Real Time Factor 1.00
- **Onboard drone camera** processed via cv_bridge — not a webcam shortcut
- **3-axis PID face tracking** — yaw, altitude, forward/back independently tuned
- **YAML parameter tuning** — change PID gains without recompiling
- **Separate PID header** `pid_controller.hpp` — clean C++ architecture
- **Landing legs** + stealth matte black visual design

---

## Architecture
```
Drone Camera (/stealth_quad/front_camera/image_raw)
      ↓
  cv_bridge → OpenCV Mat (bgr8)
      ↓
  Grayscale + equalizeHist (lighting normalization)
      ↓
  Haar Cascade detectMultiScale → Face Bounding Box
      ↓
  Error Calculation:
    error_x = face_center_x - image_center_x  → strafe
    error_y = face_center_y - image_center_y  → altitude
    error_z = face_width    - target_width     → distance
      ↓
  3x PID Controllers (gains from YAML params)
      ↓
  Gazebo SetEntityState Service → Drone moves
```

---

## Tech Stack
| Tool | Version | Purpose |
|------|---------|---------|
| ROS 2 | Humble | Robot middleware |
| Gazebo | Classic 11 | Physics simulation |
| OpenCV | 4.5 | Face detection |
| C++ | 17 | Main tracking node |
| Ubuntu | 22.04 | OS |
| GPU | GTX 1650 | Simulation rendering |

---

## Sensor Suite
| Sensor | Rate | Topic |
|--------|------|-------|
| Camera | 30 Hz | `/stealth_quad/front_camera/image_raw` |
| LiDAR | 10 Hz | `/stealth_quad/scan` |
| IMU | 200 Hz | `/stealth_quad/imu/data` |

---

## Quickstart - additionally I have made aliases for these bash terminals (d.tracker, d.build, d.sim)
```bash
# Clone
git clone git@github.com:kryptonus/Drone-Projekt.git
cd Drone-Projekt

# Build
colcon build --symlink-install
source /opt/ros/humble/setup.bash
source install/setup.bash

# Terminal 1 — Launch simulation
ros2 launch quadcopter_description display.launch.py

# Terminal 2 — Start face tracker
ros2 run quadcopter_control face_tracker_node --ros-args \
  --params-file src/quadcopter_control/config/tracker_params.yaml
```

---

## Roadmap
- [x] Stage 1: Physics URDF with full sensor suite
- [x] Stage 2: Computer vision pipeline + cv_bridge
- [x] Stage 3: C++ face tracking with PID controller
- [x] Stage 3.5: YAML parameter tuning + separate PID header
- [ ] Stage 4: C++ flight controller with velocity commands
- [ ] Stage 5: YOLOv8 object detection
- [ ] Stage 6: GPS-denied autonomous navigation

---

## Related Projects

| Repo | What it is |
|------|-----------|
| [C++ PID Controller](https://github.com/kryptonus/PID-Controller-Cpp) | The standalone PID library used in this drone — built and tested separately before being integrated |

---

## Why I Made This
I built this over a weekend (was like a hackathon XD) because it was listed as a future goal on my CV and I could not leave it as just a future plan.

**Author:** Vaishnav — IEB, TH Mannheim.
**Focus:** Defense, Autonomous Systems, Computer Vision, Robotics.
