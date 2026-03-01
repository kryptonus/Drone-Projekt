# Stealth Quadcopter — ROS 2 Humble + Gazebo Classic

A fully simulated autonomous quadcopter with AI-powered face tracking.

## Demo
- Physics-enabled drone simulation at 62 FPS
- Live camera feed with targeting HUD
- Face tracking using OpenCV + PID controller
- LiDAR 360 scanning, IMU at 200Hz, Camera at 30Hz

## Tech Stack
- ROS 2 Humble
- Gazebo Classic 11
- OpenCV 4.5
- Ubuntu 22.04, NVIDIA GTX 1650

## Run It
Terminal 1:
ros2 launch quadcopter_description display.launch.py

Terminal 2:
ros2 run quadcopter_cv face_tracker

## Sensors
- Camera: 30Hz
- LiDAR: 10Hz
- IMU: 200Hz

## Roadmap
- Stage 1: Physics URDF with sensors - DONE
- Stage 2: Computer vision pipeline - DONE
- Stage 3: Face tracking PID - DONE
- Stage 4: C++ flight controller - IN PROGRESS
- Stage 5: YOLO object detection
- Stage 6: Autonomous navigation

## Author
Vaishnav — IEB Program, TH Mannheim
Defense, AI, Robotics
