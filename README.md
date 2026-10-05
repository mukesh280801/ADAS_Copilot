# ADAS Copilot

AI-powered Advanced Driver Assistance System built with CARLA, YOLOv8, ByteTrack, lane perception, trajectory prediction, collision-risk analysis, TTC, ACC, AEB, traffic-light handling, and safe overtaking.

## Overview

ADAS Copilot is a software-based ADAS research and simulation platform designed for real-time driving assistance inside the CARLA simulator.

The system combines computer vision, object tracking, physical vehicle-state information, trajectory reasoning, and safety-control logic to continuously analyze the driving environment and control an ego vehicle.

## Key Features

- CARLA-based driving simulation
- Front RGB camera perception
- YOLOv8 real-time object detection
- ByteTrack multi-object tracking
- Lane detection and lane visualization
- Lane-departure monitoring
- Physical lane-status verification using CARLA road geometry
- Vehicle and pedestrian safety handling
- Time-to-Collision (TTC) calculation
- Physical collision-risk estimation
- Automatic Emergency Braking (AEB)
- Adaptive Cruise Control (ACC)
- Traffic-light handling
- Safe lane-change decision making
- Controlled overtaking state machine
- Trajectory prediction
- Real-time steering, throttle, and brake control
- Real-time ADAS dashboard
- CARLA Traffic Manager integration
- Multi-threaded perception and control pipeline
- Safety-priority arbitration

## System Architecture

```text
                    CARLA SIMULATOR
                           |
                           v
                    FRONT RGB CAMERA
                           |
              +------------+------------+
              |                         |
              v                         v
          YOLOv8                    Lane Detection
              |                         |
              v                         v
         ByteTrack              Lane Departure
              |                         |
              +------------+------------+
                           |
                           v
              PHYSICAL WORLD STATE
                           |
                           v
              TRAJECTORY + TTC + RISK
                           |
                           v
                  SAFETY ARBITRATION
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
         AEB              ACC          OVERTAKING
          |                |                |
          +----------------+----------------+
                           |
                           v
                    VEHICLE CONTROL
                           |
                           v
                      EGO VEHICLE
                           |
                           v
                    LIVE DASHBOARD



📌 Project Objective
To develop a real-time, software-based ADAS Copilot capable of understanding the driving environment and assisting the ego vehicle with safer driving decisions using AI and computer vision.

⚠️ Disclaimer
This project is developed for research, education, and simulation purposes using CARLA. It is not intended for deployment in real-world vehicles without extensive safety validation.