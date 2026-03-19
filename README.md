# Mobile Robots State Estimation and Simulation in Python and MuJoCo

A structured robotics project exploring state estimation for mobile robots, from linear Kalman filtering to nonlinear EKF-SLAM, culminating in a full MuJoCo-based simulation pipeline.

---

## Overview

This repository presents a progression of robotics simulations focused on:

- motion modeling  
- noisy sensing  
- probabilistic state estimation  
- mapping  

The project evolves through:

1. Linear Kalman filtering  
2. EKF localization  
3. EKF-SLAM  
4. Full MuJoCo simulation  

The goal is to demonstrate how robots estimate their state and environment under uncertainty.

---

## Repository Structure

.
├── src/
├── mujoco/
├── outputs/
│ └── gifs/
├── docs/
└── README.md


---

## Kalman Filtering (Linear System)

A double-integrator system with position and velocity:

- Linear dynamics  
- Gaussian noise  
- Standard Kalman Filter
  
State:
x = [q, q̇]

Dynamics:
xₖ₊₁ = A xₖ + B uₖ + wₖ

Measurement:
zₖ = C xₖ + vₖ

### Result

![Kalman Filter](outputs/gifs/double_integrator_4panels.gif)

→ The filter reconstructs the true motion from noisy measurements.

---

## EKF Localization

Motion (unicycle):
xₖ₊₁ = xₖ + v cosθ dt  
yₖ₊₁ = yₖ + v sinθ dt  
θₖ₊₁ = θₖ + ω dt  

Measurement (range-bearing):
r = √((lₓ-x)² + (l_y-y)²)  
β = atan2(l_y-y, lₓ-x) - θ  

→ Nonlinear → Extended Kalman Filter

### Results

![Ground Truth](outputs/gifs/ground_truth_noiseless.gif)

![EKF Localization](outputs/gifs/sim5_noisy_motion_ekf_fov.gif)

![State & Covariance](outputs/gifs/mu_sigma_evolution.gif)

→ The EKF estimates the trajectory despite noise and limited sensing.

---

## EKF-SLAM

Simultaneous estimation of:

- robot pose  
- landmark positions
  
State:
μ = [x, y, θ, l₁ₓ, l₁_y, ..., lₙₓ, lₙ_y]

Using nonlinear models and EKF linearization.

### Results

![EKF SLAM (2 Landmarks)](outputs/gifs/ekf_slam_2_landmarks.gif)

![EKF SLAM (Random Landmarks)](outputs/gifs/ekf_slam_random_4_landmarks.gif)

![EKF SLAM (Dynamic Motion)](outputs/gifs/ekf_slam_slide_model.gif)

→ The robot builds a map while localizing itself.

---

## Final MuJoCo Simulation

### `mujoco/scripts/mujoco_ekf_slam_demo.py`

The complete part.

<img width="2876" height="1704" alt="image" src="https://github.com/user-attachments/assets/17d03ca8-6946-46c5-aedd-180be575379a" />

It integrates:

- differential-drive robot control  
- exploration behavior  
- obstacle avoidance  
- limited field-of-view sensing  
- noisy motion & measurements  
- EKF-SLAM  
- real-time visualization  

### What makes it important

This is a full robotics system:

> control + perception + estimation + environment interaction

Unlike the previous scripts, this simulation shows a robot actively exploring and mapping its environment.

---

## Key Concepts

- Kalman Filter  
- Extended Kalman Filter (EKF)  
- Nonlinear kinematics  
- Range-bearing sensing  
- SLAM  
- Uncertainty modeling  
- MuJoCo simulation  

---

## Installation

```bash
pip install -r requirements.txt
```

---

Tested on:

Windows
Linux

## Suggested Reading Order

To follow the conceptual progression of the project:

1. Kalman Filter (linear systems)  
2. EKF Localization  
3. EKF-SLAM  
4. MuJoCo Simulation  

---

## Purpose

This project was developed as a robotics portfolio demonstrating:

- understanding of probabilistic state estimation  
- implementation of robotics models  
- simulation and visualization  
- system-level integration  

---

## Author

Αnastasia Chatziparaskeva

Undergraduate Electrical Engineering Student  
Focus: Robotics and Control 

Contact: 04anastasia@gmail.com 
