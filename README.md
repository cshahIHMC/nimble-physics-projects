# 🧠 Nimble Physics Projects

This repository explores the capabilities of the **[Nimble Physics](https://nimblephysics.org/)** Python package for human motion simulation using a DART differentiable physics engine at its core.  
Our primary goal is to **integrate IMU data with a full human-body skeleton** in real time, leveraging **MicroStrain IMUs** present in the **Georgia Tech wearable sensor suit** to enable accurate **motion visualization and biomechanical estimation**.

---

## 🚀 Overview

This project demonstrates:
- Real-time motion reconstruction using IMU quaternion data.
- Integration of sensor data streams with a full human skeleton defined in **Nimble Physics**.
- Explorations into using physics-based priors for improving motion estimation fidelity.
- Comparison between raw IMU-based motion and physics-consistent reconstructions.

---

## 🦾 IMU Integration

### MicroStrain IMUs
We use **MicroStrain 3DM-GX5** IMUs as part of the **Georgia Tech wearable sensor suit**.  
Each IMU provides:
- Quaternion orientation data
- Accelerometer and gyroscope readings
- Real-time data streaming via the MSCL (MicroStrain Communication Library)

Future goals include:
- Calibration of IMUs relative to the skeleton’s joint frames.
- Integration of orientation and angular velocity for full-body kinematic estimation.
- Validation against motion capture ground truth.

---

## 🎥 Demonstration

Below is a demonstration video showing **real-time skeleton visualization** driven by IMU data.  
*(Video will be added soon)*

---
## Dependencies


---

## 💻 Code Structure
