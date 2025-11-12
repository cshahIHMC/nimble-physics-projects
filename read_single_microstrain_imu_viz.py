"""
===============================================================================
Script Name: imu_orientation_viz.py
Author: Chinmay Shah
Date: Oct 2025
===============================================================================
Description:
    This script streams real-time orientation data from a single MicroStrain IMU
    and visualizes its motion in the Nimble Physics GUI. The IMU’s quaternion
    orientation is read, zeroed to the initial reference frame, and applied to
    a simple 3D box model in the simulation to represent its rotation in space.

    Key Features:
        - Connects to a MicroStrain IMU via serial interface
        - Configures and streams filtered orientation data (quaternion)
        - Converts quaternion to rotation vector and updates a simble box in the nimble gui
        - Provides real-time visualization through Nimble’s WebSocket GUI
        - Includes a least-squares velocity and acceleration solver (optional)

    Usage:
        Run this script directly to visualize the IMU orientation.
        Example:
            $ python3 reltime_single_imu_viz.py

    Requirements:
        - MicroStrain IMU driver and Python API
        - Nimble Physics (Python bindings)
        - NumPy, SciPy, PyTorch

===============================================================================
"""

import time
import sys
import os
import threading
from typing import List, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import nimblephysics as nimble
from Libraries.Microstrain import MicroStrainIMU


import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_imu_to_world_transform(acc_b, gyro_b, mag_b):
    """
    Given accelerometer, gyro, and magnetometer (in IMU frame),
    compute rotation matrix from IMU frame → anatomical world frame.
    """
    # Normalize accelerometer (gravity)
    g_b = acc_b / np.linalg.norm(acc_b)
    # print(g_b)
    y_world = g_b  # Up direction is opposite of gravity
    
    # Remove gravity from magnetometer to get horizontal projection
    mag_proj = mag_b - np.dot(mag_b, y_world) * y_world
    x_world = mag_proj / np.linalg.norm(mag_proj)  # Forward direction
    

    # Compute right direction
    z_world = np.cross(x_world, y_world)
    z_world /= np.linalg.norm(z_world)
    

    # # Re-orthogonalize (optional)
    # x_world = np.cross(y_world, z_world)
    # x_world /= np.linalg.norm(x_world)

    # Construct rotation matrix: columns are world axes in IMU frame
    R_imu_to_world = R.from_matrix(np.vstack([x_world, y_world, z_world]).T)

    return R_imu_to_world

def leastSquareVelAndAccSolver(imu_acc, imu_gyro, sensors, skeleton):
    """
    Solves least-squares estimates for joint velocities and accelerations
    based on IMU linear acceleration and angular velocity readings.
    """
    # Jacobian wrt joint velocity (maps joint velocities -> gyro readings)
    d_rot_vel_d_vel = skeleton.getGyroReadingsJacobianWrt(
        sensors, wrt=nimble.neural.WRT_VELOCITY
    )
    vel = np.linalg.lstsq(d_rot_vel_d_vel, imu_gyro, rcond=None)[0]

    # Jacobian wrt joint acceleration (maps joint accelerations -> accel readings)
    d_lin_acc_d_acc = skeleton.getAccelerometerReadingsJacobianWrt(
        sensors, wrt=nimble.neural.WRT_ACCELERATION
    )
    acc = np.linalg.lstsq(d_lin_acc_d_acc, imu_acc, rcond=None)[0]

    return vel, acc


# --------------------------------------------------------------------
# IMU helpers
# --------------------------------------------------------------------

def get_imu_data(imu):
    data = imu.get_ESTFILTER_data(20, 0)
    device_time = data[0]
    
    quat = np.array([
        data[10].as_floatAt(0),
        data[10].as_floatAt(1),
        data[10].as_floatAt(2),
        data[10].as_floatAt(3)
    ])
        
    return np.array(data[1:10]), quat

def read_sensor(data, plotter=None):
    acc_b = data[0:3] * 9.80665
    gyro_b = data[3:6]
    mag_b  = data[6:9]
    
    return acc_b, gyro_b, mag_b

def get_estfilter_data(imu):
    data = imu.get_ESTFILTER_data(20, 0)
    quat = np.array([
        data[10].as_floatAt(0),
        data[10].as_floatAt(1),
        data[10].as_floatAt(2),
        data[10].as_floatAt(3)
    ])
    return quat


def main():
    """
    Real-time visualization of MicroStrain IMU orientation streamed into
    a Nimble Physics simulated box (for orientation tracking).
    """

    # --- IMU Configuration ---
    imu = MicroStrainIMU("195772", 921600)
    NODE_RATE = 200  # IMU stream rate (Hz)

    # --- World Setup ---
    world = nimble.simulation.World()
    world.setGravity([0, 0, 0])  # No gravity for visualization

    # --- Create a simple body to represent IMU ---
    box = nimble.dynamics.Skeleton()
    boxJoint, boxBody = box.createBallJointAndBodyNodePair()
    boxShape = boxBody.createShapeNode(nimble.dynamics.BoxShape([0.05, 0.15, 0.1]))
    boxVisual = boxShape.createVisualAspect()
    boxVisual.setColor([0.5, 0.5, 0.5])
    world.addSkeleton(box)
    box.setName("IMU_Object")

    # --- Initial State ---
    initial_rotation = torch.zeros(3)
    initial_velocity = torch.zeros(world.getNumDofs(), requires_grad=True)

    # --- GUI Setup ---
    gui = nimble.NimbleGUI(world)
    gui.serve(8080)
    gui.nativeAPI().renderSkeleton(box)

    # --- Draw a basis at IMU origin ---
    box_transform = boxBody.getWorldTransform()
    box_pos = box_transform.translation()
    box_euler = nimble.math.matrixToEulerXYZ(box_transform.rotation())
    gui.nativeAPI().renderBasis(scale=0.3, pos=box_pos, euler=box_euler, prefix="IMU_basis")

    # --- Configure IMU filtering ---
    try:
        imu.configure_ESTFLTER_imu(NODE_RATE)
    except Exception as e:
        exc_type, _, tb = sys.exc_info()
        print(f"Error configuring IMU on line {tb.tb_lineno}: {e}")
        imu.set_to_idle()


    
    ### Get data from IMU
    imu_data, imu_quat_data = get_imu_data(imu)
    acc, gyro, mag = read_sensor(imu_data)
    # print(acc)
            
    R_imu_2_anatomical = compute_imu_to_world_transform(acc, gyro, mag)
            

    # --- Initialize IMU data stream ---
    time_initial = time.perf_counter()
    imu_quat_0 = R.from_quat(imu_quat_data, scalar_first=True)

    state = torch.cat((initial_rotation, initial_velocity), 0)
    
    # --- Main Streaming Loop ---
    while True:
        try:
            cur_time = time.perf_counter()
            device_time = cur_time - time_initial

            ## --- Read IMU Data ---
            imu_quat_data = get_estfilter_data(imu)
            imu_rot = R.from_quat(imu_quat_data, scalar_first=True)

            # Zero to starting orientation
            imu_zeroed_body = imu_quat_0.inv() * imu_rot # axis-angle representation
            # imu_rotvec_body = imu_zeroed.as_rotvec() 
            
            imu_rotvec_world = (R_imu_2_anatomical.inv() * imu_zeroed_body * R_imu_2_anatomical).as_rotvec()

            # --- Update simulation state ---
            state = torch.cat((torch.tensor(imu_rotvec_world), initial_velocity), 0)
            state = nimble.timestep(world, state, torch.zeros(world.getNumDofs()))

            # --- Render the updated state ---
            gui.nativeAPI().renderWorld(world)

            # --- Maintain frame rate ---
            elapsed = time.time() - cur_time
            sleep_time = max(0, (1 / NODE_RATE - 0.001) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            break
        except Exception as e:
            exc_type, _, tb = sys.exc_info()
            print(f"Error during loop at line {tb.tb_lineno}: {e}")
            imu.set_to_idle()
            break

    # --- Cleanup on Exit ---
    try:
        print("Ending IMU Stream...")
        imu.set_to_idle()
    except Exception as e:
        exc_type, _, tb = sys.exc_info()
        print(f"Error ending stream at line {tb.tb_lineno}: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
