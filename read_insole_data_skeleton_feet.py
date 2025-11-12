from xsensor_driver_no_ros import XSENSOR_Data
from xsensor_driver_no_ros import XSENSORS
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import nimblephysics as nimble
from Tools.live_plotter import LivePlotter3D
from typing import List, Tuple

def compute_imu_to_world_from_acc(acc_b):
    """
    Estimate IMU→world rotation from accelerometer (gravity direction only).
    Defines 'up' but not heading (yaw).
    """
    g_b = acc_b / np.linalg.norm(acc_b)
    y_world = g_b  # Up direction

    # Choose arbitrary x_world to define yaw = 0 (e.g., [1, 0, 0])
    x_temp = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(x_temp, y_world)) > 0.9:
        x_temp = np.array([0.0, 1.0, 0.0])
    
    # # Orthogonalize
    z_world = np.cross(x_temp, y_world)
    z_world /= np.linalg.norm(z_world)
    x_world = np.cross(y_world, z_world)
    x_world /= np.linalg.norm(x_world)

    R_imu_to_world = R.from_matrix(np.vstack([x_world, y_world, z_world]).T)
    return R_imu_to_world

def safe_axis_angle(rotvec: np.ndarray) -> Tuple[np.ndarray, float]:
    theta = np.linalg.norm(rotvec)
    if theta < 1e-8:
        return np.zeros(3), 0.0
    return rotvec / theta, theta

def get_acc_quat(insole_data):
    
    accx = insole_data.linx * 9.80665
    accy = insole_data.liny * 9.80665
    accz = insole_data.linz * 9.80665
    
    acc = np.array([
        accx,
        accy,
        accz
    ], dtype=float)
    

    # Extract quaternion as numpy array
    q = np.array([
        insole_data.qx,
        insole_data.qy,
        insole_data.qz,
        insole_data.qw
    ], dtype=float)
    
    # Normalize to unit quaternion
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Quaternion magnitude is zero — invalid data from left insole.")
    q /= norm
    # Return scipy Rotation object
    quat_rot = R.from_quat(q, scalar_first=False)
    
    return quat_rot, acc
    

def get_data(xsensors):
    """
    Reads latest left insole data from the XSensor interface and 
    returns its orientation as a scipy Rotation object.
    """
    # Publish or read data from both sides
    left_insole_data, right_insole_data = xsensors.publish_data()
    
    left_quat, left_acc = get_acc_quat(left_insole_data)
    right_quat, right_acc = get_acc_quat(right_insole_data)
    
    
    return left_quat, left_acc, right_quat, right_acc


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    sample_frequency = 200 # hz
    NUM_INSOLES = 2
    TCP_IP = '192.168.0.1'

    
    # Nimble world
    world = nimble.simulation.World()
    world.setGravity([0, 0, 0])

    opensim_model = nimble.RajagopalHumanBodyModel()
    skeleton = opensim_model.skeleton
    world.addSkeleton(skeleton)
    
    # GUI
    gui = nimble.NimbleGUI(world)
    gui.serve(8080)
    gui.nativeAPI().renderWorld(world)
    gui.nativeAPI().renderBasis(scale=0.5)

    tibia_r_node = skeleton.getBodyNode("tibia_r")
    tibia_tf = tibia_r_node.getWorldTransform()
    gui.nativeAPI().renderBasis(
        scale=0.3,
        pos=tibia_tf.translation(),
        euler=nimble.math.matrixToEulerXYZ(tibia_tf.rotation()),
        prefix="tibia_basis"
    )
    
    
    sample_period = 1.0 / sample_frequency
    next_time = time.perf_counter()
    
    xsensors = XSENSORS(num_xsensors = NUM_INSOLES)
    xsensorConnection = xsensors.start_server(tcp_ip=TCP_IP, startup=True)
    
    left_foot_quat_0, left_foot_acc_0, right_foot_quat_0, right_foot_acc_0 = get_data(xsensors=xsensors)
    
    left_foot_anatomical = compute_imu_to_world_from_acc(left_foot_acc_0)
    right_foot_anatomical = compute_imu_to_world_from_acc(right_foot_acc_0)
    
    
    ##################################################################
    # Had to do this - not ideal at all ! ###################
    # --- Add a yaw (Z-axis) correction ---
    yaw_correction_deg = 180.0  # try 90, 180, or -90 as needed
    R_z_fix = R.from_euler('z', yaw_correction_deg, degrees=True)
    
    ###################################################################
    
    while True:
        try:    
            
            left_foot_quat, left_acc, right_foot_quat, right_foot_acc = get_data(xsensors=xsensors)

            left_foot_quat_zeroed = left_foot_quat_0.inv() * left_foot_quat 
            right_foot_quat_zeroed = right_foot_quat_0.inv() * right_foot_quat  
            
            left_foot_quat_joint_frame = left_foot_anatomical * left_foot_quat_zeroed * left_foot_anatomical.inv()
            right_foot_quat_joint_frame =  right_foot_anatomical * right_foot_quat_zeroed * right_foot_anatomical.inv()
            
            feet_l_axis, feet_l_theta = safe_axis_angle(left_foot_quat_joint_frame.as_rotvec())
            feet_r_axis, feet_r_theta = safe_axis_angle(right_foot_quat_joint_frame.as_rotvec())
            
            # Compute joint angles
            pos = skeleton.getPositions()
            
            # right foot
            pos[10] = np.dot(feet_r_axis, np.array([0, 0, 1])) * feet_r_theta  # ankle angle_r
            pos[11] = np.dot(feet_r_axis, np.array([-1, 0, 0])) * feet_r_theta  # subtalar_angle_r
            # # pos[12] = np.dot(feet_l_axis,  np.array([1, 0, 0])) * feet_l_theta  # mtp_angle_r
                        
            # left foot
            pos[17] = np.dot(feet_l_axis, np.array([0, 0, 1])) * feet_l_theta  # ankle angle_l
            pos[18] = np.dot(feet_l_axis, np.array([1, 0, 0])) * feet_l_theta  # subtalar_angle_l
            # # pos[19] = np.dot(feet_l_axis,  np.array([1, 0, 0])) * feet_l_theta  # mtp_angle_l - toe movement - not used - we dont capture it only
            
            
            skeleton.setPositions(pos)
            gui.nativeAPI().renderWorld(world)
            
            # --- Maintain precise timing ---
            next_time += sample_period
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If we fell behind, skip sleep and resync
                next_time = time.perf_counter()


        except KeyboardInterrupt:
            print("exited")
            break
        except Exception as e:
            print("Triggered some exception")
            xsensors.close_server()
            break   

if __name__ == "__main__":
    raise SystemExit(main())