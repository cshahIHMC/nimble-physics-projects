import time
import sys
import os
from Microstrain import MicroStrainIMU
import threading
import nimblephysics as nimble
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple


def get_R_x(theta):
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
    return R

def get_R_y(theta):
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0,  np.cos(theta)]])
    return R

def get_R_z(theta):
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    return R


def main():
    
    # Configure the IMU's
    # imu = MicroStrainIMU("/dev/ttyACM0", 921600)
    imu = MicroStrainIMU("195772", 921600)
    imu2 = MicroStrainIMU("195778", 921600)
    NODE_RATE = 200
    
    # 1. Set up the simulation world
    world = nimble.simulation.World()
    world.setGravity([0, 0, 0]) # No gravity for this simple viz
    
    
    rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.RajagopalHumanBodyModel()
    skeleton: nimble.dynamics.Skeleton = rajagopal_opensim.skeleton
    world.addSkeleton(skeleton)
    
    
    
    ## List of all the joints
    # for i in range(skeleton.getNumBodyNodes()):
    #     print(f"{i} : {skeleton.getBodyNode(i).getName()}")
    
    
    for i in range(skeleton.getNumDofs()):
        
        print(f"{i}: {skeleton.getDofByIndex(i).getName()}")
    
    try:
        imu.configure_ESTFLTER_imu(200)
        imu2.configure_ESTFLTER_imu(200)
    except Exception as e:
        exc_type, exc_obj, tb = sys.exc_info()
        line_number = tb.tb_lineno
        print(e)
        print(exc_type.__name__)
        print(f'Error Ending Stream on line {line_number}')
        imu.set_to_idle()
        imu2.set_to_idle()
        
    time_initial = time.perf_counter()
    
    imu_timestamp, imu_accel_x, imu_accel_y, imu_accel_z, imu_gyro_x, imu_gyro_y, imu_gyro_z, imu_mag_x, imu_mag_y, imu_mag_z, imu_quat, imu_filt_quat = (imu.get_ESTFILTER_data(20, 0))
    imu2_timestamp, imu2_accel_x, imu2_accel_y, imu2_accel_z, imu2_gyro_x, imu2_gyro_y, imu2_gyro_z, imu2_mag_x, imu2_mag_y, imu2_mag_z, imu2_quat, imu2_filt_quat = (imu2.get_ESTFILTER_data(20, 0))
    # ############### Quaternion Streaming ###########################
    # Zeroing Quaternion
    imu_filt_quat_np = np.array([imu_filt_quat.as_floatAt(0), imu_filt_quat.as_floatAt(1), imu_filt_quat.as_floatAt(2), imu_filt_quat.as_floatAt(3)])
    imu_quat_0 = R.from_quat(imu_filt_quat_np, scalar_first=True)
   
    imu2_filt_quat_np = np.array([imu2_filt_quat.as_floatAt(0), imu2_filt_quat.as_floatAt(1), imu2_filt_quat.as_floatAt(2), imu2_filt_quat.as_floatAt(3)])
    imu2_quat_0 = R.from_quat(imu2_filt_quat_np, scalar_first=True)    
    
    
    # transform from IMU frame to Knee Joint frame
    transform_imu_2_knee = R.from_matrix( get_R_z(-np.pi/2) @ get_R_x(np.pi) )
    
    # Set up the GUI
    gui: nimble.NimbleGUI = nimble.NimbleGUI(world)
    gui.serve(8080)
    gui.nativeAPI().renderWorld(world)
    gui.nativeAPI().renderBasis(scale=0.5)
    
    tibia_r_node = skeleton.getBodyNode("tibia_r")
    
    # Convert the right hand rotation matrix to euler angles
    tibia_r_world_transform: nimble.math.Isometry3 = tibia_r_node.getWorldTransform()
    
    tibia_r_pos = tibia_r_world_transform.translation()
    tibia_r_euler_angles = nimble.math.matrixToEulerXYZ(tibia_r_world_transform.rotation())
    
    
    gui.nativeAPI().renderBasis(scale=0.3, pos=tibia_r_pos, euler=tibia_r_euler_angles, prefix="tibia_basis")   
    
    
    
    

    while True:
        try:
            cur_time = time.perf_counter()
            device_time = cur_time - time_initial

            imu_timestamp, imu_accel_x, imu_accel_y, imu_accel_z, imu_gyro_x, imu_gyro_y, imu_gyro_z, imu_mag_x, imu_mag_y, imu_mag_z, imu_quat, imu_filt_quat = (imu.get_ESTFILTER_data(20, 0))
            imu2_timestamp, imu2_accel_x, imu2_accel_y, imu2_accel_z, imu2_gyro_x, imu2_gyro_y, imu2_gyro_z, imu2_mag_x, imu2_mag_y, imu2_mag_z, imu2_quat, imu2_filt_quat = (imu2.get_ESTFILTER_data(20, 0))
            
            ############### Quaternion Streaming ###########################
            imu_filt_quat_np = np.array([imu_filt_quat.as_floatAt(0), imu_filt_quat.as_floatAt(1), imu_filt_quat.as_floatAt(2), imu_filt_quat.as_floatAt(3)])
            imu2_filt_quat_np = np.array([imu2_filt_quat.as_floatAt(0), imu2_filt_quat.as_floatAt(1), imu2_filt_quat.as_floatAt(2), imu2_filt_quat.as_floatAt(3)])
            
            # Convert the quat to rotation
            imu_filt_quat_rotation = R.from_quat(imu_filt_quat_np, scalar_first=True)
            imu2_filt_quat_rotation = R.from_quat(imu2_filt_quat_np, scalar_first=True)
            
            # zero the quat to starting orientation
            imu_filt_quat_zeroed =   imu_quat_0.inv() * imu_filt_quat_rotation 
            imu2_filt_quat_zeroed =   imu2_quat_0.inv() * imu2_filt_quat_rotation 
            
            
            
            # Transform from imu frame to knee joint frame
            imu_rotvec_joint_frame = transform_imu_2_knee * imu_filt_quat_zeroed * transform_imu_2_knee.inv()
            imu2_rotvec_joint_frame = transform_imu_2_knee * imu2_filt_quat_zeroed * transform_imu_2_knee.inv()
            
            
            # Convert to axis-angle (rotation vector) representation
            imu_rotvec = imu_rotvec_joint_frame.as_rotvec()  # 3D vector: axis * angle in radians
            imu2_rotvec = imu2_rotvec_joint_frame.as_rotvec()  # 3D vector: axis * angle in radians
            
            
            theta = np.linalg.norm(np.array(imu_rotvec))
            theta2 = np.linalg.norm(np.array(imu2_rotvec))
            if theta < 1e-8:
                axis = np.zeros(3)
            else:
                axis = imu_rotvec / theta
                
            if theta2 < 1e-8:
                axis2 = np.zeros(3)
            else:
                axis2 = imu2_rotvec / theta2
        
                
            joint_knee_local_axis = [0,0,-1]
            joint_hip_local_x_axis = [-1,0,0]
            joint_hip_local_y_axis = [0,-1,0]
            joint_hip_local_z_axis = [0,0,1]
            
    
    
    
            # Assuming axis is normalized
            joint_knee_angle = np.dot(axis, joint_knee_local_axis) * theta
    # 
            positions = skeleton.getPositions()
            positions[9] = joint_knee_angle
            
            positions[6] = np.dot(axis2, joint_hip_local_z_axis) * theta2
            positions[7] = np.dot(axis2, joint_hip_local_x_axis) * theta2
            positions[8] = np.dot(axis2, joint_hip_local_y_axis) * theta2
            skeleton.setPositions(positions)

            
            
            gui.nativeAPI().renderWorld(world)

        
            time_diff = time.time() - cur_time
            if time_diff < (1/NODE_RATE - 0.001):
                time.sleep((1/NODE_RATE- 0.001) - time_diff)

        except KeyboardInterrupt:
            break
        except Exception as e:
            exc_type, exc_obj, tb = sys.exc_info()
            line_number = tb.tb_lineno
            print(e)
            print(exc_type.__name__)
            print(f'Error Ending Stream on line {line_number}')
            imu.set_to_idle()
            imu2.set_to_idle()
            break



    try:
        print('Ending Stream')
        imu.set_to_idle()
        imu2.set_to_idle()
    except Exception as e:
       # pass
        exc_type, exc_obj, tb = sys.exc_info()
        line_number = tb.tb_lineno
        print(e)
        print(exc_type.__name__)
        print(f'Error Ending Stream on line {line_number}')
    
    



if __name__ == "__main__":
    raise SystemExit(main())

    
