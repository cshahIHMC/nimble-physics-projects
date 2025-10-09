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


def leastSquareVelAndAccSolver(imu_acc, imu_gyro, sensors, skeleton):
    
    # Solve for the (least-squares) joint velocities
    d_rot_vel_d_vel: np.ndarray = skeleton.getGyroReadingsJacobianWrt(sensors, wrt=nimble.neural.WRT_VELOCITY)
    vel: np.ndarray = np.linalg.lstsq(d_rot_vel_d_vel, imu_gyro, rcond=None)[0]

    # Solve for the (least-squares) joint accelerations
    d_lin_acc_d_acc: np.ndarray = skeleton.getAccelerometerReadingsJacobianWrt(sensors, wrt=nimble.neural.WRT_ACCELERATION)
    acc: np.ndarray = np.linalg.lstsq(d_lin_acc_d_acc, imu_acc, rcond=None)[0]
    
    return vel, acc


def main():
    
    # Configure the IMU's
    # imu = MicroStrainIMU("/dev/ttyACM0", 921600)
    imu = MicroStrainIMU("195772", 921600)
    NODE_RATE = 200
    
    # 1. Set up the simulation world
    world = nimble.simulation.World()
    world.setGravity([0, 0, 0]) # No gravity for this simple viz

    # 2. Create the object that will represent the IMU
    box = nimble.dynamics.Skeleton()
    boxJoint, boxBody = box.createBallJointAndBodyNodePair()
    # boxJoint, boxBody = box.createTranslationalJointAndBodyNodePair()
    boxShape = boxBody.createShapeNode(nimble.dynamics.BoxShape([.15, .1, .05]))
    boxVisual = boxShape.createVisualAspect()
    boxVisual.setColor([0.5, 0.5, 0.5])
    
    world.addSkeleton(box)
    box.setName("IMU_Object")
    
    # # Set up initial conditions for optimization
    initial_rotation: torch.Tensor = torch.tensor([0.0, 0.0, 0.0])
    initial_velocity: torch.Tensor = torch.zeros((world.getNumDofs()), requires_grad=True)
    
    # Set up the GUI
    gui: nimble.NimbleGUI = nimble.NimbleGUI(world)
    gui.serve(8080)
    
    # # Render the skeleton to the GUIright_hand
    gui.nativeAPI().renderSkeleton(box)  
    
    # Convert the right hand rotation matrix to euler angles
    box_world_transform: nimble.math.Isometry3 = boxBody.getWorldTransform()
    
    box_pos = box_world_transform.translation()
    box_euler_angles = nimble.math.matrixToEulerXYZ(box_world_transform.rotation())
    
    gui.nativeAPI().renderBasis(scale=0.3, pos=box_pos, euler=box_euler_angles, prefix="IMU_basis")   
    
    # box_joint : nimble.dynamics.Joint = box.getJoint(0)
    # translation: np.ndarray = np.array([0.0, 0.0, 0.0])
    # rotation: np.ndarray = np.eye(3)
    # imu_offset: nimble.math.Isometry3 = nimble.math.Isometry3(rotation, translation)
    # sensors: List[Tuple[nimble.dynamics.BodyNode, nimble.math.Isometry3]] = [(boxBody, imu_offset)]
    
    try:
        imu.configure_ESTFLTER_imu(200)
    except Exception as e:
        exc_type, exc_obj, tb = sys.exc_info()
        line_number = tb.tb_lineno
        print(e)
        print(exc_type.__name__)
        print(f'Error Ending Stream on line {line_number}')
        imu.set_to_idle()
    
    
    time_initial = time.perf_counter()
    
    imu_timestamp, imu_accel_x, imu_accel_y, imu_accel_z, imu_gyro_x, imu_gyro_y, imu_gyro_z, imu_mag_x, imu_mag_y, imu_mag_z, imu_quat, imu_filt_quat = (imu.get_ESTFILTER_data(20, 0))
    
    
    #################### Acc+ Gyro Streaming ############################
    # imu_acc = np.array([imu_accel_x, imu_accel_y, imu_accel_z])
    # imu_gyro = np.array([imu_gyro_x, imu_gyro_y, imu_gyro_z])
    
    # box_vel, box_acc = leastSquareVelAndAccSolver(imu_acc, imu_gyro, sensors, box)
    
    # # state: torch.Tensor = torch.cat((initial_rotation, torch.tensor(box_vel), torch.tensor(box_acc)), 0)
    # state: torch.Tensor = torch.cat((initial_rotation, torch.tensor(box_acc)), 0)
    
    
    ############### Quaternion Streaming ###########################
    # Zeroing Quaternion
    imu_filt_quat_np = np.array([imu_filt_quat.as_floatAt(0), imu_filt_quat.as_floatAt(1), imu_filt_quat.as_floatAt(2), imu_filt_quat.as_floatAt(3)])
    imu_quat_0 = R.from_quat(imu_filt_quat_np, scalar_first=True)
    
    state: torch.Tensor = torch.cat((initial_rotation, initial_velocity), 0)
            
    while True:
        try:
            cur_time = time.perf_counter()
            device_time = cur_time - time_initial

            imu_timestamp, imu_accel_x, imu_accel_y, imu_accel_z, imu_gyro_x, imu_gyro_y, imu_gyro_z, imu_mag_x, imu_mag_y, imu_mag_z, imu_quat, imu_filt_quat = (imu.get_ESTFILTER_data(20, 0))
            
            ############### Quaternion Streaming ###########################
            imu_filt_quat_np = np.array([imu_filt_quat.as_floatAt(0), imu_filt_quat.as_floatAt(1), imu_filt_quat.as_floatAt(2), imu_filt_quat.as_floatAt(3)])
            
            # Convert the quat to rotation
            imu_filt_quat_rotation = R.from_quat(imu_filt_quat_np, scalar_first=True)
            # zero the quat to starting orientation
            imu_filt_quat_zeroed = imu_quat_0.inv() * imu_filt_quat_rotation 
            
            # Imu_euler_angles = imu_filt_quat_zeroed.as_euler('xyz', degrees=False)
            
            # Convert to axis-angle (rotation vector) representation
            imu_rotvec = imu_filt_quat_zeroed.as_rotvec()  # 3D vector: axis * angle in radians


            
            state = torch.cat((torch.tensor(imu_rotvec), initial_velocity), 0)
            state = nimble.timestep(world, state, torch.zeros((world.getNumDofs())))
            
            
            #################### Acc+ Gyro Streaming ############################
            # Render the box and its basis
            # imu_acc = np.array([imu_accel_x, imu_accel_y, imu_accel_z])
            # imu_gyro = np.array([imu_gyro_x, imu_gyro_y, imu_gyro_z])
    
            # box_vel, box_acc = leastSquareVelAndAccSolver(imu_acc, imu_gyro, sensors, box)
            
            # print(box_acc)
            
            # # state: torch.Tensor = torch.cat((initial_rotation, torch.tensor(box_vel), torch.tensor(box_acc)), 0)
            # state: torch.Tensor = torch.cat((initial_rotation, torch.tensor(box_acc)), 0)
            # state = nimble.timestep(world, state, torch.zeros((world.getNumDofs())))
            
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
            break



    try:
        print('Ending Stream')
        imu.set_to_idle()
    except Exception as e:
       # pass
        exc_type, exc_obj, tb = sys.exc_info()
        line_number = tb.tb_lineno
        print(e)
        print(exc_type.__name__)
        print(f'Error Ending Stream on line {line_number}')





if __name__ == "__main__":
    raise SystemExit(main())