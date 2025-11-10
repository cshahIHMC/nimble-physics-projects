from xsensor_driver_no_ros import XSENSOR_Data
from xsensor_driver_no_ros import XSENSORS
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import nimblephysics as nimble
from Tools.live_plotter import LivePlotter3D

def infer_mapping(q_scipy: R, a_body: np.ndarray):
    """
    q_scipy: Rotation from your stream (as you're using now)
    a_body: 3-vector accelerometer in sensor/body frame, units don't matter (normalize)
    Returns which mapping matches: 'world->body' or 'body->world'
    """
    g_world = np.array([0, 0, -1.0])         # conventional down in world
    a_b = a_body / (np.linalg.norm(a_body) + 1e-12)

    # Hypothesis 1: q maps WORLD -> BODY (active)
    a_pred_b1 = q_scipy.apply(g_world)       # rotate world gravity into body
    e1 = np.linalg.norm(a_pred_b1 - a_b)

    # Hypothesis 2: q maps BODY -> WORLD (active)
    # then WORLD -> BODY is q^{-1}
    a_pred_b2 = q_scipy.inv().apply(g_world)
    e2 = np.linalg.norm(a_pred_b2 - a_b)

    return 'world->body' if e1 < e2 else 'body->world'


def get_quaternion(xsensors):
    """
    Reads latest left insole data from the XSensor interface and 
    returns its orientation as a scipy Rotation object.
    """
    # Publish or read data from both sides
    left_insole_data, right_insole_data = xsensors.publish_data()
    
    accx = right_insole_data.linx * 9.80665
    accy = right_insole_data.liny * 9.80665
    accz = right_insole_data.linz * 9.80665
    
    acc = np.array([
        accx,
        accy,
        accz
    ], dtype=float)
    

    # Extract quaternion as numpy array
    q = np.array([
        right_insole_data.qx,
        right_insole_data.qy,
        right_insole_data.qz,
        right_insole_data.qw
    ], dtype=float)
    


    # Normalize to unit quaternion
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Quaternion magnitude is zero — invalid data from left insole.")
    q /= norm
    # Return scipy Rotation object
    quat_rot = R.from_quat(q, scalar_first=False)
    
    return quat_rot, acc

def main():
    
    # Plotter for testing
    # plotter = LivePlotter3D(100)
    
        # --- World Setup ---
    world = nimble.simulation.World()
    world.setGravity([0, 0, 0])  # No gravity for visualization

    # --- Create a simple body to represent IMU ---
    box = nimble.dynamics.Skeleton()
    boxJoint, boxBody = box.createBallJointAndBodyNodePair()
    boxShape = boxBody.createShapeNode(nimble.dynamics.BoxShape([0.1, 0.15, 0.05]))
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
    
    
    # change this if change configurations with latte panda, but shouldn't need to 
    TCP_IP = '192.168.0.1'

    NUM_INSOLES = 2
    
    sample_frequency = 200 # hz
    sample_period = 1.0 / sample_frequency
    next_time = time.perf_counter()
    
    xsensors = XSENSORS(num_xsensors = NUM_INSOLES)
    xsensorConnection = xsensors.start_server(tcp_ip=TCP_IP, startup=True)
    
    feet_quat_0, acc_0 = get_quaternion(xsensors=xsensors)
    
    
    ##################################################################
    # Had to do this - not ideal at all ! ###################
    # --- Add a yaw (Z-axis) correction ---
    yaw_correction_deg = 180.0  # try 90, 180, or -90 as needed
    R_z_fix = R.from_euler('z', yaw_correction_deg, degrees=True)
    
    ###################################################################

    
    while True:
        try:
            

            
            feet_quat, acc = get_quaternion(xsensors=xsensors)
       
            feet_quat_zeroed = feet_quat_0.inv() * feet_quat 
            
            ## Fixing this x y issue
            feet_rotvec = (R_z_fix * feet_quat_zeroed).as_rotvec() 


        
            
            # --- Update simulation state ---
            state = torch.cat((torch.tensor(feet_rotvec), initial_velocity), 0)
            state = nimble.timestep(world, state, torch.zeros(world.getNumDofs()))

            # --- Render the updated state ---
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