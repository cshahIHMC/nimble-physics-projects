import time
import sys
import numpy as np
from Microstrain import MicroStrainIMU
import nimblephysics as nimble
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple
# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

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
    R_imu_to_world = R.from_matrix(np.vstack([x_world, y_world, z_world]))

    return R_imu_to_world


def rotation_matrix_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rotation_matrix_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotation_matrix_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def safe_axis_angle(rotvec: np.ndarray) -> Tuple[np.ndarray, float]:
    theta = np.linalg.norm(rotvec)
    if theta < 1e-8:
        return np.zeros(3), 0.0
    return rotvec / theta, theta

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

def report_error(e):
    exc_type, _, tb = sys.exc_info()
    print(f"[{exc_type.__name__}] {e} (line {tb.tb_lineno})")

# ------------------------------------------------------------------
# Zeroing functions
# ------------------------------------------------------------------
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
    R_imu_to_world = R.from_matrix(np.vstack([x_world, y_world, z_world]))

    return R_imu_to_world



# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    NODE_RATE = 200
    
    pelvis_imu = MicroStrainIMU("195772", 921600)

    # shank_r_imu = MicroStrainIMU("195773", 921600)
    thigh_r_imu = MicroStrainIMU("195778", 921600)
    
    # thigh_l_imu = MicroStrainIMU("195775", 921600)
    # shank_l_imu = MicroStrainIMU("196864", 921600)

    try:
        pelvis_imu.configure_ESTFLTER_imu(NODE_RATE)
        
        # shank_r_imu.configure_ESTFLTER_imu(NODE_RATE)
        thigh_r_imu.configure_ESTFLTER_imu(NODE_RATE)
        
        # thigh_l_imu.configure_ESTFLTER_imu(NODE_RATE)
        # shank_l_imu.configure_ESTFLTER_imu(NODE_RATE)
    except Exception as e:
        report_error(e)
        pelvis_imu.set_to_idle()
        
        # shank_r_imu.set_to_idle()
        thigh_r_imu.set_to_idle()
        
        # thigh_l_imu.set_to_idle()
        # shank_l_imu.set_to_idle()
        
        return
    
    
    # Nimble world
    world = nimble.simulation.World()
    world.setGravity([0, 0, 0])

    opensim_model = nimble.RajagopalHumanBodyModel()
    skeleton = opensim_model.skeleton
    world.addSkeleton(skeleton)

    # Precompute transforms
    # transform_right_imu_to_world = R.from_matrix(rotation_matrix_z(-np.pi / 2) @ rotation_matrix_x(np.pi))
    
    # if right imu is at front
    transform_right_imu_to_world = R.from_matrix(rotation_matrix_x(np.pi / 2) @ rotation_matrix_y(np.pi/2))
    
    transform_right_imu_to_world_inv = transform_right_imu_to_world.inv()
    
    transform_left_imu_to_world = R.from_matrix(rotation_matrix_z(-np.pi / 2))
    transform_left_imu_to_world_inv = transform_left_imu_to_world.inv()
    
    transform_pelvis_imu_to_world = R.from_matrix(rotation_matrix_x(-np.pi / 2) @ rotation_matrix_y(-np.pi/2))
    transform_pelvis_imu_to_world_inv = transform_pelvis_imu_to_world.inv()


    # Get Data from the imu
    pelvis_imu_data, pelvis_imu_quat_data = get_imu_data(pelvis_imu)
    pelvis_acc, pelvis_gyro, pelvis_mag = read_sensor(pelvis_imu_data)
    R_pelvis_anatomical = compute_imu_to_world_transform(pelvis_acc, pelvis_gyro, pelvis_mag )
 
    thigh_r_imu_data, thigh_r_imu_quat_data = get_imu_data(thigh_r_imu)
    thigh_r_acc, thigh_r_gyro, thigh_r_mag = read_sensor(thigh_r_imu_data)
    R_thigh_r_anatomical = compute_imu_to_world_transform(thigh_r_acc, thigh_r_gyro, thigh_r_mag)   
    
    
    # Initial quaternions (zero reference)
    pelvis_quat0 = R.from_quat(pelvis_imu_quat_data, scalar_first=True)
    
    # shank_r_quat0 = R.from_quat(get_estfilter_data(shank_r_imu), scalar_first=True)
    thigh_r_quat0 = R.from_quat(thigh_r_imu_quat_data, scalar_first=True)
    
    # shank_l_quat0 = R.from_quat(get_estfilter_data(shank_l_imu), scalar_first=True)
    # thigh_l_quat0 = R.from_quat(get_estfilter_data(thigh_l_imu), scalar_first=True)

    # GUI
    gui = nimble.NimbleGUI(world)
    gui.serve(8080)
    gui.nativeAPI().renderWorld(world)
    gui.nativeAPI().renderBasis(scale=0.5)

    tibia_r_node = skeleton.getBodyNode("tibia_l")
    tibia_tf = tibia_r_node.getWorldTransform()
    gui.nativeAPI().renderBasis(
        scale=0.3,
        pos=tibia_tf.translation(),
        euler=nimble.math.matrixToEulerXYZ(tibia_tf.rotation()),
        prefix="tibia_basis"
    )

    joint_axes = {
        
        "knee": np.array([0, 0, -1]),
        
        # for thigh imu in front config
        "r_knee": np.array([1, 0, 0]),
        
        "hip_x": np.array([1, 0, 0]),
        "hip_y": np.array([0, 1, 0]),
        "hip_z": np.array([0, 0, 1]),
        
        "r_hip_front_x": np.array([-1, 0, 0]),
        "r_hip_front_y": np.array([0, 1, 0]),
        "r_hip_front_z": np.array([0, 0, -1]),
        
        "l_hip_x": np.array([1, 0, 0]),
        "l_hip_y": np.array([0, 1, 0]),
        "pelvis_x": np.array([1, 0, 0]),
        "pelvis_y": np.array([0, 1, 0]),
        "pelvis_z": np.array([0, 0, 1])
    }
    
    
    t_prev = time.perf_counter()

    try:
        while True:
            
            # Get Pelvis data
            pelvis_quat_data = get_estfilter_data(pelvis_imu)
            pelvis_quat =  R.from_quat(pelvis_quat_data, scalar_first=True)
            
            
            # # Get Right Limb data
            # shank_r_quat_data = get_estfilter_data(shank_r_imu)
            thigh_r_quat_data = get_estfilter_data(thigh_r_imu)
            
            # shank_r_quat = R.from_quat(shank_r_quat_data, scalar_first=True)
            thigh_r_quat = R.from_quat(thigh_r_quat_data, scalar_first=True)
            
            # # Get Left limb data
            # shank_l_quat_data = get_estfilter_data(shank_l_imu)
            # thigh_l_quat_data = get_estfilter_data(thigh_l_imu)
            
            # shank_l_quat = R.from_quat(shank_l_quat_data, scalar_first=True)
            # thigh_l_quat = R.from_quat(thigh_l_quat_data, scalar_first=True)            

            # Zero relative rotation
            # Zero pelvis
            pelvis_quat_zeroed = pelvis_quat0.inv() * pelvis_quat
            
            # # Zero right side
            # shank_r_quat_zeroed = shank_r_quat0.inv() * shank_r_quat
            thigh_r_quat_zeroed = thigh_r_quat0.inv() * thigh_r_quat
            
            # # Zero Left Side
            # shank_l_quat_zeroed = shank_l_quat0.inv() * shank_l_quat
            # thigh_l_quat_zeroed = thigh_l_quat0.inv() * thigh_l_quat
            
            
            # # Shank relative to thigh
            # shank_r_quat_rel = thigh_r_quat_zeroed.inv() * shank_r_quat_zeroed
            # shank_r_quat_rel = (thigh_r_quat0.inv() * shank_r_quat0).inv() * (thigh_r_quat.inv() * shank_r_quat)
            # shank_l_quat_rel = thigh_l_quat_zeroed.inv() * shank_l_quat_zeroed
            
            # thigh_r_quat_rel = (pelvis_quat0.inv() * thigh_r_quat0).inv() * (pelvis_quat.inv() * thigh_r_quat)
            # thigh_l_quat_rel = (pelvis_quat0.inv() * thigh_l_quat0).inv() * (pelvis_quat.inv() * thigh_l_quat)
            
            thigh_r_quat_rel = (thigh_r_quat0).inv() * (thigh_r_quat)
            # thigh_l_quat_rel = (thigh_l_quat0).inv() * (thigh_l_quat)

            # Transform to joint frame
            # pelvis_quat_joint_frame = transform_pelvis_imu_to_world * pelvis_quat_zeroed * transform_pelvis_imu_to_world_inv
            
            # shank_r_quat_joint_frame = transform_right_imu_to_world * shank_r_quat_rel * transform_right_imu_to_world_inv
            # thigh_r_quat_joint_frame = transform_right_imu_to_world * thigh_r_quat_rel * transform_right_imu_to_world_inv
            
            # thigh_l_quat_joint_frame = transform_left_imu_to_world * thigh_l_quat_rel * transform_left_imu_to_world_inv
            # shank_l_quat_joint_frame = transform_left_imu_to_world * shank_l_quat_rel * transform_left_imu_to_world_inv
            
            
            ##################### New Zeroing approach  ##############################
            pelvis_quat_joint_frame = R_pelvis_anatomical * pelvis_quat_zeroed * R_pelvis_anatomical.inv()
            thigh_r_quat_joint_frame = R_thigh_r_anatomical * thigh_r_quat_rel * R_thigh_r_anatomical.inv()
            
            ### Now I can safely calculate the relative values
            thigh_r_quat_joint_frame_rel = pelvis_quat_joint_frame.inv() * thigh_r_quat_joint_frame
            
            ###############################################################3
            
            pelvis_axis, pelvis_theta = safe_axis_angle(pelvis_quat_joint_frame.as_rotvec())

            # shank_r_axis, shank_r_theta = safe_axis_angle(shank_r_quat_joint_frame.as_rotvec())
            thigh_r_axis, thigh_r_theta = safe_axis_angle(thigh_r_quat_joint_frame_rel.as_rotvec())
            
            # shank_l_axis, shank_l_theta = safe_axis_angle(shank_l_quat_joint_frame.as_rotvec())
            # thigh_l_axis, thigh_l_theta = safe_axis_angle(thigh_l_quat_joint_frame.as_rotvec())

            # Compute joint angles
            pos = skeleton.getPositions()
            
            
            # pelvis
            pos[0] = np.dot(pelvis_axis, joint_axes["pelvis_z"]) * pelvis_theta
            pos[1] = np.dot(pelvis_axis, joint_axes["pelvis_x"]) * pelvis_theta
            pos[2] = np.dot(pelvis_axis, joint_axes["pelvis_y"]) * pelvis_theta
            
            # Right Side
            # pos[9] = np.dot(shank_r_axis, joint_axes["r_knee"]) * shank_r_theta
            pos[6] = np.dot(thigh_r_axis, joint_axes["hip_z"]) * thigh_r_theta
            pos[7] = np.dot(thigh_r_axis, joint_axes["hip_x"]) * thigh_r_theta
            pos[8] = np.dot(thigh_r_axis, joint_axes["hip_y"]) * thigh_r_theta
            
            
            # Imu location on front of thigh
            # pos[6] = np.dot(thigh_r_axis, joint_axes["r_hip_front_z"]) * thigh_r_theta
            # pos[7] = np.dot(thigh_r_axis, joint_axes["r_hip_front_x"]) * thigh_r_theta
            # pos[8] = np.dot(thigh_r_axis, joint_axes["r_hip_front_y"]) * thigh_r_theta
                        
            # Left Side
            # pos[16] = np.dot(shank_l_axis, joint_axes["knee"]) * shank_l_theta
            # pos[13] = np.dot(thigh_l_axis, joint_axes["hip_z"]) * thigh_l_theta
            # pos[14] = np.dot(thigh_l_axis, joint_axes["l_hip_x"]) * thigh_l_theta
            # pos[15] = np.dot(thigh_l_axis, joint_axes["l_hip_y"]) * thigh_l_theta
                       
            skeleton.setPositions(pos)

            gui.nativeAPI().renderWorld(world)

            # Simple rate limiter
            elapsed = time.perf_counter() - t_prev
            sleep_time = (1.0 / NODE_RATE) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            t_prev = time.perf_counter()

    except KeyboardInterrupt:
        print("\nTerminated by user.")
    except Exception as e:
        report_error(e)
    finally:
        print("Ending Stream.")
        pelvis_imu.set_to_idle()
        
        # shank_r_imu.set_to_idle()
        thigh_r_imu.set_to_idle()
        
        # shank_l_imu.set_to_idle()
        # thigh_l_imu.set_to_idle()


if __name__ == "__main__":
    raise SystemExit(main())
