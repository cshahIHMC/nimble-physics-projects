import time
import sys
import numpy as np
from Libraries.Microstrain import MicroStrainIMU
import nimblephysics as nimble
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple

# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def safe_axis_angle(rotvec: np.ndarray) -> Tuple[np.ndarray, float]:
    theta = np.linalg.norm(rotvec)
    if theta < 1e-8:
        return np.zeros(3), 0.0
    return rotvec / theta, theta

def report_error(e):
    exc_type, _, tb = sys.exc_info()
    print(f"[{exc_type.__name__}] {e} (line {tb.tb_lineno})")

# Microstrain IMU data
def get_microstrain_imu_data(imu):
    data = imu.get_ESTFILTER_data(20, 0)
    device_time = data[0]
    
    quat = np.array([
        data[10].as_floatAt(0),
        data[10].as_floatAt(1),
        data[10].as_floatAt(2),
        data[10].as_floatAt(3)
    ])
        
    return np.array(data[1:10]), R.from_quat(quat, scalar_first=True)

def get_microstrain_acc_gyro_mag(data, plotter=None):
    acc_b = data[0:3] * 9.80665
    gyro_b = data[3:6]
    mag_b  = data[6:9]
    
    return acc_b, gyro_b, mag_b

def get_microstrain_quat(imu):
    data = imu.get_ESTFILTER_data(20, 0)
    quat = np.array([
        data[10].as_floatAt(0),
        data[10].as_floatAt(1),
        data[10].as_floatAt(2),
        data[10].as_floatAt(3)
    ])
    return R.from_quat(quat, scalar_first=True)

# ------------------------------------------------------------------
# Zeroing functions
# ------------------------------------------------------------------
def compute_imu_to_world_transform(acc_b, mag_b):
    """
    Compute the rotation matrix (scipy Rotation) that maps from IMU frame → anatomical world frame.
    
    For MicroStrain IMUs (pelvis + limbs): use accelerometer + magnetometer.
    For Insoles (no magnetometer): use gravity + pelvis forward direction to resolve yaw.
    """

    # --------------------------------------------------------------
    # 1. World UP direction from accelerometer (IMU frame)
    # --------------------------------------------------------------
    y_world = acc_b / np.linalg.norm(acc_b)
    
    print(y_world)

    # --------------------------------------------------------------
    # 2. MICROSTRAIN IMUs (with magnetometer)
    # --------------------------------------------------------------
    # Project magnetometer into horizontal plane
    mag_proj = mag_b - np.dot(mag_b, y_world) * y_world
    x_world = mag_proj / np.linalg.norm(mag_proj)  # forward (horizontal)
    
    # Orthogonal right axis
    z_world = np.cross(x_world, y_world)
    z_world /= np.linalg.norm(z_world)
    
    # Build rotation: columns = world axes in IMU frame
    R_imu_to_world = R.from_matrix(np.column_stack([x_world, y_world, z_world]))
    
    return R_imu_to_world

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    
    # Setup all the system constants
    sample_frequency = 200 # hz

    # Setup Microstrain IMU's
    pelvis_imu = MicroStrainIMU("195772", 921600)

    thigh_r_imu = MicroStrainIMU("195778", 921600)
    shank_r_imu = MicroStrainIMU("195773", 921600)
    foot_r_imu =  MicroStrainIMU("", 921600)
    
    thigh_l_imu = MicroStrainIMU("195775", 921600)
    shank_l_imu = MicroStrainIMU("196864", 921600)
    foot_l_imu = MicroStrainIMU("", 921600)

    try:
        pelvis_imu.configure_ESTFLTER_imu(sample_frequency)
        
        thigh_r_imu.configure_ESTFLTER_imu(sample_frequency)
        shank_r_imu.configure_ESTFLTER_imu(sample_frequency)
        foot_r_imu.configure_ESTFLTER_imu(sample_frequency)
        
        
        thigh_l_imu.configure_ESTFLTER_imu(sample_frequency)
        shank_l_imu.configure_ESTFLTER_imu(sample_frequency)
        foot_l_imu.configure_ESTFLTER_imu(sample_frequency)
        
    except Exception as e:
        report_error(e)
        pelvis_imu.set_to_idle()
        
        thigh_r_imu.set_to_idle()
        shank_r_imu.set_to_idle()
        foot_r_imu.set_to_idle()
        
        thigh_l_imu.set_to_idle()
        shank_l_imu.set_to_idle()
        foot_l_imu.set_to_idle()
        
        return
    
    try:
    
        # Get Data from the imu and generate the anatomical frames
        pelvis_imu_data, pelvis_imu_quat_0 = get_microstrain_imu_data(pelvis_imu)
        pelvis_acc, pelvis_gyro, pelvis_mag = get_microstrain_acc_gyro_mag(pelvis_imu_data)
        R_pelvis_anatomical = compute_imu_to_world_transform(pelvis_acc, pelvis_mag)
    
        thigh_r_imu_data, thigh_r_imu_quat_0 = get_microstrain_imu_data(thigh_r_imu)
        thigh_r_acc, thigh_r_gyro, thigh_r_mag = get_microstrain_acc_gyro_mag(thigh_r_imu_data)
        R_thigh_r_anatomical = compute_imu_to_world_transform(thigh_r_acc, thigh_r_mag)   
        
        shank_r_imu_data, shank_r_imu_quat_0  = get_microstrain_imu_data(shank_r_imu)
        shank_r_acc, shank_r_gyro, shank_r_mag = get_microstrain_acc_gyro_mag(shank_r_imu_data)
        R_shank_r_anatomical = compute_imu_to_world_transform(shank_r_acc, shank_r_mag)   

        foot_r_imu_data, foot_r_imu_quat_0  = get_microstrain_imu_data(foot_r_imu)
        foot_r_acc, foot_r_gyro, foot_r_mag = get_microstrain_acc_gyro_mag(foot_r_imu_data)
        R_foot_r_anatomical = compute_imu_to_world_transform(foot_r_acc, foot_r_mag)   
        
        thigh_l_imu_data, thigh_l_imu_quat_0 = get_microstrain_imu_data(thigh_l_imu)
        thigh_l_acc, thigh_l_gyro, thigh_l_mag = get_microstrain_acc_gyro_mag(thigh_l_imu_data)
        R_thigh_l_anatomical = compute_imu_to_world_transform(thigh_l_acc, thigh_l_mag)   
        
        shank_l_imu_data, shank_l_imu_quat_0  = get_microstrain_imu_data(shank_l_imu)
        shank_l_acc, shank_l_gyro, shank_l_mag = get_microstrain_acc_gyro_mag(shank_l_imu_data)
        R_shank_l_anatomical = compute_imu_to_world_transform(shank_l_acc, shank_l_mag)   

        foot_l_imu_data, foot_l_imu_quat_0  = get_microstrain_imu_data(foot_l_imu)
        foot_l_acc, foot_l_gyro, foot_l_mag = get_microstrain_acc_gyro_mag(foot_l_imu_data)
        R_foot_l_anatomical = compute_imu_to_world_transform(foot_l_acc, foot_l_mag)   
        
        
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

        joint_axes = {

            "pelvis_x": np.array([1, 0, 0]),
            "pelvis_y": np.array([0, 1, 0]),
            "pelvis_z": np.array([0, 0, 1]),

            "hip_x": np.array([1, 0, 0]),
            "hip_y": np.array([0, 1, 0]),
            "hip_z": np.array([0, 0, 1]),

            ####### Investigate later why I need the -1 here once I have brought all imu's to their anatomical frame ##############
            "r_knee": np.array([0, 0, -1]),

            # For left side
            "l_hip_x": np.array([-1, 0, 0]),
            "l_hip_y": np.array([0, -1, 0]),

            "l_knee": np.array([0, 0, -1]),

            ######## Feet Joint axes ###########
            "ankle_z": np.array([0, 0, 1]),
            "r_ankle_x": np.array([-1, 0, 0]),
            "l_ankle_x": np.array([1, 0, 0]),

        }    

        ## Setup timer
        t_prev = time.perf_counter()


        while True:
            
            
            # Get all the realtime data
            # Microstrain IMU's
            pelvis_quat = get_microstrain_quat(pelvis_imu)
            thigh_r_quat = get_microstrain_quat(thigh_r_imu)
            shank_r_quat = get_microstrain_quat(shank_r_imu)
            foot_r_quat = get_microstrain_quat(foot_r_imu)
            thigh_l_quat = get_microstrain_quat(thigh_l_imu)
            shank_l_quat = get_microstrain_quat(shank_l_imu)
            foot_l_quat = get_microstrain_quat(foot_l_imu)
            
            # Zero all the frames w.r.t starting frame
            pelvis_quat_zeroed = pelvis_imu_quat_0.inv() * pelvis_quat  
            
            thigh_r_quat_zeroed = thigh_r_imu_quat_0.inv() * thigh_r_quat     
            shank_r_quat_zeroed = shank_r_imu_quat_0.inv() * shank_r_quat
            foot_r_quat_zeroed = foot_r_imu_quat_0.inv() * foot_r_quat
            
            thigh_l_quat_zeroed = thigh_l_imu_quat_0.inv() * thigh_l_quat   
            shank_l_quat_zeroed = shank_l_imu_quat_0.inv() * shank_l_quat     
            foot_l_quat_zeroed = foot_l_imu_quat_0.inv() * foot_l_quat    
            
            # Change reference frame to the anatomical frame           
            pelvis_quat_joint_frame = R_pelvis_anatomical.inv() * pelvis_quat_zeroed * R_pelvis_anatomical
            
            thigh_r_quat_joint_frame = R_thigh_r_anatomical.inv() * thigh_r_quat_zeroed * R_thigh_r_anatomical
            shank_r_quat_joint_frame = R_shank_r_anatomical.inv() * shank_r_quat_zeroed * R_shank_r_anatomical
            foot_r_quat_joint_frame = R_foot_r_anatomical.inv() * foot_r_quat_zeroed * R_foot_r_anatomical
            
            thigh_l_quat_joint_frame = R_thigh_l_anatomical.inv() * thigh_l_quat_zeroed * R_thigh_l_anatomical
            shank_l_quat_joint_frame = R_shank_l_anatomical.inv() * shank_l_quat_zeroed * R_shank_l_anatomical 
            foot_l_quat_joint_frame = R_foot_l_anatomical.inv() * foot_l_quat_zeroed * R_foot_l_anatomical    
            

            # Calculate joint angles w.r.t parent segments
            thigh_r_quat_joint_frame_rel = pelvis_quat_joint_frame.inv() * thigh_r_quat_joint_frame
            shank_r_quat_joint_frame_rel = thigh_r_quat_joint_frame.inv() * shank_r_quat_joint_frame
            foot_r_quat_joint_frame_rel = shank_r_quat_joint_frame.inv() * foot_r_quat_joint_frame
            
            thigh_l_quat_joint_frame_rel = pelvis_quat_joint_frame.inv() * thigh_l_quat_joint_frame            
            shank_l_quat_joint_frame_rel = thigh_l_quat_joint_frame.inv() * shank_l_quat_joint_frame
            foot_l_quat_joint_frame_rel = shank_l_quat_joint_frame.inv() * foot_l_quat_joint_frame

            # Calculating the joint angles 
            
            pelvis_axis, pelvis_theta = safe_axis_angle(pelvis_quat_joint_frame.as_rotvec())
            thigh_r_axis, thigh_r_theta = safe_axis_angle(thigh_r_quat_joint_frame_rel.as_rotvec())
            shank_r_axis, shank_r_theta = safe_axis_angle(shank_r_quat_joint_frame_rel.as_rotvec())
            feet_r_axis, feet_r_theta = safe_axis_angle(foot_r_quat_joint_frame_rel.as_rotvec())
            
            thigh_l_axis, thigh_l_theta = safe_axis_angle(thigh_l_quat_joint_frame_rel.as_rotvec())
            shank_l_axis, shank_l_theta = safe_axis_angle(shank_l_quat_joint_frame_rel.as_rotvec())
            feet_l_axis, feet_l_theta = safe_axis_angle(foot_l_quat_joint_frame_rel.as_rotvec())
            
            # Compute joint angles
            pos = skeleton.getPositions()
            
            # pelvis
            pos[0] = np.dot(pelvis_axis, joint_axes["pelvis_z"]) * pelvis_theta
            pos[1] = np.dot(pelvis_axis, joint_axes["pelvis_x"]) * pelvis_theta
            pos[2] = np.dot(pelvis_axis, joint_axes["pelvis_y"]) * pelvis_theta
            
            # # Right Side
            pos[6] = np.dot(thigh_r_axis, joint_axes["hip_z"]) * thigh_r_theta
            pos[7] = np.dot(thigh_r_axis, joint_axes["hip_x"]) * thigh_r_theta
            pos[8] = np.dot(thigh_r_axis, joint_axes["hip_y"]) * thigh_r_theta
            pos[9] = np.dot(shank_r_axis, joint_axes["r_knee"]) * shank_r_theta
            pos[10] = np.dot(feet_r_axis, joint_axes["ankle_z"]) * feet_r_theta  # ankle angle_r
            pos[11] = np.dot(feet_r_axis, joint_axes["r_ankle_x"]) * feet_r_theta  # subtalar_angle_r
            # # pos[12] = np.dot(feet_l_axis,  np.array([1, 0, 0])) * feet_l_theta  # mtp_angle_r
           
                                    
            # Left Side
            pos[13] = np.dot(thigh_l_axis, joint_axes["hip_z"]) * thigh_l_theta
            pos[14] = np.dot(thigh_l_axis, joint_axes["l_hip_x"]) * thigh_l_theta
            pos[15] = np.dot(thigh_l_axis, joint_axes["l_hip_y"]) * thigh_l_theta
            pos[16] = np.dot(shank_l_axis, joint_axes["l_knee"]) * shank_l_theta
            pos[17] = np.dot(feet_l_axis, joint_axes["ankle_z"]) * feet_l_theta  # ankle angle_l
            pos[18] = np.dot(feet_l_axis, joint_axes["l_ankle_x"]) * feet_l_theta  # subtalar_angle_l
            # # pos[19] = np.dot(feet_l_axis,  np.array([1, 0, 0])) * feet_l_theta  # mtp_angle_l - toe movement - not used - we dont capture it only
            
            skeleton.setPositions(pos)
            gui.nativeAPI().renderWorld(world)
            
            # Simple rate limiter
            elapsed = time.perf_counter() - t_prev
            sleep_time = (1.0 / sample_frequency) - elapsed
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
        
        thigh_r_imu.set_to_idle()
        shank_r_imu.set_to_idle()
        foot_r_imu.set_to_idle()
         
        shank_l_imu.set_to_idle()
        thigh_l_imu.set_to_idle()   
        foot_l_imu.set_to_idle()


if __name__ == "__main__":
    raise SystemExit(main())           