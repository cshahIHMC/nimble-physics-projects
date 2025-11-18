"""
===============================================================================
Script Name: realtime_lowerlimb_imu_insole_viz.py
Author: Chinmay Shah (cleanup & comments assisted by ChatGPT)
Date: Nov 2025
===============================================================================

Description:
    Streams real-time orientation data from:
        - MicroStrain IMUs (pelvis, thighs, shanks)
        - XSENSOR insoles (feet with built-in IMU)
    and maps them onto the Rajagopal human model in Nimble Physics.

    The pipeline is:

        1) Initialize and configure all sensors.
        2) Grab an initial frame from each IMU/insole.
           - Compute "anatomical" frames using accel + mag (for MicroStrain).
           - For the feet, use gravity + pelvis-forward to resolve yaw.
        3) In the live loop:
           - Read current quaternions from all sensors.
           - Zero them w.r.t. their starting quaternions.
           - Change reference from sensor frame → anatomical frame.
           - Compute joint-relative quaternions (child w.r.t parent).
           - Convert quats → axis-angle, project on anatomical joint axes.
           - Write those angles into Nimble's Rajagopal skeleton.
           - Render.

    Coordinate-frame conventions (high level):
        - IMU frame: whatever the sensor provides.
        - World/anatomical frame: constructed from gravity + magnetometer
          (for MicroStrain) or gravity + pelvis direction (for insoles).
        - Joint frame: child segment expressed relative to parent.

Usage:
    Run directly:

        $ python3 realtime_lowerlimb_imu_insole_viz.py

    Make sure:
        - MicroStrain devices are connected with correct IDs.
        - XSENSOR insole server is reachable at the given TCP_IP.

===============================================================================
"""
import sys
import time
from typing import List, Tuple

import nimblephysics as nimble
import numpy as np
from Libraries.Microstrain import MicroStrainIMU
from Libraries.xsensor_driver_no_ros import XSENSORS
from scipy.spatial.transform import Rotation as R


DEBUG_PRINT = True
# =============================================================================
# Utility functions
# =============================================================================

def safe_axis_angle(rotvec: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Convert a rotation vector to (axis, angle) with a robust check for small angles.

    Parameters
    ----------
    rotvec : np.ndarray, shape (3,)
        Rotation vector representation (axis * angle).

    Returns
    -------
    axis : np.ndarray, shape (3,)
        Unit rotation axis. If angle is ~0, returns [0, 0, 0].
    theta : float
        Rotation angle in radians. If norm < 1e-8, returns 0.
    """
    theta = np.linalg.norm(rotvec)
    if theta < 1e-8:
        return np.zeros(3), 0.0
    return rotvec / theta, theta


def report_error(e: Exception) -> None:
    """
    Print an exception with line number for quick debugging.
    """
    exc_type, _, tb = sys.exc_info()
    print(f"[{exc_type.__name__}] {e} (line {tb.tb_lineno})")


# =============================================================================
# XSENSOR insole helper functions
# =============================================================================

def get_insole_acc_quat(insole_data) -> Tuple[R, np.ndarray]:
    """
    Extract accelerometer + quaternion from a single insole data message.

    Parameters
    ----------
    insole_data
        Object returned by XSENSORS, expected to have attributes:
        - linx, liny, linz  (linear acceleration, presumably in g's)
        - qx, qy, qz, qw    (quaternion, scalar-last: [x, y, z, w])

    Returns
    -------
    quat_rot : scipy.spatial.transform.Rotation
        Orientation as a Rotation object.
    acc : np.ndarray, shape (3,)
        Linear acceleration in m/s^2 in the insole frame.
    """
    # Convert accel to m/s^2
    acc = np.array(
        [
            insole_data.linx * 9.80665,
            insole_data.liny * 9.80665,
            insole_data.linz * 9.80665,
        ],
        dtype=float,
    )

    # Quaternion: XSENSOR gives [x, y, z, w] (scalar-last)
    q = np.array(
        [insole_data.qx, insole_data.qy, insole_data.qz, insole_data.qw],
        dtype=float,
    )

    # Normalize quaternion
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Quaternion magnitude is zero — invalid data from insole.")
    q /= norm

    # For XSENSOR: scalar_last = True → use scalar_first=False
    quat_rot = R.from_quat(q, scalar_first=False)

    return quat_rot, acc


def get_insole_data(xsensors: XSENSORS) -> Tuple[R, np.ndarray, R, np.ndarray]:
    """
    Read the latest left/right insole data and return (quat, acc) for both.

    Parameters
    ----------
    xsensors : XSENSORS
        Insole interface object.

    Returns
    -------
    left_quat : Rotation
    left_acc : np.ndarray, shape (3,)
    right_quat : Rotation
    right_acc : np.ndarray, shape (3,)
    """
    left_insole_data, right_insole_data = xsensors.publish_data()

    left_quat, left_acc = get_insole_acc_quat(left_insole_data)
    right_quat, right_acc = get_insole_acc_quat(right_insole_data)

    return left_quat, left_acc, right_quat, right_acc

# =============================================================================
# MicroStrain IMU helper functions
# =============================================================================

def get_microstrain_imu_data(imu: MicroStrainIMU) -> Tuple[np.ndarray, R]:
    """
    Get one ESTFILTER data packet and extract 9-axis data + quaternion.

    Parameters
    ----------
    imu : MicroStrainIMU

    Returns
    -------
    data_1_to_9 : np.ndarray, shape (9,)
        Usually [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z].
    quat_rot : Rotation
        IMU orientation as a Rotation object with scalar-first convention [w, x, y, z].
    """
    data = imu.get_ESTFILTER_data(20, 0)

    quat = np.array(
        [
            data[10].as_floatAt(0),
            data[10].as_floatAt(1),
            data[10].as_floatAt(2),
            data[10].as_floatAt(3),
        ]
    )

    # MicroStrain: we are treating this as [w, x, y, z]
    quat_rot = R.from_quat(quat, scalar_first=True)

    return np.array(data[1:10]), quat_rot


def get_microstrain_acc_gyro_mag(
    data_1_to_9: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the 9-element MicroStrain data array into accel, gyro, mag.

    Parameters
    ----------
    data_1_to_9 : np.ndarray, shape (9,)
        [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z]

    Returns
    -------
    acc_b : np.ndarray, shape (3,)
        Linear acceleration in m/s^2 in IMU frame.
    gyro_b : np.ndarray, shape (3,)
        Angular velocity in rad/s (or deg/s depending on MicroStrain config).
    mag_b : np.ndarray, shape (3,)
        Magnetometer vector in IMU frame.
    """
    acc_b = data_1_to_9[0:3] * 9.80665
    gyro_b = data_1_to_9[3:6]
    mag_b = data_1_to_9[6:9]
    return acc_b, gyro_b, mag_b


def get_microstrain_quat(imu: MicroStrainIMU) -> R:
    """
    Get only the quaternion from the MicroStrain ESTFILTER packet.

    Parameters
    ----------
    imu : MicroStrainIMU

    Returns
    -------
    quat_rot : Rotation
        Current orientation as Rotation object.
    """
    data = imu.get_ESTFILTER_data(20, 0)
    quat = np.array(
        [
            data[10].as_floatAt(0),
            data[10].as_floatAt(1),
            data[10].as_floatAt(2),
            data[10].as_floatAt(3),
        ]
    )
    return R.from_quat(quat, scalar_first=True)

# =============================================================================
# Frame-construction / zeroing helpers
# =============================================================================

def compute_imu_to_world_transform(
    acc_b: np.ndarray,
    mag_b: np.ndarray | None = None,
    pelvis_R_anatomical: R | None = None,
    left_foot: bool = False,
) -> R:
    """
    Compute the rotation that maps IMU frame → anatomical world frame.

    Two modes:
        1) MicroStrain IMUs (pelvis, thighs, shanks):
            - Use (acc_b, mag_b).
            - Gravity gives world "up", magnetometer resolves yaw.

        2) Insoles (no magnetometer):
            - Use (acc_b) + pelvis_R_anatomical to borrow yaw from pelvis.
            - For left foot, flip x/z to get mirrored axes.

    Parameters
    ----------
    acc_b : np.ndarray, shape (3,)
        Linear acceleration in IMU frame (includes gravity).
    mag_b : np.ndarray, shape (3,), optional
        Magnetometer vector in IMU frame. If provided, MicroStrain logic is used.
    pelvis_R_anatomical : Rotation, optional
        Pelvis anatomical/world frame used to transfer yaw to insoles.
        Required if mag_b is None.
    left_foot : bool, default False
        If True, mirror x/z axes for left foot.

    Returns
    -------
    R_imu_to_world : Rotation
        Rotation mapping vectors in IMU frame to anatomical world frame.
    """
    # 1) World up direction (IMU frame) from gravity
    y_world = acc_b / np.linalg.norm(acc_b)

    # ----------------------------------------------------------------------
    # Case 1: MicroStrain IMUs (have magnetometer) → use mag to define yaw
    # ----------------------------------------------------------------------
    if mag_b is not None:
        # Project mag onto horizontal plane (remove vertical component)
        mag_proj = mag_b - np.dot(mag_b, y_world) * y_world
        x_world = mag_proj / np.linalg.norm(mag_proj)  # forward, in horizontal plane

        # Right axis from x × y
        z_world = np.cross(x_world, y_world)
        z_world /= np.linalg.norm(z_world)

        # Columns = world axes expressed in IMU frame
        R_imu_to_world = R.from_matrix(np.column_stack([x_world, y_world, z_world]))
        return R_imu_to_world

    # ----------------------------------------------------------------------
    # Case 2: Insoles (no magnetometer) → borrow yaw from pelvis
    # ----------------------------------------------------------------------
    if pelvis_R_anatomical is None:
        raise ValueError(
            "compute_imu_to_world_transform(): pelvis_R_anatomical must be provided "
            "when magnetometer is not available."
        )

    # Pelvis forward direction in world frame (X-axis in world)
    pelvis_forward_world = pelvis_R_anatomical.inv().apply([1.0, 0.0, 0.0])
    pelvis_forward_world /= np.linalg.norm(pelvis_forward_world)

    # Project pelvis forward into this IMU's horizontal plane
    x_world = pelvis_forward_world - np.dot(pelvis_forward_world, y_world) * y_world
    x_world /= np.linalg.norm(x_world)

    # Anatomical right axis
    z_world = np.cross(x_world, y_world)
    z_world /= np.linalg.norm(z_world)

    # For the left foot, mirror the x/z axes to match left/right anatomy
    if left_foot:
        x_world = -x_world
        z_world = -z_world

    R_imu_to_world = R.from_matrix(np.column_stack([x_world, y_world, z_world]))
    return R_imu_to_world




# =============================================================================
# Main
# =============================================================================

def main() -> int:
    
    
    # ---------------------------------------------------------------------
    # System Constants
    # ---------------------------------------------------------------------
    sample_frequency = 200 # hz
    NUM_INSOLES = 2
    TCP_IP = '192.168.0.1'
    
    # ---------------------------------------------------------------------
    # 1. Sensor setup
    # ---------------------------------------------------------------------
    
    # Setup Microstrain IMU's
    pelvis_imu = MicroStrainIMU("195772", 921600)

    shank_r_imu = MicroStrainIMU("195773", 921600)
    thigh_r_imu = MicroStrainIMU("195778", 921600)
    
    thigh_l_imu = MicroStrainIMU("195775", 921600)
    shank_l_imu = MicroStrainIMU("", 921600)

    try:
        # Configure all MicroStrain IMUs for ESTFILTER streaming
        for imu in [pelvis_imu, shank_r_imu, thigh_r_imu, thigh_l_imu, shank_l_imu]:
            imu.configure_ESTFLTER_imu(sample_frequency)
    except Exception as e:
        report_error(e)
        for imu in [pelvis_imu, shank_r_imu, thigh_r_imu, thigh_l_imu, shank_l_imu]:
            imu.set_to_idle()
        return 1
    
    
    # Setup xsensor insoles and IMU's
    xsensors = XSENSORS(num_xsensors = NUM_INSOLES)
    xsensorConnection = xsensors.start_server(tcp_ip=TCP_IP, startup=True)    
    
    # ---------------------------------------------------------------------
    # 2. Initial calibration: zero quats + compute anatomical frames
    # ---------------------------------------------------------------------
    
    try:
    
        # Get Data from the imu and generate the anatomical frames
        pelvis_imu_data, pelvis_quat_0 = get_microstrain_imu_data(pelvis_imu)
        pelvis_acc, pelvis_gyro, pelvis_mag = get_microstrain_acc_gyro_mag(pelvis_imu_data)
        R_pelvis_anatomical = compute_imu_to_world_transform(pelvis_acc, pelvis_mag)
    
        thigh_r_imu_data, thigh_r_quat_0 = get_microstrain_imu_data(thigh_r_imu)
        thigh_r_acc, thigh_r_gyro, thigh_r_mag = get_microstrain_acc_gyro_mag(thigh_r_imu_data)
        R_thigh_r_anatomical = compute_imu_to_world_transform(thigh_r_acc, thigh_r_mag)   
        
        shank_r_imu_data, shank_r_quat_0  = get_microstrain_imu_data(shank_r_imu)
        shank_r_acc, shank_r_gyro, shank_r_mag = get_microstrain_acc_gyro_mag(shank_r_imu_data)
        R_shank_r_anatomical = compute_imu_to_world_transform(shank_r_acc, shank_r_mag)   

        thigh_l_imu_data, thigh_l_quat_0 = get_microstrain_imu_data(thigh_l_imu)
        thigh_l_acc, thigh_l_gyro, thigh_l_mag = get_microstrain_acc_gyro_mag(thigh_l_imu_data)
        R_thigh_l_anatomical = compute_imu_to_world_transform(thigh_l_acc, thigh_l_mag)   
        
        shank_l_imu_data, shank_l_quat_0  = get_microstrain_imu_data(shank_l_imu)
        shank_l_acc, shank_l_gyro, shank_l_mag = get_microstrain_acc_gyro_mag(shank_l_imu_data)
        R_shank_l_anatomical = compute_imu_to_world_transform(shank_l_acc, shank_l_mag)   
        
        foot_l_quat_0, foot_l_acc_0, foot_r_quat_0, foot_r_acc_0 = get_insole_data(xsensors=xsensors)
        
        foot_l_anatomical = compute_imu_to_world_transform(acc_b=foot_l_acc_0, pelvis_R_anatomical=R_pelvis_anatomical, left_foot=True)
        foot_r_anatomical = compute_imu_to_world_transform(acc_b=foot_r_acc_0, pelvis_R_anatomical=R_pelvis_anatomical)
        
    except Exception as e:
        report_error(e)
        # Put all sensors to idle and bail
        for imu in [pelvis_imu, shank_r_imu, thigh_r_imu, thigh_l_imu, shank_l_imu]:
            imu.set_to_idle()
        xsensors.close_server()
        return 1
    # ---------------------------------------------------------------------
    # 3. Nimble world & GUI setup
    # ---------------------------------------------------------------------
        
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

    # Joint axes in anatomical coordinates.
    # These define how you project axis-angle onto OpenSim DOFs.
    joint_axes: dict[str, np.ndarray] = {
        # Pelvis
        "pelvis_x": np.array([1, 0, 0]),
        "pelvis_y": np.array([0, 1, 0]),
        "pelvis_z": np.array([0, 0, 1]),

        # Right hip
        "hip_x": np.array([1, 0, 0]),
        "hip_y": np.array([0, 1, 0]),
        "hip_z": np.array([0, 0, 1]),

        # Right knee (note sign flip)
        "r_knee": np.array([0, 0, -1]),

        # Left hip (mirrored)
        "l_hip_x": np.array([-1, 0, 0]),
        "l_hip_y": np.array([0, -1, 0]),

        # Left knee
        "l_knee": np.array([0, 0, -1]),

        # Feet joint axes
        "ankle_z": np.array([0, 0, 1]),
        "r_ankle_x": np.array([1, 0, 0]),
        "l_ankle_x": np.array([-1, 0, 0]),
    }   

    # Foot mounting correction (XSENSOR vs shank frame)
    # This is your "jerry-rig" to align insole IMU to shank anatomical frame.
    R_mount = R.from_euler("xyz", [0.0, 180.0, 0.0], degrees=True)

    # Simple rate limiter
    t_prev = time.perf_counter()

    # ---------------------------------------------------------------------
    # 4. Main streaming loop
    # ---------------------------------------------------------------------
    try:
        while True:
            
            
            # -------------------------------------------------------------
            # 4.1 Read all current quats from IMUs & insoles
            # -------------------------------------------------------------
            pelvis_quat = get_microstrain_quat(pelvis_imu)
            thigh_r_quat = get_microstrain_quat(thigh_r_imu)
            shank_r_quat = get_microstrain_quat(shank_r_imu)
            thigh_l_quat = get_microstrain_quat(thigh_l_imu)
            shank_l_quat = get_microstrain_quat(shank_l_imu)

            foot_l_quat, foot_l_acc, foot_r_quat, foot_r_acc = get_insole_data(
                xsensors=xsensors
            )

            # -------------------------------------------------------------
            # 4.2 Zero frames w.r.t. initial quaternions
            #     q_zeroed = q0^{-1} * q
            # -------------------------------------------------------------
            pelvis_quat_zeroed = pelvis_quat_0.inv() * pelvis_quat
            thigh_r_quat_zeroed = thigh_r_quat_0.inv() * thigh_r_quat
            shank_r_quat_zeroed = shank_r_quat_0.inv() * shank_r_quat
            thigh_l_quat_zeroed = thigh_l_quat_0.inv() * thigh_l_quat
            shank_l_quat_zeroed = shank_l_quat_0.inv() * shank_l_quat

            foot_l_quat_zeroed = foot_l_quat_0.inv() * foot_l_quat
            foot_r_quat_zeroed = foot_r_quat_0.inv() * foot_r_quat

            # -------------------------------------------------------------
            # 4.3 Change reference frame to anatomical frame:
            #
            #     q_anat = R_anat^{-1} * q_zeroed * R_anat
            #
            # This is a similarity transform that re-expresses the rotation
            # in the anatomical frame instead of sensor frame.
            # -------------------------------------------------------------
            pelvis_quat_joint_frame = (
                R_pelvis_anatomical.inv() * pelvis_quat_zeroed * R_pelvis_anatomical
            )
            thigh_r_quat_joint_frame = (
                R_thigh_r_anatomical.inv()
                * thigh_r_quat_zeroed
                * R_thigh_r_anatomical
            )
            shank_r_quat_joint_frame = (
                R_shank_r_anatomical.inv()
                * shank_r_quat_zeroed
                * R_shank_r_anatomical
            )
            thigh_l_quat_joint_frame = (
                R_thigh_l_anatomical.inv()
                * thigh_l_quat_zeroed
                * R_thigh_l_anatomical
            )
            shank_l_quat_joint_frame = (
                R_shank_l_anatomical.inv()
                * shank_l_quat_zeroed
                * R_shank_l_anatomical
            )

            # foot are expressed using shank anatomical frames plus R_mount
            foot_l_quat_joint_frame = (
                R_mount.inv()
                * R_shank_l_anatomical.inv()
                * foot_l_quat_zeroed
                * R_shank_l_anatomical
                * R_mount
            )
            foot_r_quat_joint_frame = (
                R_mount.inv()
                * R_shank_r_anatomical.inv()
                * foot_r_quat_zeroed
                * R_shank_r_anatomical
                * R_mount
            )
            
            # -------------------------------------------------------------
            # 4.4 Joint-relative rotations:
            #
            #     q_child_rel = q_parent^{-1} * q_child
            # -------------------------------------------------------------
            thigh_r_quat_rel = pelvis_quat_joint_frame.inv() * thigh_r_quat_joint_frame
            shank_r_quat_rel = thigh_r_quat_joint_frame.inv() * shank_r_quat_joint_frame

            thigh_l_quat_rel = pelvis_quat_joint_frame.inv() * thigh_l_quat_joint_frame
            shank_l_quat_rel = thigh_l_quat_joint_frame.inv() * shank_l_quat_joint_frame

            foot_r_quat_rel = shank_r_quat_joint_frame.inv() * foot_r_quat_joint_frame
            foot_l_quat_rel = shank_l_quat_joint_frame.inv() * foot_l_quat_joint_frame

            # -------------------------------------------------------------
            # 4.5 Axis-angle for each joint
            # -------------------------------------------------------------
            pelvis_axis, pelvis_theta = safe_axis_angle(
                pelvis_quat_joint_frame.as_rotvec()
            )
            thigh_r_axis, thigh_r_theta = safe_axis_angle(thigh_r_quat_rel.as_rotvec())
            shank_r_axis, shank_r_theta = safe_axis_angle(shank_r_quat_rel.as_rotvec())
            thigh_l_axis, thigh_l_theta = safe_axis_angle(thigh_l_quat_rel.as_rotvec())
            shank_l_axis, shank_l_theta = safe_axis_angle(shank_l_quat_rel.as_rotvec())

            foot_l_axis, foot_l_theta = safe_axis_angle(foot_l_quat_rel.as_rotvec())
            foot_r_axis, foot_r_theta = safe_axis_angle(foot_r_quat_rel.as_rotvec())

            # Optional debug print for right foot joint
            if DEBUG_PRINT:
                axis_dbg, theta_dbg = safe_axis_angle(foot_r_quat_rel.as_rotvec())
                print(
                    "Axis-Angle(deg about X,Y,Z):",
                    np.degrees(axis_dbg * theta_dbg),
                )
                
                
            # -------------------------------------------------------------
            # 4.6 Map axis-angle onto Rajagopal joint DOFs
            #
            #     angle_dof = (joint_axis · rot_axis) * rot_angle
            #
            # NOTE: pos indices (0..19) must match the Rajagopal DOF ordering.
            # -------------------------------------------------------------
            
            # Compute joint angles
            pos = skeleton.getPositions()
            
            # pelvis
            pos[0] = np.dot(pelvis_axis, joint_axes["pelvis_z"]) * pelvis_theta
            pos[1] = np.dot(pelvis_axis, joint_axes["pelvis_x"]) * pelvis_theta
            pos[2] = np.dot(pelvis_axis, joint_axes["pelvis_y"]) * pelvis_theta
            
            # Right Side
            pos[6] = np.dot(thigh_r_axis, joint_axes["hip_z"]) * thigh_r_theta
            pos[7] = np.dot(thigh_r_axis, joint_axes["hip_x"]) * thigh_r_theta
            pos[8] = np.dot(thigh_r_axis, joint_axes["hip_y"]) * thigh_r_theta
            pos[9] = np.dot(shank_r_axis, joint_axes["r_knee"]) * shank_r_theta
            pos[10] = np.dot(foot_r_axis, joint_axes["ankle_z"]) * foot_r_theta  # ankle angle_r
            pos[11] = np.dot(foot_r_axis, joint_axes["r_ankle_x"]) * foot_r_theta  # subtalar_angle_r
            # # pos[12] = np.dot(foot_l_axis,  np.array([1, 0, 0])) * foot_l_theta  # mtp_angle_r
           
                                    
            # Left Side
            pos[13] = np.dot(thigh_l_axis, joint_axes["hip_z"]) * thigh_l_theta
            pos[14] = np.dot(thigh_l_axis, joint_axes["l_hip_x"]) * thigh_l_theta
            pos[15] = np.dot(thigh_l_axis, joint_axes["l_hip_y"]) * thigh_l_theta
            pos[16] = np.dot(shank_l_axis, joint_axes["l_knee"]) * shank_l_theta
            pos[17] = np.dot(foot_l_axis, joint_axes["ankle_z"]) * foot_l_theta  # ankle angle_l
            pos[18] = np.dot(foot_l_axis, joint_axes["l_ankle_x"]) * foot_l_theta  # subtalar_angle_l
            # # pos[19] = np.dot(foot_l_axis,  np.array([1, 0, 0])) * foot_l_theta  # mtp_angle_l - toe movement - not used - we dont capture it only
            
        
            skeleton.setPositions(pos)

            gui.nativeAPI().renderWorld(world)
            
            
            # -------------------------------------------------------------
            # 4.7 Rate limiting to ~SAMPLE_FREQUENCY
            # -------------------------------------------------------------
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
        print("Ending stream.")
        # Make sure everything is safely idled/closed
        for imu in [pelvis_imu, shank_r_imu, thigh_r_imu, thigh_l_imu, shank_l_imu]:
            imu.set_to_idle()
        xsensors.close_server()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())