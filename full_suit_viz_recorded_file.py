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
import os

import nimblephysics as nimble
import numpy as np
from Libraries.Microstrain import MicroStrainIMU
from Libraries.xsensor_driver_no_ros import XSENSORS
from scipy.spatial.transform import Rotation as R
import pandas as pd

DEBUG_PRINT = False
START_FRAME = 20
USE_XSENSOR_INSOLE = False

IMU_NAME_MAP = {
    1: "pelvis",
    2: "thigh_r",
    3: "shank_r",
    4: "thigh_l",
    5: "shank_l",
}

if USE_XSENSOR_INSOLE is False:
    IMU_NAME_MAP = {
    1: "pelvis",
    2: "thigh_r",
    3: "shank_r",
    4: "thigh_l",
    5: "shank_l",
    6: "foot_r",
    7: "foot_l"
}

IMU_FIELDS = [
    "timestamp",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
    "quat", "est_quat",
    "deltaThetax", "deltaThetay", "deltaThetaz",
    "deltaVelx", "deltaVely", "deltaVelz",
]


# ---------------------------------------------------------------------------
# LOAD FULL CSV FILE
# ---------------------------------------------------------------------------
def load_trial(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV file and return a full pandas DataFrame.

    Parameters
    ----------
    csv_path : str
        Path to the CSV saved by your logging script.

    Returns
    -------
    df : pandas.DataFrame
    """
    df = pd.read_csv(csv_path)
    df = remap_imu_columns(df)
    print(f"[load_trial] Loaded {csv_path} with shape {df.shape}")
    return df

def remap_imu_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}

    for idx, body_name in IMU_NAME_MAP.items():
        for f in IMU_FIELDS:
            old = f"imu{idx}_{f}"
            new = f"{body_name}_{f}"
            if old in df.columns:
                renamed[old] = new

    return df.rename(columns=renamed)

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

def get_insole_state(df, side: str, frame: int):
    """
    side: 'L' or 'R'
    frame: row index

    Returns:
        R_quat : Rotation object
        accel  : np.array([ax, ay, az])
        gyro   : np.array([gx, gy, gz])
    """

    side = side.upper()
    if side not in ["L", "R"]:
        raise ValueError("side must be 'L' or 'R'")

    # Prefix
    prefix = f"{side}_insole"

    # ---------------------------------------
    # Acceleration
    # ---------------------------------------
    accel = np.array([
        float(df[f"{prefix}_accel_x"].iloc[frame]) * 9.80665,
        float(df[f"{prefix}_accel_y"].iloc[frame]) * 9.80665,
        float(df[f"{prefix}_accel_z"].iloc[frame]) * 9.80665,
    ])

    # ---------------------------------------
    # Gyro
    # ---------------------------------------
    gyro = np.array([
        float(df[f"{prefix}_gyro_x"].iloc[frame]),
        float(df[f"{prefix}_gyro_y"].iloc[frame]),
        float(df[f"{prefix}_gyro_z"].iloc[frame]),
    ])

    # ---------------------------------------
    # Quaternion (qx, qy, qz, qw)
    # scalar_first = False (correct for this ordering)
    # ---------------------------------------
    quat = np.array([
        float(df[f"{prefix}_qx"].iloc[frame]),
        float(df[f"{prefix}_qy"].iloc[frame]),
        float(df[f"{prefix}_qz"].iloc[frame]),
        float(df[f"{prefix}_qw"].iloc[frame]),
    ])

    R_quat = R.from_quat(quat, scalar_first=False)

    return R_quat, accel
# =============================================================================
# MicroStrain IMU helper functions
# =============================================================================

# ---------------------------------------------------------------
# Safe quaternion parser: handles "[0.1 0.2 0.3 0.4]" and variants
# ---------------------------------------------------------------
def parse_quat(q):
    if isinstance(q, (list, tuple, np.ndarray)):
        return np.array(q, dtype=float)

    if isinstance(q, str):
        clean = q.replace(",", "").replace("[", "").replace("]", "")
        return np.fromstring(clean, sep=" ")

    raise ValueError(f"Cannot parse quaternion from type {type(q)}")


# ---------------------------------------------------------------
# Main helper: extract IMU quat, accel, mag at a given frame
# ---------------------------------------------------------------
def get_imu_state(df, imu_name: str, frame: int):
    """
    imu_name: "pelvis", "thigh_r", "shank_r", "thigh_l", "shank_l"
    frame: row index
    
    Returns:
        R_quat      (Rotation object)
        accel       np.array([ax, ay, az])
        mag         np.array([mx, my, mz])
    """

    # -------- Parse quaternion safely --------
    quat_raw = df[f"{imu_name}_est_quat"].iloc[frame]
    quat = parse_quat(quat_raw)
    R_quat = R.from_quat(quat, scalar_first=True)

    # -------- Acceleration --------
    accel = np.array([
        float(df[f"{imu_name}_accel_x"].iloc[frame]) * 9.80665,
        float(df[f"{imu_name}_accel_y"].iloc[frame]) * 9.80665,
        float(df[f"{imu_name}_accel_z"].iloc[frame]) * 9.80665,
    ])

    # -------- Magnetometer --------
    mag = np.array([
        float(df[f"{imu_name}_mag_x"].iloc[frame]),
        float(df[f"{imu_name}_mag_y"].iloc[frame]),
        float(df[f"{imu_name}_mag_z"].iloc[frame]),
    ])

    return R_quat, accel, mag

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
    # Load File and get a pandas dataframe
    # ---------------------------------------------------------------------
    trial = input("Enter trial file name (e.g., trial2.csv): ").strip()
    if not trial.endswith(".csv"):
        trial += ".csv"
    logs_root = os.path.join(os.getcwd(), "logs")
    
    # --- Search for this file inside logs/ recursively ---
    matches = []
    for root, _, files in os.walk(logs_root):
        if trial in files:
            matches.append(os.path.join(root, trial))

    if not matches:
        raise FileNotFoundError(f"'{trial}' not found in any folder inside logs/")
    
    file_path = matches[0]
    print(file_path)
    
    # --- Load & print basic data ---
    df = load_trial(file_path)
    
    # ---------------------------------------------------------------------
    # 2. Initial calibration: zero quats + compute anatomical frames
    # ---------------------------------------------------------------------
    
    # Get Data from the imu and generate the anatomical frames
    pelvis_quat_0, pelvis_acc, pelvis_mag = get_imu_state(df, "pelvis", START_FRAME) 
    R_pelvis_anatomical = compute_imu_to_world_transform(pelvis_acc, pelvis_mag)

    thigh_r_quat_0, thigh_r_acc, thigh_r_mag = get_imu_state(df, "thigh_r", START_FRAME) 
    R_thigh_r_anatomical = compute_imu_to_world_transform(thigh_r_acc, thigh_r_mag)
    
    shank_r_quat_0, shank_r_acc, shank_r_mag = get_imu_state(df, "shank_r", START_FRAME) 
    R_shank_r_anatomical = compute_imu_to_world_transform(shank_r_acc, shank_r_mag)
    
    thigh_l_quat_0, thigh_l_acc, thigh_l_mag = get_imu_state(df, "thigh_l", START_FRAME) 
    R_thigh_l_anatomical = compute_imu_to_world_transform(thigh_l_acc, thigh_l_mag)
    
    shank_l_quat_0, shank_l_acc, shank_l_mag = get_imu_state(df, "shank_l", START_FRAME) 
    R_shank_l_anatomical = compute_imu_to_world_transform(shank_l_acc, shank_l_mag)
                
    
    if USE_XSENSOR_INSOLE:
        foot_r_quat_0, foot_r_acc_0 = get_insole_state(df, "R", START_FRAME)
        foot_l_quat_0, foot_l_acc_0 = get_insole_state(df, "L", START_FRAME)

        R_foot_r_anatomical = compute_imu_to_world_transform(acc_b=foot_r_acc_0, pelvis_R_anatomical=R_pelvis_anatomical)
        R_foot_l_anatomical = compute_imu_to_world_transform(acc_b=foot_l_acc_0, pelvis_R_anatomical=R_pelvis_anatomical, left_foot=True)
    else:
        
        foot_r_quat_0, foot_r_acc, foot_r_mag = get_imu_state(df, "foot_r", START_FRAME) 
        R_foot_r_anatomical = compute_imu_to_world_transform(foot_r_acc, foot_r_mag)
    
        
        foot_l_quat_0, foot_l_acc, foot_l_mag = get_imu_state(df, "foot_l", START_FRAME) 
        R_foot_l_anatomical = compute_imu_to_world_transform(foot_l_acc, foot_l_mag)
                
        
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
    for FRAME in range(START_FRAME+1, df.shape[0]):
        
        
        # -------------------------------------------------------------
        # 4.1 Read all current quats from IMUs & insoles
        # -------------------------------------------------------------
        pelvis_quat, _, _ = get_imu_state(df, "pelvis", FRAME)
        thigh_r_quat, _, _ = get_imu_state(df, "thigh_r", FRAME)
        shank_r_quat, _, _ = get_imu_state(df, "shank_r", FRAME)
        thigh_l_quat, _, _ = get_imu_state(df, "thigh_l", FRAME)
        shank_l_quat, _, _ = get_imu_state(df, "shank_l", FRAME)
        
        if USE_XSENSOR_INSOLE:
            foot_r_quat, _ = get_insole_state(df, "R", FRAME)
            foot_l_quat, _ = get_insole_state(df, "L", FRAME)
        else:
            foot_r_quat, _, _ = get_imu_state(df, "foot_r", FRAME)
            foot_l_quat, _, _  = get_imu_state(df, "foot_l", FRAME)
            
        
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
        if USE_XSENSOR_INSOLE:
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
        else:
            foot_l_quat_joint_frame = (
                R_foot_l_anatomical.inv()
                * foot_l_quat_zeroed
                * R_foot_l_anatomical
            )
            foot_r_quat_joint_frame = (
                R_foot_r_anatomical.inv()
                * foot_r_quat_zeroed
                * R_foot_r_anatomical
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

    return 0

if __name__ == "__main__":
    raise SystemExit(main())