import time
import sys
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
from scipy.spatial.transform import Rotation as R

import nimblephysics as nimble
from Libraries.Microstrain import MicroStrainIMU

# ==============================================================================
# Utility functions
# ==============================================================================

def safe_axis_angle(rotvec: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Convert a rotation vector (axis * angle) into (unit_axis, angle).

    SciPy's Rotation.as_rotvec() returns:
        rotvec = axis_unit * theta

    where:
        - axis_unit is a unit vector (3,)
        - theta is the rotation magnitude in radians (scalar)
        - rotvec is a 3D vector (3,) with norm = theta

    This function:
        - Computes theta = ||rotvec||
        - Returns (axis_unit, theta)
        - Handles the near-zero rotation case robustly.

    Parameters
    ----------
    rotvec : np.ndarray
        3D rotation vector.

    Returns
    -------
    axis_unit : np.ndarray
        Unit rotation axis. Zero vector if theta is tiny.
    theta : float
        Rotation magnitude in radians.
    """
    theta = np.linalg.norm(rotvec)
    if theta < 1e-8:
        # No reliable axis if angle is ~0, so return zero axis and angle.
        return np.zeros(3), 0.0
    return rotvec / theta, theta


def report_error(e: Exception) -> None:
    """
    Print an error with the exception type and the line where it occurred.
    """
    exc_type, _, tb = sys.exc_info()
    print(f"[{exc_type.__name__}] {e} (line {tb.tb_lineno})")

# ==============================================================================
# MicroStrain IMU helpers
# ==============================================================================

def get_microstrain_imu_data(imu: MicroStrainIMU):
    """
    Read ESTFILTER data from a MicroStrain IMU and return:
        - raw numeric data[1:10] as a numpy array
        - orientation as a scipy Rotation (quaternion)

    Assumes:
        data[10] holds quaternion components (scalar-first convention).
    """
    data = imu.get_ESTFILTER_data(20, 0)

    # Quaternion: scalar-first (w, x, y, z)
    quat = np.array([
        data[10].as_floatAt(0),
        data[10].as_floatAt(1),
        data[10].as_floatAt(2),
        data[10].as_floatAt(3),
    ])

    # Return sensor-specific numeric fields (acc, gyro, mag, etc.)
    return np.array(data[1:10]), R.from_quat(quat, scalar_first=True)


def get_microstrain_acc_gyro_mag(data_1_to_9: np.ndarray):
    """
    Extract accelerometer, gyroscope, and magnetometer from data[1:10].

    Parameters
    ----------
    data_1_to_9 : np.ndarray
        Array of shape (9,) containing [acc_x, acc_y, acc_z,
                                        gyro_x, gyro_y, gyro_z,
                                        mag_x, mag_y, mag_z].

    Returns
    -------
    acc_b : np.ndarray
        Accelerometer in IMU/body frame (m/s^2), gravity-included.
    gyro_b : np.ndarray
        Gyro in IMU/body frame (rad/s).
    mag_b : np.ndarray
        Magnetometer in IMU/body frame (arbitrary units).
    """
    acc_b = data_1_to_9[0:3] * 9.80665  # Convert g to m/s^2
    gyro_b = data_1_to_9[3:6]
    mag_b = data_1_to_9[6:9]
    return acc_b, gyro_b, mag_b

def get_microstrain_quat(imu: MicroStrainIMU) -> R:
    """
    Fast helper for streaming: directly return the current quaternion
    as a scipy Rotation.
    """
    data = imu.get_ESTFILTER_data(20, 0)
    quat = np.array([
        data[10].as_floatAt(0),
        data[10].as_floatAt(1),
        data[10].as_floatAt(2),
        data[10].as_floatAt(3),
    ])
    return R.from_quat(quat, scalar_first=True)


# ------------------------------------------------------------------
# Zeroing functions
# ------------------------------------------------------------------
def compute_imu_to_world_transform(acc_b: np.ndarray, mag_b: np.ndarray) -> R:
    """
    Compute a rotation (scipy Rotation) that maps from IMU frame → local anatomical
    "world" frame at the time of calibration.

    Intuition:
    ----------
    We want to build a local 'anatomical world' frame for each IMU using:
        - gravity (from accelerometer) to define the UP direction
        - magnetometer to define the FORWARD direction in the horizontal plane

    Steps:
    ------
    1) Use accelerometer (which measures -g in static pose) to define 'world up'
       in the IMU frame. Depending on sign conventions, this will be consistent
       across sensors as long as you are consistent in how you interpret it.

    2) Project the magnetometer onto the horizontal plane orthogonal to 'up'
       to define a forward axis.

    3) Use cross products to build a right-handed orthonormal basis:
           x_world: forward (horizontal)
           y_world: up
           z_world: right

       Note: All these are expressed in the IMU frame.

    4) Stack these as columns to form a rotation matrix:
           R_imu_to_world = [ x_world | y_world | z_world ]

       This maps from local 'world' coordinates to IMU coordinates
       when used in the usual linear-algebra sense; but we treat
       it as IMU→world when we apply it in the sandwich transform:
           R_world^T * R_imu * R_world
       which effectively changes reference frame from IMU axes to
       anatomical axes.

    Parameters
    ----------
    acc_b : np.ndarray
        Accelerometer vector in IMU frame (m/s^2).
    mag_b : np.ndarray
        Magnetometer vector in IMU frame.

    Returns
    -------
    R_imu_to_world : scipy.spatial.transform.Rotation
        Rotation object encoding the anatomical 'world' frame for this IMU.
    """

    # -------------------------------
    # 1. World UP direction from accelerometer in IMU frame
    # -------------------------------
    y_world = acc_b / np.linalg.norm(acc_b)

    # Debug: see which way 'up' is pointing in IMU frame
    # print("y_world (IMU frame UP):", y_world)

    # -------------------------------
    # 2. Use magnetometer to get horizontal FORWARD direction
    # -------------------------------
    # Remove vertical component of mag to project onto horizontal plane
    mag_proj = mag_b - np.dot(mag_b, y_world) * y_world
    x_world = mag_proj / np.linalg.norm(mag_proj)  # forward (horizontal)
    
    # -------------------------------
    # 3. Orthogonal RIGHT axis using cross products
    # -------------------------------
    z_world = np.cross(x_world, y_world)
    z_world /= np.linalg.norm(z_world)
    
    # -------------------------------
    # 4. Build rotation matrix (columns are the 'world' axes in IMU frame)
    # -------------------------------
    R_imu_to_world = R.from_matrix(np.column_stack([x_world, y_world, z_world]))

    return R_imu_to_world

# ==============================================================================
# Data structure for IMU + calibration
# ==============================================================================

@dataclass
class SegmentIMU:
    """
    Represents one body segment's IMU and its calibration.

    Attributes
    ----------
    name : str
        Segment name (e.g., 'pelvis', 'thigh_r', etc.) for debugging.
    imu : MicroStrainIMU
        The IMU object.
    quat_0 : R
        Initial quaternion at calibration time (used for zeroing).
    R_anatomical : R
        IMU→world anatomical transform at calibration time.

    Methods
    -------
    zero_and_to_joint_frame():
        Read current quaternion, zero it w.r.t. quat_0 and transform to
        the anatomical frame of this segment.
    """
    name: str
    imu: MicroStrainIMU
    quat_0: R
    R_anatomical: R

    def read_current_quat(self) -> R:
        """Return current IMU orientation as scipy Rotation."""
        return get_microstrain_quat(self.imu)

    def zero_and_to_joint_frame(self) -> R:
        """
        Convert the current IMU quaternion into the segment's anatomical joint frame.

        Steps:
        ------
        1) Zero w.r.t. initial pose:
               q_zeroed = q0^{-1} * q_current

        2) Change reference frame from IMU axes to anatomical axes using the
           'sandwich' transform:
               q_joint = R_anatomical^{-1} * q_zeroed * R_anatomical

           Here R_anatomical encodes how the IMU axes relate to a canonical
           segment frame at calibration.
        """
        q_current = self.read_current_quat()
        q_zeroed = self.quat_0.inv() * q_current
        q_joint = self.R_anatomical.inv() * q_zeroed * self.R_anatomical
        return q_joint


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    
    # ------------------------------------------------------------------
    # Constants / configuration
    # ------------------------------------------------------------------
    sample_frequency = 200  # Hz
    dt_target = 1.0 / sample_frequency

    # ------------------------------------------------------------------
    # Setup MicroStrain IMUs
    # ------------------------------------------------------------------
    pelvis_imu = MicroStrainIMU("195772", 921600)

    thigh_r_imu = MicroStrainIMU("195778", 921600)
    shank_r_imu = MicroStrainIMU("195773", 921600)
    foot_r_imu =  MicroStrainIMU("", 921600)
    
    thigh_l_imu = MicroStrainIMU("195775", 921600)
    shank_l_imu = MicroStrainIMU("196864", 921600)
    foot_l_imu = MicroStrainIMU("", 921600)
    
    all_imus = [
        pelvis_imu,
        thigh_r_imu, shank_r_imu, foot_r_imu,
        thigh_l_imu, shank_l_imu, foot_l_imu,
    ]
        

    # Configure ESTFILTER for all IMUs
    try:
        for imu in all_imus:
            imu.configure_ESTFLTER_imu(sample_frequency)
    except Exception as e:
        report_error(e)
        for imu in all_imus:
            imu.set_to_idle()
        return

    # ------------------------------------------------------------------
    # Initial calibration: get initial quaternion + anatomical transform
    # ------------------------------------------------------------------
    try:
        # Pelvis
        pelvis_data, pelvis_quat_0 = get_microstrain_imu_data(pelvis_imu)
        pelvis_acc, pelvis_gyro, pelvis_mag = get_microstrain_acc_gyro_mag(pelvis_data)
        R_pelvis_anatomical = compute_imu_to_world_transform(pelvis_acc, pelvis_mag)

        # Right thigh
        thigh_r_data, thigh_r_quat_0 = get_microstrain_imu_data(thigh_r_imu)
        thigh_r_acc, thigh_r_gyro, thigh_r_mag = get_microstrain_acc_gyro_mag(thigh_r_data)
        R_thigh_r_anatomical = compute_imu_to_world_transform(thigh_r_acc, thigh_r_mag)

        # Right shank
        shank_r_data, shank_r_quat_0 = get_microstrain_imu_data(shank_r_imu)
        shank_r_acc, shank_r_gyro, shank_r_mag = get_microstrain_acc_gyro_mag(shank_r_data)
        R_shank_r_anatomical = compute_imu_to_world_transform(shank_r_acc, shank_r_mag)

        # Right foot
        foot_r_data, foot_r_quat_0 = get_microstrain_imu_data(foot_r_imu)
        foot_r_acc, foot_r_gyro, foot_r_mag = get_microstrain_acc_gyro_mag(foot_r_data)
        R_foot_r_anatomical = compute_imu_to_world_transform(foot_r_acc, foot_r_mag)

        # Left thigh
        thigh_l_data, thigh_l_quat_0 = get_microstrain_imu_data(thigh_l_imu)
        thigh_l_acc, thigh_l_gyro, thigh_l_mag = get_microstrain_acc_gyro_mag(thigh_l_data)
        R_thigh_l_anatomical = compute_imu_to_world_transform(thigh_l_acc, thigh_l_mag)

        # Left shank
        shank_l_data, shank_l_quat_0 = get_microstrain_imu_data(shank_l_imu)
        shank_l_acc, shank_l_gyro, shank_l_mag = get_microstrain_acc_gyro_mag(shank_l_data)
        R_shank_l_anatomical = compute_imu_to_world_transform(shank_l_acc, shank_l_mag)

        # Left foot
        foot_l_data, foot_l_quat_0 = get_microstrain_imu_data(foot_l_imu)
        foot_l_acc, foot_l_gyro, foot_l_mag = get_microstrain_acc_gyro_mag(foot_l_data)
        R_foot_l_anatomical = compute_imu_to_world_transform(foot_l_acc, foot_l_mag)

    except Exception as e:
        report_error(e)
        for imu in all_imus:
            imu.set_to_idle()
        return

    # Wrap all in SegmentIMU objects for clean real-time usage
    pelvis_seg = SegmentIMU("pelvis", pelvis_imu, pelvis_quat_0, R_pelvis_anatomical)

    thigh_r_seg = SegmentIMU("thigh_r", thigh_r_imu, thigh_r_quat_0, R_thigh_r_anatomical)
    shank_r_seg = SegmentIMU("shank_r", shank_r_imu, shank_r_quat_0, R_shank_r_anatomical)
    foot_r_seg  = SegmentIMU("foot_r",  foot_r_imu,  foot_r_quat_0,  R_foot_r_anatomical)

    thigh_l_seg = SegmentIMU("thigh_l", thigh_l_imu, thigh_l_quat_0, R_thigh_l_anatomical)
    shank_l_seg = SegmentIMU("shank_l", shank_l_imu, shank_l_quat_0, R_shank_l_anatomical)
    foot_l_seg  = SegmentIMU("foot_l",  foot_l_imu,  foot_l_quat_0,  R_foot_l_anatomical)
 
        
    # ------------------------------------------------------------------
    # Nimble world and skeleton setup
    # ------------------------------------------------------------------
    world = nimble.simulation.World()
    world.setGravity([0, 0, 0])  # No dynamics integration, just kinematics

    opensim_model = nimble.RajagopalHumanBodyModel()
    skeleton = opensim_model.skeleton
    world.addSkeleton(skeleton)

    # GUI setup (render 3D world + basis)
    gui = nimble.NimbleGUI(world)
    gui.serve(8080)
    gui.nativeAPI().renderWorld(world)
    gui.nativeAPI().renderBasis(scale=0.5)

    # Debug visualization of tibia basis
    tibia_r_node = skeleton.getBodyNode("tibia_r")
    tibia_tf = tibia_r_node.getWorldTransform()
    gui.nativeAPI().renderBasis(
        scale=0.3,
        pos=tibia_tf.translation(),
        euler=nimble.math.matrixToEulerXYZ(tibia_tf.rotation()),
        prefix="tibia_basis"
    )

  # ------------------------------------------------------------------
    # Joint axis definitions in anatomical frame
    # These define how axis-angle maps onto specific OpenSim joint DOFs.
    # ------------------------------------------------------------------
    joint_axes: Dict[str, np.ndarray] = {
        # Pelvis rotations about global axes
        "pelvis_x": np.array([1.0, 0.0, 0.0]),
        "pelvis_y": np.array([0.0, 1.0, 0.0]),
        "pelvis_z": np.array([0.0, 0.0, 1.0]),

        # Right hip DOFs (flexion/extension, adduction/abduction, internal/external)
        "hip_x": np.array([1.0, 0.0, 0.0]),
        "hip_y": np.array([0.0, 1.0, 0.0]),
        "hip_z": np.array([0.0, 0.0, 1.0]),

        # Right knee flexion axis (note the -1 sign tweak from your script)
        "r_knee": np.array([0.0, 0.0, -1.0]),

        # Left hip axes (flipped sign to match Rajagopal conventions)
        "l_hip_x": np.array([-1.0, 0.0, 0.0]),
        "l_hip_y": np.array([0.0, -1.0, 0.0]),

        # Left knee flexion axis
        "l_knee": np.array([0.0, 0.0, -1.0]),

        # Feet / ankle / subtalar axes
        "ankle_z": np.array([0.0, 0.0, 1.0]),
        "r_ankle_x": np.array([1.0, 0.0, 0.0]),
        "l_ankle_x": np.array([-1.0, 0.0, 0.0]),
    }
    
    # ------------------------------------------------------------------
    # Main real-time loop
    # ------------------------------------------------------------------
    t_prev = time.perf_counter()
    
    try:
        while True:
            
            # ----------------------------------------------------------
            # 1. Read current IMU orientations and convert to joint frames
            # ----------------------------------------------------------
            pelvis_q_joint = pelvis_seg.zero_and_to_joint_frame()

            thigh_r_q_joint = thigh_r_seg.zero_and_to_joint_frame()
            shank_r_q_joint = shank_r_seg.zero_and_to_joint_frame()
            foot_r_q_joint  = foot_r_seg.zero_and_to_joint_frame()

            thigh_l_q_joint = thigh_l_seg.zero_and_to_joint_frame()
            shank_l_q_joint = shank_l_seg.zero_and_to_joint_frame()
            foot_l_q_joint  = foot_l_seg.zero_and_to_joint_frame()
            
            # ----------------------------------------------------------
            # 2. Compute relative joint orientations (child w.r.t parent)
            # ----------------------------------------------------------
            # Pelvis is already in global joint frame
            pelvis_q_rel = pelvis_q_joint

            # Right side
            thigh_r_q_rel = pelvis_q_joint.inv() * thigh_r_q_joint
            shank_r_q_rel = thigh_r_q_joint.inv() * shank_r_q_joint
            foot_r_q_rel  = shank_r_q_joint.inv() * foot_r_q_joint

            # Left side
            thigh_l_q_rel = pelvis_q_joint.inv() * thigh_l_q_joint
            shank_l_q_rel = thigh_l_q_joint.inv() * shank_l_q_joint
            foot_l_q_rel  = shank_l_q_joint.inv() * foot_l_q_joint
            # ----------------------------------------------------------
            # 3. Convert relative quaternions to axis-angle representations
            # ----------------------------------------------------------
            pelvis_axis, pelvis_theta   = safe_axis_angle(pelvis_q_rel.as_rotvec())

            thigh_r_axis, thigh_r_theta = safe_axis_angle(thigh_r_q_rel.as_rotvec())
            shank_r_axis, shank_r_theta = safe_axis_angle(shank_r_q_rel.as_rotvec())
            feet_r_axis,  feet_r_theta  = safe_axis_angle(foot_r_q_rel.as_rotvec())

            thigh_l_axis, thigh_l_theta = safe_axis_angle(thigh_l_q_rel.as_rotvec())
            shank_l_axis, shank_l_theta = safe_axis_angle(shank_l_q_rel.as_rotvec())
            feet_l_axis,  feet_l_theta  = safe_axis_angle(foot_l_q_rel.as_rotvec())

            # ----------------------------------------------------------
            # 4. Map axis-angle onto OpenSim joint DOFs
            #
            # NOTE: The indices here assume RajagopalHumanBodyModel's
            #       dof ordering. Double-check indices if they ever change.
            # ----------------------------------------------------------
            pos = skeleton.getPositions()

            # Pelvis DOFs: [0, 1, 2] usually correspond to rotation about z, x, y
            pos[0] = np.dot(pelvis_axis, joint_axes["pelvis_z"]) * pelvis_theta
            pos[1] = np.dot(pelvis_axis, joint_axes["pelvis_x"]) * pelvis_theta
            pos[2] = np.dot(pelvis_axis, joint_axes["pelvis_y"]) * pelvis_theta

            # Right side hip (3 DOFs) + knee + ankle/subtalar
            # These indices (6–11) are from your original mapping.
            pos[6]  = np.dot(thigh_r_axis, joint_axes["hip_z"]) * thigh_r_theta
            pos[7]  = np.dot(thigh_r_axis, joint_axes["hip_x"]) * thigh_r_theta
            pos[8]  = np.dot(thigh_r_axis, joint_axes["hip_y"]) * thigh_r_theta
            pos[9]  = np.dot(shank_r_axis, joint_axes["r_knee"]) * shank_r_theta
            pos[10] = np.dot(feet_r_axis,  joint_axes["ankle_z"]) * feet_r_theta     # ankle_angle_r
            pos[11] = np.dot(feet_r_axis,  joint_axes["r_ankle_x"]) * feet_r_theta   # subtalar_angle_r
            # pos[12] could be mtp_angle_r (toe), if you ever want it.

            # Left side hip (3 DOFs) + knee + ankle/subtalar
            pos[13] = np.dot(thigh_l_axis, joint_axes["hip_z"])   * thigh_l_theta
            pos[14] = np.dot(thigh_l_axis, joint_axes["l_hip_x"]) * thigh_l_theta
            pos[15] = np.dot(thigh_l_axis, joint_axes["l_hip_y"]) * thigh_l_theta
            pos[16] = np.dot(shank_l_axis, joint_axes["l_knee"])  * shank_l_theta
            pos[17] = np.dot(feet_l_axis,  joint_axes["ankle_z"]) * feet_l_theta     # ankle_angle_l
            pos[18] = np.dot(feet_l_axis,  joint_axes["l_ankle_x"]) * feet_l_theta   # subtalar_angle_l
            # pos[19] could be mtp_angle_l (toe), not used.

            skeleton.setPositions(pos)

            # ----------------------------------------------------------
            # 5. Render one frame
            # ----------------------------------------------------------
            gui.nativeAPI().renderWorld(world)

            # ----------------------------------------------------------
            # 6. Simple rate limiter to approximate 'sample_frequency'
            # ----------------------------------------------------------
            elapsed = time.perf_counter() - t_prev
            sleep_time = dt_target - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            t_prev = time.perf_counter()

    except KeyboardInterrupt:
        print("\nTerminated by user.")
    except Exception as e:
        report_error(e)
    finally:
        print("Ending Stream.")
        for imu in all_imus:
            imu.set_to_idle()


if __name__ == "__main__":
    raise SystemExit(main())