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
# Main
# ------------------------------------------------------------------
def main():
    NODE_RATE = 200

    imu1 = MicroStrainIMU("195772", 921600)
    imu2 = MicroStrainIMU("195778", 921600)

    try:
        imu1.configure_ESTFLTER_imu(NODE_RATE)
        imu2.configure_ESTFLTER_imu(NODE_RATE)
    except Exception as e:
        report_error(e)
        imu1.set_to_idle()
        imu2.set_to_idle()
        return
    
    
    # Nimble world
    world = nimble.simulation.World()
    world.setGravity([0, 0, 0])

    opensim_model = nimble.RajagopalHumanBodyModel()
    skeleton = opensim_model.skeleton
    world.addSkeleton(skeleton)

    # Precompute transforms
    transform_imu_to_knee = R.from_matrix(rotation_matrix_z(-np.pi / 2) @ rotation_matrix_x(np.pi))
    transform_imu_to_knee_inv = transform_imu_to_knee.inv()

    # Initial quaternions (zero reference)
    imu1_quat0 = R.from_quat(get_estfilter_data(imu1), scalar_first=True)
    imu2_quat0 = R.from_quat(get_estfilter_data(imu2), scalar_first=True)

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
        "knee": np.array([0, 0, -1]),
        "hip_x": np.array([-1, 0, 0]),
        "hip_y": np.array([0, -1, 0]),
        "hip_z": np.array([0, 0, 1]),
    }
    
    
    t_prev = time.perf_counter()

    try:
        while True:
            quat1 = get_estfilter_data(imu1)
            quat2 = get_estfilter_data(imu2)

            r1 = R.from_quat(quat1, scalar_first=True)
            r2 = R.from_quat(quat2, scalar_first=True)

            # Zero relative rotation
            r1_zeroed = imu1_quat0.inv() * r1
            r2_zeroed = imu2_quat0.inv() * r2

            # Transform to joint frame
            r1_joint = transform_imu_to_knee * r1_zeroed * transform_imu_to_knee_inv
            r2_joint = transform_imu_to_knee * r2_zeroed * transform_imu_to_knee_inv

            axis1, theta1 = safe_axis_angle(r1_joint.as_rotvec())
            axis2, theta2 = safe_axis_angle(r2_joint.as_rotvec())

            # Compute joint angles
            pos = skeleton.getPositions()
            pos[9] = np.dot(axis1, joint_axes["knee"]) * theta1
            pos[6] = np.dot(axis2, joint_axes["hip_z"]) * theta2
            pos[7] = np.dot(axis2, joint_axes["hip_x"]) * theta2
            pos[8] = np.dot(axis2, joint_axes["hip_y"]) * theta2
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
        imu1.set_to_idle()
        imu2.set_to_idle()


if __name__ == "__main__":
    raise SystemExit(main())
