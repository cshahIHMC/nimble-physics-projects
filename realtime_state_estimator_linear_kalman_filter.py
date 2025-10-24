#!/usr/bin/env python3
"""
Linear Kalman Filter (KF) for Position & Velocity with Quaternion Orientation
==============================================================================

This version:
  • Keeps position & velocity inside a linear Kalman filter (6-state)
  • Represents orientation as a quaternion updated from gyro data
  • Rotates accelerometer readings into the world frame using that quaternion

Author: (your name)
"""

import numpy as np
import time
import sys
from Microstrain import MicroStrainIMU
from scipy.spatial.transform import Rotation as R
from Tools.live_plotter import LivePlotter3D
import collections
import nimblephysics as nimble
import torch



# --------------------------------------------------------------------
# Quaternion utilities
# --------------------------------------------------------------------
def quat_normalize(q):
    return q / (np.linalg.norm(q) + 1e-12)

def quat_multiply(q1, q2):
    """Hamilton product q = q1 ⊗ q2 (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_from_omega_dt(omega, dt):
    """Convert angular velocity (rad/s) to delta-quaternion over dt."""
    theta = np.linalg.norm(omega * dt)
    if theta < 1e-12:
        return np.array([1, 0, 0, 0])
    axis = omega / np.linalg.norm(omega)
    half = 0.5 * theta
    return np.array([np.cos(half), *(np.sin(half) * axis)])

def rotmat_from_quat(q):
    """Rotation matrix world<-body."""
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_hemisphere(q):
    # Ensure scalar part non-negative to avoid 180° Euler flips
    return q if q[0] >= 0 else -q

# --------------------------------------------------------------------
# Mahony Filter for orientation fusion (gyro + accel + mag)
# --------------------------------------------------------------------
class MahonyFilter:
    def __init__(self, kp=2.0, ki=0.03, g=9.80665):
        self.kp = kp
        self.ki = ki
        self.g = g
        self.bias = np.zeros(3)
        self.int_err = np.zeros(3)

    def step(self, q, gyro_b, acc_b, mag_b, dt):
        """Returns updated quaternion [w,x,y,z]"""
        q = q / (np.linalg.norm(q) + 1e-12)

        # Normalize accel
        acc_norm = np.linalg.norm(acc_b)
        if 0.5*self.g < acc_norm < 1.5*self.g:
            acc_b = acc_b / (acc_norm + 1e-12)
        else:
            acc_b = None

        # Normalize mag
        if mag_b is not None:
            mag_b = mag_b / (np.linalg.norm(mag_b) + 1e-12)

        # Expected gravity in body frame
        w, x, y, z = q
        g_b = np.array([
            2*(x*z - w*y),
            2*(y*z + w*x),
            w*w - x*x - y*y + z*z
        ])

        # Compute error
        err = np.zeros(3)
        if acc_b is not None:
            err += np.cross(g_b, acc_b)

        # Gyro correction
        self.int_err += err * dt
        gyro_corr = gyro_b - self.bias + self.kp * err + self.ki * self.int_err

        # Integrate corrected gyro
        dq = quat_from_omega_dt(gyro_corr, dt)
        q_new = quat_normalize(quat_multiply(q, dq))
        return q_new
    
    
# --------------------------------------------------------------------
# 6-state linear Kalman filter for position & velocity
# --------------------------------------------------------------------
class KalmanFilter6D:
    """
    State:  x = [p_x, p_y, p_z, v_x, v_y, v_z]^T
    Control: a_w  (linear acceleration in world)
    """

    def __init__(self, dt, process_var=0.2, meas_var=1.0):
        self.dt = dt

        # State transition F (constant velocity + acceleration input)
        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i+3] = dt

        # Control matrix B maps acceleration to Δv and Δp
        self.B = np.zeros((6, 3))
        for i in range(3):
            self.B[i, i]   = 0.5 * dt**2
            self.B[i+3, i] = dt

        # Observation matrix H: we measure position directly
        self.H = np.zeros((3, 6))
        self.H[:, 0:3] = np.eye(3)

        # Noise covariances
        self.Q = np.eye(6) * process_var       # model/process noise
        self.R = np.eye(3) * meas_var          # measurement noise
        self.P = np.eye(6) * 1.0               # state covariance
        self.x = np.zeros((6, 1))              # [pos; vel]

    # ---- Prediction ------------------------------------------------
    def predict(self, accel_world):
        u = accel_world.reshape(3, 1)
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q

    # ---- Update ----------------------------------------------------
    def update(self, z_meas):
        """z_meas: measured position (3×1)"""
        z = z_meas.reshape(3, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        p = self.x[0:3].flatten()
        v = self.x[3:6].flatten()
        return p, v


# --------------------------------------------------------------------
# IMU helpers
# --------------------------------------------------------------------
def get_imu_data(imu):
    data = imu.get_ESTFILTER_data(20, 0)
    device_time = data[0]
    return np.array(data[1:10]), device_time

def read_sensor(data, plotter=None):
    acc_b = data[0:3] * 9.80665
    gyro_b = data[3:6]
    mag_b  = data[6:9]
    # print("acc: ", acc_b )
    # print("gyro: ", gyro_b )
    # print("mag: ", mag_b)
    
    # plotter.update(np.array(acc_b))
    
    return acc_b, gyro_b, mag_b


# --------------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------------
def main():
    NODE_RATE = 200
    dt_target = 1.0 / NODE_RATE
    
    plotter = LivePlotter3D(300)

    imu = MicroStrainIMU("195772", 921600)
    try:
        print("Did I configure?")
        imu.configure_ESTFLTER_imu(NODE_RATE)
    except Exception as e:
        exc_type, _, tb = sys.exc_info()
        print(f"Error configuring IMU on line {tb.tb_lineno}: {e}")
        imu.set_to_idle()

    # Initialize filters
    kf = KalmanFilter6D(dt=dt_target)
    q_wb = np.array([0, 1, 0, 0])   # world<-body
    mahony = MahonyFilter(kp=2.0, ki=0.05)
    
    # ---------------------------------------------------------
    # INITIAL ORIENTATION CALIBRATION (makes start = identity)
    # ---------------------------------------------------------
    print("Calibrating initial orientation... hold IMU still for ~0.5s")
    q_tmp = np.array([1,0,0,0], dtype=float)
    samples = 100
    for _ in range(samples):
        data = imu.get_ESTFILTER_data(20, 0)
        data_np = np.array(data[1:10])
        acc_b = data_np[0:3] * 9.80665          # m/s^2
        gyro_b = data_np[3:6]                    # likely deg/s from MicroStrain
        mag_b  = data_np[6:9]

        # Convert gyro to rad/s (CRITICAL for your integration math)
        q_tmp = mahony.step(q_tmp, gyro_b, acc_b, mag_b, 1.0/float(NODE_RATE))

    # Reference so that first pose becomes identity
    q_ref = quat_conjugate(q_tmp)
    print("Reference orientation set (start will be 0,0,0).")


    print("Running linear KF with quaternion orientation.")
    print("time, px, py, pz, vx, vy, vz, qw, qx, qy, qz")

    # ----------------------------------------------------------------
    # Timing and tracking setup
    # ----------------------------------------------------------------
    N = 200  # averaging window
    loop_intervals = collections.deque(maxlen=N)
    device_intervals = collections.deque(maxlen=N)

    start_time = time.perf_counter()
    last_loop_time = start_time
    last_device_time = None
    g_w = np.array([0, 0, -9.81])
    counter = 0
    
    # ----------------------------------------------------------------
    # Setting up nimble physics simulation environment
    # ----------------------------------------------------------------
    
    # --- World Setup ---
    world = nimble.simulation.World()
    world.setGravity([0, 0, 0])  # No gravity for visualization

    # --- Create a simple body to represent IMU ---
    box = nimble.dynamics.Skeleton()
    boxJoint, boxBody = box.createFreeJointAndBodyNodePair()
    boxShape = boxBody.createShapeNode(nimble.dynamics.BoxShape([0.15, 0.1, 0.05]))
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
    
    initial_position = torch.tensor([0,0,0])
    initial_rotation = torch.tensor([0,0,0])
    state = torch.cat((initial_position, initial_rotation), 0)

    # ----------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------
    try:
        while True:
            loop_start = time.perf_counter()
            dt_loop = loop_start - last_loop_time
            last_loop_time = loop_start

            # 1) Read IMU data
            data = (imu.get_ESTFILTER_data(20, 0))
            device_time = data[0]  
            data_np = np.array(data[1:10])
            acc_b = data_np[0:3] * 9.80665
            gyro_b = data_np[3:6]
            mag_b  = data_np[6:9]
        

            # Track device time difference for frequency estimation
            if last_device_time is not None:
                device_intervals.append(device_time - last_device_time)
            last_device_time = device_time

            # 2) Update orientation (Mahony)
            q_wb = mahony.step(q_wb, gyro_b, acc_b, mag_b, dt_loop)
            
            # Apply reference so start is identity and rotations are about the sensor's own axes
            q_wb = quat_multiply(q_ref, q_wb)
            q_wb = quat_hemisphere(q_wb)
            

            # 3) Rotate accel into world frame
            R_wb = rotmat_from_quat(q_wb)
            acc_w = R_wb @ acc_b + np.array([0, 0, 9.80665])

            # 4) KF predict step
            kf.predict(acc_w)

            # 5) Optional fake position measurement (for stability)
            if np.random.rand() < 0.05:
                z_meas = kf.x[0:3].flatten() + np.random.randn(3) * 0.1
                kf.update(z_meas)

            # 6) Get estimates
            p, v = kf.get_state()

            # 7) Timing bookkeeping
            loop_intervals.append(dt_loop)
            counter += 1

            # ----------------------------------------------------------------
            # Print every N iterations
            # ----------------------------------------------------------------
            if counter % N == 0 and len(loop_intervals) == N:
                print("In thie loop")
                avg_loop_rate = len(loop_intervals) / sum(loop_intervals)

                if len(device_intervals) > 0:
                    avg_device_period = sum(device_intervals) / len(device_intervals)
                    avg_device_rate = 1.0 / avg_device_period if avg_device_period > 0 else 0.0
                else:
                    avg_device_rate = 0.0

                elapsed = loop_start - start_time

                print(f"\nElapsed: {elapsed:.3f} s | "
                      f"Loop rate: {avg_loop_rate:.2f} Hz | "
                      f"Device rate: {avg_device_rate:.2f} Hz")

                print(f"Pos [m]: ({p[0]: .3f}, {p[1]: .3f}, {p[2]: .3f}) | "
                      f"Vel [m/s]: ({v[0]: .3f}, {v[1]: .3f}, {v[2]: .3f}) | "
                      f"Quat: ({q_wb[0]: .4f}, {q_wb[1]: .4f}, {q_wb[2]: .4f}, {q_wb[3]: .4f})")
                
            
            
            imu_rotvec = torch.tensor(R.from_quat(q_wb).as_rotvec())
            
            
            plotter.update(R.from_quat(q_wb).as_euler('zyx', degrees=True))
            # --- Update simulation state ---
            position = torch.tensor(p)
            zeros1 = torch.tensor([0,0,0])
            zeros2 = torch.tensor([0,0,0])
            state = torch.cat((imu_rotvec, position, zeros1, zeros2), 0)
            state = nimble.timestep(world, state, torch.zeros(world.getNumDofs()))

            # --- Render the updated state ---
            gui.nativeAPI().renderWorld(world)



            # ----------------------------------------------------------------
            # Enforce 200 Hz target rate
            # ----------------------------------------------------------------
            elapsed_loop = time.perf_counter() - loop_start
            sleep_time = dt_target - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        exc_type, _, tb = sys.exc_info()
        print(f"Error during loop at line {tb.tb_lineno}: {e}")
    finally:
        try:
            imu.set_to_idle()
            print("IMU set to idle mode.")
        except Exception as e:
            print(f"Warning: Failed to set IMU to idle: {e}")


if __name__ == "__main__":
    main()
