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
    return np.array(data[1:10])

def read_sensor(data, plotter):
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
    
    # Setup the live plotter for debugging
    plotter = LivePlotter3D(window=5, title="Live Acc/Gyro/Mag Example")

    NODE_RATE = 200
    dt = 1.0 / NODE_RATE

    imu = MicroStrainIMU("195772", 921600)
    try:
        imu.configure_ESTFLTER_imu(NODE_RATE)
    except Exception as e:
        exc_type, _, tb = sys.exc_info()
        print(f"Error configuring IMU on line {tb.tb_lineno}: {e}")
        imu.set_to_idle()

    # Initialize Kalman filter and orientation quaternion
    kf = KalmanFilter6D(dt=dt)
    q_wb = np.array([1, 0, 0, 0])   # world<-body
    
    mahony = MahonyFilter(kp=2.0, ki=0.05)


    print("Running linear KF with quaternion orientation.")
    print("time, px, py, pz, vx, vy, vz, qw, qx, qy, qz")

    last = time.time()
    g_w = np.array([0, 0, -9.81])   # gravity in world

    try:
        while True:
            now = time.time()
            if now - last < dt:
                time.sleep(dt - (now - last))
            dt_i = np.clip(time.time() - last, 0, 0.02)
            last = now

            # 1) Read IMU data
            data_np = get_imu_data(imu)
            acc_b, gyro_b, mag_b = read_sensor(data_np, plotter)

            # 2) Update quaternion orientation by integrating gyro
            q_wb = mahony.step(q_wb, gyro_b, acc_b, mag_b, dt_i)
            
            # Suppose q_wb = np.array([w, x, y, z])
            # r = R.from_quat([q_wb[1], q_wb[2], q_wb[3], q_wb[0]])  # SciPy expects [x, y, z, w]
            # euler = r.as_euler('xyz', degrees=True) 
            # print(" Orientation: ", euler)
            # plotter.update(euler)

            # 3) Rotate accel to world frame
            R_wb = rotmat_from_quat(q_wb)
            # print("1: ", acc_b)
            acc_w = R_wb @ acc_b + np.array([0, 0, 9.80665])  # add gravity back
            # print(acc_w)

            # 4) KF predict step
            kf.predict(acc_w)

            # 5) Optional fake position measurement (to show update)
            if np.random.rand() < 0.05:
                z_meas = kf.x[0:3].flatten() + np.random.randn(3) * 0.1
                kf.update(z_meas)

            # 6) Get estimates
            p, v = kf.get_state()

            # 7) Print
            print(f"{now:.3f}, {p[0]: .3f}, {p[1]: .3f}, {p[2]: .3f}, "
                  f"{v[0]: .3f}, {v[1]: .3f}, {v[2]: .3f}, "
                  f"{q_wb[0]: .4f}, {q_wb[1]: .4f}, {q_wb[2]: .4f}, {q_wb[3]: .4f}")

    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        exc_type, _, tb = sys.exc_info()
        print(f"Error during loop at line {tb.tb_lineno}: {e}")
        imu.set_to_idle()

    try:
        imu.set_to_idle()
    except:
        pass


if __name__ == "__main__":
    main()
