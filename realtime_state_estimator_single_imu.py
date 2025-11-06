#!/usr/bin/env python3
"""
IMU State Estimation with an Unscented Kalman Filter (UKF)
=========================================================

This script reads (or simulates) IMU data — accelerometer, gyroscope, and magnetometer —
and estimates the system state consisting of:
    - Position in world frame p_w (m)
    - Velocity in world frame v_w (m/s)
    - Orientation as a unit quaternion q_wb (world<-body)

It uses a two-layer approach common in inertial navigation systems (INS):

1) **Attitude UKF (AHRS)**
   - State: orientation quaternion q_wb (4), gyro bias b_g (3), accel bias b_a (3) → 10D state.
   - Process model: integrates gyro (minus bias) to propagate orientation; biases are random walks.
   - Measurement model: uses normalized accelerometer (gravity direction) and normalized magnetometer
     (Earth field direction) to correct orientation and biases. This reduces sensitivity to linear
     accelerations because we only use *direction* of gravity and magnetic field.

2) **INS kinematics (dead reckoning)**
   - Uses the UKF’s orientation and accel bias to compute world-frame linear acceleration and then
     integrates to velocity and position.
   - NOTE: Without an external position/velocity aid (e.g., GPS, mocap, wheel odometry, zero-velocity updates),
     position will drift over time. You can add such measurements in the `INS update hooks` section.

Replace `read_sensor()` with your device’s API (e.g., MicroStrain, BNO, ICM-based boards). The function
should return raw accelerometer (m/s^2), gyroscope (rad/s), and magnetometer (uT or arbitrary units).

Dependencies: numpy, scipy

Usage:
    python imu_ukf_state_estimator.py

Author: (you)
"""

from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R
from Microstrain import MicroStrainIMU
import sys

# ---------------------------------------------------------------
# Utility: Quaternion helpers (world<-body)
# ---------------------------------------------------------------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q = q1 ⊗ q2 (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_from_omega_dt(omega: np.ndarray, dt: float) -> np.ndarray:
    """Exponential map: integrate small-angle rotation vector (rad) to quaternion.
    omega: 3D angular rate (rad/s) in body frame.
    dt: timestep in seconds
    """
    rotvec = omega * dt
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotvec / theta
    half = 0.5 * theta
    return np.array([np.cos(half), *(np.sin(half) * axis)])

def rotmat_from_quat(q: np.ndarray) -> np.ndarray:
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()  # scipy uses (x,y,z,w)

# ---------------------------------------------------------------
# Sigma-point utilities for UKF (Van der Merwe scaled UKF)
# ---------------------------------------------------------------

def sigma_points(x: np.ndarray, P: np.ndarray, alpha=1e-3, beta=2.0, kappa=0.0):
    """Compute sigma points and weights.
    Returns: X (n x (2n+1)), Wm, Wc
    """
    n = x.size
    lam = alpha**2 * (n + kappa) - n
    c = n + lam
    # Ensure symmetry and numerical stability
    P = 0.5 * (P + P.T)
    min_eig = np.min(np.real(np.linalg.eigvals(P)))
    if min_eig < 1e-9:
        P += np.eye(P.shape[0]) * (1e-9 - min_eig)
    
    U = np.linalg.cholesky(c * P)

    X = np.zeros((n, 2*n + 1))
    X[:, 0] = x
    for i in range(n):
        X[:, i+1]     = x + U[:, i]
        X[:, i+1+n]   = x - U[:, i]

    Wm = np.full(2*n + 1, 1.0 / (2*c))
    Wc = np.full(2*n + 1, 1.0 / (2*c))
    Wm[0] = lam / c
    Wc[0] = lam / c + (1 - alpha**2 + beta)
    return X, Wm, Wc, c

# ---------------------------------------------------------------
# UKF for Attitude + Biases (10D state)
#   state z = [q_wb(4), b_g(3), b_a(3)]
#   process: q_k+1 = q_k ⊗ exp((omega - b_g)*dt), biases random walk
#   measure: unit accel dir and unit mag dir in body
# ---------------------------------------------------------------

@dataclass
class AttitudeUKFConfig:
    alpha: float = 5e-2    # spread of sigma points (larger for quaternion robustness)
    beta: float  = 2.0     # optimal for Gaussian
    kappa: float = 0.0
    # Process noise std devs
    gyro_rw: float = 0.05       # rad/sqrt(s) gyro noise driving attitude
    bg_rw: float   = 5e-4       # rad/s^2 gyro bias random walk
    ba_rw: float   = 1e-3       # m/s^2/s accel bias random walk
    # Measurement noise (unit-vector errors)
    accel_dir_std: float = 0.05 # radians approx, used as dir noise
    mag_dir_std: float   = 0.1  # radians approx

class AttitudeUKF:
    def __init__(self, cfg: AttitudeUKFConfig):
        self.cfg = cfg
        # state: [q(4), bg(3), ba(3)]
        self.z = np.zeros(10)
        self.z[:4] = np.array([1,0,0,0], dtype=float)
        self.P = np.eye(10) * 0.1

    def state_parts(self, z: np.ndarray):
        q = quat_normalize(z[:4])
        bg = z[4:7]
        ba = z[7:10]
        return q, bg, ba

    def compose_state(self, q: np.ndarray, bg: np.ndarray, ba: np.ndarray) -> np.ndarray:
        z = np.zeros(10)
        z[:4] = quat_normalize(q)
        z[4:7] = bg
        z[7:10] = ba
        return z

    def predict(self, omega_meas: np.ndarray, dt: float):
        cfg = self.cfg
        z = self.z
        P = self.P
        n = z.size
        X, Wm, Wc, c = sigma_points(z, P, cfg.alpha, cfg.beta, cfg.kappa)

        # Process noise Q (discrete)
        Q = np.zeros((n, n))
        # attitude drive noise approximated via gyro noise
        Q[0:4, 0:4] += np.eye(4) * (cfg.gyro_rw**2 * dt)  # heuristic in quat space
        Q[4:7, 4:7] += np.eye(3) * (cfg.bg_rw**2 * dt)
        Q[7:10,7:10]+= np.eye(3) * (cfg.ba_rw**2 * dt)

        # Propagate each sigma point through process model
        X_prop = np.zeros_like(X)
        for i in range(X.shape[1]):
            qi, bgi, bai = self.state_parts(X[:, i])
            # gyro bias corrected
            omega_b = omega_meas - bgi
            dq = quat_from_omega_dt(omega_b, dt)
            q_next = quat_mul(qi, dq)
            q_next = quat_normalize(q_next)
            bg_next = bgi  # random walk handled by Q
            ba_next = bai
            X_prop[:, i] = self.compose_state(q_next, bg_next, ba_next)

        # Recombine mean in quaternion manifold
        # We use iterative averaging on SO(3) via small-angle updates
        q_mean = X_prop[0:4, 0].copy()
        for _ in range(10):  # fixed-point iterations
            E = np.zeros(3)
            for i in range(X_prop.shape[1]):
                qi = X_prop[0:4, i]
                # rotation from q_mean to qi: q_err = qi ⊗ q_mean^{-1}
                q_mean_inv = np.array([q_mean[0], -q_mean[1], -q_mean[2], -q_mean[3]])
                q_err = quat_mul(qi, q_mean_inv)
                q_err = quat_normalize(q_err)
                # map to rotvec (approx 2*v for small angles):
                rot = R.from_quat([q_err[1], q_err[2], q_err[3], q_err[0]])
                rotvec = rot.as_rotvec()
                E += Wm[i] * rotvec
            # update mean
            dq = R.from_rotvec(E).as_quat()  # (x,y,z,w)
            dq = np.array([dq[3], dq[0], dq[1], dq[2]])
            q_mean = quat_normalize(quat_mul(dq, q_mean))
            if np.linalg.norm(E) < 1e-9:
                break

        # Other components are Euclidean
        bg_mean = np.sum(X_prop[4:7, :] * Wm, axis=1)
        ba_mean = np.sum(X_prop[7:10, :] * Wm, axis=1)

        z_pred = self.compose_state(q_mean, bg_mean, ba_mean)

        # Covariance reconstruction
        P_pred = np.zeros((n, n))
        for i in range(X_prop.shape[1]):
            # orientation error in tangent space
            qi = X_prop[0:4, i]
            q_mean_inv = np.array([q_mean[0], -q_mean[1], -q_mean[2], -q_mean[3]])
            q_err = quat_mul(qi, q_mean_inv)
            rot = R.from_quat([q_err[1], q_err[2], q_err[3], q_err[0]])
            e_q = rot.as_rotvec()  # 3
            e_bg = X_prop[4:7, i] - bg_mean  # 3
            e_ba = X_prop[7:10, i] - ba_mean # 3
            e = np.concatenate([e_q, e_bg, e_ba])  # 9D error vector
            P_pred[:9, :9] += Wc[i] * np.outer(e, e)
        # map 9x9 back into 10x10 by embedding and adding Q
        # We store covariance in hybrid form:
        #   block for tangent orientation (3) + bg (3) + ba (3), and small quat noise on diag
        # For simplicity, we keep full 10x10 with small diag for quat entries.
        # Embed:
        P_out = np.eye(10) * 1e-6
        P_out[0:3, 0:3] = P_pred[0:3, 0:3]  # orientation tangent
        P_out[4:7, 4:7] = P_pred[3:6, 3:6]
        P_out[7:10,7:10]= P_pred[6:9, 6:9]
        P = P_out + Q

        self.z = z_pred
        self.P = P

    def update(self, acc_meas: np.ndarray, mag_meas: np.ndarray, g_w: np.ndarray, m_w: np.ndarray):
        """Measurement uses unit directions of gravity and magnetic field in body frame.
        z_meas = [a_hat_b, m_hat_b] (6x1)
        h(x): rotate world unit vectors into body: a_hat_b = R_bw * g_hat_w, m_hat_b = R_bw * m_hat_w
        """
        cfg = self.cfg
        z = self.z
        P = self.P
        n = z.size
        # normalize measurements
        a_hat_b_meas = acc_meas / (np.linalg.norm(acc_meas) + 1e-12)
        m_hat_b_meas = mag_meas / (np.linalg.norm(mag_meas) + 1e-12)
        y_meas = np.concatenate([a_hat_b_meas, m_hat_b_meas])  # 6

        # Sigma points
        X, Wm, Wc, c = sigma_points(z, P, cfg.alpha, cfg.beta, cfg.kappa)

        # Predict measurements
        Z = np.zeros((6, X.shape[1]))
        q_mean, _, _ = self.state_parts(z)
        for i in range(X.shape[1]):
            qi, _, _ = self.state_parts(X[:, i])
            R_wb = rotmat_from_quat(qi)
            R_bw = R_wb.T
            a_hat_b = R_bw @ (g_w / np.linalg.norm(g_w))
            m_hat_b = R_bw @ (m_w / np.linalg.norm(m_w))
            Z[:, i] = np.hstack([a_hat_b, m_hat_b])

        z_pred = np.sum(Z * Wm, axis=1)

        # Innovation covariance S and cross-covariance Pxz
        S = np.zeros((6, 6))
        Pxz = np.zeros((10, 6))
        for i in range(X.shape[1]):
            dz = (Z[:, i] - z_pred)
            # state deviation: orientation tangent + bg + ba
            qi = X[0:4, i]
            q_mean_inv = np.array([q_mean[0], -q_mean[1], -q_mean[2], -q_mean[3]])
            q_err = quat_mul(qi, q_mean_inv)
            rot = R.from_quat([q_err[1], q_err[2], q_err[3], q_err[0]])
            e_q = rot.as_rotvec()
            e_bg = X[4:7, i] - z[4:7]
            e_ba = X[7:10, i] - z[7:10]
            ex = np.zeros(10)
            ex[0:3] = e_q
            ex[4:7] = e_bg
            ex[7:10] = e_ba
            S += Wc[i] * np.outer(dz, dz)
            Pxz += Wc[i] * np.outer(ex, dz)

        # Measurement noise Rm
        Rm = np.eye(6)
        Rm[0:3, 0:3] *= cfg.accel_dir_std**2
        Rm[3:6, 3:6] *= cfg.mag_dir_std**2
        S += Rm

        # Kalman gain
        K = Pxz @ np.linalg.inv(S)
        innov = y_meas - z_pred
        dx = K @ innov  # 10

        # Apply correction: orientation via small-angle, biases additive
        q, bg, ba = self.state_parts(z)
        dtheta = dx[0:3]
        dq = R.from_rotvec(dtheta).as_quat()  # (x,y,z,w)
        dq = np.array([dq[3], dq[0], dq[1], dq[2]])
        q_upd = quat_normalize(quat_mul(dq, q))
        bg_upd = bg + dx[4:7]
        ba_upd = ba + dx[7:10]

        self.z = self.compose_state(q_upd, bg_upd, ba_upd)
        
        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        min_eig = np.min(np.real(np.linalg.eigvals(self.P)))
        if min_eig < 1e-9:
            self.P += np.eye(self.P.shape[0]) * (1e-9 - min_eig)



    @property
    def q_wb(self) -> np.ndarray:
        return quat_normalize(self.z[:4])

    @property
    def bg(self) -> np.ndarray:
        return self.z[4:7]

    @property
    def ba(self) -> np.ndarray:
        return self.z[7:10]

# ---------------------------------------------------------------
# Sensor I/O placeholder
# ---------------------------------------------------------------

def get_imu_data(imu) -> np.ndarray:
    data = imu.get_ESTFILTER_data(20, 0)
    return np.array(data[1:10])


def read_sensor(data:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace this with your actual sensor read.
    Returns:
        acc_b (3,)  : accelerometer in m/s^2 (body frame)
        gyro_b (3,) : gyroscope in rad/s (body frame)
        mag_b (3,)  : magnetometer in arbitrary units (body frame)
    """
    acc_b = data[0:3] * 9.80655
    # print(acc_b)
    gyro = data[3:6]
    # print(gyro)
    mag_b = data[6:9]
    # print(mag_b)

    return acc_b, gyro, mag_b

# ---------------------------------------------------------------
# Main loop: run AHRS UKF + integrate INS
# ---------------------------------------------------------------

def main():
    cfg = AttitudeUKFConfig()
    ahrs = AttitudeUKF(cfg)
    
    # --- IMU Configuration ---
    imu = MicroStrainIMU("195772", 921600)
    NODE_RATE = 200  # IMU stream rate (Hz)
            
            
    # --- Configure IMU filtering ---
    try:
        imu.configure_ESTFLTER_imu(NODE_RATE)
    except Exception as e:
        exc_type, _, tb = sys.exc_info()
        print(f"Error configuring IMU on line {tb.tb_lineno}: {e}")
        imu.set_to_idle()
                

    # World constants (set your local magnetic field direction; magnitude unimportant)
    g_w = np.array([0, 0, -9.81])
    m_w = np.array([0.40, 0.00, 0.10])  # rough North + down declination; adjust for your locale

    # INS states
    p_w = np.zeros(3)
    v_w = np.zeros(3)

    # Timing
    rate_hz = 200.0
    dt = 1.0 / rate_hz
    last = time.time()

    print("Running UKF-based IMU state estimator. Press Ctrl+C to stop.\n")
    print(f"Columns: time, px, py, pz, vx, vy, vz, qw, qx, qy, qz")

    try:
        while True:
            now = time.time()
            # Maintain (approximately) constant dt
            if now - last < dt:
                time.sleep(dt - (now - last))
                now = time.time()
            dt_i = np.clip(now - last, 0, 0.02)
            last = now

            # 1) Read sensors
            data_np = get_imu_data(imu)
            acc_b, gyro_b, mag_b = read_sensor(data_np)

            # 2) Attitude UKF predict with gyro
            ahrs.predict(gyro_b, dt_i)

            # 3) Attitude UKF update with accel+mag directions
            ahrs.update(acc_b, mag_b, g_w, m_w)

            # 4) INS propagation using debiased accel
            q = ahrs.q_wb  # world<-body
            R_wb = rotmat_from_quat(q)
            a_lin_w = R_wb @ (acc_b - ahrs.ba) + np.array([0, 0, 9.81])  # add gravity to get linear accel

            # Integrate velocity and position
            v_w += a_lin_w * dt_i
            p_w += v_w * dt_i

            # Output state vector
            print(f"{now:.3f}, {p_w[0]: .3f}, {p_w[1]: .3f}, {p_w[2]: .3f}, "
                  f"{v_w[0]: .3f}, {v_w[1]: .3f}, {v_w[2]: .3f}, "
                  f"{q[0]: .4f}, {q[1]: .4f}, {q[2]: .4f}, {q[3]: .4f}")

            # --- INS update hooks (optional) ---
            # If you have external aiding (GPS/mocap/VO/ZUPT), insert an EKF/UKF correction here
            # to update p_w and v_w. For example, when stationary, set R for a ZUPT on v_w.

    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        exc_type, _, tb = sys.exc_info()
        print(f"Error during loop at line {tb.tb_lineno}: {e}")
        imu.set_to_idle()
    
    # --- Cleanup on Exit ---
    try:
        print("Ending IMU Stream...")
        imu.set_to_idle()
    except Exception as e:
        exc_type, _, tb = sys.exc_info()
        print(f"Error ending stream at line {tb.tb_lineno}: {e}")



if __name__ == "__main__":
    main()
