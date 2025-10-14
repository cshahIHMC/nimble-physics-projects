import numpy as np
import nimblephysics as nimble
import time

# --- Assumptions: skeleton and sensors already exist ---
# skeleton: nimble.dynamics.Skeleton
# sensors: List[Tuple[BodyNode, Isometry3]]

dt = 0.01  # EKF timestep, set to your IMU sample period (e.g. 0.01s => 100 Hz)

num_dofs = skeleton.getNumDofs()
state_dim = 3 * num_dofs + 3  # q, qdot, qddot, b_mag

# Initial state
x = np.zeros((state_dim,))
# initialize q from skeleton current positions
x[:num_dofs] = skeleton.getPositions()
x[num_dofs:2*num_dofs] = skeleton.getVelocities()
x[2*num_dofs:3*num_dofs] = skeleton.getAccelerations()
x[3*num_dofs:] = np.array([0.2, 0.0, 0.4])  # initial guess for world magnetic field (example)

# Initial covariance
P = np.eye(state_dim) * 1e-3

# Process noise (tune)
Q_pos = np.eye(num_dofs) * 1e-6
Q_vel = np.eye(num_dofs) * 1e-4
Q_acc = np.eye(num_dofs) * 1e-2
Q_bmag = np.eye(3) * 1e-4
Q = np.block([
    [Q_pos, np.zeros((num_dofs, num_dofs)), np.zeros((num_dofs, num_dofs)), np.zeros((num_dofs,3))],
    [np.zeros((num_dofs, num_dofs)), Q_vel, np.zeros((num_dofs, num_dofs)), np.zeros((num_dofs,3))],
    [np.zeros((num_dofs, num_dofs)), np.zeros((num_dofs, num_dofs)), Q_acc, np.zeros((num_dofs,3))],
    [np.zeros((3, num_dofs)), np.zeros((3,num_dofs)), np.zeros((3,num_dofs)), Q_bmag]
])

# Measurement noise (tune)
num_sensors = len(sensors)
one_sensor_type_dim = num_sensors * 3
measurement_dim = one_sensor_type_dim * 3  # gyro + accel + mag
R_gyro = np.eye(one_sensor_type_dim) * (1e-4)   # gyro noise
R_acc = np.eye(one_sensor_type_dim) * (1e-2)    # accel noise
R_mag = np.eye(one_sensor_type_dim) * (1e-2)    # mag noise
R = np.block([
    [R_gyro, np.zeros((one_sensor_type_dim, one_sensor_type_dim)), np.zeros((one_sensor_type_dim, one_sensor_type_dim))],
    [np.zeros((one_sensor_type_dim, one_sensor_type_dim)), R_acc, np.zeros((one_sensor_type_dim, one_sensor_type_dim))],
    [np.zeros((one_sensor_type_dim, one_sensor_type_dim)), np.zeros((one_sensor_type_dim, one_sensor_type_dim)), R_mag]
])

# Helper: build state transition matrix A (linearized approximate)
def build_state_transition(dt):
    A = np.zeros((state_dim, state_dim))
    # next q as a function of q, qdot, qddot
    A[:num_dofs, :num_dofs] = np.eye(num_dofs)
    A[:num_dofs, num_dofs:2*num_dofs] = dt * np.eye(num_dofs)
    A[:num_dofs, 2*num_dofs:3*num_dofs] = 0.5 * dt * dt * np.eye(num_dofs)  # using 0.5*dt^2
    # velocities
    A[num_dofs:2*num_dofs, num_dofs:2*num_dofs] = np.eye(num_dofs)
    A[num_dofs:2*num_dofs, 2*num_dofs:3*num_dofs] = dt * np.eye(num_dofs)
    # accelerations (assume roughly constant)
    A[2*num_dofs:3*num_dofs, 2*num_dofs:3*num_dofs] = np.eye(num_dofs)
    # magnetic field stays constant (identity)
    A[3*num_dofs:, 3*num_dofs:] = np.eye(3)
    return A

A = build_state_transition(dt)

# Measurement prediction: uses skeleton to compute predicted sensors given q,qdot,qddot
def predict_measurements_from_state(x_state):
    # set skeleton to the state so Nimble returns correct predicted sensors
    q = x_state[:num_dofs]
    qdot = x_state[num_dofs:2*num_dofs]
    qddot = x_state[2*num_dofs:3*num_dofs]
    bmag = x_state[3*num_dofs:]

    skeleton.setPositions(q)
    skeleton.setVelocities(qdot)
    skeleton.setAccelerations(qddot)

    gyro_pred = skeleton.getGyroReadings(sensors)          # shape (num_sensors*3,)
    accel_pred = skeleton.getAccelerometerReadings(sensors)
    mag_pred = skeleton.getMagnetometerReadings(sensors, worldMagneticField=bmag) if hasattr(skeleton, 'getMagnetometerReadings') else skeleton.getMagnetometerReadings(sensors)

    # stack: [gyro; accel; mag]
    h = np.concatenate([gyro_pred, accel_pred, mag_pred])
    return h

# Build measurement Jacobian H using Nimble Jacobians (linearization at current state)
def build_measurement_jacobian():
    # gyro w.r.t position and velocity
    Jg_pos = skeleton.getGyroReadingsJacobianWrt(sensors, wrt=nimble.neural.WRT_POSITION)   # (gdim, num_dofs)
    Jg_vel = skeleton.getGyroReadingsJacobianWrt(sensors, wrt=nimble.neural.WRT_VELOCITY)   # (gdim, num_dofs)

    # accel w.r.t position, velocity, acceleration
    Ja_pos = skeleton.getAccelerometerReadingsJacobianWrt(sensors, wrt=nimble.neural.WRT_POSITION)
    Ja_vel = skeleton.getAccelerometerReadingsJacobianWrt(sensors, wrt=nimble.neural.WRT_VELOCITY)
    Ja_acc = skeleton.getAccelerometerReadingsJacobianWrt(sensors, wrt=nimble.neural.WRT_ACCELERATION)

    # mag
    Jm_pos = skeleton.getMagnetometerReadingsJacobianWrt(sensors, wrt=nimble.neural.WRT_POSITION)
    Jm_bmag = skeleton.getMagnetometerReadingsJacobianWrtMagneticField(sensors)  # (mdim, 3)

    gdim = Jg_pos.shape[0]
    mdim = Jm_pos.shape[0]
    adim = Ja_pos.shape[0]
    # sanity check: should be equal to one_sensor_type_dim
    assert gdim == one_sensor_type_dim and adim == one_sensor_type_dim and mdim == one_sensor_type_dim

    # Measurement matrix blocks
    H = np.zeros((measurement_dim, state_dim))

    # Gyros: rows 0:gdim
    H[0:gdim, 0:num_dofs] = Jg_pos
    H[0:gdim, num_dofs:2*num_dofs] = Jg_vel
    # gyros have no direct dependence on qddot or bmag in this simplified linearization

    # Accels: rows gdim : gdim+adim
    a_row_start = gdim
    H[a_row_start:a_row_start+adim, 0:num_dofs] = Ja_pos
    H[a_row_start:a_row_start+adim, num_dofs:2*num_dofs] = Ja_vel
    H[a_row_start:a_row_start+adim, 2*num_dofs:3*num_dofs] = Ja_acc

    # Mags: rows gdim+adim : end
    m_row_start = gdim + adim
    H[m_row_start:m_row_start+mdim, 0:num_dofs] = Jm_pos
    H[m_row_start:m_row_start+mdim, 3*num_dofs:3*num_dofs+3] = Jm_bmag

    return H

# Main EKF update step (call each time you get a measurement z)
def ekf_step(x, P, z):
    # 1) Predict
    A = build_state_transition(dt)
    x_pred = A @ x
    P_pred = A @ P @ A.T + Q

    # 2) Set skeleton to predicted q,qdot,qddot for linearization
    skeleton.setPositions(x_pred[:num_dofs])
    skeleton.setVelocities(x_pred[num_dofs:2*num_dofs])
    skeleton.setAccelerations(x_pred[2*num_dofs:3*num_dofs])

    # 3) Predict measurements and Jacobian
    h_pred = predict_measurements_from_state(x_pred)
    H = build_measurement_jacobian()

    # 4) Innovation
    y = z - h_pred

    # 5) Kalman gain
    S = H @ P_pred @ H.T + R
    # stable inversion: use np.linalg.solve when possible
    K = P_pred @ H.T @ np.linalg.inv(S)

    # 6) Update state & covariance
    x_upd = x_pred + K @ y
    I = np.eye(state_dim)
    P_upd = (I - K @ H) @ P_pred

    # 7) apply updated q/qdot/qddot to skeleton for visualization/usage
    skeleton.setPositions(x_upd[:num_dofs])
    skeleton.setVelocities(x_upd[num_dofs:2*num_dofs])
    skeleton.setAccelerations(x_upd[2*num_dofs:3*num_dofs])
    # optionally store updated world magnetic field somewhere if you need it:
    # world_b = x_upd[3*num_dofs:]

    return x_upd, P_upd

# --- Example realtime loop consuming IMU data ---
# read_imu() must return (accel, gyro, mag) as numpy arrays of shape (3,)
def read_imu():
    # user must replace with their own IO code
    raise NotImplementedError("Replace read_imu() with your IMU reading function")

try:
    while True:
        accel, gyro, mag = read_imu()  # each shape (3,)
        if accel is None:
            continue

        # Build measurement z stacked as [gyro_sensor1,...,gyro_sensorN, accel..., mag...]
        # If you have multiple sensors, collect vectors accordingly.
        # Here assuming sensors are in same order as used in skeleton functions and each returns (N*3,)
        z = np.concatenate([gyro, accel, mag])

        x, P = ekf_step(x, P, z)

        # optional: render GUI or print some states
        # gui.render() or print(x[:num_dofs])

        time.sleep(dt)

except KeyboardInterrupt:
    print("Stopping EKF loop")