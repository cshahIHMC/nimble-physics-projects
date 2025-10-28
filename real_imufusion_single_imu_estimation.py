import imufusion
import imumaster
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from Microstrain import MicroStrainIMU
from Tools.live_plotter import LivePlotter3D



# --------------------------------------------------------------------
# IMU helpers
# --------------------------------------------------------------------

def get_imu_data(imu):
    data = imu.get_ESTFILTER_data(20, 0)
    device_time = data[0]
    return np.array(data[1:10])

def read_sensor(data, plotter=None):
    acc_b = data[0:3] * 9.80665
    gyro_b = data[3:6]
    mag_b  = data[6:9]
    
    return acc_b, gyro_b, mag_b


# ---------------------------------------------------------------
# Main loop: runs the imu fusion 
# ---------------------------------------------------------------

def main():
    
    sample_frequency = 200  #200Hz
    dt_target = 1.0 / sample_frequency
    
    plotter = LivePlotter3D(100)
    
    ### Setup IMU to collect data
    imu = MicroStrainIMU("195772", 921600)
    
    try:
        imu.configure_ESTFLTER_imu(sample_frequency)
    except Exception as e:
        exc_type, _, tb = sys.exc_info()
        print(f"Error configuring IMU on line {tb.tb_lineno}: {e}")
        imu.set_to_idle()    
        
    # ### Instantiate the algorithms
    # offset = imufusion.Offset(sample_frequency)
    # ahrs = imufusion.Ahrs()
    
    # ahrs.settings = imufusion.Settings(
    # imufusion.CONVENTION_NED,  # convention
    # 10.0,  # gain
    # 2000,  # gyroscope range
    # 10,  # acceleration rejection
    # 10,  # magnetic rejection
    # 5 * sample_frequency,  # recovery trigger period = 5 seconds
    # )
    
    
    # Initialize the timer
    start_time = time.perf_counter()
    last_loop_time = start_time
    
    orientation = imumaster.Orientation(sample_rate=sample_frequency, frame='NED', method='EKF')
    
    
    try:
        while True:
            
            ### Get data from IMU
            imu_data = get_imu_data(imu)
            acc, gyro, mag = read_sensor(imu_data)

            ### Get the delta time
            loop_start = time.perf_counter()
            dt_loop = loop_start - last_loop_time
            last_loop_time = loop_start
            
            ### Bias corrected gyro
            # gyro_bias_corrected = offset.update(gyro)
            
            # ahrs.update(gyro_bias_corrected, acc, mag, dt_loop)
            
            # # Suppose this is your AHRS quaternion
            # q = ahrs.quaternion  # imufusion.Quaternion

            # # Invert (take conjugate)
            # q_inv = imufusion.Quaternion(np.array([q.w, -q.x, -q.y, -q.z], dtype=np.float32))
 
            # euler = q.to_euler()
            
            # q_estimate = orientation.EKF(acc, gyro, mag)
            # euler = orientation.eulerangle(q_estimate)
            
            # print("Euler Angles: ", euler)
            # plotter.update(euler)
            
            ### Enforce 200 Hz target rate
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
