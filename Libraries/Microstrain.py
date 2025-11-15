import sys
import time

# import the microstrain library pack

sys.path.append('/usr/lib/python3.13/dist-packages')
import mscl


class MicroStrainIMU:
    """ 
    A class to interface with the MicroStrain IMU 3DM-GX5-AHRS via serial connection RS232.

    This class will open a connection with an IMU start the IMU streaming and retrieve data from
    the IMU and specify which channels of data the IMU should return.
    """

    def __init__(self, imu_id: str, baud_rate: int) -> None:
        """
        Initializes the connection with the IMU with a specified port and  baud rate.

        """

        self.baud_rate = baud_rate
        self.serial_connection = None
        
        devices = self.check_connected_imus()
     
        
        try:
            # Iterate through IMU IDs detected by device
            for serial_port, device_info in devices.items():
                connected_device = device_info.serial().split(".")[1]
                print(connected_device)

                # If IMU ID is in list of IMU IDs from config, include in dict
                if imu_id == connected_device:
                    self.port = serial_port

        except Exception:
            print("Handling...")
            print(
                "Tip: check that you set all the serial numbers of the IMUs properly!"
            )
            for serial_port in devices.keys():
                tmp_connection = mscl.Connection.Serial(serial_port)
                tmp_connection.disconnect()

        print(self.port)
        connection = mscl.Connection.Serial(self.port, self.baud_rate)
        
     

        # create a node which is used to interface with the IMU
        self.node = mscl.InertialNode(connection)
        
        print(f"Connected IMUs: {self.node}")

        # do a test ping to see if connected
        success = self.node.ping()
        print(f"Ping IMU {self.port} {success}")

        # on initalization set the IMU to idel state
        
        #self.node.setToIdle()

        # set initial values used for data logging to zero
        
        self.timestamp = 0
        self.accel_x = 0
        self.accel_y = 0
        self.accel_z = 0
        self.gyro_x = 0
        self.gyro_y = 0
        self.gyro_z =0
        self.mag_x = 0
        self.mag_y = 0
        self.mag_z = 0
        self.quat = 0
        self.ESTquat = 0
        self.deltaThetax = 0
        self.deltaThetay = 0
        self.deltaThetaz = 0
        self.deltaVelx = 0
        self.deltaVely = 0
        self.deltaVelz = 0
        # measures from estimated filtered values
        self.estLinAccel_x = 0
        self.estLinAccel_y = 0
        self.estLinAccel_z = 0
        self.estAngRate_x = 0
        self.estAngRate_y = 0
        self.estAngRate_z = 0
        self.estGyroBias_x = 0
        self.estGyroBias_y = 0
        self.estGyroBias_z = 0
        self.estMagEast = 0
        self.estMagDown = 0
        self.estMagNorth = 0
        self.estMagIncl = 0
        self.estMagDecl = 0
        self.estCompAccel_x = 0
        self.estCompAccel_y = 0
        self.estCompAccel_z = 0
        self.estQuat = 0
        self.estTimeWeek = 0
        self.estTimeTow = 0
    
    def set_to_idle(self):
        """ 
        Sets the state of the IMU to idle
        """
        self.node.setToIdle()
        
        
    def check_connected_imus(self) -> dict:
        """Check for connected IMUs and return a dictionary of mscl DeviceInfo objects

        Returns:
            connected_devices (dict): Dictionary of mscl DeviceInfo objects
            with serial ports as keys.
        """
        connected_devices = mscl.Devices.listInertialDevices()
        # print(f"Connected serial ports: {connected_devices.keys()}")

        return connected_devices

            
    def start_data_streaming(self, imu_class):
        """ 
        Puts the IMU in streaming mode for the type of class you want
        """

        if 'AHRS' in imu_class:
            class_type = mscl.MipTypes.CLASS_AHRS_IMU
        if 'ESTFILTER' in imu_class:
            class_type = mscl.MipTypes.ESTFILTER

        self.node.enableDataStream(class_type)
        
        
    def configure_ESTFLTER_imu(self, sample_rate: int):
        
        self.sample_rate = sample_rate
        
        if self.node.features().supportsCategory(mscl.MipTypes.CLASS_AHRS_IMU):
            ahrsImuChs = mscl.MipChannels()
            ahrsImuChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_SENSOR_SCALED_MAG_VEC, mscl.SampleRate.Hertz(self.sample_rate)))
            ahrsImuChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_SENSOR_SCALED_ACCEL_VEC, mscl.SampleRate.Hertz(self.sample_rate)))
            ahrsImuChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_SENSOR_SCALED_GYRO_VEC, mscl.SampleRate.Hertz(self.sample_rate)))
            
            ## Attempt to get delta theta and delta velocity
            ahrsImuChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_SENSOR_DELTA_THETA_VEC, mscl.SampleRate.Hertz(self.sample_rate)))
            ahrsImuChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_SENSOR_DELTA_VELOCITY_VEC, mscl.SampleRate.Hertz(self.sample_rate)))
            
            ahrsImuChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_SENSOR_ORIENTATION_QUATERNION, mscl.SampleRate.Hertz(sample_rate)))
            
            ahrsImuChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_SENSOR_GPS_CORRELATION_TIMESTAMP, mscl.SampleRate.Hertz(self.sample_rate)))
            

        # if node supports Estimation Filter
        if self.node.features().supportsCategory(mscl.MipTypes.CLASS_ESTFILTER):
                estFilterChs = mscl.MipChannels()
                # estFilterChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_ESTFILTER_ESTIMATED_GYRO_BIAS, mscl.SampleRate.Hertz(sample_rate)))
                # estFilterChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_ESTFILTER_MAGNETIC_MODEL_SLN, mscl.SampleRate.Hertz(sample_rate)))
                # estFilterChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_ESTFILTER_GPS_TIMESTAMP, mscl.SampleRate.Hertz(sample_rate)))
        
                # estFilterChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_ESTFILTER_COMPENSATED_ACCEL, mscl.SampleRate.Hertz(sample_rate)))
        
                estFilterChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_ESTFILTER_ESTIMATED_ORIENT_QUATERNION, mscl.SampleRate.Hertz(self.sample_rate)))
                # estFilterChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_ESTFILTER_ESTIMATED_LINEAR_ACCEL, mscl.SampleRate.Hertz(sample_rate)))
                # estFilterChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_ESTFILTER_ESTIMATED_ORIENT_QUATERNION, mscl.SampleRate.Hertz(sample_rate)))
                #apply to the node
                
                # estFilterChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_SENSOR_SCALED_ACCEL_VEC, mscl.SampleRate.Hertz(self.sample_rate)))
                # estFilterChs.append(mscl.MipChannel(mscl.MipTypes.CH_FIELD_SENSOR_SCALED_GYRO_VEC, mscl.SampleRate.Hertz(self.sample_rate)))
                
         # activate the channles
        self.node.setActiveChannelFields(mscl.MipTypes.CLASS_AHRS_IMU, ahrsImuChs)
        self.node.setActiveChannelFields(mscl.MipTypes.CLASS_ESTFILTER, estFilterChs)

        # start streaming
        self.start_data_streaming('AHRS')
        self.node.enableDataStream(mscl.MipTypes.CLASS_ESTFILTER)
        
    def get_ESTFILTER_data(self, measure_time_out: int, max_packet_num: int):
        # reutrns outputs of sensor values from standard set configeration which is time stamp accel, gyro and mag values
        #pack_num = self.node.totalPackets()
        #print(f'total packets: {pack_num}')
        packets = self.node.getDataPackets(measure_time_out, max_packet_num)
        
        while (len(packets) < 1):
            print(len(packets))
            packets = self.node.getDataPackets(measure_time_out, max_packet_num)
        
        for packet in packets:
            # get all the datapoints in one packet
            points = packet.data()
            for dataPoint in points:              
                # print(dataPoint.channelName(), " ", dataPoint.as_string())
                if 'AccelX' in dataPoint.channelName():
                   self.accel_x = dataPoint.as_float()
                elif 'AccelY' in dataPoint.channelName():
                    self.accel_y = dataPoint.as_float()
                elif 'AccelZ' in dataPoint.channelName():
                    self.accel_z = dataPoint.as_float()
                elif 'scaledGyroX' in dataPoint.channelName():
                    self.gyro_x = dataPoint.as_float()
                elif 'scaledGyroY' in dataPoint.channelName():
                    self.gyro_y = dataPoint.as_float()
                elif 'scaledGyroZ' in dataPoint.channelName():
                    self.gyro_z = dataPoint.as_float()
                elif 'TimestampTow' in dataPoint.channelName():
                    self.timestamp = dataPoint.as_float()
                elif 'MagX' in dataPoint.channelName():
                    self.mag_x = dataPoint.as_float()
                elif 'MagY' in dataPoint.channelName():
                    self.mag_y = dataPoint.as_float()
                elif 'MagZ' in dataPoint.channelName():
                    self.mag_z = dataPoint.as_float()
                elif 'deltaThetaX' in dataPoint.channelName():
                    self.deltaThetax = dataPoint.as_float()
                elif 'deltaThetaY' in dataPoint.channelName():
                    self.deltaThetay = dataPoint.as_float()
                elif 'deltaThetaZ' in dataPoint.channelName():
                    self.deltaThetaz = dataPoint.as_float()
                elif 'deltaVelX' in dataPoint.channelName():
                    self.deltaVelx = dataPoint.as_float()
                elif 'deltaVelY' in dataPoint.channelName():
                    self.deltaVely = dataPoint.as_float()
                elif 'deltaVelZ' in dataPoint.channelName():
                    self.deltaVelz = dataPoint.as_float()
                elif 'orientQuaternion' in dataPoint.channelName():
                    self.quat = dataPoint.as_Vector()
                elif 'estOrientQuaternion' in dataPoint.channelName():
                    self.ESTquat = dataPoint.as_Vector()
                else:
                    pass
                
             
            # return (self.timestamp, self.accel_x, self.accel_y, self.accel_z, self.gyro_x, self.gyro_y, self.gyro_z, self.quat, self.ESTquat)
            return (self.timestamp, self.accel_x, self.accel_y, self.accel_z, self.gyro_x, self.gyro_y, self.gyro_z, self.mag_x, self.mag_y, self.mag_z, self.quat, self.ESTquat, self.deltaThetax, self.deltaThetay, self.deltaThetaz, self.deltaVelx, self.deltaVely, self.deltaVelz)

# if __name__ == "__main__":

#     # set up logger
#     from logger import Logger

#     data_log = Logger(log_path = "./logs", file_name = "example_logger", buffer_size = 5)

#     # create varriable to track that matches the data you want to get off the IMU
#     imu1_accel_x = 0
#     imu1_accel_y = 0
#     imu1_accel_z = 0
#     imu1_gyro_x = 0
#     imu1_gyro_y = 0
#     imu1_gyro_z = 0
#     imu1_timestamp = 0
    
#     data_log.track_variable(lambda: imu1_accel_x, "imu1_accel_x")
#     data_log.track_variable(lambda: imu1_accel_y, "imu1_accel_y")
#     data_log.track_variable(lambda: imu1_accel_z, "imu1_accel_z")
#     data_log.track_variable(lambda: imu1_gyro_x, "imu1_gyro_x")
#     data_log.track_variable(lambda: imu1_gyro_y, "imu1_gyro_y")
#     data_log.track_variable(lambda: imu1_gyro_z, "imu1_gyro_z")
#     data_log.track_variable(lambda: imu1_timestamp, "imu1_timestamp")


#     # create varriable to track that matches the data you want to get off the IMU
#     imu2_accel_x = 0
#     imu2_accel_y = 0
#     imu2_accel_z = 0
#     imu2_gyro_x = 0
#     imu2_gyro_y = 0
#     imu2_gyro_z = 0
#     imu2_timestamp = 0
    
#     data_log.track_variable(lambda: imu2_accel_x, "imu2_accel_x")
#     data_log.track_variable(lambda: imu2_accel_y, "imu2_accel_y")
#     data_log.track_variable(lambda: imu2_accel_z, "imu2_accel_z")
#     data_log.track_variable(lambda: imu2_gyro_x, "imu2_gyro_x")
#     data_log.track_variable(lambda: imu2_gyro_y, "imu2_gyro_y")
#     data_log.track_variable(lambda: imu2_gyro_z, "imu2_gyro_z")
#     data_log.track_variable(lambda: imu2_timestamp, "imu2_timestamp")

#     data_log.info("started tracking variables.")
    
#     imu = MicroStrainIMU("/dev/ttyACM1", 921600)
#     imu.configure_AHRS_imu_and_start_streaming(250)
    
#     imu2 = MicroStrainIMU("/dev/ttyACM0", 921600)
#     imu2.configure_AHRS_imu_and_start_streaming(250)
    
#     t_initial = time.time()
#     try:
#         while True:
#             cur_time = time.time()
#             imu1_timestamp, imu1_accel_x, imu1_accel_y, imu1_accel_z, imu1_gyro_x, imu1_gyro_y, imu1_gyro_z, imu1_mag_x, imu1_mag_y, imu1_mag_z = (imu.get_AHRS_data(20, 0))
            
#             imu2_timestamp, imu2_accel_x, imu2_accel_y, imu2_accel_z, imu2_gyro_x, imu2_gyro_y, imu2_gyro_z, imu2_mag_x, imu2_mag_y, imu2_mag_z, quat, deltaThetax, deltaThetay, deltaThetaz, deltaVelx, deltaVely, deltaVelz = imu2.get_AHRS_data_with_quat(20, 0)
            
#             data_log.update()
#             time_diff = time.time() - cur_time
#             print(f'time dif: {time_diff}')
#     except KeyboardInterrupt:
#         print('Ending Stream')
#         imu.set_to_idle()
#         imu2.set_to_idle()

#         # end the data logger
#         data_log.info("Finished Logging.")
#         data_log.close()