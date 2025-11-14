#!/usr/bin/env python3

import sys
import time
import json
from dataclasses import dataclass
import numpy as np
import sys
import socket

NODE_RATE = 200
NUM_INSOLES = 2

# SET TCP IP for Windows microprocessor
TCP_IP =  '192.168.0.2' #'169.254.234.108' # LattePanda V1 `espresso`
# TCP_IP = '169.254.108.116' # LattePanda Delta 3 `cortado`
# TCP_IP = '169.254.176.177' # LattePanda Delta 3 `cappuccino`
TCP_PORT = 5005
BUFFER_SIZE = 400 # 340

@dataclass
class XSENSOR_Data:
    # Insole side
    side:str = 'u' # `u` = unknown, `l` = left, `r` = right

    # Force & COP
    force:float = 0.0
    COPx:float = 0.0
    COPz:float = 0.0

    # Linear acceleration
    linx:float = 0.0
    liny:float = 0.0
    linz:float = 0.0

    # Gyro (angular velocity)
    angx:float = 0.0
    angy:float = 0.0
    angz:float = 0.0

    # Orientation (Quaternion)
    qx:float = 0.0
    qy:float = 0.0
    qz:float = 0.0
    qw:float = 0.0

    timestamp:float = 0.0

    @property
    def acceleration(self):
        return [self.linx,self.liny,self.linz]

    @property
    def gyro(self):
        return [self.angx,self.angy,self.angz]

    @property
    def quaternion(self):
        return [self.qx,self.qy,self.qz,self.qw]


class XSENSORS:
    def __init__(self, num_xsensors=NUM_INSOLES):
        self.num_xsensors = num_xsensors
        self.insole_data = []
        self.insole_data_buffer = {'l': XSENSOR_Data(), 
                            'r': XSENSOR_Data(), 
                            'u': XSENSOR_Data()}
        self.unconnected = True


    def start_server(self, tcp_ip=TCP_IP, tcp_port=TCP_PORT, startup=''):
        error_os = 0
        while True:
            try:
                self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.s.connect((tcp_ip, tcp_port))
                break
            except ConnectionRefusedError:
                print('Connection refused! Retrying...')
                self.unconnected = True
            except ConnectionResetError:
                print('Connection reset! Retrying...')
                self.unconnected = True
            except OSError:
                if error_os == 0:
                    print('Connection broken! Retrying...')
                    self.unconnected = True
                    time.sleep(15)
                    error_os += 1
                else:
                    print('\nLattePanda Windows microprocessor not connecting. Try restarting it.\n')
                    self.unconnected = True
                    time.sleep(15)
            else:
                print('\nLattePanda Windows microprocessor not connecting. Try restarting it.\n')
                self.unconnected = True
                time.sleep(15)
            time.sleep(5)


    def pull_data(self):
        if not hasattr(self, "buffer"):
            self.buffer = ""
        try:
            data = self.s.recv(BUFFER_SIZE).decode()
            if not data:
                return
            self.buffer += data

            while True:
                try:
                    data_obj, index = json.JSONDecoder().raw_decode(self.buffer)
                    self.buffer = self.buffer[index:].lstrip()
                    if "a" in data_obj:
                        if len(data_obj["a"]) == 1:
                            print(data_obj["a"][0])
                        else:
                            if self.unconnected:
                                self.s.settimeout(10)
                                self.unconnected = False
                                print('\nConnection established!\n')
                            self.insole_data = data_obj["a"]
                except ValueError:
                    break
                    print('value error: data: ', data)
        except Exception as e:
            print("error", e)
            print(data)
#    def pull_data(self):
#        data = self.s.recv(BUFFER_SIZE)
#        #print(data)
#        data_arr = json.loads(data.decode())
#        #print('data decoded')
#        if len(data_arr.get("a")) == 1:
#            print(data_arr.get("a")[0])
#        else:
#            if self.unconnected:
#                self.s.settimeout(10)
#                self.unconnected = False
#                print('\nConnection established!\n')
#            self.insole_data = data_arr.get("a")


    def close_server(self):
        try:
            self.s.close()
        except:
            print('Tried unsuccessfully to close server. Server may not have been running.')
            pass


    def publish_data(self):
        try :
            self.pull_data()
        except ConnectionResetError:
            self.start_server(startup=False)
            self.unconnected = True
        except socket.timeout:
            print('Connection timed out...')
            self.start_server(startup=False)
            self.unconnected = True 
        except json.decoder.JSONDecodeError as e:
            if "Extra data:" not in str(e):
                self.start_server(startup=False)
                self.unconnected = True


        if len(self.insole_data) > 0:
        # TODO: Add better fix for the case where re-enumeration of sensors works
            for insole in range(self.num_xsensors):
                publisher_name = 'insole_publisher' + str(insole)
                
                combined_insole_data = XSENSOR_Data()

                if self.insole_data[insole * 14 + 1] == 'r':
                    side = 'r'
                elif self.insole_data[insole * 14 + 1] == 'l':
                    side = 'l'
                else:
                    side = 'u'

                combined_insole_data.timestamp = time.perf_counter()
                combined_insole_data.side = side
                combined_insole_data.force = float(self.insole_data[insole * 14 + 2])
                combined_insole_data.COPx = float(self.insole_data[insole * 14 + 3])
                combined_insole_data.COPz = float(self.insole_data[insole * 14 + 4])
                combined_insole_data.linx = float(self.insole_data[insole * 14 + 5])
                combined_insole_data.liny = float(self.insole_data[insole * 14 + 6])
                combined_insole_data.linz = float(self.insole_data[insole * 14 + 7])
                combined_insole_data.angx = float(self.insole_data[insole * 14 + 8])
                combined_insole_data.angy = float(self.insole_data[insole * 14 +9])
                combined_insole_data.angz = float(self.insole_data[insole * 14 + 10])
                combined_insole_data.qx = float(self.insole_data[insole * 14 + 11])
                combined_insole_data.qy = float(self.insole_data[insole * 14 + 12])
                combined_insole_data.qz = float(self.insole_data[insole * 14 + 13])
                combined_insole_data.qw = float(self.insole_data[insole * 14 + 14])

                self.insole_data_buffer[side] = combined_insole_data
                     
           # print(f"left: {self.insole_data_buffer['l'].force:.2f} | right: {self.insole_data_buffer['r'].force:.2f}")
            # return left and right insole data sets 
            return (self.insole_data_buffer['l'], self.insole_data_buffer['r'])


if __name__=="__main__":
    print("What is your Windows microprocessor's Ethernet IP address?")
    TCP_IP = input("(should be on the top of the unit): ")
    
    if type(TCP_IP) is not str:
        TCP_IP = str(TCP_IP)

    if len(TCP_IP) <= 10:
        print('Incorrect format for Ethernet IP address. Defaulting to `168.254.234.108`...')
        TCP_IP = '169.254.234.108'

    NUM_INSOLES = 2
    NODE_RATE = 200

    xsensors = XSENSORS(num_xsensors=NUM_INSOLES)
    xsensors.start_server(tcp_ip=TCP_IP, startup=True)
    
    while True:
        try:
            curr_time = time.time()
            left_data, right_data = xsensors.publish_data()
            print(f'left: {left_data.force}')
            print(f'right: {right_data.force}')
            time.sleep(1/NODE_RATE)
            print(time.time() - curr_time) 
        except KeyboardInterrupt:
            break

    print("\n\nExiting XSENSOR driver...\n")
    
    try:
        xsensors.close_server()
    except:
        pass
