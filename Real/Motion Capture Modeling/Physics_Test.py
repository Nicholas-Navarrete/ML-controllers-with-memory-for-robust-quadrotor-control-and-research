import numpy as np
import math
import time
import keyboard

import logging
import time
from threading import Thread
import motioncapture
import csv
import pandas as pd
import cflib
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper
from vicon_dssdk import ViconDataStream

'''
This program is designed to test VICON motion capture system by logging kinematic data at a set frequency.
'''




uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

TimeSize=5

# The host name or ip address of the mocap system
host_name = '128.101.167.111'

# The type of the mocap system
# Valid options are: 'vicon', 'optitrack', 'optitrack_closed_source', 'qualisys', 'nokov', 'vrpn', 'motionanalysis'
mocap_system_type = 'vicon'

# The name of the rigid body that represents the Crazyflie (VICON object name)
drone_object_name = 'Drone4'



logging.basicConfig(level=logging.ERROR)

class RollingBuffer():
    """ 
    This is a rolling buffer class where the rows are the entries and the colums are the values in the entry of the buffer
    """

    # TODO add filo=false logic
    def __init__(self, Length, n_elements=1, filo = True):
        """Initialize the numpy rolling buffer with length and defult first in last out"""
        self.buffer = np.zeros([Length, n_elements])
        self.filo=filo

    def _input(self,data):
        """ Data is input one at a time """
        for y in range(len(self.buffer)-1):
            self.buffer[-1-y,:]=self.buffer[-2-y,:] ## Shift buffer FILO
        self.buffer[0,:]=data

    def _readBuffer(self):
        return self.buffer
        

class DroneController:
    """Example that connects to a Crazyflie and ramps the motors up/down and
    the disconnects"""

    def __init__(self, link_uri):
        """ Initialize and run the example with the specified link_uri """
        self.ZeroTime = time.time()
        self._mc = motioncapture.connect(mocap_system_type, {'hostname': host_name})
        self.prev_time = 0
        self.control = True
        self.HOVER_THRUST = 57834

        ### ADDED FOR PHYSICS LOGGING ###
        self.roll_data=[]
        self.pitch_data=[]
        self.yaw_data=[]
        self.time_data=[]

        # Initialize the Vicon motion capture system
        self.client = ViconDataStream.Client()
        self.client.Connect(host_name)
        self.client.SetStreamMode(ViconDataStream.Client.StreamMode.EClientPull)
        self.client.EnableSegmentData()
        
        self._connected(link_uri)

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""

        # Start a separate thread to do the motor test.
        # Do not hijack the calling thread!
        self.input_thread = Thread(target=self._watch_for_key_press, daemon=True)
        self.input_thread.start()
        self.control_thread = Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the specified address)"""
        print('Connection to %s failed: %s' % (link_uri, msg))

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        print('Disconnected from %s' % link_uri)

    def _watch_for_key_press(self):
        """ Non system dependent way to do a non-blocking read of keyboard inputs """
        while True:
            if keyboard.is_pressed("esc"):
                self.control = False
                # Read in the key that was pressed so it doesn't appear in the terminal when the program exits
                break
            time.sleep(0.1)

    def _get_position_data(self, rigid_body_name):
        """ Reads position data from the motion capture system and returns x,y,z, yaw, and a timestamp """
        # Get the position data from the motion capture system
        self.client.GetFrame()
        pos = self.client.GetSegmentGlobalTranslation(drone_object_name, drone_object_name)[0]
        rpy = self.client.GetSegmentGlobalRotationEulerXYZ(drone_object_name, drone_object_name)[0]
        return pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]

    def PID_control(self, x_pos, y_pos, z_pos, x_goal, y_goal, z_goal, Yaw, times):
        """ Takes in positional data and returns control parameters (roll, pitch, thrust) """
        # Gains
        Kpx=1.5291
        Kpy=Kpx
        Kpz=3.5
        Kdx=0.8155
        Kdy=Kdx
        kdz=0.4104

        # Apply Negative FB
        x_error=x_goal-x_pos
        y_error=y_goal-y_pos
        z_error=z_goal-z_pos

        cos_yaw = math.cos(Yaw)
        sin_yaw = math.sin(Yaw)
        
        for i in range(len(x_error)):
            global_errors=np.array([[x_error[i]],[y_error[i]],[z_error[i]]])
            RelativeErrorRotation = np.array([[cos_yaw, sin_yaw, 0], [-sin_yaw, cos_yaw, 0], [0,0,1]])
            
            Local_errors=RelativeErrorRotation@global_errors

            x_error[i] = Local_errors[0][0]
            y_error[i] = Local_errors[1][0]
            z_error[i] = Local_errors[2][0]
        # print(x_error[-1], "\t", y_error[-1], "\t", z_error[-1], "\t", Yaw, "          ", end="\r")

        # Calculate Derivatives
        dx=np.mean(np.gradient(x_error)/np.gradient(times))
        dy=np.mean(np.gradient(y_error)/np.gradient(times))
        dz=np.mean(np.gradient(z_error)/np.gradient(times))

        # pitch is theta (rotation about the y axis where x is pointing forwards on the aircraft)
        # roll is phi (rotation about the x axis where x is pointing forwards on the aircraft)
        theta = Kpx*x_error[-1] + Kdx*dx
        phi = Kpy*y_error[-1] + Kdy*dy
        thrust = Kpz*z_error[-1] + kdz*dz

        theta = np.rad2deg(theta)
        phi = np.rad2deg(phi)

        # Setting saturation
        if theta > 15:
            theta = 15
        elif theta < -15:
            theta = -15

        if phi > 15:
            phi = 15
        elif phi < -15:
            phi = -15

        if thrust > 2.3346:
            thrust = 2.23346
        elif thrust < 0:
            thrust = 0
        thrust = thrust*(50000/2.3346) + 10001
        
        # Drone's z-axis points down, so flip the sign of the roll to compensate
        return (theta, -phi, int(thrust))

    def _control_loop(self):
        """ The main loop of the controller code """
        # These goals are in standard unts for mocap (should be in meters)
        # Initial goal states
        self.RunTime = time.time() - self.ZeroTime
        self.timer = self.RunTime

        self.x_buffer = []
        self.y_buffer = []
        self.z_buffer = []
        self.roll_buffer = []
        self.pitch_buffer = []
        self.yaw_buffer = []
        NUM_SAMPLES = 0
        print("\n"+"Press [ESC] to stop sampling.")
        while self.control:
            
            self.RunTime = time.time() - self.ZeroTime
            #print("RunTime: ", self.RunTime, end="\r")
            # Control frequency
            frequency = 120.0
            
            # Get position data from the motion capture system every control cycle
            if self.RunTime - self.timer > 1/frequency:
                print("Sample Count: ", NUM_SAMPLES, end="\r")
                NUM_SAMPLES += 1
                X, Y, Z, R, P, Yaw = self._get_position_data(drone_object_name)
                self.x_buffer.append(X)
                self.y_buffer.append(Y)
                self.z_buffer.append(Z)
                self.roll_buffer.append(R)
                self.pitch_buffer.append(P)
                self.yaw_buffer.append(Yaw)

                self.timer = self.RunTime
            

           
        # This is where we would put any code to be run before the program ends
        self._print_physics_data()
        
        print("\nKill button pressed, shutting down")


    def _print_physics_data(self):
        filename = 'x3y1z1_'
        # Export each buffer to its own CSV file
        pd.DataFrame(self.x_buffer, columns=['x']).to_csv(filename+'x_buffer.csv', index=False)
        pd.DataFrame(self.y_buffer, columns=['y']).to_csv(filename+'y_buffer.csv', index=False)
        pd.DataFrame(self.z_buffer, columns=['z']).to_csv(filename+'z_buffer.csv', index=False)
        pd.DataFrame(self.roll_buffer, columns=['roll']).to_csv(filename+'roll_buffer.csv', index=False)
        pd.DataFrame(self.pitch_buffer, columns=['pitch']).to_csv(filename+'pitch_buffer.csv', index=False)
        pd.DataFrame(self.yaw_buffer, columns=['yaw']).to_csv(filename+'yaw_buffer.csv', index=False)


if __name__ == '__main__':
    # Initialize the low-level drivers
    #cflib.crtp.init_drivers()

    controller = DroneController(uri)
    # Wait for the input thread to finish before exiting
    controller.input_thread.join()
    controller.control_thread.join()


    

