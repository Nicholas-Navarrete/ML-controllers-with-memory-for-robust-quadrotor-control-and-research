"""
This code is based off of various examples from the crazyflie-lib-python repository https://github.com/bitcraze/crazyflie-lib-python

Start with the drone on and the crazyradio dongle plugged in to the computer running the program.
The host name variable below is the ip of the computer running the Vicon motion capture software.
You can connect to the Vicon computer over ethernet or wifi (though ethernet will obviously be faster).

At any point while the program is running, you can press Esc to stop the drone's motors and kill the
program. The keypress thread only polls every.

CF2X with long markermount and 5 IR markers has mass: 38.48g
HoverRPM = RPM When Thrust = 38.48g/4 = 9.62g
M1 HoverRPM = 55700 thrust was approx 9.6g ; 55700 had an rpm of 13931; max rpm was
M2 HoverRPM = 55800 thrust was approx 9.68g; 55800 had an rpm of 14168; max rpm was 14408
M3 HoverRPM = 62000 thrust was approx 9.6g
M4 HoverRPM = 65000 thrust was approx 9.4g 14286
"""

import numpy as np
import math
import time
import keyboard


from vicon_dssdk import ViconDataStream
import logging
import time
from time import perf_counter
from threading import Thread
import motioncapture

import cflib
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper

from stable_baselines3 import PPO
from sympy import intervals

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

import numpy as np
import sys

import numpy as np
import sys

from collections import deque

history = [deque(maxlen=5) for _ in range(4)]  # store last 5 values

def colorize(pct):
    """Return colored percentage string with green→yellow→red transition in 80–90%."""
    reset = "\033[0m"
    
    if pct < 80:
        color = "\033[92m"   # Green
    elif pct >= 90:
        color = "\033[91m"   # Red
    else:
        # Map 80–90 → [0,1]
        t = (pct - 80) / 10.0
        if t < 0.5:
            color = "\033[93m"  # Yellow (midway)
        else:
            color = "\033[91m"  # Red (approaching 90%)
    
    return f"{color}{pct:3d}%{reset}"

def display_quadrotor(action):
    """
    Display quadrotor motors in X configuration with smooth color scaling in 80–90%.
    Overwrites the same spot in the console each update.
    action: numpy array or list with 4 motor values (0–65534).
    """
    # Normalize 0–65534 to percentage
    global history
    
    percentages = [(a / 65534) * 100 for a in action[0,0]]
    
    # Update history
    for i, p in enumerate(percentages):
        history[i].append(p)
    
    # Compute rolling average
    smoothed = [sum(h)/len(h) for h in history]
    
    m1, m2, m3, m4 = [colorize(int(p)) for p in smoothed]

    lines = [
        f" m4({m4})        m1({m1})",
        f"      \\      /",
        f"         X",
        f"      /      \\",
        f" m3({m3})        m2({m2})"
    ]

    print("\n".join(lines))
    sys.stdout.write("\033[5A")
    sys.stdout.flush()


        
class DroneController:
    """Example that connects to a Crazyflie and ramps the motors up/down and
    the disconnects"""

    def __init__(self, link_uri, model_name=None):
        """ Initialize and run the example with the specified link_uri """
        
        self.ZeroTime = time.time()
        self._mc = motioncapture.connect(mocap_system_type, {'hostname': host_name})
        self.prev_time = 0
        self.control = True

        # Initialize the position and rotation buffers
        self.velBufSize = 10
        self.xVel=np.zeros(self.velBufSize)
        self.yVel=np.zeros(self.velBufSize)
        self.zVel=np.zeros(self.velBufSize)
        self.yawRate=np.zeros(self.velBufSize)
        self.rollRate=np.zeros(self.velBufSize)
        self.pitchRate=np.zeros(self.velBufSize)
        
        self.model=None

        self.CONTROLFREQUENCY = 120.0 #Hz
        self.G = 9.8
        self.M=0.027
        self.THRUST2WEIGHT_RATIO = 2.25
        self.GRAVITY = self.G*self.M
        self.KF = 3.16e-10

        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4*self.KF))
        # self.HOVER_RPM = 49786 # This is the hover RPM for the Crazyflie 2.1 with the small MoCap Deck
        # self.HOVER_RPM = 55996 # This is the hover RPM for the Crazyflie 2.1 with the large MoCap Deck

        self.HOVER_RPM = 35000#55996 # This is the hover RPM for the Crazyflie 2.1
        self.MAX_RPM = 65534 # This is the maximum RPM for the Crazyflie 2.1


        if model_name:
            self.model=PPO.load(model_name)
            
        self._cf = Crazyflie(rw_cache='./cache')
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)
        #self._cf.param.set_value('flightmode.stabmoderoll/pitch', 1) ## Setting the CF to be controlled by attitude rate
        self._cf.open_link(link_uri)

        print('Connecting to %s' % link_uri)

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""

        # Start a separate thread to do the motor test.
        # Do not hijack the calling thread!
        print('Connected to %s' % link_uri)
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
            # Sleep to avoid taking processor time away from the controller thread
            time.sleep(0.1)

    def PPO_control(self):
        actions, _ = self.model.predict(self.obs)
        return actions
    
    def _get_observations(self):
        ''' This function gets and returns the observations from the motion capture system.
            The observations are in the format expected by the PPO model.
            The observations vector is:
            [X, Y, Z, R, P, Y, VX, VY, VZ, WX, WY, WZ]
        '''
        # Get the position data from the motion capture system
        while True:
            self._mc.waitForNextFrame()
            for name, obj in self._mc.rigidBodies.items():
                if name == drone_object_name:
                    # Get a timestamp to go with the frame
                    timestamp = time.time()
                    if timestamp != self.prev_time:
                        # Position data is sent as [x, y, z]
                        pos = obj.position
                        # Rotation data is sent as a quaternion object with w,x,y,z values
                        q_w = obj.rotation.w
                        q_x = obj.rotation.x
                        q_y = obj.rotation.y
                        q_z = obj.rotation.z
                        # quanternion to euler angles conversion
                        rpy = np.array([np.arctan2(2*(q_w*q_z + q_x*q_y), 1 - 2*(q_y*q_y + q_z*q_z)), 
                                        np.arcsin(2*(q_w*q_y - q_x*q_z)), 
                                        np.arctan2(2*(q_w*q_x + q_y*q_z), 1 - 2*(q_x*q_x + q_y*q_y))])
                        if (abs(rpy[0]) > np.pi/3) or (abs(rpy[1]) > np.pi/3): # If the drone is flipped, shut off the motors and end the program
                            self.control = False
                            print("Drone flipped, shutting down")
                            
                        # Calculate the velocity and angular velocity
                        self.xVel = np.hstack(([pos[0]], self.xVel[0:self.velBufSize-1]))
                        self.yVel = np.hstack(([pos[1]], self.yVel[0:self.velBufSize-1]))
                        self.zVel = np.hstack(([pos[2]], self.zVel[0:self.velBufSize-1]))
                        self.yawRate = np.hstack(([rpy[2]], self.yawRate[0:self.velBufSize-1]))
                        self.rollRate = np.hstack(([rpy[0]], self.rollRate[0:self.velBufSize-1]))
                        self.pitchRate = np.hstack(([rpy[1]], self.pitchRate[0:self.velBufSize-1]))

                        Vx = np.mean(np.gradient(self.xVel)/(1/self.CONTROLFREQUENCY))
                        Vy = np.mean(np.gradient(self.yVel)/(1/self.CONTROLFREQUENCY))
                        Vz = np.mean(np.gradient(self.zVel)/(1/self.CONTROLFREQUENCY))
                        Wx = np.mean(np.gradient(self.rollRate)/(1/self.CONTROLFREQUENCY))
                        Wy = np.mean(np.gradient(self.pitchRate)/(1/self.CONTROLFREQUENCY))
                        Wz = np.mean(np.gradient(self.yawRate)/(1/self.CONTROLFREQUENCY))



                        self.prev_time = timestamp
                        return np.array([[[pos[0], pos[1], pos[2], 
                                            rpy[0], rpy[1], rpy[2], 
                                            Vx, Vy, Vz, 
                                            Wx, Wy, Wz]]])
    
    def _preprocess_action(self, action):
        ''' This Function Preprocess the action from the PPO model to the Crazyflie setpoint format.
            PPO outputs the action in the format: RPMm1 = self.hover_rpm *(1+0.05*thrust)
            Where the crazyflie setpoint wants rpms
        '''
        
        #print(self.RunTime ,int(action[0,0,0]), int(-action[0,0,1]), int(action[0,0,2]), int(action[0,0,3]), "\t", end="\r")
        # The action is in the format: [[[m1, m2, m3, m4]]]

        """Note: the motors layout changed so that m1 and m3 flip positions & m4 and m2 flip positions"""
        m3 = self.HOVER_RPM + self.HOVER_RPM* 0.05 * action[0,0,0]
        m4 = self.HOVER_RPM + self.HOVER_RPM* 0.05 * action[0,0,1]
        m1 = self.HOVER_RPM + self.HOVER_RPM* 0.05 * action[0,0,2]
        m2 = self.HOVER_RPM + self.HOVER_RPM* 0.05 * action[0,0,3]



        # Make sure the motors are not above the maximum RPM
        m1 = min(m1, self.MAX_RPM)
        m2 = min(m2, self.MAX_RPM)
        m3 = min(m3, self.MAX_RPM)
        m4 = min(m4, self.MAX_RPM)
        # Return the action in the format: [m1, m2, m3, m4]
        return np.array([[[m1, m2, m3, m4]]])

    def _control_loop(self): 
        """ The main loop of the controller code """
        # These goals are in standard unts for mocap (should be in meters)
        self.RunTime = time.time() - self.ZeroTime
        self.timer = self.RunTime
        Frequencies=[]
        # Unlock startup thrust protection
        self._cf.commander.send_setpoint(0, 0, 0, 0)

        while self.control:
            
            self.RunTime = time.time() - self.ZeroTime

            # Wait for control cycle
            if 1 == 1: #self.RunTime - self.timer > 1/self.CONTROLFREQUENCY:
                self.timer = self.RunTime
                Frequencies.append(perf_counter())
                obs = self._get_observations()

                #Apply Setpoints
                obs = np.array([[[obs[0,0,0], obs[0,0,1], 1-obs[0,0,2],
                                  obs[0,0,3], obs[0,0,4], obs[0,0,5],
                                 obs[0,0,6], obs[0,0,7], obs[0,0,8],
                                 obs[0,0,9], obs[0,0,10], obs[0,0,11]]]])

                action, _states = self.model.predict(obs, deterministic=True)
                action = self._preprocess_action(action)
                #display_quadrotor(action)
                #print("Action: ", int(action[0,0,0]), int(-action[0,0,1]), int(action[0,0,2]), int(action[0,0,3]), "\t", end="\r")
                
                #self._cf.commander.send_setpoint(int(action[0,0,0]), int(action[0,0,1]), int(action[0,0,2]), int(action[0,0,3]))
                #Measuring real control frequency
                
                


        # This is where we would put any code to be run before the program ends
        print("\nKill button pressed, shutting down")
        self._cf.commander.send_setpoint(0, 0, 0, 0)
        Frequencies = np.diff(Frequencies)          # time differences
        print(f"Average Latency: {Frequencies.mean():.8f} seconds")
        Frequencies = 1 / Frequencies             # instantaneous frequencies
        print("Average Control Frequency: ", np.mean(Frequencies), "Hz")
        self._cf.close_link()


if __name__ == '__main__':
    print('starting')
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()
    print('drivers initialized')
    controller = DroneController(uri, model_name = "SimToReal22")

