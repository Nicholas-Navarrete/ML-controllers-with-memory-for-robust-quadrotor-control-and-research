"""
Nicholas Navarrete

Agent Control 2 Implements the VICON dssdk to get the position of the drone and uses a trained PPO model
with xyz rpy Vxyz Wxyz as the observation space to control the drone in position hold.

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
import pandas as pd
import numpy as np
import time
import keyboard
from vicon_dssdk import ViconDataStream
import logging
from time import perf_counter
from threading import Thread
import cflib
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper
from stable_baselines3 import PPO
from sympy import intervals
import sys
from collections import deque
from stable_baselines3.common.policies import ActorCriticPolicy


uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

# The host name or ip address of the mocap system
host_name = '128.101.167.111'

# The type of the mocap system
# Valid options are: 'vicon', 'optitrack', 'optitrack_closed_source', 'qualisys', 'nokov', 'vrpn', 'motionanalysis'
mocap_system_type = 'vicon'

# The name of the rigid body that represents the Crazyflie (VICON object name)
drone_object_name = 'Drone7'

logging.basicConfig(level=logging.ERROR)



####### The following code is for displaying the quadrotor motors in the terminal #######

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


###### The following code is for controlling the drone #######
class DroneController:
    """Example that connects to a Crazyflie and ramps the motors up/down and
    the disconnects"""

    def __init__(self, link_uri, model_name=None):
        """ Initialize and run the example with the specified link_uri """
        self.obs_buffer=[]
        self.Action_buffer=[]
        self.ZeroTime = perf_counter()

        # Initialize the Vicon motion capture system
        self.client = ViconDataStream.Client()
        self.client.Connect(host_name)
        self.client.SetStreamMode(ViconDataStream.Client.StreamMode.EClientPull)
        self.client.EnableSegmentData()

        
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

        self.HOVER_RPM = 55996 # This is the hover RPM for the Crazyflie 2.1
        self.HOVER_RPM = 59000
        self.MAX_RPM = 65534 # This is the maximum RPM for the Crazyflie 2.1

       
        if 'policy' not in model_name:
            self.model=PPO.load(model_name, device = 'cuda')
        else:
            self.model=ActorCriticPolicy.load(model_name, device = 'cuda')    
        print("Model Loaded "+self.model.__class__.__name__)
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

    def save_buffers_to_csv(self, filename="drone_data.csv"):
        """
        Save observation and action buffers to a CSV file.
        Each row corresponds to one timestep.
        """
        if not self.obs_buffer or not self.Action_buffer:
            print("No data to save.")
            return

        # Flatten observation and action arrays
        obs_array = np.concatenate(self.obs_buffer, axis=0).reshape(len(self.obs_buffer), -1)
        act_array = np.concatenate(self.Action_buffer, axis=0).reshape(len(self.Action_buffer), -1)

        # Ensure equal length
        min_len = min(len(obs_array), len(act_array))
        obs_array = obs_array[:min_len]
        act_array = act_array[:min_len]

        # Create column names
        obs_cols = [
            'X','Y','Z','Roll','Pitch','Yaw',
            'Vx','Vy','Vz','Wx','Wy','Wz'
        ]
        act_cols = ['Act_m1','Act_m2','Act_m3','Act_m4']

        df = pd.DataFrame(
            np.hstack((obs_array, act_array)),
            columns=obs_cols + act_cols
        )

        df.to_csv(filename, index=False)
        print(f"✅ Saved buffers to {filename}")

    def _get_observations(self):
        ''' This function gets and returns the observations from the motion capture system.
            The observations are in the format expected by the PPO model.
            The observations vector is:
            [X, Y, Z, R, P, Y, VX, VY, VZ, WX, WY, WZ]
        '''
        # Get the position data from the motion capture system
        self.client.GetFrame()
        pos = self.client.GetSegmentGlobalTranslation(drone_object_name, drone_object_name)[0]
        rpy = self.client.GetSegmentGlobalRotationEulerXYZ(drone_object_name, drone_object_name)[0]


        if (abs(rpy[0]) > np.pi/2) or (abs(rpy[1]) > np.pi/2): # If the drone is flipped, shut off the motors and end the program
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

        
        m1 = self.HOVER_RPM + self.HOVER_RPM* 0.1 * action[0,0,0]
        m2 = self.HOVER_RPM + self.HOVER_RPM* 0.1 * action[0,0,1]
        m3 = self.HOVER_RPM + self.HOVER_RPM* 0.1 * action[0,0,2]
        m4 = self.HOVER_RPM + self.HOVER_RPM* 0.1 * action[0,0,3]



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
            Target = [0.0, 0.0, 1.5]
            # Wait for control cycle
            if  1==1:#self.RunTime - self.timer > 1/self.CONTROLFREQUENCY:
                self.timer = self.RunTime
                Frequencies.append(perf_counter())
                obs = self._get_observations()

                #Apply Setpoints (Errors)
                obs = np.array([[[Target[0]- obs[0,0,0], Target[1]-obs[0,0,1], Target[2]-obs[0,0,2],
                                  obs[0,0,3], obs[0,0,4], obs[0,0,5],
                                 obs[0,0,6], obs[0,0,7], obs[0,0,8],
                                 obs[0,0,9], obs[0,0,10], obs[0,0,11]]]])
                
                

                action, _states = self.model.predict(obs, deterministic=True)  ### This is taking a lot of time
                action = self._preprocess_action(action)
                self.obs_buffer.append(obs)
                self.Action_buffer.append(action)
                #display_quadrotor(action)
                #print("Action: ", int(action[0,0,0]), int(-action[0,0,1]), int(action[0,0,2]), int(action[0,0,3]), "\t", end="\r")
                
                self._cf.commander.send_setpoint(int(action[0,0,0]), int(action[0,0,1]), int(action[0,0,2]), int(action[0,0,3]))
                #Measuring real control frequency
                
                


        # This is where we would put any code to be run before the program ends
        print("\nKill button pressed, shutting down")
        self._cf.commander.send_setpoint(0, 0, 0, 0)
        Frequencies = np.diff(Frequencies)          # time differences
        print(f"Average Latency: {Frequencies.mean():.8f} seconds")
        Frequencies = 1 / Frequencies             # instantaneous frequencies
        print("Average Control Frequency: ", np.mean(Frequencies), "Hz")
        print("Control Frequency Variance:", np.var(Frequencies), "Hz^2")
        self.save_buffers_to_csv("obs_action_data.csv")
        self._cf.close_link()


if __name__ == '__main__':
    
    print('starting')
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()
    print('drivers initialized')
    controller = DroneController(uri, model_name = "SimToReal60_policy.zip")

