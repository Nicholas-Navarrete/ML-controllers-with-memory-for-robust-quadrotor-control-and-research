"""
This program is built to test the motors of the crazyflie, you can change which motors are being tested by modifying the 
sendsetpoint command, the self.thrust variable is being updated by user input in a separate thread. This variable
is the value that is written to the motors directly.

"""

import numpy as np
import math
import time
import keyboard

import logging
import time
from threading import Thread
import motioncapture

import cflib
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper

from stable_baselines3 import PPO

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
        
class DroneController:
    """Example that connects to a Crazyflie and ramps the motors up/down and
    the disconnects"""

    def __init__(self, link_uri, model_name=None):
        """ Initialize and run the example with the specified link_uri """
        # Shared variable
        self.thrust = 55996
        self.ZeroTime = time.time()
        #self._mc = motioncapture.connect(mocap_system_type, {'hostname': host_name})
        self.prev_time = 0
        self.control = True

        # Initialize the position and rotation buffers
        self.velBufSize = 5
        self.xVel=np.zeros(self.velBufSize)
        self.yVel=np.zeros(self.velBufSize)
        self.zVel=np.zeros(self.velBufSize)
        self.yawRate=np.zeros(self.velBufSize)
        self.rollRate=np.zeros(self.velBufSize)
        self.pitchRate=np.zeros(self.velBufSize)
        
        self.model=None

        self.CONTROLFREQUENCY = 200.0 #Hz
        self.G = 9.8
        self.M=0.027
        self.THRUST2WEIGHT_RATIO = 2.25
        self.GRAVITY = self.G*self.M
        self.KF = 3.16e-10

        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4*self.KF))
        # self.HOVER_RPM = 49786 # This is the hover RPM for the Crazyflie 2.1 with the small MoCap Deck
        # self.HOVER_RPM = 55996 # This is the hover RPM for the Crazyflie 2.1 with the large MoCap Deck

        self.HOVER_RPM = 55996 # This is the hover RPM for the Crazyflie 2.1
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
        Thread(target=self.thrust_input, daemon=True).start()

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

    def thrust_input(self):
        while True:
            try:
                val = input("Enter thrust value: ")
                if val.isdigit():
                    self.thrust = int(val)
                    print(f"Updated thrust to {self.thrust}")
            except Exception as e:
                print(f"Input error: {e}")

    def _get_position_data(self, rigid_body_name):
        """ Reads position data from the motion capture system and returns x,y,z, yaw, and a timestamp """
        # This implementation WILL cause the program to hang here if the drone moves out of range of the cameras
        while True:
            self._mc.waitForNextFrame()
            for name, obj in self._mc.rigidBodies.items():
                if name == rigid_body_name:
                    # Get a timestamp to go with the frame
                    # Vicon does not send a timestamp with the frame, so this is the next best way
                    timestamp = time.time()
                    if timestamp != self.prev_time:
                        # Position data is sent as [x, y, z]
                        pos = obj.position
                        # Rotation data is sent as a quaternion object with w,x,y,z values
                        q_w = obj.rotation.w
                        q_x = obj.rotation.x
                        q_y = obj.rotation.y
                        q_z = obj.rotation.z

                        # Sometimes Vicon sends data packets that have x, y, and z as either extremely large values or all zero (they're floats so something like 1e-30)
                        # Discard that packet and wait for the next one
                        # This is a temporary solution until a more reliable way to detect those bad packets is found
                        # while ((np.abs(pos[0]) < 1.0e-9 or np.abs(pos[0]) > 1.0e4) or (np.abs(pos[1]) < 1.0e-9 or np.abs(pos[1]) > 1.0e4) or (np.abs(pos[2]) < 1.0e-9 or np.abs(pos[2]) > 1.0e4)):
                        if np.abs(pos[0]) > 0.000000001 and np.abs(pos[0]) < 10000 and np.abs(pos[1]) > 0.000000001 and np.abs(pos[1]) < 10000 and np.abs(pos[2]) > 0.000000001 and np.abs(pos[2]) < 10000:
                            yaw = math.atan2(2*(q_w*q_z + q_x*q_y), 1 - 2*(q_y*q_y + q_z*q_z))
                            self.prev_time = timestamp
                            return pos[0], pos[1], pos[2], yaw, timestamp

    def PID_control(self, x_pos, y_pos, z_pos, x_goal, y_goal, z_goal, Yaw, times):
        """ Takes in positional data and returns control parameters (roll, pitch, thrust) """
        # Gains
        Kpx=1.5291
        Kpy=Kpx
        Kpz=2.4
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

    def PPO_control(self):
        actions, _ = self.model.predict(self.obs)
        return actions

    #def _get_Observations(self, rigid_body_name):  ##### TODO UPDATE THE FUNCTION TO RETURN OBSERVATIONS IN THE FORMAT OF PPO
        """ Reads position data from the motion capture system and returns x,y,z, yaw, and a timestamp 

            Observation vector X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ
        """
        # This implementation WILL cause the program to hang here if the drone moves out of range of the cameras
        while True:
            self._mc.waitForNextFrame()
            for name, obj in self._mc.rigidBodies.items():
                if name == rigid_body_name:
                    # Get a timestamp to go with the frame
                    # Vicon does not send a timestamp with the frame, so this is the next best way
                    timestamp = time.time()
                    if timestamp != self.prev_time:
                        # Position data is sent as [x, y, z]
                        pos = obj.position
                        # Rotation data is sent as a quaternion object with w,x,y,z values
                        q_w = obj.rotation.w
                        q_x = obj.rotation.x
                        q_y = obj.rotation.y
                        q_z = obj.rotation.z
                        rpy=np.array()
                        # Sometimes Vicon sends data packets that have x, y, and z as either extremely large values or all zero (they're floats so something like 1e-30)
                        # Discard that packet and wait for the next one
                        # This is a temporary solution until a more reliable way to detect those bad packets is found
                        # while ((np.abs(pos[0]) < 1.0e-9 or np.abs(pos[0]) > 1.0e4) or (np.abs(pos[1]) < 1.0e-9 or np.abs(pos[1]) > 1.0e4) or (np.abs(pos[2]) < 1.0e-9 or np.abs(pos[2]) > 1.0e4)):
                        if np.abs(pos[0]) > 0.000000001 and np.abs(pos[0]) < 10000 and np.abs(pos[1]) > 0.000000001 and np.abs(pos[1]) < 10000 and np.abs(pos[2]) > 0.000000001 and np.abs(pos[2]) < 10000:
                            self.prev_time = timestamp
                            self.PositionBuffer._input(np.array([pos,rpy]))
                            # Quanternion for model is likely x y z w from the pybullet docs
                            return np.array([pos[0],pos[1],pos[2],q_x,q_y,q_z,q_w,Vx,Vy,Vz,Wx,Wy,Wz]) 

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
        m1 = self.HOVER_RPM + self.HOVER_RPM* 0.05 * action[0,0,0]
        m2 = self.HOVER_RPM + self.HOVER_RPM* 0.05 * action[0,0,1]
        m3 = self.HOVER_RPM + self.HOVER_RPM* 0.05 * action[0,0,2]
        m4 = self.HOVER_RPM + self.HOVER_RPM* 0.05 * action[0,0,3]
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

        # Unlock startup thrust protection
        self._cf.commander.send_setpoint(0, 0, 0, 0)

        while self.control:
            
            self.RunTime = time.time() - self.ZeroTime
            

            # Wait for control cycle
            if self.RunTime - self.timer > 1/self.CONTROLFREQUENCY:
                # if self.RunTime<5:
                #     thrust = int(self.RunTime*(self.MAX_RPM/5))
                #     self._cf.commander.send_setpoint(0, 0, 0, thrust)
                #     self.timer = self.RunTime
                # elif self.RunTime<10:
                #     thrust = int(self.MAX_RPM - (self.RunTime-5)*(self.MAX_RPM/5))
                #     self._cf.commander.send_setpoint(0, 0, 0, thrust)
                #     self.timer = self.RunTime
                # else:
                #     self.control = False
                self._cf.commander.send_setpoint(0, 0, 0, self.thrust)
                

        # This is where we would put any code to be run before the program ends
        print("\nKill button pressed, shutting down")
        self._cf.commander.send_setpoint(0, 0, 0, 0)
        self._cf.close_link()


if __name__ == '__main__':
    print('starting')
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()
    print('drivers initialized')
    controller = DroneController(uri, model_name = "SimToReal12")

