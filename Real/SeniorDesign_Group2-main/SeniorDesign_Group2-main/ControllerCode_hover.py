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

uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

TimeSize=5

# The host name or ip address of the mocap system
host_name = '128.101.167.111'

# The type of the mocap system
# Valid options are: 'vicon', 'optitrack', 'optitrack_closed_source', 'qualisys', 'nokov', 'vrpn', 'motionanalysis'
mocap_system_type = 'vicon'

# The name of the rigid body that represents the Crazyflie (VICON object name)
drone_object_name = 'drone2'



logging.basicConfig(level=logging.ERROR)



class DroneController:
    """Example that connects to a Crazyflie and ramps the motors up/down and
    the disconnects"""

    def __init__(self, link_uri):
        """ Initialize and run the example with the specified link_uri """
        self.ZeroTime = time.time()
        self._mc = motioncapture.connect(mocap_system_type, {'hostname': host_name})
        self.prev_time = 0
        self.control = True

        self._cf = Crazyflie(rw_cache='./cache')

        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        self._cf.open_link(link_uri)

        print('Connecting to %s' % link_uri)

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
        while True:
            self._mc.waitForNextFrame()
            for name, obj in self._mc.rigidBodies.items():
                if name == rigid_body_name:
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

    def _control_loop(self):
        """ The main loop of the controller code """
        # These goals are in standard unts for mocap (should be in meters)
        # Initial goal states
        RunTime = time.time() - self.ZeroTime

        # Initial goal states
        x_goal = 0
        y_goal = 0
        z_goal = 0

        # Initialize position and time buffers
        x_pos = np.zeros(TimeSize)
        y_pos = np.zeros(TimeSize)
        z_pos = np.zeros(TimeSize)
        # Time buffers should be different so there are no divide by zero errors at startup
        times = np.array([0.01, 0.02, 0.04, 0.05, 0.07])

        # Unlock startup thrust protection
        self._cf.commander.send_setpoint(0, 0, 0, 0)

        while self.control:
            RunTime = time.time() - self.ZeroTime

            # Set goal positions as a function of time
            if RunTime < 10:
                x_goal = 0
                y_goal = 0
                z_goal = 1.5

            else:
                x_goal = 0
                y_goal = 0
                z_goal = 1.5 - 0.2*(RunTime - 10)
                if(z_goal < 0):
                    z_goal = 0
                    self.control = False

            
            X_new, Y_new, Z_new, Yaw, new_time = self._get_position_data(drone_object_name)
            
            # Not sure if we actually need this check
            if ((X_new != x_pos[-1]) and np.abs(X_new)<=10) or ((Y_new != y_pos[-1]) and np.abs(Y_new)<=10) or ((Z_new != z_pos[-1]) and np.abs(X_new)<=10):
                for i in range(TimeSize-1):
                    x_pos[i]=x_pos[i+1]
                x_pos[-1]=X_new
                
                for i in range(TimeSize-1):
                    y_pos[i]=y_pos[i+1]
                y_pos[-1]=Y_new

                for i in range(TimeSize-1):
                    z_pos[i]=z_pos[i+1]
                z_pos[-1]=Z_new

                for i in range(TimeSize-1):
                    times[i]=times[i+1]
                times[-1]=new_time

            pitch, roll, thrust = self.PID_control(x_pos, y_pos, z_pos, x_goal, y_goal, z_goal, Yaw, times)
            
            self._cf.commander.send_setpoint(roll, pitch, 0, thrust)

        # This is where we would put any code to be run before the program ends
        print("\nKill button pressed, shutting down")
        self._cf.commander.send_setpoint(0, 0, 0, 0)
        self._cf.close_link()


if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    controller = DroneController(uri)

