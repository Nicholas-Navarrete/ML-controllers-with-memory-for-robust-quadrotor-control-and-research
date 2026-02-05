
"""
Minimum required code to read in position data of an object from VICON Tracker
Prints the [x, y, z] position and the w,x,y,z values of the quaternion
"""

import time
from threading import Thread

import motioncapture

# The host name or ip address of the mocap system
host_name = '128.101.167.111'

# The type of the mocap system
# Valid options are: 'vicon', 'optitrack', 'optitrack_closed_source', 'qualisys', 'nokov', 'vrpn', 'motionanalysis'
mocap_system_type = 'vicon'

# The name of the rigid body that represents the Crazyflie (VICON object name)
rigid_body_name = 'drone2'



class MocapWrapper(Thread):
    def __init__(self, body_name):
        Thread.__init__(self)

        self.body_name = body_name
        self._stay_open = True

        self.start()

    def close(self):
        self._stay_open = False

    def run(self):
        mc = motioncapture.connect(mocap_system_type, {'hostname': host_name})
        while self._stay_open:
            mc.waitForNextFrame()
            for name, obj in mc.rigidBodies.items():
                if name == self.body_name:
                    print("XYZ:", obj.position)
                    print("Rot:", obj.rotation.x, obj.rotation.y, obj.rotation.z, obj.rotation.w)


if __name__ == '__main__':
    print("test2")
    # Connect to the mocap system
    mocap_wrapper = MocapWrapper(rigid_body_name)

    # Sleep for 10 seconds
    # mocap_wrapper gathers data in a seperate thread while this happens
    time.sleep(10)

    mocap_wrapper.close()
