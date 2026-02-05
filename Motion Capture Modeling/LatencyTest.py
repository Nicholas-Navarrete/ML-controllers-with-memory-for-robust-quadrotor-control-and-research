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

from collections import deque

if __name__ == '__main__':
    #_cf = Crazyflie(rw_cache='./cache')
    #_cf.open_link(uri)
    _mc = motioncapture.connect(mocap_system_type, {'hostname': host_name})
    model=PPO.load("SimToReal22")
    obs = np.array([[0,0,0,0,0,0,0,0,0,0,0,0]])
    Times=[]
    print("Starting latency test, press 'esc' to stop...")
    while not keyboard.is_pressed("esc"):
        _mc.waitForNextFrame()
        actions, _ = model.predict(obs)
        Times.append(time.time())
        print(len(Times), end='\r')

    Times = Times[-TimeSize:]
    Latencies = np.diff(Times)
    print(f"Average Latency: {Latencies.mean():.8f} seconds")
    Frequencies = np.diff(Times)          # time differences
    Frequencies = 1 / Frequencies       # instantaneous frequencies
    print("Average Control Frequency: ", np.mean(Frequencies), "Hz")