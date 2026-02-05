import gymnasium as gym
import argparse
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.policies import ActorCriticPolicy
import keyboard
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from stable_baselines3.common.env_util import make_vec_env
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
from datetime import datetime
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
import numpy as np
import sys
from collections import deque
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

'''
This program tests a trained LSTM + MLP model or policy on the HoverAviary environment with a GUI.
This can be useful to visualize how well the model performs in simulation.
Nicholas Navarrete
'''

HOVER_RPM=55996
MAX_RPM=65534

history = [deque(maxlen=5) for _ in range(4)]

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
    
    percentages = [(a / 65534) * 100 for a in action[0]]
    
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

def _preprocess_action(action):
        ''' This Function Preprocess the action from the PPO model to the Crazyflie setpoint format.
            PPO outputs the action in the format: RPMm1 = self.hover_rpm *(1+0.05*thrust)
            Where the crazyflie setpoint wants rpms
        '''
        
        #print(self.RunTime ,int(action[0,0,0]), int(-action[0,0,1]), int(action[0,0,2]), int(action[0,0,3]), "\t", end="\r")
        # The action is in the format: [[[m1, m2, m3, m4]]]

        """Note: the motors layout changed so that m1 and m3 flip positions & m4 and m2 flip positions"""
        m1 = HOVER_RPM + HOVER_RPM* 0.05 * action[0,0]
        m2 = HOVER_RPM + HOVER_RPM* 0.05 * action[0,1]
        m3 = HOVER_RPM + HOVER_RPM* 0.05 * action[0,2]
        m4 = HOVER_RPM + HOVER_RPM* 0.05 * action[0,3]



        # Make sure the motors are not above the maximum RPM
        m1 = min(m1, MAX_RPM)
        m2 = min(m2, MAX_RPM)
        m3 = min(m3, MAX_RPM)
        m4 = min(m4, MAX_RPM)
        # Return the action in the format: [m1, m2, m3, m4]
        return np.array([[m1, m2, m3, m4]])


Sim_Freq=12000
control_freq = 120 #in Hz

DEFAULT_OBS = ObservationType.REAL2 # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'



initial_pos = np.array([[0,0,0]])
initial_pos.reshape(1,3)
env = make_vec_env(HoverAviary,env_kwargs=dict(obs=ObservationType.REAL3,
                                                act=ActionType.RPM, 
                                                ctrl_freq=control_freq, 
                                                pyb_freq=Sim_Freq, 
                                                initial_xyzs = initial_pos, 
                                                drone_model = DroneModel.CF2X, 
                                                gui = True, 
                                                randomize_init_rpy=True,
                                                kf_Variance=1e-7),n_envs=1)
obs = env.reset()


name = "PPO_LSTM_10"
if "policy" not in name:
    model = RecurrentPPO.load(name,print_system_info=True)
else:
    model = RecurrentActorCriticPolicy.load(name)

# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)

while not keyboard.is_pressed('esc'):
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, reward, dones, info = env.step(action)
    episode_starts = dones
    
    print(str(action[0])+'\r')
    display_quadrotor(_preprocess_action(action))
    if keyboard.is_pressed('r'):
        obs = env.reset()