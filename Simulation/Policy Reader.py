import gymnasium as gym
import argparse
from sb3_contrib import RecurrentPPO
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
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

'''
This program loads a trained RL policy and prints out the parameters of its policy and value networks.
It supports both standard PPO and RecurrentPPO models.
Nicholas Navarrete
'''

# Input the path to the model below, it can be a model zip or an actor critic policy
name = "Trained Models/Thesis Models/PPO_LSTM_10.zip"

if "LSTM" in name:
    if "policy" not in name:
        model = RecurrentPPO.load(name,print_system_info=True)
        model_params = model.policy.state_dict()
    else:
        model = RecurrentPPO.load(name)
        model_params = model.policy.state_dict()
else:    
    if "policy" not in name:
        model = PPO.load(name,print_system_info=True)
        model_params = model.policy.state_dict()
    else:
        model = ActorCriticPolicy.load(name)
        model_params = model.state_dict()


# Print the matrices for the policy and value networks
print("##################################################")
for name, param in model_params.items():
    print(f"Parameter Name: {name}")
    print(f"Shape: {param.shape}")
    # Print data for illustration
    print(f"Data:\n{param.data}\n")
    print(f"The rank of the array is: {np.linalg.matrix_rank(param.data)}")
    print("*" * 20+"LAYER SEPERATOR"+"*" * 20)