import gymnasium as gym
import argparse
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
import keyboard

from stable_baselines3.common.env_util import make_vec_env
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

obs_buffer = []
env = HoverAviary()

# COLLECT DATA
number_of_samples = 1000
for i in range(number_of_samples):
    obs, info = env.reset()

    obs_buffer.append(env._getDroneStateVector(nth_drone=0))

print(obs_buffer)
# PLOT AS PDF (probability distribution function)
# plt.plot(obs_buffer)
# plt.title('Continuous Probability Distribution (KDE)')
# plt.xlabel('Value (z)')
# plt.ylabel('Density')
# plt.grid(True)
# plt.show()