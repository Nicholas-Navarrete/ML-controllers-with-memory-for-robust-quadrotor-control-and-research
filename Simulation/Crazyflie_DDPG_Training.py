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

env = HoverAviary()

target_reward = 400




callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
eval_callback = EvalCallback(env,callback_on_new_best=callback_on_best,verbose=1,
eval_freq=int(1000),deterministic=True,render=False)



#### Train the model #######################################
model = DDPG('MlpPolicy', env, verbose=1,learning_rate=0.0003,device="cuda")
model.learn(total_timesteps= 1e5,
            callback=eval_callback,
            log_interval=10, progress_bar= True)

model.save("DDPG_HoverAviary")

env = HoverAviary(gui=True)
obs,info = env.reset()

input("Training is Done! Press enter to continue")

while not keyboard.is_pressed('esc'):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()