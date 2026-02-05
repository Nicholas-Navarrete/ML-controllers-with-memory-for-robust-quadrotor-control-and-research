import gymnasium as gym
import argparse
from stable_baselines3 import DQN
from stable_baselines3 import PPO
import keyboard
import numpy as np
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from stable_baselines3.common.env_util import make_vec_env
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
from datetime import datetime
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from sb3_contrib import RecurrentPPO
import matplotlib.pyplot as plt
import pandas as pd
import os

modelName = "SimToReal_63.zip"
NumberOfSimulations = 50000
Controller_Freq = 120
LSTM_flag = False


kf_variance_set = np.logspace(-13, -5, num=100)
success_rates = []

for kf_variance in kf_variance_set:
    print(f"Testing KF Variance: {kf_variance}")
    env = HoverAviary(gui=False, 
                    ctrl_freq=Controller_Freq, 
                    pyb_freq=Controller_Freq*100,
                    obs=ObservationType.REAL3, 
                    act=ActionType.RPM,
                    drone_model = DroneModel.CF2X, 
                    initial_xyzs=np.array([[0,0,2]]), 
                    Inefficient_Motors=True,
                    kf_Variance=kf_variance)

    obs,info = env.reset()
    lstm_states = None

    if LSTM_flag:
        model = RecurrentPPO.load(modelName, env=env)
    else:
        model = PPO.load(modelName, env=env)

    Failures = 0
    for _ in range(NumberOfSimulations):
        if LSTM_flag:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=np.array([False]), deterministic=True)
        else:
            action, lstm_states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            if info['out_of_bounds']:
                Failures += 1
            obs, info = env.reset()

    success_rate = (NumberOfSimulations - Failures) / NumberOfSimulations
    success_rates.append(success_rate)


######################################
######## Plotting Results ############
######################################


# Create results directory if it doesn't exist
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Save results to CSV
results_df = pd.DataFrame({
    "KF_Variance": kf_variance_set,
    "Success_Rate": success_rates
})

csv_filename = os.path.join(results_dir, f"success_rate_vs_kfvariance_{modelName.replace('.zip', '')}.csv")
results_df.to_csv(csv_filename, index=False)
print(f"Results saved to {csv_filename}")

# Plot success rate vs KF variance
plt.figure(figsize=(8, 5))
plt.plot(kf_variance_set, success_rates, marker='o', linestyle='-', linewidth=2, markersize=8)

# Formatting
plt.xscale('log')
plt.xlabel("KF Variance", fontsize=12)
plt.ylabel("Success Rate", fontsize=12)
plt.title(f"Success Rate vs KF Variance\nModel: {modelName} (n={NumberOfSimulations})", fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()

# Save and show plot
plot_filename = os.path.join(results_dir, f"success_rate_vs_kfvariance_{modelName.replace('.zip', '')}.png")
plt.savefig(plot_filename, dpi=300)
plt.show()

print(f"Plot saved to {plot_filename}")
