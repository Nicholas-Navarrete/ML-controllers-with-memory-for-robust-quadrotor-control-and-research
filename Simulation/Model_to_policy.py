"""
export_policy.py
----------------
This script loads a trained PPO or PPO_LSTM model (with its environment)
and saves only the policy network for portable inference.

Usage:
    python export_policy.py --model_path PPO_SimToReal.zip --output PPO_SimToReal_policy
    python export_policy.py --model_path PPO_LSTM_SimToReal.zip --output PPO_LSTM_policy --lstm
Nicholas Navarrete
"""

import argparse
import numpy as np
from stable_baselines3 import PPO
try:
    from sb3_contrib import RecurrentPPO  # For PPO_LSTM support
except ImportError:
    RecurrentPPO = None
    print("WARNING: sb3_contrib not installed — PPO_LSTM models will not be supported.")

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


def main(args):
    # === Step 1: Recreate the same environment ===
    Sim_Freq = 12100
    control_freq = 121
    training_initial_pos = np.array([[0, 0, 2]])

    env = HoverAviary(
        gui=False,
        obs=ObservationType.REAL,
        act=ActionType.RPM,
        ctrl_freq=control_freq,
        pyb_freq=Sim_Freq,
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        Inefficient_Motors=True,
        initial_xyzs=training_initial_pos
    )

    # === Step 2: Load PPO or PPO_LSTM model ===
    print(f"Loading model from: {args.model_path}")
    if args.lstm:
        if RecurrentPPO is None:
            raise ImportError("sb3_contrib is required for PPO_LSTM models. Please install it via: pip install sb3-contrib")
        model = RecurrentPPO.load(args.model_path, env=env)
        print("Loaded PPO_LSTM model.")
    else:
        model = PPO.load(args.model_path, env=env)
        print("Loaded PPO model.")

    # === Step 3: Save only the policy ===
    output_path = args.output
    model.policy.save(output_path)
    print(f"Policy successfully saved as: {output_path}.zip")

    # === Step 4: Optional — Test a single prediction ===
    obs, _ = env.reset()
    if args.lstm:
        # For recurrent policy, initialize hidden states and set episode start flag
        lstm_states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        action, lstm_states = model.policy.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    else:
        action, _ = model.policy.predict(obs, deterministic=True)

    print(f"Sample action from loaded policy: {action}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PPO or PPO_LSTM policy for portable inference.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved PPO or PPO_LSTM model (e.g., PPO_SimToReal.zip)")
    parser.add_argument("--output", type=str, default="PPO_policy", help="Output filename for saved policy")
    parser.add_argument("--lstm", action="store_true", help="Use this flag if the model is PPO_LSTM (RecurrentPPO)")
    args = parser.parse_args()
    main(args)
