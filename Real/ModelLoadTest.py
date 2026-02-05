from stable_baselines3 import PPO

model = PPO.load(".\SimToReal45.zip")
print("Model Loaded "+model.__class__.__name__)