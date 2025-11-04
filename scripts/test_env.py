from gym_env import SolarBatteryEnv
import pandas as pd

# --- load training data ---
data = pd.read_csv(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\processed\train.csv")

# --- create environment ---
env = SolarBatteryEnv(data)

# --- reset environment ---
obs, info = env.reset()

# --- take a few sample steps ---
for _ in range(5):
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)
    print(info)
    if terminated:
        break
