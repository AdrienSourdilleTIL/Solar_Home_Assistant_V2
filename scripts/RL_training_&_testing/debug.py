from stable_baselines3 import SAC
from gym_env import SolarBatteryEnv
import pandas as pd
import numpy as np
from pathlib import Path

# --- Load dataset ---
df_path = Path(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\processed\train.csv")
df = pd.read_csv(df_path).reset_index(drop=True)

# --- Instantiate environment ---
env = SolarBatteryEnv(df)

# --- Print environment observation space ---
print("✅ Environment observation space shape:", env.observation_space.shape)

# --- Sample an observation ---
obs, _ = env.reset()
print("Sample observation shape:", obs.shape)
print("Sample observation (first 10 values):", obs[:10])

# --- Load trained SAC model ---
model_path = Path(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\solar_batt_agent_weekly_lagged.zip")
model = SAC.load(model_path)

# --- Print model observation space ---
if hasattr(model, 'observation_space'):
    model_obs_space = model.observation_space
else:
    # Older versions may have it in policy
    model_obs_space = model.policy.observation_space
print("✅ Model expected observation space:", model_obs_space)
print("Model expected observation shape:", model_obs_space.shape)

# --- Compare shapes ---
if env.observation_space.shape == model_obs_space.shape:
    print("✅ Environment and model observation shapes match.")
else:
    print("❌ Mismatch detected! Environment vs model:", env.observation_space.shape, "vs", model_obs_space.shape)
