from gym_env import SolarBatteryEnv
import pandas as pd
from stable_baselines3 import SAC
import numpy as np

# --- Load dataset ---
df = pd.read_csv(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\processed\train.csv")

# --- Initialize environment ---
env = SolarBatteryEnv(df)

print("Environment observation space shape:", env.observation_space.shape)

# --- Get a sample observation from the env ---
obs, _ = env.reset()
print("Sample observation shape:", obs.shape)
print("Sample observation values (first 10):", obs[:10])

# --- Load trained model ---
model_path = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\solar_batt_agent_weekly_lagged.zip"
model = SAC.load(model_path)

# --- Inspect model input space ---
print("Model expected observation space:", model.observation_space)
print("Model expected observation shape:", model.observation_space.shape)

# --- Test if the env obs matches model obs shape ---
try:
    action, _ = model.predict(obs, deterministic=True)
    print("✅ Model can process environment observation")
except ValueError as e:
    print("❌ Model cannot process environment observation:", e)
