import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from gym_env import SolarBatteryEnv  # your environment

# --- Load dataset ---
data_path = Path(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\processed\train.csv")
df = pd.read_csv(data_path).reset_index(drop=True)

# --- Feature cleanup ---
# Remove redundant irradiance columns if present
for col in ["Gb", "Gd", "Gr"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# --- Create cyclical encodings ---
if "hour" in df.columns:
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df = df.drop(columns=["hour"])

if "day_of_week" in df.columns:
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df = df.drop(columns=["day_of_week"])

# --- Lagged features for past 3 timesteps ---
lag_features = ["P", "consumption_kWh", "buy_price", "sell_price"]
for col in lag_features:
    if col in df.columns:
        for lag in range(1, 4):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

df = df.dropna().reset_index(drop=True)

# --- Optional: custom environment wrapper for weekly random starts ---
class RandomStartWeeklyEnv(SolarBatteryEnv):
    def __init__(self, data, battery_capacity=10.0, max_charge_rate=5.0, timestep_h=1.0, episode_length=24*7):
        super().__init__(data, battery_capacity, max_charge_rate, timestep_h)
        self.episode_length = episode_length
        self.steps_in_episode = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = np.random.randint(0, len(self.data) - self.episode_length)
        self.steps_in_episode = 0
        return self._get_obs(), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.steps_in_episode += 1
        if self.steps_in_episode >= self.episode_length:
            terminated = True
        return obs, reward, terminated, truncated, info

# --- Helper to wrap environment with Monitor ---
def make_env():
    env = RandomStartWeeklyEnv(
        data=df,
        battery_capacity=10.0,
        max_charge_rate=5.0,
        timestep_h=1.0,
        episode_length=24*7
    )
    return Monitor(env)

# --- Vectorized environments ---
vec_env = make_vec_env(make_env, n_envs=4)

# --- SAC agent ---
model = SAC(
    "MlpPolicy",
    vec_env,
    verbose=1,
    batch_size=128,
    learning_rate=3e-4,
    gamma=0.999,  # Increased from 0.99 to better value delayed rewards
    tensorboard_log="./solar_batt_tensorboard/"
)

# --- Train the agent ---
model.learn(total_timesteps=200000)
model.save("./solar_batt_agent_weekly_lagged")
print("Training complete and model saved!")
