from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from gym_env import SolarBatteryEnv
import pandas as pd
import numpy as np
from pathlib import Path

# --- Load dataset ---
path = Path(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\processed\train.csv")
train_df = pd.read_csv(path).reset_index(drop=True)

# --- Feature cleanup ---
# Remove redundant irradiance columns
train_df = train_df.drop(columns=[c for c in ["Gb", "Gd", "Gr"] if c in train_df.columns])

# Create cyclical encodings for hour and day_of_week if available
if "hour" in train_df.columns:
    train_df["hour_sin"] = np.sin(2 * np.pi * train_df["hour"] / 24)
    train_df["hour_cos"] = np.cos(2 * np.pi * train_df["hour"] / 24)
    train_df = train_df.drop(columns=["hour"])

if "day_of_week" in train_df.columns:
    train_df["day_sin"] = np.sin(2 * np.pi * train_df["day_of_week"] / 7)
    train_df["day_cos"] = np.cos(2 * np.pi * train_df["day_of_week"] / 7)
    train_df = train_df.drop(columns=["day_of_week"])


# --- Lagged features (past 3 hours of key signals) ---
lag_features = ["P_pv", "P_load", "price", "soc"]
for col in lag_features:
    if col in train_df.columns:
        for lag in range(1, 4):  # 3-hour memory
            train_df[f"{col}_lag{lag}"] = train_df[col].shift(lag)
train_df = train_df.dropna().reset_index(drop=True)


# --- Environment with weekly episodes ---
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


# --- Helper to wrap with Monitor ---
def make_env():
    env = RandomStartWeeklyEnv(
        train_df,
        battery_capacity=10.0,
        max_charge_rate=5.0,
        timestep_h=1.0,
        episode_length=24*7
    )
    return Monitor(env)


# --- Vectorized environment ---
vec_env = make_vec_env(make_env, n_envs=4)

# --- SAC agent ---
model = SAC(
    "MlpPolicy",
    vec_env,
    verbose=1,
    batch_size=128,
    learning_rate=3e-4,
    gamma=0.99,
    tensorboard_log="./solar_batt_tensorboard/"
)

# --- Train ---
model.learn(total_timesteps=200_000)
model.save("./solar_batt_agent_weekly_lagged")
print("✅ Training complete and model saved with lagged features & cyclical time encoding.")
