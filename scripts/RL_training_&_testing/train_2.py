from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from gym_env import SolarBatteryEnv
import pandas as pd
from pathlib import Path
import numpy as np

# --- Load full training dataset ---
path = Path(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\processed\train.csv")
train_df = pd.read_csv(path).reset_index(drop=True)

# --- Weekly episode environment ---
class RandomStartWeeklyEnv(SolarBatteryEnv):
    def __init__(self, data, battery_capacity=10.0, max_charge_rate=5.0, timestep_h=1.0, episode_length=24*7):
        super().__init__(data, battery_capacity, max_charge_rate, timestep_h)
        self.episode_length = episode_length
        self.steps_in_episode = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Random start to capture seasonal/weekend variations
        self.idx = np.random.randint(0, len(self.data) - self.episode_length)
        self.steps_in_episode = 0
        return self._get_obs(), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.steps_in_episode += 1
        # End episode after one week
        if self.steps_in_episode >= self.episode_length:
            terminated = True
        return obs, reward, terminated, truncated, info

# --- Function to create monitored environments ---
def make_env():
    env = RandomStartWeeklyEnv(
        train_df,
        battery_capacity=10.0,
        max_charge_rate=5.0,
        timestep_h=1.0,
        episode_length=24*7
    )
    return Monitor(env)  # logs stats for TensorBoard

# --- Vectorized environment for SAC ---
vec_env = make_vec_env(make_env, n_envs=4)  # 4 envs for stability

# --- Initialize SAC agent ---
policy_kwargs = dict(net_arch=[256, 256, 128])  # deeper policy for weekly patterns

model = SAC(
    "MlpPolicy",
    vec_env,
    verbose=1,
    batch_size=256,       # larger batch size for stability
    learning_rate=3e-4,
    gamma=0.99,
    tensorboard_log="./solar_batt_tensorboard/",
    policy_kwargs=policy_kwargs
)

# --- Train agent ---
model.learn(total_timesteps=200_000)  # longer training for weekly episodes

# --- Save trained model ---
model.save("./solar_batt_agent_weekly")
print("✅ Weekly training complete and model saved.")
