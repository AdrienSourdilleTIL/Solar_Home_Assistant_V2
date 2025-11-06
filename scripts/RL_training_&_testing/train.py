from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from gym_env import SolarBatteryEnv
import pandas as pd
from pathlib import Path

# --- Load full training dataset ---
path = Path(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\processed\train.csv")
train_df = pd.read_csv(path).reset_index(drop=True)

# --- Wrap environment ---
# The env will loop automatically when reset()
class LoopingSolarEnv(SolarBatteryEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated:
            obs, _ = self.reset()  # reset if episode ends
        return obs, reward, terminated, truncated, info

env = LoopingSolarEnv(train_df, battery_capacity=10.0, max_charge_rate=5.0, timestep_h=1.0)

# --- Vectorized env ---
vec_env = make_vec_env(lambda: env, n_envs=1)

# --- Initialize SAC agent ---
model = SAC(
    "MlpPolicy",
    vec_env,
    verbose=1,
    batch_size=128,          # smaller batch for stability
    learning_rate=3e-4,
    gamma=0.99,
    tensorboard_log="./solar_batt_tensorboard/"
)

# --- Train on dataset multiple times ---
model.learn(total_timesteps=500_000)  # allows agent to see the dataset many times

# --- Save trained model ---
model.save("./solar_batt_agent_full")

print("✅ Training complete and model saved.")
