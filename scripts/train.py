from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from gym_env import SolarBatteryEnv
import pandas as pd
from pathlib import Path

# --- Load full training dataset ---
path = Path(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\processed\train.csv")
train_df = pd.read_csv(path)

# --- Use the entire dataset ---
train_sample = train_df.reset_index(drop=True)

# --- Wrap environment ---
env = SolarBatteryEnv(train_sample, battery_capacity=10.0, max_charge_rate=5.0, timestep_h=1.0)

# --- Vectorized env ---
vec_env = make_vec_env(lambda: env, n_envs=1)

# --- Initialize SAC agent ---
model = SAC(
    "MlpPolicy",
    vec_env,
    verbose=1,
    batch_size=256,
    learning_rate=3e-4,
    gamma=0.99,
    tensorboard_log="./solar_batt_tensorboard/"
)

# --- Train on entire dataset ---
model.learn(total_timesteps=len(train_sample))  # 1 step per row in dataset

# --- Save trained model ---
model.save("./solar_batt_agent_full")
