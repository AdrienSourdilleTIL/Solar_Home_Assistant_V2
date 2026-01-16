"""
Train SAC agent with simplified environment (v2)
"""

import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_env_v2 import SolarBatteryEnvV2

# Load training data
train_path = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\processed\train.csv"
train_df = pd.read_csv(train_path).reset_index(drop=True)

# Feature cleanup
for col in ["Gb", "Gd", "Gr"]:
    if col in train_df.columns:
        train_df = train_df.drop(columns=[col])

# Cyclical encodings
if "hour" in train_df.columns:
    train_df["hour_sin"] = np.sin(2 * np.pi * train_df["hour"] / 24)
    train_df["hour_cos"] = np.cos(2 * np.pi * train_df["hour"] / 24)
    train_df = train_df.drop(columns=["hour"])

if "day_of_week" in train_df.columns:
    train_df["day_sin"] = np.sin(2 * np.pi * train_df["day_of_week"] / 7)
    train_df["day_cos"] = np.cos(2 * np.pi * train_df["day_of_week"] / 7)
    train_df = train_df.drop(columns=["day_of_week"])

# Lagged features
lag_features = ["P", "consumption_kWh", "buy_price", "sell_price"]
for col in lag_features:
    if col in train_df.columns:
        for lag in range(1, 4):
            train_df[f"{col}_lag{lag}"] = train_df[col].shift(lag)
train_df = train_df.dropna().reset_index(drop=True)

print(f"Training data shape: {train_df.shape}")

# Create environment
def make_env():
    return SolarBatteryEnvV2(
        data=train_df,
        battery_capacity=10.0,
        max_charge_rate=5.0,
        timestep_h=1.0,
        eta=0.95,
        degradation_cost=0.001
    )

vec_env = DummyVecEnv([make_env])

# SAC agent with gamma=0.999
model = SAC(
    "MlpPolicy",
    vec_env,
    verbose=1,
    batch_size=128,
    learning_rate=3e-4,
    gamma=0.999,  # Higher gamma to value future rewards
    tensorboard_log="./solar_batt_tensorboard_v2/"
)

print("\n" + "="*80)
print("TRAINING SIMPLIFIED ENVIRONMENT (V2)")
print("="*80)
print("Action space: 2 actions (charge_frac, use_batt_frac)")
print("Normalization: Buy and sell prices by same factor")
print("Reward: Natural rewards only (no shaping)")
print("Gamma: 0.999")
print("="*80 + "\n")

# Train
model.learn(total_timesteps=200000)
model.save("./solar_batt_agent_v2")
print("\nTraining complete and model saved!")
