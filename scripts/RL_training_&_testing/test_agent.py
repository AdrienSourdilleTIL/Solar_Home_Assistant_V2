import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from gym_env import SolarBatteryEnv

# --- Load test dataset ---
test_path = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\processed\test.csv"
test_df = pd.read_csv(test_path).reset_index(drop=True)

# --- Feature cleanup ---
for col in ["Gb", "Gd", "Gr"]:
    if col in test_df.columns:
        test_df = test_df.drop(columns=[col])

# --- Cyclical encodings ---
if "hour" in test_df.columns:
    test_df["hour_sin"] = np.sin(2 * np.pi * test_df["hour"] / 24)
    test_df["hour_cos"] = np.cos(2 * np.pi * test_df["hour"] / 24)
    test_df = test_df.drop(columns=["hour"])

if "day_of_week" in test_df.columns:
    test_df["day_sin"] = np.sin(2 * np.pi * test_df["day_of_week"] / 7)
    test_df["day_cos"] = np.cos(2 * np.pi * test_df["day_of_week"] / 7)
    test_df = test_df.drop(columns=["day_of_week"])

# --- Lagged features ---
lag_features = ["P", "consumption_kWh", "buy_price", "sell_price"]
for col in lag_features:
    if col in test_df.columns:
        for lag in range(1, 4):
            test_df[f"{col}_lag{lag}"] = test_df[col].shift(lag)
test_df = test_df.dropna().reset_index(drop=True)

# --- Create test environment ---
env = SolarBatteryEnv(test_df)

# --- Load trained model ---
model_path = r"./solar_batt_agent_weekly_lagged.zip"
model = SAC.load(model_path, env=env)

# --- Run the model on the test set ---
obs, _ = env.reset()
done = False
rewards = []
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    done = terminated or truncated

# --- Compute cumulative reward ---
cumulative_reward = np.cumsum(rewards)

# --- Plot cumulative reward ---
plt.figure(figsize=(12, 5))
plt.plot(cumulative_reward, label="Cumulative Reward")
plt.xlabel("Timestep")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward of SAC Agent on Test Dataset")
plt.legend()
plt.grid(True)
plt.show()
