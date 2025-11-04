import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from gym_env import SolarBatteryEnv
from stable_baselines3 import SAC
import numpy as np

# --- Load test dataset ---
test_path = Path(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\processed\test.csv")
test_df = pd.read_csv(test_path).reset_index(drop=True)

# --- Instantiate environments ---
env_agent = SolarBatteryEnv(test_df, battery_capacity=10.0, max_charge_rate=5.0, timestep_h=1.0)
env_random = SolarBatteryEnv(test_df, battery_capacity=10.0, max_charge_rate=5.0, timestep_h=1.0)
env_rule = SolarBatteryEnv(test_df, battery_capacity=10.0, max_charge_rate=5.0, timestep_h=1.0)

# --- Load trained model ---
model = SAC.load(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\solar_batt_agent_full.zip")

# --- Run Trained Agent ---
obs, _ = env_agent.reset()
rewards_agent = []
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_agent.step(action)
    rewards_agent.append(reward)
    if terminated:
        break

# --- Run Random Policy ---
obs, _ = env_random.reset()
rewards_random = []
while True:
    action = env_random.action_space.sample()
    obs, reward, terminated, truncated, info = env_random.step(action)
    rewards_random.append(reward)
    if terminated:
        break

# --- Run Rule-Based Policy ---
obs, _ = env_rule.reset()
rewards_rule = []
while True:
    row = test_df.iloc[env_rule.idx]
    pv = max(row["P"], 0.0)
    consumption = row["consumption_kWh"]

    # --- PV allocation ---
    pv_to_house = min(pv, consumption)
    pv_remaining = max(pv - pv_to_house, 0.0)
    pv_to_batt = min(pv_remaining, env_rule.battery_capacity - env_rule.soc)
    pv_to_grid = max(0.0, pv_remaining - pv_to_batt)

    # --- Battery discharge ---
    deficit = consumption - pv_to_house
    discharge_to_house = min(env_rule.soc, deficit) if deficit > 0 else 0.0
    batt_to_house_frac = discharge_to_house / env_rule.soc if env_rule.soc > 0 else 0.0

    # --- Grid charging fraction ---
    grid_to_batt_frac = 1.0 if env_rule.soc + pv_to_batt < env_rule.battery_capacity else 0.0

    # --- Compute PV fractions ---
    pv_to_house_frac = pv_to_house / pv if pv > 0 else 0.0
    pv_to_batt_frac = pv_to_batt / pv if pv > 0 else 0.0

    action = np.array([pv_to_house_frac, pv_to_batt_frac, batt_to_house_frac, grid_to_batt_frac], dtype=np.float32)
    obs, reward, terminated, truncated, info = env_rule.step(action)
    rewards_rule.append(reward)
    if terminated:
        break

# --- Compute cumulative rewards ---
cumulative_agent = np.cumsum(rewards_agent)
cumulative_random = np.cumsum(rewards_random)
cumulative_rule = np.cumsum(rewards_rule)

# --- Plot cumulative rewards ---
plt.figure(figsize=(14,7))
plt.plot(cumulative_agent, label="Trained Agent", color='blue')
plt.plot(cumulative_random, label="Random Policy", color='red')
plt.plot(cumulative_rule, label="Rule-Based Policy", color='green')
plt.xlabel("Timestep")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Comparison: Trained vs Random vs Rule-Based")
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\outputs\cumulative_rewards_comparison.png", dpi=300)
plt.close()