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
env_rule = SolarBatteryEnv(test_df, battery_capacity=10.0, max_charge_rate=5.0, timestep_h=1.0)
env_forecast_rule = SolarBatteryEnv(test_df, battery_capacity=10.0, max_charge_rate=5.0, timestep_h=1.0)

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

# --- Run Original Rule-Based Policy ---
obs, _ = env_rule.reset()
rewards_rule = []
while True:
    row = test_df.iloc[env_rule.idx]
    pv = max(row["P"], 0.0)
    consumption = row["consumption_kWh"]
    soc = env_rule.soc
    batt_cap = env_rule.battery_capacity
    max_rate = env_rule.max_charge_rate

    pv_to_house_frac = 0.0
    pv_to_batt_frac = 0.0
    batt_to_house_frac = 0.0
    grid_to_batt_frac = 0.0

    if pv >= consumption:
        pv_to_house_frac = consumption / pv
        excess_pv = pv - consumption
        batt_room = batt_cap - soc
        pv_to_batt = min(excess_pv, max_rate, batt_room)
        pv_to_batt_frac = pv_to_batt / pv
    else:
        pv_to_house_frac = pv / consumption if consumption > 0 else 0.0
        deficit = consumption - pv
        discharge = min(deficit, soc, max_rate)
        batt_to_house_frac = discharge / soc if soc > 0 else 0.0

    action = np.array([pv_to_house_frac, pv_to_batt_frac, batt_to_house_frac, grid_to_batt_frac], dtype=np.float32)
    obs, reward, terminated, truncated, info = env_rule.step(action)
    rewards_rule.append(reward)
    if terminated:
        break


# --- Run Forecast-Aware Rule-Based Policy ---
obs, _ = env_forecast_rule.reset()
rewards_forecast_rule = []

while True:
    row = test_df.iloc[env_forecast_rule.idx]
    pv = max(row["P"], 0.0)
    consumption = row["consumption_kWh"]
    soc = env_forecast_rule.soc
    batt_cap = env_forecast_rule.battery_capacity
    max_rate = env_forecast_rule.max_charge_rate
    hour = row["hour"]
    buy_price = row["buy_price"]
    sell_price = row["sell_price"]

    # --- Aggregate short-term forecast signals (next 6 hours for smoother behavior) ---
    future_pv = np.mean([row[f"pv_forecast_{i}"] for i in range(1, 7)])
    future_load = np.mean([row[f"load_forecast_{i}"] for i in range(1, 7)])

    pv_to_house_frac = 0.0
    pv_to_batt_frac = 0.0
    batt_to_house_frac = 0.0
    grid_to_batt_frac = 0.0

    batt_room = batt_cap - soc
    future_deficit = future_load - future_pv

    # --- Compute dynamic reserve based on PV outlook and time ---
    if future_pv > future_load:
        reserve = 0.05 * batt_cap  # sunny hours coming
    elif 17 <= hour <= 22:
        reserve = 0.25 * batt_cap  # evening: hold more
    else:
        reserve = 0.15 * batt_cap  # neutral baseline

    # --- Decision logic ---
    if pv >= consumption:
        # Self-consume first
        pv_to_house_frac = consumption / pv
        surplus = pv - consumption

        # If future PV is high, save room — charge less now
        charge_factor = 0.5 if future_pv > future_load * 1.2 else 1.0

        # Charge battery (limited by available capacity, max_rate, and outlook)
        pv_to_batt = min(surplus * charge_factor, batt_room, max_rate)
        pv_to_batt_frac = pv_to_batt / pv

        # Sell remaining PV if sell_price high
        if sell_price > buy_price * 0.8 and batt_room < 1.0:
            pv_to_batt_frac *= 0.8  # slightly prefer selling when profitable

    else:
        # PV < consumption
        pv_to_house_frac = pv / consumption if consumption > 0 else 0.0
        deficit = consumption - pv

        # If future PV is low and future load high → discharge aggressively
        if future_deficit > 0:
            discharge_factor = 1.0
        else:
            discharge_factor = 0.6

        discharge = min(deficit, max(0, soc - reserve), max_rate) * discharge_factor
        batt_to_house_frac = discharge / soc if soc > 0 else 0.0
        deficit -= discharge

        # Optional: grid charging when cheap and low PV ahead
        if buy_price < sell_price * 0.6 and future_pv < future_load * 0.8 and batt_room > 0.5 * batt_cap:
            grid_to_batt_frac = 0.5  # opportunistic charging

    action = np.array([
        pv_to_house_frac,
        pv_to_batt_frac,
        batt_to_house_frac,
        grid_to_batt_frac
    ], dtype=np.float32)

    obs, reward, terminated, truncated, info = env_forecast_rule.step(action)
    rewards_forecast_rule.append(reward)
    if terminated:
        break


# --- Compute cumulative rewards ---
cumulative_agent = np.cumsum(rewards_agent)
cumulative_rule = np.cumsum(rewards_rule)
cumulative_forecast_rule = np.cumsum(rewards_forecast_rule)

# --- Plot cumulative rewards ---
plt.figure(figsize=(14, 7))
plt.plot(cumulative_agent, label="Trained Agent", color="blue")
plt.plot(cumulative_rule, label="Simple Rule-Based", color="green")
plt.plot(cumulative_forecast_rule, label="Forecast-Aware Rule-Based", color="orange")
plt.xlabel("Timestep")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Comparison: Agent vs Rule-Based Policies (Forecast-Aware)")
plt.legend()
plt.grid(True)
plt.savefig(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\outputs\cumulative_rewards_comparison.png", dpi=300)
plt.close()
