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

# --- Load newly trained model with env alignment ---
model = SAC.load(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\solar_batt_agent_weekly_lagged.zip", env=env_agent)

# --- Run Trained Agent ---
obs, _ = env_agent.reset()
rewards_agent = []
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_agent.step(action)
    rewards_agent.append(reward)
    done = terminated

# --- Run Original Rule-Based Policy ---
obs, _ = env_rule.reset()
rewards_rule = []
done = False
while not done:
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
    done = terminated

# --- Run Forecast-Aware Rule-Based Policy ---
obs, _ = env_forecast_rule.reset()
rewards_forecast_rule = []
done = False
while not done:
    row = test_df.iloc[env_forecast_rule.idx]
    pv = max(row["P"], 0.0)
    consumption = row["consumption_kWh"]
    soc = env_forecast_rule.soc
    batt_cap = env_forecast_rule.battery_capacity
    max_rate = env_forecast_rule.max_charge_rate
    hour = row["hour"]
    buy_price = row["buy_price"]
    sell_price = row["sell_price"]

    future_pv = np.mean([row[f"pv_forecast_{i}"] for i in range(1, 7)])
    future_load = np.mean([row[f"load_forecast_{i}"] for i in range(1, 7)])

    pv_to_house_frac = 0.0
    pv_to_batt_frac = 0.0
    batt_to_house_frac = 0.0
    grid_to_batt_frac = 0.0

    batt_room = batt_cap - soc
    future_deficit = future_load - future_pv

    # Dynamic reserve
    if future_pv > future_load:
        reserve = 0.05 * batt_cap
    elif 17 <= hour <= 22:
        reserve = 0.25 * batt_cap
    else:
        reserve = 0.15 * batt_cap

    if pv >= consumption:
        pv_to_house_frac = consumption / pv
        surplus = pv - consumption
        charge_factor = 0.5 if future_pv > future_load * 1.2 else 1.0
        pv_to_batt = min(surplus * charge_factor, batt_room, max_rate)
        pv_to_batt_frac = pv_to_batt / pv
        if sell_price > buy_price * 0.8 and batt_room < 1.0:
            pv_to_batt_frac *= 0.8
    else:
        pv_to_house_frac = pv / consumption if consumption > 0 else 0.0
        deficit = consumption - pv
        discharge_factor = 1.0 if future_deficit > 0 else 0.6
        discharge = min(deficit, max(0, soc - reserve), max_rate) * discharge_factor
        batt_to_house_frac = discharge / soc if soc > 0 else 0.0
        deficit -= discharge
        if buy_price < sell_price * 0.6 and future_pv < future_load * 0.8 and batt_room > 0.5 * batt_cap:
            grid_to_batt_frac = 0.5

    action = np.array([pv_to_house_frac, pv_to_batt_frac, batt_to_house_frac, grid_to_batt_frac], dtype=np.float32)
    obs, reward, terminated, truncated, info = env_forecast_rule.step(action)
    rewards_forecast_rule.append(reward)
    done = terminated

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
