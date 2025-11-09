import pandas as pd
from pathlib import Path
from gym_env import SolarBatteryEnv
from stable_baselines3 import SAC
import numpy as np

# --- Load test dataset ---
test_path = Path(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\processed\test.csv")
test_df = pd.read_csv(test_path).reset_index(drop=True)

# --- Instantiate environment ---
env = SolarBatteryEnv(test_df, battery_capacity=10.0, max_charge_rate=5.0, timestep_h=1.0)

# --- Load trained model ---
model = SAC.load(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\solar_batt_agent_full.zip")

# --- Run Trained Agent ---
obs, _ = env.reset()
records = []

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    row = {
        "datetime": test_df.iloc[env.idx-1]["datetime"],  # current timestep
        "pv_to_house_frac": float(action[0]),
        "pv_to_batt_frac": float(action[1]),
        "batt_to_house_frac": float(action[2]),
        "grid_to_batt_frac": float(action[3]),
        "soc_kWh": info["soc_kWh"],
        "grid_to_house_kWh": info["grid_to_house_kWh"],
        "discharge_to_house_kWh": info["discharge_to_house_kWh"],
        "pv_to_batt_kWh": info["pv_to_batt_kWh"],
        "grid_to_batt_kWh": info["grid_to_batt_kWh"],
        "pv_to_house_kWh": info["pv_to_house_kWh"],
        "pv_to_grid_kWh": info["pv_to_grid_kWh"],
        "discharge_to_grid_kWh": info["discharge_to_grid_kWh"],   
        "house_energy_kWh": info["pv_to_house_kWh"] + info["discharge_to_house_kWh"] + info["grid_to_house_kWh"],
        "battery_energy_kWh": info["pv_to_batt_kWh"] + info["grid_to_batt_kWh"],
        "grid_energy_kWh": info["pv_to_grid_kWh"] + info["discharge_to_grid_kWh"],
        "consumption_kWh": test_df.iloc[env.idx-1]["consumption_kWh"],
        "pv_production_kWh": max(test_df.iloc[env.idx-1]["P"], 0.0),
        "step_cost_eur": info["cost_eur"]
    }
    records.append(row)
    
    if terminated:
        break

# --- Save to CSV ---
output_df = pd.DataFrame(records)
output_path = Path(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\outputs\agent_step_data.csv")
output_df.to_csv(output_path, index=False)

print(f"Saved {len(output_df)} steps to {output_path}")
print(output_df.head())
