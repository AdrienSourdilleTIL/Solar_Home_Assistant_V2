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
    
    # Use info from env to avoid 1-hour offset
    row = {
        "datetime": info["datetime"],  # current timestep from environment

        # Action fractions
        "pv_to_house_frac": float(action[0]),
        "pv_to_batt_frac": float(action[1]),
        "batt_to_house_frac": float(action[2]),
        "grid_to_batt_frac": float(action[3]),

        # Battery state
        "soc_kWh": info["soc_kWh"],

        # House energy components
        "pv_to_house_kWh": info["pv_to_house_kWh"],
        "discharge_to_house_kWh": info["discharge_to_house_kWh"],
        "grid_to_house_kWh": info["grid_to_house_kWh"],

        # Battery energy components
        "pv_to_batt_kWh": info["pv_to_batt_kWh"],
        "grid_to_batt_kWh": info["grid_to_batt_kWh"],

        # Grid energy components
        "pv_to_grid_kWh": info["pv_to_grid_kWh"],
        "discharge_to_grid_kWh": info["discharge_to_grid_kWh"],

        # Original consumption and PV production
        "consumption_kWh": test_df.loc[test_df["datetime"] == info["datetime"], "consumption_kWh"].values[0],
        "pv_production_kWh": max(test_df.loc[test_df["datetime"] == info["datetime"], "P"].values[0], 0.0),

        # Cost
        "step_cost_eur": info["cost_eur"],
        "degradation_penalty": info["degradation_penalty"]
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
