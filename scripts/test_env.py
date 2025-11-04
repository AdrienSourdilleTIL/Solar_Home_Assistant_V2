import pandas as pd
from pathlib import Path
from gym_env import SolarBatteryEnv

# --- Load dataset ---
path = Path(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\processed\train.csv")
data = pd.read_csv(path)

# --- Take a consecutive sample (e.g., 500 timesteps starting at index 1000) ---
start_idx = 1000
end_idx = start_idx + 500
data_sample = data.iloc[start_idx:end_idx].reset_index(drop=True)

# --- Instantiate environment ---
env = SolarBatteryEnv(data_sample, battery_capacity=10.0, max_charge_rate=5.0, timestep_h=1.0)

# --- Reset environment ---
obs, _ = env.reset()
print(f"Initial SOC: {env.soc:.2f} kWh\n")

# --- Run a few random steps ---
for t in range(10):
    action = env.action_space.sample()  # random exploratory action
    obs, reward, done, trunc, info = env.step(action)

    print(f"Step {t}:")
    print(f"  Reward: {reward:.4f}")
    print(f"  SOC: {env.soc:.2f} kWh ({env.soc / env.battery_capacity:.2%})")
    print(f"  Grid -> House: {info['grid_to_house_kWh']:.3f} | Grid -> Batt: {info['grid_to_batt_kWh']:.3f}")
    print(f"  PV -> House: {info['pv_to_house_kWh']:.3f} | PV -> Batt: {info['pv_to_batt_kWh']:.3f} | PV -> Grid: {info['pv_to_grid_kWh']:.3f}")
    print(f"  Batt -> House: {info['discharge_to_house_kWh']:.3f} | Batt -> Grid: {info['discharge_to_grid_kWh']:.3f}")
    print(f"  Cost: {info['cost_eur']:.4f} | Deg. Penalty: {info['degradation_penalty']:.6f}\n")

    if done:
        break

env.render()
