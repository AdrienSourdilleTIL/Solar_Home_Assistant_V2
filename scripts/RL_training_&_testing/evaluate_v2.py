"""
Evaluate the V2 agent on test set
"""

import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import SAC
from gym_env_v2 import SolarBatteryEnvV2
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DATA_PATH = PROJECT_ROOT / "data" / "main" / "processed" / "test.csv"
MODEL_PATH = Path("./solar_batt_agent_v2.zip")
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "agent_step_data_v2.csv"

print("="*80)
print("EVALUATING V2 AGENT (Simplified Action Space)")
print("="*80)

# Load test data
print("\n[1/4] Loading test data...")
test_df = pd.read_csv(TEST_DATA_PATH).reset_index(drop=True)

# Feature cleanup
for col in ["Gb", "Gd", "Gr"]:
    if col in test_df.columns:
        test_df = test_df.drop(columns=[col])

# Cyclical encodings
if "hour" in test_df.columns:
    test_df["hour_sin"] = np.sin(2 * np.pi * test_df["hour"] / 24)
    test_df["hour_cos"] = np.cos(2 * np.pi * test_df["hour"] / 24)
    test_df = test_df.drop(columns=["hour"])

if "day_of_week" in test_df.columns:
    test_df["day_sin"] = np.sin(2 * np.pi * test_df["day_of_week"] / 7)
    test_df["day_cos"] = np.cos(2 * np.pi * test_df["day_of_week"] / 7)
    test_df = test_df.drop(columns=["day_of_week"])

# Lagged features
lag_features = ["P", "consumption_kWh", "buy_price", "sell_price"]
for col in lag_features:
    if col in test_df.columns:
        for lag in range(1, 4):
            test_df[f"{col}_lag{lag}"] = test_df[col].shift(lag)
test_df = test_df.dropna().reset_index(drop=True)

print(f"  Loaded {len(test_df):,} timesteps")

# Create environment
print("\n[2/4] Creating environment...")
env = SolarBatteryEnvV2(
    data=test_df,
    battery_capacity=10.0,
    max_charge_rate=5.0,
    timestep_h=1.0,
    eta=0.95,
    degradation_cost=0.001
)

# Load trained model
print("\n[3/4] Loading trained model...")
model = SAC.load(MODEL_PATH)

# Run evaluation
print("\n[4/4] Running evaluation on test set...")
obs, _ = env.reset()
done = False

step_records = []
total_reward = 0
step_count = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # Record step data
    step_records.append({
        'datetime': info['datetime'],
        'step': step_count,
        # Actions
        'charge_batt_frac': info['charge_batt_frac'],
        'use_batt_frac': info['use_batt_frac'],
        # State
        'soc_kWh': info['soc_kWh'],
        'soc_ratio': info['soc_ratio'],
        # Energy flows
        'pv_production_kWh': info['pv_production_kWh'],
        'consumption_kWh': info['consumption_kWh'],
        'pv_to_house_kWh': info['pv_to_house_kWh'],
        'pv_to_batt_kWh': info['pv_to_batt_kWh'],
        'pv_to_grid_kWh': info['pv_to_grid_kWh'],
        'batt_to_house_kWh': info['batt_to_house_kWh'],
        'grid_to_house_kWh': info['grid_to_house_kWh'],
        # Costs
        'step_cost_eur': info['cost_eur'],
        'degradation_penalty': info['degradation_penalty'],
        'reward': reward
    })

    total_reward += reward
    step_count += 1
    done = terminated or truncated

    if step_count % 1000 == 0:
        print(f"  Progress: {step_count:,}/{len(test_df):,} steps")

# Save results
print(f"\n[5/5] Saving results...")
results_df = pd.DataFrame(step_records)
results_df.to_csv(OUTPUT_PATH, index=False)

print(f"\n  Saved to: {OUTPUT_PATH}")
print(f"\n  Total steps: {step_count:,}")
print(f"  Total reward: {total_reward:.2f}")
print(f"  Total cost: {results_df['step_cost_eur'].sum():.2f} EUR")
print(f"  Total degradation: {results_df['degradation_penalty'].sum():.2f} EUR")
print(f"  Combined cost: {results_df['step_cost_eur'].sum() + results_df['degradation_penalty'].sum():.2f} EUR")

# Quick metrics
total_sold = results_df['pv_to_grid_kWh'].sum()
total_pv = results_df['pv_production_kWh'].sum()
total_bought = results_df['grid_to_house_kWh'].sum()
avg_soc = results_df['soc_kWh'].mean()

print(f"\n  Key Metrics:")
print(f"    Total PV production: {total_pv:.1f} kWh")
print(f"    Total sold to grid: {total_sold:.1f} kWh ({100*total_sold/total_pv:.1f}% of PV)")
print(f"    Total bought from grid: {total_bought:.1f} kWh")
print(f"    Average battery SOC: {avg_soc:.2f} kWh ({100*avg_soc/10:.0f}% capacity)")

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
