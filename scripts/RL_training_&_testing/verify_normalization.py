"""
Verify that the normalization fix is actually working correctly
"""

import pandas as pd
import numpy as np
from pathlib import Path
from gym_env import SolarBatteryEnv

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DATA_PATH = PROJECT_ROOT / "data" / "main" / "processed" / "test.csv"

print("="*80)
print("NORMALIZATION VERIFICATION")
print("="*80)

# Load test data with preprocessing (same as training)
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

# Create environment
env = SolarBatteryEnv(
    data=test_df,
    battery_capacity=10.0,
    max_charge_rate=5.0,
    timestep_h=1.0,
    eta=0.95,
    degradation_cost=0.001
)

print("\n[1] Normalization Factors:")
print("-"*80)

if 'buy_price' in env.norm_factors:
    print(f"  buy_price norm factor: {env.norm_factors['buy_price']:.6f}")
if 'sell_price' in env.norm_factors:
    print(f"  sell_price norm factor: {env.norm_factors['sell_price']:.6f}")

print("\n[2] Raw Price Data (first 10 timesteps):")
print("-"*80)
print(f"  buy_price:  {test_df['buy_price'].iloc[:10].values}")
print(f"  sell_price: {test_df['sell_price'].iloc[:10].values}")

print("\n[3] Normalized Prices (what agent sees):")
print("-"*80)

# Get observation
obs, _ = env.reset()

# Find indices of buy_price and sell_price in observation
if 'buy_price' in env.state_cols:
    buy_idx = env.state_cols.index('buy_price')
    print(f"  buy_price index in obs: {buy_idx}")
    print(f"  Normalized buy_price[0]: {obs[buy_idx]:.6f}")

if 'sell_price' in env.state_cols:
    sell_idx = env.state_cols.index('sell_price')
    print(f"  sell_price index in obs: {sell_idx}")
    print(f"  Normalized sell_price[0]: {obs[sell_idx]:.6f}")

# Check ratio
if 'buy_price' in env.state_cols and 'sell_price' in env.state_cols:
    ratio = obs[sell_idx] / obs[buy_idx]
    print(f"\n  Ratio (normalized sell/buy): {ratio:.6f}")
    print(f"  Expected ratio: {test_df['sell_price'].iloc[0] / test_df['buy_price'].iloc[0]:.6f}")

    if abs(ratio - (test_df['sell_price'].iloc[0] / test_df['buy_price'].iloc[0])) < 0.01:
        print("\n  SUCCESS: Price ratio is preserved!")
    else:
        print("\n  PROBLEM: Price ratio is NOT preserved!")

print("\n[4] Checking multiple timesteps:")
print("-"*80)

for i in range(5):
    obs, _ = env.reset()
    # Move to timestep i
    for _ in range(i):
        obs, _, _, _, _ = env.step(np.array([0.5, 0.5, 0.5, 0.0]))

    if 'buy_price' in env.state_cols and 'sell_price' in env.state_cols:
        buy_norm = obs[buy_idx]
        sell_norm = obs[sell_idx]
        buy_raw = test_df['buy_price'].iloc[i]
        sell_raw = test_df['sell_price'].iloc[i]

        print(f"\n  Timestep {i}:")
        print(f"    Raw:        buy={buy_raw:.4f}, sell={sell_raw:.4f}, ratio={sell_raw/buy_raw:.4f}")
        print(f"    Normalized: buy={buy_norm:.4f}, sell={sell_norm:.4f}, ratio={sell_norm/buy_norm:.4f}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)

if 'buy_price' in env.norm_factors and 'sell_price' in env.norm_factors:
    if env.norm_factors['buy_price'] == env.norm_factors['sell_price']:
        print("\nSUCCESS: Both prices use the same normalization factor")
        print(f"  Common factor: {env.norm_factors['buy_price']:.6f}")
        print("  This preserves the economic relationship between prices")
    else:
        print("\nPROBLEM: Prices still use different normalization factors!")
        print(f"  buy_price factor: {env.norm_factors['buy_price']:.6f}")
        print(f"  sell_price factor: {env.norm_factors['sell_price']:.6f}")
else:
    print("\nWARNING: Could not verify - price columns not found in state_cols")
    print(f"  state_cols: {env.state_cols}")

print("\n" + "="*80)
