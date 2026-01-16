"""
Compare V2 RL Agent vs Rule-Based Policy
=========================================

Evaluate both approaches on the same test data and compare:
1. Physical energy flows
2. Decision patterns
3. Total costs
4. Battery usage patterns
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

print("="*80)
print("COMPARING V2 RL AGENT vs RULE-BASED POLICY")
print("="*80)

# ============================================================================
# Load and prepare test data
# ============================================================================
print("\n[1/5] Loading test data...")
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

# ============================================================================
# Evaluate RL Agent
# ============================================================================
print("\n[2/5] Evaluating RL Agent...")
env_rl = SolarBatteryEnvV2(
    data=test_df,
    battery_capacity=10.0,
    max_charge_rate=5.0,
    timestep_h=1.0,
    eta=0.95,
    degradation_cost=0.001
)

model = SAC.load(MODEL_PATH)
obs, _ = env_rl.reset()
done = False

rl_records = []
step_count = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_rl.step(action)

    rl_records.append({
        'timestep': step_count,
        'charge_batt_frac': info['charge_batt_frac'],
        'use_batt_frac': info['use_batt_frac'],
        'soc_kWh': info['soc_kWh'],
        'pv_to_house_kWh': info['pv_to_house_kWh'],
        'pv_to_batt_kWh': info['pv_to_batt_kWh'],
        'pv_to_grid_kWh': info['pv_to_grid_kWh'],
        'batt_to_house_kWh': info['batt_to_house_kWh'],
        'grid_to_house_kWh': info['grid_to_house_kWh'],
        'cost_eur': info['cost_eur'],
        'degradation_penalty': info['degradation_penalty']
    })

    step_count += 1
    done = terminated or truncated

rl_df = pd.DataFrame(rl_records)
print(f"  Completed {step_count:,} steps")

# ============================================================================
# Evaluate Rule-Based Policy
# ============================================================================
print("\n[3/5] Evaluating Rule-Based Policy...")

class SimpleRulePolicy:
    """
    Simple rule-based policy:
    - Always use PV for house first (done automatically by env)
    - Surplus PV: charge battery if SOC < 80%, else sell
    - Deficit: use battery if SOC > 30%, else buy from grid
    """
    def predict(self, obs, soc, deterministic=True):
        # Extract SOC from observation or use passed value
        soc_ratio = soc / 10.0  # Normalize to [0, 1]

        # Rule 1: Charge battery if below 80%
        if soc_ratio < 0.8:
            charge_batt_frac = 1.0  # Store all surplus
        else:
            charge_batt_frac = 0.0  # Sell surplus

        # Rule 2: Use battery if above 30%
        if soc_ratio > 0.3:
            use_batt_frac = 1.0  # Use battery for deficit
        else:
            use_batt_frac = 0.0  # Buy from grid

        return np.array([charge_batt_frac, use_batt_frac])

env_rule = SolarBatteryEnvV2(
    data=test_df,
    battery_capacity=10.0,
    max_charge_rate=5.0,
    timestep_h=1.0,
    eta=0.95,
    degradation_cost=0.001
)

rule_policy = SimpleRulePolicy()
obs, _ = env_rule.reset()
done = False

rule_records = []
step_count = 0

while not done:
    # Rule policy needs current SOC to make decisions
    action = rule_policy.predict(obs, env_rule.soc, deterministic=True)
    obs, reward, terminated, truncated, info = env_rule.step(action)

    rule_records.append({
        'timestep': step_count,
        'charge_batt_frac': info['charge_batt_frac'],
        'use_batt_frac': info['use_batt_frac'],
        'soc_kWh': info['soc_kWh'],
        'pv_to_house_kWh': info['pv_to_house_kWh'],
        'pv_to_batt_kWh': info['pv_to_batt_kWh'],
        'pv_to_grid_kWh': info['pv_to_grid_kWh'],
        'batt_to_house_kWh': info['batt_to_house_kWh'],
        'grid_to_house_kWh': info['grid_to_house_kWh'],
        'cost_eur': info['cost_eur'],
        'degradation_penalty': info['degradation_penalty']
    })

    step_count += 1
    done = terminated or truncated

rule_df = pd.DataFrame(rule_records)
print(f"  Completed {step_count:,} steps")

# ============================================================================
# Merge with original data for context
# ============================================================================
print("\n[4/5] Merging with original data...")
original_data = pd.read_csv(TEST_DATA_PATH).iloc[:len(rl_df)].reset_index(drop=True)

rl_df['datetime'] = original_data['datetime']
rl_df['pv_production_kWh'] = original_data['P']
rl_df['consumption_kWh'] = original_data['consumption_kWh']
rl_df['buy_price'] = original_data['buy_price']
rl_df['sell_price'] = original_data['sell_price']

rule_df['datetime'] = original_data['datetime']
rule_df['pv_production_kWh'] = original_data['P']
rule_df['consumption_kWh'] = original_data['consumption_kWh']
rule_df['buy_price'] = original_data['buy_price']
rule_df['sell_price'] = original_data['sell_price']

# ============================================================================
# Compare Results
# ============================================================================
print("\n[5/5] Computing comparison...")

print("\n" + "="*80)
print("PHYSICAL ENERGY FLOWS")
print("="*80)

# RL Agent flows
rl_pv_total = rl_df['pv_production_kWh'].sum()
rl_consumption_total = rl_df['consumption_kWh'].sum()
rl_pv_to_house = rl_df['pv_to_house_kWh'].sum()
rl_pv_to_batt = rl_df['pv_to_batt_kWh'].sum()
rl_pv_to_grid = rl_df['pv_to_grid_kWh'].sum()
rl_batt_to_house = rl_df['batt_to_house_kWh'].sum()
rl_grid_to_house = rl_df['grid_to_house_kWh'].sum()
rl_avg_soc = rl_df['soc_kWh'].mean()

print("\nRL AGENT:")
print(f"  PV Production: {rl_pv_total:.1f} kWh")
print(f"  Consumption: {rl_consumption_total:.1f} kWh")
print(f"\n  PV Allocation:")
print(f"    -> House: {rl_pv_to_house:.1f} kWh ({100*rl_pv_to_house/rl_pv_total:.1f}%)")
print(f"    -> Battery: {rl_pv_to_batt:.1f} kWh ({100*rl_pv_to_batt/rl_pv_total:.1f}%)")
print(f"    -> Grid (sold): {rl_pv_to_grid:.1f} kWh ({100*rl_pv_to_grid/rl_pv_total:.1f}%)")
print(f"\n  House Supply:")
print(f"    <- PV: {rl_pv_to_house:.1f} kWh ({100*rl_pv_to_house/rl_consumption_total:.1f}%)")
print(f"    <- Battery: {rl_batt_to_house:.1f} kWh ({100*rl_batt_to_house/rl_consumption_total:.1f}%)")
print(f"    <- Grid: {rl_grid_to_house:.1f} kWh ({100*rl_grid_to_house/rl_consumption_total:.1f}%)")
print(f"\n  Battery:")
print(f"    Average SOC: {rl_avg_soc:.2f} kWh ({100*rl_avg_soc/10:.0f}%)")
print(f"    Total charged: {rl_pv_to_batt:.1f} kWh")
print(f"    Total discharged: {rl_batt_to_house:.1f} kWh")

# Rule-based flows
rule_pv_total = rule_df['pv_production_kWh'].sum()
rule_consumption_total = rule_df['consumption_kWh'].sum()
rule_pv_to_house = rule_df['pv_to_house_kWh'].sum()
rule_pv_to_batt = rule_df['pv_to_batt_kWh'].sum()
rule_pv_to_grid = rule_df['pv_to_grid_kWh'].sum()
rule_batt_to_house = rule_df['batt_to_house_kWh'].sum()
rule_grid_to_house = rule_df['grid_to_house_kWh'].sum()
rule_avg_soc = rule_df['soc_kWh'].mean()

print("\nRULE-BASED POLICY:")
print(f"  PV Production: {rule_pv_total:.1f} kWh")
print(f"  Consumption: {rule_consumption_total:.1f} kWh")
print(f"\n  PV Allocation:")
print(f"    -> House: {rule_pv_to_house:.1f} kWh ({100*rule_pv_to_house/rule_pv_total:.1f}%)")
print(f"    -> Battery: {rule_pv_to_batt:.1f} kWh ({100*rule_pv_to_batt/rule_pv_total:.1f}%)")
print(f"    -> Grid (sold): {rule_pv_to_grid:.1f} kWh ({100*rule_pv_to_grid/rule_pv_total:.1f}%)")
print(f"\n  House Supply:")
print(f"    <- PV: {rule_pv_to_house:.1f} kWh ({100*rule_pv_to_house/rule_consumption_total:.1f}%)")
print(f"    <- Battery: {rule_batt_to_house:.1f} kWh ({100*rule_batt_to_house/rule_consumption_total:.1f}%)")
print(f"    <- Grid: {rule_grid_to_house:.1f} kWh ({100*rule_grid_to_house/rule_consumption_total:.1f}%)")
print(f"\n  Battery:")
print(f"    Average SOC: {rule_avg_soc:.2f} kWh ({100*rule_avg_soc/10:.0f}%)")
print(f"    Total charged: {rule_pv_to_batt:.1f} kWh")
print(f"    Total discharged: {rule_batt_to_house:.1f} kWh")

# ============================================================================
# Cost Comparison
# ============================================================================
print("\n" + "="*80)
print("COST COMPARISON")
print("="*80)

# RL costs
rl_revenue = (rl_df['pv_to_grid_kWh'] * rl_df['sell_price']).sum()
rl_cost_grid = (rl_df['grid_to_house_kWh'] * rl_df['buy_price']).sum()
rl_net_cost = rl_cost_grid - rl_revenue
rl_degradation = rl_df['degradation_penalty'].sum()
rl_total = rl_net_cost + rl_degradation

print("\nRL AGENT:")
print(f"  Revenue (selling): +EUR {rl_revenue:.2f}")
print(f"  Cost (buying): -EUR {rl_cost_grid:.2f}")
print(f"  Net electricity: EUR {rl_net_cost:.2f}")
print(f"  Battery degradation: EUR {rl_degradation:.2f}")
print(f"  TOTAL ANNUAL COST: EUR {rl_total:.2f}")

# Rule costs
rule_revenue = (rule_df['pv_to_grid_kWh'] * rule_df['sell_price']).sum()
rule_cost_grid = (rule_df['grid_to_house_kWh'] * rule_df['buy_price']).sum()
rule_net_cost = rule_cost_grid - rule_revenue
rule_degradation = rule_df['degradation_penalty'].sum()
rule_total = rule_net_cost + rule_degradation

print("\nRULE-BASED POLICY:")
print(f"  Revenue (selling): +EUR {rule_revenue:.2f}")
print(f"  Cost (buying): -EUR {rule_cost_grid:.2f}")
print(f"  Net electricity: EUR {rule_net_cost:.2f}")
print(f"  Battery degradation: EUR {rule_degradation:.2f}")
print(f"  TOTAL ANNUAL COST: EUR {rule_total:.2f}")

# Delta
delta = rule_total - rl_total
print("\n" + "="*80)
print("DELTA (Rule - RL):")
print("="*80)
print(f"  RL saves EUR {delta:.2f} per year ({100*delta/rule_total:.1f}% better)")

# ============================================================================
# Decision Pattern Analysis
# ============================================================================
print("\n" + "="*80)
print("DECISION PATTERNS")
print("="*80)

print("\nRL AGENT:")
print(f"  Avg charge_batt_frac: {rl_df['charge_batt_frac'].mean():.3f}")
print(f"  Avg use_batt_frac: {rl_df['use_batt_frac'].mean():.3f}")
print(f"  Times battery charged: {(rl_pv_to_batt > 0).sum()} / {len(rl_df)} steps")
print(f"  Times battery used: {(rl_batt_to_house > 0).sum()} / {len(rl_df)} steps")

print("\nRULE-BASED POLICY:")
print(f"  Avg charge_batt_frac: {rule_df['charge_batt_frac'].mean():.3f}")
print(f"  Avg use_batt_frac: {rule_df['use_batt_frac'].mean():.3f}")
print(f"  Times battery charged: {(rule_pv_to_batt > 0).sum()} / {len(rule_df)} steps")
print(f"  Times battery used: {(rule_batt_to_house > 0).sum()} / {len(rule_df)} steps")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
print(f"\nSummary: RL agent achieves EUR {delta:.2f} annual savings")
print(f"through better battery management and strategic energy allocation.")
print("\n" + "="*80)
