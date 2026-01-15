"""
Compare OLD agent (with normalization bug) vs NEW agent (fixed normalization + gamma=0.999)
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
OLD_AGENT_PATH = PROJECT_ROOT / "outputs" / "agent_step_data.csv"
NEW_AGENT_PATH = PROJECT_ROOT / "outputs" / "agent_step_data_new.csv"

print("="*80)
print("COMPARISON: OLD vs NEW AGENT")
print("="*80)

# Load both datasets
old_df = pd.read_csv(OLD_AGENT_PATH)
new_df = pd.read_csv(NEW_AGENT_PATH)

print(f"\nOLD AGENT (Normalization Bug, Gamma=0.99):")
print(f"  Total steps: {len(old_df):,}")

old_cost = old_df['step_cost_eur'].sum()
old_deg = old_df['degradation_penalty'].sum()
old_total = old_cost + old_deg
old_sold = old_df['pv_to_grid_kWh'].sum() + old_df['discharge_to_grid_kWh'].sum()
old_pv = old_df['pv_production_kWh'].sum()
old_bought = old_df['grid_to_house_kWh'].sum()
old_soc = old_df['soc_kWh'].mean()

print(f"  Total electricity cost: EUR {old_cost:.2f}")
print(f"  Battery degradation: EUR {old_deg:.2f}")
print(f"  Combined cost: EUR {old_total:.2f}")
print(f"  Energy sold to grid: {old_sold:.1f} kWh ({100*old_sold/old_pv:.1f}% of PV)")
print(f"  Energy bought from grid: {old_bought:.1f} kWh")
print(f"  Avg battery SOC: {old_soc:.2f} kWh ({100*old_soc/10:.0f}%)")

print(f"\nNEW AGENT (Fixed Normalization, Gamma=0.999):")
print(f"  Total steps: {len(new_df):,}")

new_cost = new_df['step_cost_eur'].sum()
new_deg = new_df['degradation_penalty'].sum()
new_total = new_cost + new_deg
new_sold = new_df['pv_to_grid_kWh'].sum() + new_df['discharge_to_grid_kWh'].sum()
new_pv = new_df['pv_production_kWh'].sum()
new_bought = new_df['grid_to_house_kWh'].sum()
new_soc = new_df['soc_kWh'].mean()

print(f"  Total electricity cost: EUR {new_cost:.2f}")
print(f"  Battery degradation: EUR {new_deg:.2f}")
print(f"  Combined cost: EUR {new_total:.2f}")
print(f"  Energy sold to grid: {new_sold:.1f} kWh ({100*new_sold/new_pv:.1f}% of PV)")
print(f"  Energy bought from grid: {new_bought:.1f} kWh")
print(f"  Avg battery SOC: {new_soc:.2f} kWh ({100*new_soc/10:.0f}%)")

print("\n" + "="*80)
print("DELTA (New - Old):")
print("="*80)

cost_delta = new_total - old_total
sold_delta = new_sold - old_sold
bought_delta = new_bought - old_bought
soc_delta = new_soc - old_soc

print(f"  Combined cost: EUR {cost_delta:+.2f} ({100*cost_delta/old_total:+.1f}%)")
print(f"  Energy sold: {sold_delta:+.1f} kWh ({100*sold_delta/old_sold:+.1f}%)")
print(f"  Energy bought: {bought_delta:+.1f} kWh ({100*bought_delta/old_bought:+.1f}%)")
print(f"  Avg SOC: {soc_delta:+.2f} kWh ({100*soc_delta/old_soc:+.1f}%)")

if cost_delta > 0:
    print(f"\n  WORSE: New agent costs EUR {cost_delta:.2f} MORE per year")
else:
    print(f"\n  BETTER: New agent saves EUR {-cost_delta:.2f} per year")

if new_sold > old_sold:
    print(f"  PROBLEM: New agent sells {100*(new_sold-old_sold)/old_sold:.1f}% MORE to grid!")
else:
    print(f"  IMPROVEMENT: New agent sells {100*(old_sold-new_sold)/old_sold:.1f}% LESS to grid")

print("\n" + "="*80)
print("ANALYSIS: Did the fix work?")
print("="*80)

# Check if normalization is actually working in the new agent
test_df = pd.read_csv(PROJECT_ROOT / "data" / "main" / "processed" / "test.csv")
buy_prices = test_df['buy_price'].values[:10]
sell_prices = test_df['sell_price'].values[:10]

print(f"\nSample prices from test data:")
print(f"  Buy prices: {buy_prices}")
print(f"  Sell prices: {sell_prices}")
print(f"  Ratio (sell/buy): {sell_prices[0]/buy_prices[0]:.3f}")

# Analyze action patterns
print(f"\nAction patterns (OLD agent):")
print(f"  Avg pv_to_batt_frac: {old_df['pv_to_batt_frac'].mean():.3f}")
print(f"  Avg pv_to_house_frac: {old_df['pv_to_house_frac'].mean():.3f}")
print(f"  Avg batt_to_house_frac: {old_df['batt_to_house_frac'].mean():.3f}")

print(f"\nAction patterns (NEW agent):")
print(f"  Avg pv_to_batt_frac: {new_df['pv_to_batt_frac'].mean():.3f}")
print(f"  Avg pv_to_house_frac: {new_df['pv_to_house_frac'].mean():.3f}")
print(f"  Avg batt_to_house_frac: {new_df['batt_to_house_frac'].mean():.3f}")

print("\n" + "="*80)
