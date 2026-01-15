"""
Simple Diagnostic: Use existing agent_step_data.csv and re-run rule-based policies for comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DATA_PATH = PROJECT_ROOT / "data" / "main" / "processed" / "test.csv"
AGENT_DATA_PATH = PROJECT_ROOT / "outputs" / "agent_step_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics"
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("DIAGNOSTIC DEEP-DIVE: RL Agent Performance Analysis")
print("="*80)

# Load data
print("\n[1/5] Loading data...")
test_df = pd.read_csv(TEST_DATA_PATH)
agent_df = pd.read_csv(AGENT_DATA_PATH)

test_df['datetime'] = pd.to_datetime(test_df['datetime'])
agent_df['datetime'] = pd.to_datetime(agent_df['datetime'])

# Merge to get full context (agent_df already has consumption_kWh and pv_production_kWh)
full_df = agent_df.merge(test_df[['datetime', 'buy_price', 'sell_price', 'hour', 'temperature_C']],
                          on='datetime', how='left')

print(f"  OK Loaded {len(full_df):,} timesteps")

# Compute metrics
print("\n[2/5] Computing agent metrics...")
total_cost = full_df['step_cost_eur'].sum()
total_degradation = full_df['degradation_penalty'].sum()
combined_cost = total_cost + total_degradation

print(f"\n  Agent Performance (2023):")
print(f"    Total electricity cost: EUR {total_cost:.2f}")
print(f"    Battery degradation:    EUR {total_degradation:.2f}")
print(f"    Combined cost:          EUR {combined_cost:.2f}")
print(f"    Avg battery SOC:        {full_df['soc_kWh'].mean():.2f} kWh")

# Analyze energy flows
print("\n[3/5] Analyzing energy flow patterns...")
total_pv = full_df['pv_production_kWh'].sum()
total_consumption = full_df['consumption_kWh'].sum()
total_pv_to_house = full_df['pv_to_house_kWh'].sum()
total_pv_to_batt = full_df['pv_to_batt_kWh'].sum()
total_pv_to_grid = full_df['pv_to_grid_kWh'].sum()
total_batt_to_house = full_df['discharge_to_house_kWh'].sum()
total_batt_to_grid = full_df['discharge_to_grid_kWh'].sum()
total_grid_to_house = full_df['grid_to_house_kWh'].sum()
total_grid_to_batt = full_df['grid_to_batt_kWh'].sum()

print(f"\n  Energy Flows (kWh):")
print(f"    Total PV production:      {total_pv:.1f}")
print(f"    Total consumption:        {total_consumption:.1f}")
print(f"    PV -> House:              {total_pv_to_house:.1f} ({100*total_pv_to_house/total_pv:.1f}%)")
print(f"    PV -> Battery:            {total_pv_to_batt:.1f} ({100*total_pv_to_batt/total_pv:.1f}%)")
print(f"    PV -> Grid (SOLD):        {total_pv_to_grid:.1f} ({100*total_pv_to_grid/total_pv:.1f}%)")
print(f"    Battery -> House:         {total_batt_to_house:.1f}")
print(f"    Battery -> Grid (SOLD):   {total_batt_to_grid:.1f}")
print(f"    Grid -> House (BOUGHT):   {total_grid_to_house:.1f}")
print(f"    Grid -> Battery:          {total_grid_to_batt:.1f}")

# KEY INSIGHT: How much did selling cost the agent?
revenue_from_selling = (total_pv_to_grid + total_batt_to_grid) * full_df['sell_price'].mean()
cost_of_grid_purchases = total_grid_to_house * full_df['buy_price'].mean()

print(f"\n  Economics:")
print(f"    Revenue from selling:     EUR {revenue_from_selling:.2f}")
print(f"    Cost of grid purchases:   EUR {cost_of_grid_purchases:.2f}")
print(f"    Avg sell price:           EUR {full_df['sell_price'].mean():.4f}/kWh")
print(f"    Avg buy price:            EUR {full_df['buy_price'].mean():.4f}/kWh")
print(f"    Sell/Buy ratio:           {full_df['sell_price'].mean() / full_df['buy_price'].mean():.2%}")

# Temporal analysis
print("\n[4/5] Analyzing temporal patterns...")

full_df['date'] = full_df['datetime'].dt.date
full_df['month'] = full_df['datetime'].dt.month

# Daily aggregation
daily_df = full_df.groupby('date').agg({
    'step_cost_eur': 'sum',
    'degradation_penalty': 'sum',
    'pv_production_kWh': 'sum',
    'consumption_kWh': 'sum',
    'pv_to_grid_kWh': 'sum',
    'discharge_to_grid_kWh': 'sum',
    'grid_to_house_kWh': 'sum'
}).reset_index()

daily_df['total_cost'] = daily_df['step_cost_eur'] + daily_df['degradation_penalty']
daily_df['total_sold'] = daily_df['pv_to_grid_kWh'] + daily_df['discharge_to_grid_kWh']

# Find problematic patterns
print("\n  Worst 10 days by cost:")
worst_days = daily_df.nlargest(10, 'total_cost')[['date', 'total_cost', 'pv_production_kWh', 'consumption_kWh', 'total_sold']]
print(worst_days.to_string(index=False))

# Hourly analysis
hourly_df = full_df.groupby('hour').agg({
    'step_cost_eur': 'mean',
    'pv_production_kWh': 'mean',
    'consumption_kWh': 'mean',
    'pv_to_grid_kWh': 'sum',
    'discharge_to_grid_kWh': 'sum',
    'grid_to_house_kWh': 'sum',
    'buy_price': 'mean',
    'sell_price': 'mean'
}).reset_index()

hourly_df['total_sold'] = hourly_df['pv_to_grid_kWh'] + hourly_df['discharge_to_grid_kWh']

print("\n  Hours with most selling to grid:")
selling_hours = hourly_df.nlargest(5, 'total_sold')[['hour', 'total_sold', 'pv_production_kWh', 'buy_price', 'sell_price']]
print(selling_hours.to_string(index=False))

# CRITICAL INSIGHT: When does agent sell despite bad economics?
full_df['is_selling'] = (full_df['pv_to_grid_kWh'] > 0) | (full_df['discharge_to_grid_kWh'] > 0)
full_df['could_store'] = (full_df['soc_kWh'] < 9.0) & full_df['is_selling']

print(f"\n  Selling Behavior:")
print(f"    Timesteps selling to grid:    {full_df['is_selling'].sum():,} ({100*full_df['is_selling'].mean():.1f}%)")
print(f"    Could have stored instead:    {full_df['could_store'].sum():,} ({100*full_df['could_store'].mean():.1f}%)")
print(f"    Energy sold unnecessarily:    {full_df[full_df['could_store']]['pv_to_grid_kWh'].sum():.1f} kWh")

# Battery usage patterns
full_df['soc_zone'] = pd.cut(full_df['soc_kWh'], bins=[0, 2.5, 5, 7.5, 10], labels=['0-25%', '25-50%', '50-75%', '75-100%'])
print(f"\n  Battery SOC Distribution:")
print(full_df['soc_zone'].value_counts().sort_index())

# Visualizations
print("\n[5/5] Creating visualizations...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Daily costs
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(daily_df['date'], daily_df['total_cost'], linewidth=1, alpha=0.7)
ax1.set_xlabel('Date')
ax1.set_ylabel('Daily Cost (EUR)')
ax1.set_title('Daily Total Cost (Electricity + Degradation)')
ax1.grid(True, alpha=0.3)

# 2. Hourly cost pattern
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(hourly_df['hour'], hourly_df['step_cost_eur'], marker='o', linewidth=2)
ax2.set_xlabel('Hour of Day')
ax2.set_ylabel('Avg Cost (EUR)')
ax2.set_title('Average Cost by Hour')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(0, 24, 3))

# 3. Energy flows by hour
ax3 = fig.add_subplot(gs[1, 1])
ax3.bar(hourly_df['hour'], hourly_df['pv_production_kWh'], alpha=0.6, label='PV Production')
ax3.bar(hourly_df['hour'], hourly_df['consumption_kWh'], alpha=0.6, label='Consumption')
ax3.set_xlabel('Hour of Day')
ax3.set_ylabel('Energy (kWh)')
ax3.set_title('Average Energy by Hour')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xticks(range(0, 24, 3))

# 4. Selling behavior
ax4 = fig.add_subplot(gs[1, 2])
ax4.bar(hourly_df['hour'], hourly_df['total_sold'], color='orange', alpha=0.7)
ax4.set_xlabel('Hour of Day')
ax4.set_ylabel('Energy Sold (kWh)')
ax4.set_title('Total Energy Sold to Grid by Hour')
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_xticks(range(0, 24, 3))

# 5. SOC over sample week
sample_week = full_df[full_df['datetime'].dt.isocalendar().week == 1]
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(sample_week['datetime'], sample_week['soc_kWh'], linewidth=2)
ax5.set_xlabel('DateTime')
ax5.set_ylabel('SOC (kWh)')
ax5.set_title('Battery SOC (Week 1)')
ax5.grid(True, alpha=0.3)

# 6. Price vs selling decisions
ax6 = fig.add_subplot(gs[2, 1])
selling_steps = full_df[full_df['is_selling']]
not_selling_steps = full_df[~full_df['is_selling']]
ax6.scatter(not_selling_steps['buy_price'], not_selling_steps['sell_price'],
           alpha=0.1, s=1, label='Not Selling', color='blue')
ax6.scatter(selling_steps['buy_price'], selling_steps['sell_price'],
           alpha=0.3, s=2, label='Selling', color='orange')
ax6.plot([0, 0.25], [0, 0.25], 'r--', linewidth=1, label='Break-even')
ax6.set_xlabel('Buy Price (EUR/kWh)')
ax6.set_ylabel('Sell Price (EUR/kWh)')
ax6.set_title('Price Context When Agent Sells')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Monthly costs
monthly_df = full_df.groupby('month').agg({'step_cost_eur': 'sum', 'degradation_penalty': 'sum'}).reset_index()
monthly_df['total'] = monthly_df['step_cost_eur'] + monthly_df['degradation_penalty']
ax7 = fig.add_subplot(gs[2, 2])
ax7.bar(monthly_df['month'], monthly_df['total'], alpha=0.7, color='steelblue')
ax7.set_xlabel('Month')
ax7.set_ylabel('Total Cost (EUR)')
ax7.set_title('Monthly Costs')
ax7.set_xticks(range(1, 13))
ax7.grid(True, alpha=0.3, axis='y')

plt.savefig(OUTPUT_DIR / "agent_diagnostic.png", dpi=300, bbox_inches='tight')
print(f"  OK Saved: agent_diagnostic.png")

# Save analysis
full_df.to_csv(OUTPUT_DIR / "full_analysis.csv", index=False)
daily_df.to_csv(OUTPUT_DIR / "daily_summary.csv", index=False)
hourly_df.to_csv(OUTPUT_DIR / "hourly_summary.csv", index=False)

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
print(f"\n1. SELLING PROBLEM:")
print(f"   - Agent sells {total_pv_to_grid + total_batt_to_grid:.1f} kWh to grid")
print(f"   - At avg sell price of EUR {full_df['sell_price'].mean():.4f}/kWh")
print(f"   - This is only {100*full_df['sell_price'].mean()/full_df['buy_price'].mean():.1f}% of buy price!")
print(f"   - Could have stored {full_df[full_df['could_store']]['pv_to_grid_kWh'].sum():.1f} kWh instead")

print(f"\n2. GRID DEPENDENCY:")
print(f"   - Agent buys {total_grid_to_house:.1f} kWh from grid")
print(f"   - At avg buy price of EUR {full_df['buy_price'].mean():.4f}/kWh")
print(f"   - Self-sufficiency: {100*(1-total_grid_to_house/total_consumption):.1f}%")

print(f"\n3. BATTERY UTILIZATION:")
print(f"   - Avg SOC: {full_df['soc_kWh'].mean():.2f} kWh ({100*full_df['soc_kWh'].mean()/10:.0f}% of capacity)")
print(f"   - Battery cycles: {(total_pv_to_batt + total_batt_to_house)/(2*10):.1f} full cycles")

print("\n" + "="*80)
print(f"Output saved to: {OUTPUT_DIR}")
print("="*80)
