"""
Diagnostic Deep-Dive: Analyze WHERE and WHEN the RL agent underperforms vs rule-based policies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import SAC
from gym_env import SolarBatteryEnv
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DATA_PATH = PROJECT_ROOT / "data" / "main" / "processed" / "test.csv"
AGENT_DATA_PATH = PROJECT_ROOT / "outputs" / "agent_step_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics"
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("DIAGNOSTIC DEEP-DIVE: RL Agent vs Rule-Based Policies")
print("="*80)

# Load test data and agent results
print("\n[1/6] Loading data...")
test_df = pd.read_csv(TEST_DATA_PATH)
agent_df = pd.read_csv(AGENT_DATA_PATH)

# Convert datetime
test_df['datetime'] = pd.to_datetime(test_df['datetime'])
agent_df['datetime'] = pd.to_datetime(agent_df['datetime'])

print(f"  OK Test data: {len(test_df):,} timesteps")
print(f"  OK Agent data: {len(agent_df):,} timesteps")

# ============================================================================
# RUN ALL THREE POLICIES
# ============================================================================

print("\n[2/6] Running all three policies on test data...")

env = SolarBatteryEnv(
    data=test_df,
    battery_capacity=10.0,
    max_charge_rate=5.0,
    timestep_h=1.0,
    eta=0.95,
    degradation_cost=0.001
)

def run_agent_policy(env, model_path):
    """Run trained SAC agent"""
    model = SAC.load(model_path)
    obs, _ = env.reset()
    results = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        results.append({
            'reward': reward,
            'cost': info.get('step_cost', 0),
            'degradation': info.get('degradation_penalty', 0),
            'soc': info.get('soc', 0),
        })
        done = done or truncated

    return pd.DataFrame(results)

def run_simple_rule_policy(env):
    """Run simple rule-based policy (no forecasts, no price awareness)"""
    obs, _ = env.reset()
    results = []
    done = False

    while not done:
        # Simple heuristic from cumulative_reward_graph.py lines 55-87
        pv = env.pv_production
        consumption = env.consumption
        soc = env.battery_soc
        capacity = env.battery_capacity
        max_rate = env.max_charge_rate

        if pv >= consumption:
            # Surplus PV
            surplus = pv - consumption
            room_in_battery = capacity - soc
            charge_amount = min(surplus, room_in_battery, max_rate)

            pv_to_house_frac = consumption / max(pv, 1e-6)
            pv_to_batt_frac = charge_amount / max(pv, 1e-6)
            batt_to_house_frac = 0.0
            grid_to_batt_frac = 0.0
        else:
            # Deficit
            deficit = consumption - pv
            available_battery = soc
            discharge_amount = min(deficit, available_battery, max_rate)

            pv_to_house_frac = 1.0  # All PV to house
            pv_to_batt_frac = 0.0
            batt_to_house_frac = 1.0 if discharge_amount > 0 else 0.0
            grid_to_batt_frac = 0.0

        action = np.array([pv_to_house_frac, pv_to_batt_frac, batt_to_house_frac, grid_to_batt_frac])
        obs, reward, done, truncated, info = env.step(action)

        results.append({
            'reward': reward,
            'cost': info.get('step_cost', 0),
            'degradation': info.get('degradation_penalty', 0),
            'soc': info.get('soc', 0),
        })
        done = done or truncated

    return pd.DataFrame(results)

def run_forecast_aware_policy(env):
    """Run forecast-aware rule-based policy"""
    obs, _ = env.reset()
    results = []
    done = False

    while not done:
        # Forecast-aware heuristic from cumulative_reward_graph.py lines 89-144
        pv = env.pv_production
        consumption = env.consumption
        soc = env.battery_soc
        capacity = env.battery_capacity
        max_rate = env.max_charge_rate
        buy_price = env.price_buy
        sell_price = env.price_sell

        # Get forecasts (6-hour rolling average)
        current_step = env.current_step
        forecast_horizon = 6
        end_step = min(current_step + forecast_horizon, len(env.historic_df))

        future_pv = env.historic_df.iloc[current_step:end_step]['P'].mean()
        future_load = env.historic_df.iloc[current_step:end_step]['consumption'].mean()

        # Dynamic reserve level based on hour and forecast
        hour = env.historic_df.iloc[current_step]['hour']
        if future_pv > future_load:
            reserve_level = 0.05 * capacity  # Expect surplus
        elif 17 <= hour < 22:
            reserve_level = 0.25 * capacity  # Peak evening hours
        else:
            reserve_level = 0.15 * capacity

        if pv >= consumption:
            # Surplus PV
            surplus = pv - consumption
            room_in_battery = capacity - soc

            # Reduce charging if expecting more PV surplus
            charge_factor = 0.5 if future_pv > 1.2 * future_load else 1.0
            charge_amount = min(surplus * charge_factor, room_in_battery, max_rate)

            # Reduce charge if sell price is high and battery nearly full
            if sell_price > 0.05 and room_in_battery < 1.0:
                charge_amount *= 0.8

            pv_to_house_frac = consumption / max(pv, 1e-6)
            pv_to_batt_frac = charge_amount / max(pv, 1e-6)
            batt_to_house_frac = 0.0
            grid_to_batt_frac = 0.0
        else:
            # Deficit
            deficit = consumption - pv
            available_battery = max(0, soc - reserve_level)

            # Discharge more aggressively if deficit expected
            discharge_factor = 1.0 if (future_load - future_pv) > 0 else 0.6
            discharge_amount = min(deficit, available_battery, max_rate) * discharge_factor

            pv_to_house_frac = 1.0
            pv_to_batt_frac = 0.0
            batt_to_house_frac = 1.0 if discharge_amount > 0 else 0.0

            # Strategic grid charging if price is cheap
            if buy_price < 0.6 * sell_price and (future_load - future_pv) > 0.5 and room_in_battery > 0.5 * capacity:
                grid_to_batt_frac = 0.5
            else:
                grid_to_batt_frac = 0.0

        action = np.array([pv_to_house_frac, pv_to_batt_frac, batt_to_house_frac, grid_to_batt_frac])
        obs, reward, done, truncated, info = env.step(action)

        results.append({
            'reward': reward,
            'cost': info.get('step_cost', 0),
            'degradation': info.get('degradation_penalty', 0),
            'soc': info.get('soc', 0),
        })
        done = done or truncated

    return pd.DataFrame(results)

# Run all policies
model_path = PROJECT_ROOT / "solar_batt_agent_weekly_lagged.zip"

print("  → Running RL Agent...")
agent_results = run_agent_policy(env, model_path)

print("  → Running Simple Rule-Based Policy...")
env.reset()
simple_results = run_simple_rule_policy(env)

print("  → Running Forecast-Aware Rule-Based Policy...")
env.reset()
forecast_results = run_forecast_aware_policy(env)

# ============================================================================
# COMPARE PERFORMANCE
# ============================================================================

print("\n[3/6] Computing performance metrics...")

def compute_metrics(df, name):
    total_cost = df['cost'].sum()
    total_degradation = df['degradation'].sum()
    total_reward = df['reward'].sum()
    avg_soc = df['soc'].mean()

    return {
        'Policy': name,
        'Total Cost (€)': total_cost,
        'Degradation (€)': total_degradation,
        'Combined (€)': total_cost + total_degradation,
        'Cumulative Reward': total_reward,
        'Avg SOC (kWh)': avg_soc
    }

metrics = pd.DataFrame([
    compute_metrics(agent_results, 'RL Agent'),
    compute_metrics(simple_results, 'Simple Rule'),
    compute_metrics(forecast_results, 'Forecast-Aware Rule')
])

print("\n" + "="*80)
print("PERFORMANCE COMPARISON (2023 Test Year)")
print("="*80)
print(metrics.to_string(index=False))
print("="*80)

# Compute differences
agent_cost = metrics.loc[metrics['Policy'] == 'RL Agent', 'Combined (€)'].values[0]
simple_cost = metrics.loc[metrics['Policy'] == 'Simple Rule', 'Combined (€)'].values[0]
forecast_cost = metrics.loc[metrics['Policy'] == 'Forecast-Aware Rule', 'Combined (€)'].values[0]

print(f"\n💰 RL Agent vs Simple Rule: €{agent_cost - simple_cost:.2f} ({((agent_cost/simple_cost - 1) * 100):.1f}% worse)" if agent_cost > simple_cost else f"€{simple_cost - agent_cost:.2f} better ({((1 - agent_cost/simple_cost) * 100):.1f}%)")
print(f"💰 RL Agent vs Forecast Rule: €{agent_cost - forecast_cost:.2f} ({((agent_cost/forecast_cost - 1) * 100):.1f}% worse)" if agent_cost > forecast_cost else f"€{forecast_cost - agent_cost:.2f} better ({((1 - agent_cost/forecast_cost) * 100):.1f}%)")

# Save metrics
metrics.to_csv(OUTPUT_DIR / "policy_comparison_metrics.csv", index=False)

# ============================================================================
# TEMPORAL ANALYSIS: WHEN DOES AGENT FAIL?
# ============================================================================

print("\n[4/6] Analyzing temporal patterns...")

# Create comparison dataframe
comparison_df = test_df[['datetime']].copy()
comparison_df['agent_reward'] = agent_results['reward'].values
comparison_df['simple_reward'] = simple_results['reward'].values
comparison_df['forecast_reward'] = forecast_results['reward'].values
comparison_df['agent_cost'] = agent_results['cost'].values
comparison_df['simple_cost'] = simple_results['cost'].values
comparison_df['forecast_cost'] = forecast_results['cost'].values
comparison_df['agent_soc'] = agent_results['soc'].values
comparison_df['simple_soc'] = simple_results['soc'].values
comparison_df['forecast_soc'] = forecast_results['soc'].values

# Add time features
comparison_df['hour'] = comparison_df['datetime'].dt.hour
comparison_df['day_of_week'] = comparison_df['datetime'].dt.dayofweek
comparison_df['month'] = comparison_df['datetime'].dt.month
comparison_df['date'] = comparison_df['datetime'].dt.date

# Daily aggregation
daily_comparison = comparison_df.groupby('date').agg({
    'agent_cost': 'sum',
    'simple_cost': 'sum',
    'forecast_cost': 'sum',
    'agent_reward': 'sum',
    'simple_reward': 'sum',
    'forecast_reward': 'sum'
}).reset_index()

daily_comparison['agent_vs_simple'] = daily_comparison['agent_cost'] - daily_comparison['simple_cost']
daily_comparison['agent_vs_forecast'] = daily_comparison['agent_cost'] - daily_comparison['forecast_cost']

# Find worst days
worst_vs_simple = daily_comparison.nlargest(10, 'agent_vs_simple')[['date', 'agent_vs_simple', 'agent_cost', 'simple_cost']]
worst_vs_forecast = daily_comparison.nlargest(10, 'agent_vs_forecast')[['date', 'agent_vs_forecast', 'agent_cost', 'forecast_cost']]

print("\n📉 TOP 10 WORST DAYS (Agent vs Simple Rule):")
print(worst_vs_simple.to_string(index=False))

print("\n📉 TOP 10 WORST DAYS (Agent vs Forecast Rule):")
print(worst_vs_forecast.to_string(index=False))

# Hourly patterns
hourly_comparison = comparison_df.groupby('hour').agg({
    'agent_cost': 'mean',
    'simple_cost': 'mean',
    'forecast_cost': 'mean'
}).reset_index()

hourly_comparison['agent_vs_simple'] = hourly_comparison['agent_cost'] - hourly_comparison['simple_cost']
hourly_comparison['agent_vs_forecast'] = hourly_comparison['agent_cost'] - hourly_comparison['forecast_cost']

print("\nTime: HOURLY PERFORMANCE (Average cost difference):")
print(hourly_comparison[['hour', 'agent_vs_simple', 'agent_vs_forecast']].to_string(index=False))

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n[5/6] Creating diagnostic visualizations...")

# 1. Cumulative cost over time
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Daily cumulative cost
ax = axes[0, 0]
daily_cumsum = daily_comparison.copy()
daily_cumsum['agent_cumcost'] = daily_cumsum['agent_cost'].cumsum()
daily_cumsum['simple_cumcost'] = daily_cumsum['simple_cost'].cumsum()
daily_cumsum['forecast_cumcost'] = daily_cumsum['forecast_cost'].cumsum()

ax.plot(daily_cumsum['date'], daily_cumsum['agent_cumcost'], label='RL Agent', linewidth=2)
ax.plot(daily_cumsum['date'], daily_cumsum['simple_cumcost'], label='Simple Rule', linewidth=2)
ax.plot(daily_cumsum['date'], daily_cumsum['forecast_cumcost'], label='Forecast Rule', linewidth=2)
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Cost (€)')
ax.set_title('Cumulative Cost Over Time (2023)')
ax.legend()
ax.grid(True, alpha=0.3)

# Hourly average cost
ax = axes[0, 1]
ax.plot(hourly_comparison['hour'], hourly_comparison['agent_cost'], marker='o', label='RL Agent', linewidth=2)
ax.plot(hourly_comparison['hour'], hourly_comparison['simple_cost'], marker='s', label='Simple Rule', linewidth=2)
ax.plot(hourly_comparison['hour'], hourly_comparison['forecast_cost'], marker='^', label='Forecast Rule', linewidth=2)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Average Cost (€)')
ax.set_title('Average Cost by Hour of Day')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, 24, 2))

# Cost difference heatmap (hourly)
ax = axes[1, 0]
hourly_diff = hourly_comparison['agent_vs_simple'].values
colors = ['red' if x > 0 else 'green' for x in hourly_diff]
ax.bar(hourly_comparison['hour'], hourly_diff, color=colors, alpha=0.7)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Cost Difference (€)')
ax.set_title('Agent Performance vs Simple Rule by Hour (Red = Worse, Green = Better)')
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(range(0, 24, 2))

# Monthly comparison
ax = axes[1, 1]
monthly_comparison = comparison_df.groupby('month').agg({
    'agent_cost': 'sum',
    'simple_cost': 'sum',
    'forecast_cost': 'sum'
}).reset_index()

x = np.arange(len(monthly_comparison))
width = 0.25
ax.bar(x - width, monthly_comparison['agent_cost'], width, label='RL Agent')
ax.bar(x, monthly_comparison['simple_cost'], width, label='Simple Rule')
ax.bar(x + width, monthly_comparison['forecast_cost'], width, label='Forecast Rule')
ax.set_xlabel('Month')
ax.set_ylabel('Total Cost (€)')
ax.set_title('Monthly Cost Comparison')
ax.set_xticks(x)
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "temporal_analysis.png", dpi=300, bbox_inches='tight')
print(f"  OK Saved: temporal_analysis.png")

# 2. Battery SOC comparison
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Sample 7 days for visibility
sample_days = comparison_df[comparison_df['datetime'].dt.date.isin(daily_comparison.nlargest(7, 'agent_vs_simple')['date'].values)]

ax = axes[0]
ax.plot(sample_days['datetime'], sample_days['agent_soc'], label='RL Agent', linewidth=1.5, alpha=0.8)
ax.plot(sample_days['datetime'], sample_days['simple_soc'], label='Simple Rule', linewidth=1.5, alpha=0.8)
ax.plot(sample_days['datetime'], sample_days['forecast_soc'], label='Forecast Rule', linewidth=1.5, alpha=0.8)
ax.set_xlabel('DateTime')
ax.set_ylabel('Battery SOC (kWh)')
ax.set_title('Battery State of Charge: 7 Worst Days for Agent')
ax.legend()
ax.grid(True, alpha=0.3)

# SOC distribution
ax = axes[1]
ax.hist([agent_results['soc'], simple_results['soc'], forecast_results['soc']],
        bins=50, label=['RL Agent', 'Simple Rule', 'Forecast Rule'], alpha=0.6)
ax.set_xlabel('Battery SOC (kWh)')
ax.set_ylabel('Frequency')
ax.set_title('Battery SOC Distribution (Full Year)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "battery_soc_analysis.png", dpi=300, bbox_inches='tight')
print(f"  OK Saved: battery_soc_analysis.png")

# ============================================================================
# FAILURE MODE ANALYSIS
# ============================================================================

print("\n[6/6] Identifying failure modes...")

# Find timesteps where agent significantly underperforms
comparison_df['agent_underperformance'] = comparison_df['agent_cost'] - comparison_df['simple_cost']
worst_timesteps = comparison_df.nlargest(100, 'agent_underperformance')

# Merge with test data for full context
worst_timesteps_full = worst_timesteps.merge(
    test_df[['datetime', 'P', 'consumption', 'price_buy', 'price_sell', 'temperature', 'hour']],
    on='datetime'
)

print("\n🔍 CHARACTERISTICS OF WORST 100 TIMESTEPS:")
print(f"  Average PV production: {worst_timesteps_full['P'].mean():.3f} kW")
print(f"  Average consumption: {worst_timesteps_full['consumption'].mean():.3f} kW")
print(f"  Average buy price: €{worst_timesteps_full['price_buy'].mean():.4f}/kWh")
print(f"  Average sell price: €{worst_timesteps_full['price_sell'].mean():.4f}/kWh")
print(f"  Most common hours: {worst_timesteps_full['hour'].value_counts().head(5).to_dict()}")
print(f"  Agent avg SOC: {worst_timesteps['agent_soc'].mean():.2f} kWh")
print(f"  Simple rule avg SOC: {worst_timesteps['simple_soc'].mean():.2f} kWh")

# Save detailed comparison
comparison_df.to_csv(OUTPUT_DIR / "hourly_comparison.csv", index=False)
worst_timesteps_full.to_csv(OUTPUT_DIR / "worst_100_timesteps.csv", index=False)

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
print(f"\n📁 Results saved to: {OUTPUT_DIR}")
print(f"  - policy_comparison_metrics.csv")
print(f"  - hourly_comparison.csv")
print(f"  - worst_100_timesteps.csv")
print(f"  - temporal_analysis.png")
print(f"  - battery_soc_analysis.png")
print("\n" + "="*80)
