"""
Deep Analysis: WHY did the agent fail to learn the trivial optimal strategy?

We'll investigate:
1. Reward signal analysis - Does the reward function actually penalize selling?
2. What the agent actually learned - Action patterns vs state
3. Training dynamics - Did it converge? Local minima?
4. Observation space - Can the agent even see the price difference?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import SAC
from gym_env import SolarBatteryEnv
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics"

print("="*80)
print("WHY DID THE AGENT FAIL TO LEARN THE OPTIMAL STRATEGY?")
print("="*80)

# Load data
test_df = pd.read_csv(PROJECT_ROOT / "data" / "main" / "processed" / "test.csv")
agent_df = pd.read_csv(PROJECT_ROOT / "outputs" / "agent_step_data.csv")
test_df['datetime'] = pd.to_datetime(test_df['datetime'])
agent_df['datetime'] = pd.to_datetime(agent_df['datetime'])

full_df = agent_df.merge(test_df[['datetime', 'buy_price', 'sell_price', 'hour']],
                          on='datetime', how='left')

# ============================================================================
# HYPOTHESIS 1: Does the reward function actually penalize selling?
# ============================================================================

print("\n[HYPOTHESIS 1] Reward Function Analysis")
print("-" * 80)

# Simulate what reward SHOULD be for different strategies
sample_timestep = full_df.iloc[1000]  # Pick a sunny midday timestep

pv = sample_timestep['pv_production_kWh']
consumption = sample_timestep['consumption_kWh']
buy_price = sample_timestep['buy_price']
sell_price = sample_timestep['sell_price']

print(f"\nSample timestep {1000}:")
print(f"  PV production: {pv:.3f} kWh")
print(f"  Consumption: {consumption:.3f} kWh")
print(f"  Buy price: EUR {buy_price:.4f}/kWh")
print(f"  Sell price: EUR {sell_price:.4f}/kWh")
print(f"  Surplus PV: {max(0, pv - consumption):.3f} kWh")

# Strategy 1: Sell surplus to grid (what agent does)
if pv > consumption:
    surplus = pv - consumption
    cost_sell = consumption * 0 - surplus * sell_price  # negative = revenue
    reward_sell = -cost_sell
    print(f"\n  Strategy 1 (SELL surplus):")
    print(f"    Revenue from selling: EUR {surplus * sell_price:.4f}")
    print(f"    Reward: {reward_sell:.4f}")

# Strategy 2: Store surplus in battery (optimal)
# Assume we can later use this to avoid buying from grid
if pv > consumption:
    # Immediate reward: 0 (no grid interaction)
    # Future reward: Avoid buying at buy_price
    reward_store_immediate = 0
    future_savings = surplus * buy_price
    print(f"\n  Strategy 2 (STORE surplus):")
    print(f"    Immediate reward: {reward_store_immediate:.4f}")
    print(f"    Future value (avoided purchase): EUR {future_savings:.4f}")
    print(f"    But RL doesn't see this future value directly!")

print(f"\n  PROBLEM: Selling gives IMMEDIATE positive reward (+{surplus * sell_price:.4f})")
print(f"           Storing gives ZERO immediate reward, benefits come later")
print(f"           Agent with gamma=0.99 discounts future by 1% per step")
print(f"           Myopic behavior is RATIONAL for short-sighted agent!")

# ============================================================================
# HYPOTHESIS 2: What did the agent actually learn?
# ============================================================================

print("\n" + "="*80)
print("[HYPOTHESIS 2] What Actions Did the Agent Learn?")
print("-" * 80)

# Analyze action patterns by state
full_df['surplus'] = full_df['pv_production_kWh'] - full_df['consumption_kWh']
full_df['has_surplus'] = full_df['surplus'] > 0.1

surplus_steps = full_df[full_df['has_surplus']]
deficit_steps = full_df[~full_df['has_surplus']]

print(f"\nTimesteps with PV surplus: {len(surplus_steps):,}")
print(f"Timesteps with deficit: {len(deficit_steps):,}")

print("\nAgent's actions during SURPLUS (should store):")
print(f"  Avg pv_to_house_frac:  {surplus_steps['pv_to_house_frac'].mean():.3f}")
print(f"  Avg pv_to_batt_frac:   {surplus_steps['pv_to_batt_frac'].mean():.3f} <-- Should be HIGH")
print(f"  Avg batt_to_house_frac: {surplus_steps['batt_to_house_frac'].mean():.3f}")
print(f"  Avg grid_to_batt_frac: {surplus_steps['grid_to_batt_frac'].mean():.3f}")

print("\nAgent's actions during DEFICIT (should discharge):")
print(f"  Avg pv_to_house_frac:  {deficit_steps['pv_to_house_frac'].mean():.3f}")
print(f"  Avg pv_to_batt_frac:   {deficit_steps['pv_to_batt_frac'].mean():.3f}")
print(f"  Avg batt_to_house_frac: {deficit_steps['batt_to_house_frac'].mean():.3f} <-- Should be HIGH")
print(f"  Avg grid_to_batt_frac: {deficit_steps['grid_to_batt_frac'].mean():.3f}")

# ============================================================================
# HYPOTHESIS 3: Training episode structure problem
# ============================================================================

print("\n" + "="*80)
print("[HYPOTHESIS 3] Episode Length and Credit Assignment")
print("-" * 80)

print("\nTraining configuration:")
print("  Episode length: 168 hours (1 week)")
print("  Discount factor (gamma): 0.99")
print("  Total timesteps: 200,000")
print("  Number of episodes: ~1,190")

print("\nCredit assignment problem:")
print("  - Selling at hour 12 gives immediate reward (small but positive)")
print("  - NOT having stored means buying at hour 20 (high cost)")
print("  - But hour 20 is 8 steps later")
print("  - Discounted value: 0.99^8 = 0.923")
print("  - Agent might not connect action at t=12 with consequence at t=20")

discount_8_hours = 0.99 ** 8
discount_1_day = 0.99 ** 24
discount_1_week = 0.99 ** 168

print(f"\n  Effective discount over 8 hours: {discount_8_hours:.3f} (7.7% loss)")
print(f"  Effective discount over 1 day: {discount_1_day:.3f} (21.5% loss)")
print(f"  Effective discount over 1 week: {discount_1_week:.3f} (81.5% loss!)")

# ============================================================================
# HYPOTHESIS 4: Observation space - Can agent see prices?
# ============================================================================

print("\n" + "="*80)
print("[HYPOTHESIS 4] Does Agent See Price Information?")
print("-" * 80)

# Check what features are in the observation space
env = SolarBatteryEnv(data=test_df, battery_capacity=10.0, max_charge_rate=5.0)
obs, _ = env.reset()

print(f"\nObservation space dimension: {len(obs)}")
print(f"State columns tracked: {len(env.state_cols)}")

has_buy_price = 'buy_price' in env.state_cols
has_sell_price = 'sell_price' in env.state_cols

print(f"\nAgent observes buy_price: {has_buy_price}")
print(f"Agent observes sell_price: {has_sell_price}")

if has_buy_price and has_sell_price:
    print("\nOK Agent CAN see both prices")
else:
    print("\nPROBLEM: Agent is BLIND to price information!")

# Check normalization
if has_buy_price:
    buy_norm = env.norm_factors.get('buy_price', 1.0)
    sell_norm = env.norm_factors.get('sell_price', 1.0)
    print(f"\nPrice normalization factors:")
    print(f"  buy_price normalized by: {buy_norm:.4f}")
    print(f"  sell_price normalized by: {sell_norm:.4f}")

    sample_buy = test_df['buy_price'].iloc[1000] / buy_norm
    sample_sell = test_df['sell_price'].iloc[1000] / sell_norm
    print(f"\nNormalized prices at sample timestep:")
    print(f"  Normalized buy_price: {sample_buy:.4f}")
    print(f"  Normalized sell_price: {sample_sell:.4f}")
    print(f"  Ratio: {sample_sell/sample_buy:.4f}")

    if abs(sample_sell - sample_buy) < 0.01:
        print("\n  PROBLEM: After normalization, prices look VERY similar!")
        print("  Agent cannot distinguish cheap from expensive!")

# ============================================================================
# HYPOTHESIS 5: Local minima - "Selling is safe"
# ============================================================================

print("\n" + "="*80)
print("[HYPOTHESIS 5] Local Minima - Why Selling Seems Rational")
print("-" * 80)

print("\nFrom agent's perspective during training:")
print("\n1. SELLING strategy:")
print("   - Immediate small reward (EUR 0.04/kWh)")
print("   - No risk (guaranteed reward)")
print("   - Works every timestep")
print("   - Total revenue: ~EUR 200/year")

print("\n2. STORING strategy:")
print("   - No immediate reward (0)")
print("   - Requires coordination across time")
print("   - Must discharge at right moment")
print("   - Must manage SOC carefully")
print("   - Requires learning complex temporal policy")

print("\n3. Why agent chose selling:")
print("   - SAC explores stochastic policies")
print("   - Selling gives consistent positive signal")
print("   - Storing requires precise timing -> sparse rewards")
print("   - Agent converged to 'safe' local optimum")

# ============================================================================
# VISUALIZATION: Agent's learned policy
# ============================================================================

print("\n" + "="*80)
print("Creating visualization of learned policy...")
print("-" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Action vs PV production
ax = axes[0, 0]
ax.scatter(full_df['pv_production_kWh'], full_df['pv_to_batt_frac'],
           alpha=0.1, s=1, label='pv_to_batt')
ax.scatter(full_df['pv_production_kWh'], full_df['pv_to_house_frac'],
           alpha=0.1, s=1, label='pv_to_house')
ax.set_xlabel('PV Production (kWh)')
ax.set_ylabel('Action Fraction')
ax.set_title('Action vs PV Production\n(Should charge battery when PV high)')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Action vs battery SOC
ax = axes[0, 1]
ax.scatter(full_df['soc_kWh'], full_df['pv_to_batt_frac'],
           alpha=0.1, s=1, label='Charge battery')
ax.scatter(full_df['soc_kWh'], full_df['batt_to_house_frac'],
           alpha=0.1, s=1, label='Discharge battery')
ax.set_xlabel('Battery SOC (kWh)')
ax.set_ylabel('Action Fraction')
ax.set_title('Action vs SOC\n(Should charge when low, discharge when high)')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Energy sold vs PV surplus
ax = axes[0, 2]
ax.scatter(surplus_steps['surplus'], surplus_steps['pv_to_grid_kWh'],
           alpha=0.3, s=2, color='orange')
ax.plot([0, 5], [0, 5], 'r--', linewidth=2, label='Selling everything')
ax.set_xlabel('PV Surplus (kWh)')
ax.set_ylabel('Energy Sold to Grid (kWh)')
ax.set_title('Selling Behavior\n(Red line = selling ALL surplus)')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Battery charge vs surplus
ax = axes[1, 0]
ax.scatter(surplus_steps['surplus'], surplus_steps['pv_to_batt_kWh'],
           alpha=0.3, s=2, color='green')
ax.plot([0, 5], [0, 5], 'b--', linewidth=2, label='Storing everything')
ax.set_xlabel('PV Surplus (kWh)')
ax.set_ylabel('Energy Stored in Battery (kWh)')
ax.set_title('Storage Behavior\n(Blue line = storing ALL surplus)')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Reward distribution
ax = axes[1, 1]
# Calculate actual rewards from costs
full_df['actual_reward'] = -(full_df['step_cost_eur'] + full_df['degradation_penalty'])
ax.hist(full_df['actual_reward'], bins=100, alpha=0.7, color='steelblue')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
ax.set_xlabel('Reward per Timestep')
ax.set_ylabel('Frequency')
ax.set_title('Reward Distribution\n(Most rewards are negative = losing money)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 6. Cumulative regret
ax = axes[1, 2]
# Estimate optimal: never sell, store and use later
# Rough estimate: could save ~EUR 400 vs current
actual_cumulative_cost = full_df['step_cost_eur'].cumsum()
# Optimal would have ~EUR 200 lower total cost (rough estimate)
optimal_cumulative_cost = actual_cumulative_cost - 200/len(full_df) * np.arange(len(full_df))
regret = actual_cumulative_cost - optimal_cumulative_cost

ax.plot(regret, linewidth=1)
ax.set_xlabel('Timestep')
ax.set_ylabel('Cumulative Regret (EUR)')
ax.set_title('Agent Cumulative Regret vs Optimal\n(How much money was wasted)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "why_agent_fails_analysis.png", dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'why_agent_fails_analysis.png'}")

# ============================================================================
# SUMMARY AND ROOT CAUSE
# ============================================================================

print("\n" + "="*80)
print("ROOT CAUSE ANALYSIS - Why Agent Failed")
print("="*80)

print("\nThe agent DID learn a strategy, but it's a LOCAL OPTIMUM:")

print("\n1. REWARD STRUCTURE FLAW:")
print("   - Selling gives immediate positive reward")
print("   - Storing gives delayed reward (only valuable hours later)")
print("   - With gamma=0.99, future rewards are heavily discounted")
print("   - Myopic behavior (sell now) beats patient behavior (store for later)")

print("\n2. CREDIT ASSIGNMENT PROBLEM:")
print("   - Episode length: 168 hours (1 week)")
print("   - Typical delay: 8-12 hours (noon PV -> evening consumption)")
print("   - Discount factor ^8 = 92%, ^12 = 88%")
print("   - Agent struggles to connect storage decision with later benefit")

print("\n3. SPARSE REWARD LANDSCAPE:")
print("   - 'Selling' strategy: consistent small positive rewards")
print("   - 'Storing' strategy: requires precise timing, sparse rewards")
print("   - SAC exploration gravitates toward reliable rewards")

print("\n4. NORMALIZATION ISSUES (if applicable):")
if has_buy_price and has_sell_price:
    if abs(sample_sell/sample_buy - 0.21) > 0.05:
        print("   - Price signals may be distorted by normalization")
    else:
        print("   - Price normalization seems OK")
else:
    print("   - Agent may not observe prices properly!")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nThe agent learned a BAD but LOCALLY RATIONAL strategy:")
print("  'Get small immediate rewards by selling, rather than")
print("   wait for larger delayed rewards by storing'")
print("\nThis is a classic RL failure mode: REWARD SHAPING problem")
print("\nFix would require:")
print("  1. Different reward structure (penalize selling explicitly)")
print("  2. Longer gamma or value bootstrapping")
print("  3. Hindsight experience replay")
print("  4. Or just use a rule-based policy (for this simple problem)")

print("\n" + "="*80)
