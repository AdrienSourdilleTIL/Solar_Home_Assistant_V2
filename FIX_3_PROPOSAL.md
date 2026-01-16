# Fix #3 Proposal: Comprehensive Solution

**Date**: 2026-01-15
**Status**: Proposed (awaiting approval)

---

## Executive Summary

Fix #1 showed 13% improvement but still sells 72% of PV. Fix #2 failed due to removing normalization.

**Fix #3 Strategy**: Combine the best elements from both fixes while addressing the core problems:
1. Use Fix #1's price normalization (both prices by same factor)
2. Keep Fix #2's softmax action space (explicit control over selling)
3. Keep gamma=0.999
4. **NEW**: Add intelligent reward shaping to break the "selling is safe" local minimum

---

## Root Cause Analysis: Why Agents Still Fail

### Problem 1: Normalization ✓ SOLVED in Fix #1
- Fix #1 correctly normalizes both prices by same factor (0.21)
- Ratio preserved: sell/buy = 0.19/0.90 = 21%
- Agent CAN perceive economic difference

### Problem 2: Action Space Design ⚠️ PARTIALLY ADDRESSED in Fix #2
**Original Action Space** (Fix #1):
```python
action[0] = pv_to_house_frac  # Fraction of PV to house
action[1] = pv_to_batt_frac   # Fraction of PV to battery
action[2] = batt_to_house_frac # Fraction of battery discharge to house
action[3] = grid_to_batt_frac  # Charging from grid (removed)

# Residual selling: pv_to_grid = pv * (1 - action[0] - action[1])
# Problem: Agent has NO explicit control over selling!
```

**Fix #2 Action Space** (Softmax):
```python
action[0:3] = PV allocation logits [house, batt, grid]
softmax(logits) → [pv_to_house_frac, pv_to_batt_frac, pv_to_grid_frac]
action[3] = batt_to_house_frac

# Now agent explicitly controls selling!
# But: Fix #2 failed because prices weren't normalized (tiny values 0.04-0.21)
```

**Why Softmax + No Normalization Failed**:
- Neural networks need features in similar ranges
- Prices (0.04-0.21) dwarfed by other features (consumption ~1-5 kWh, SOC 0-10 kWh)
- Network couldn't learn price sensitivity
- Result: Agent ignored prices, barely used battery (5% SOC), bought more from grid

### Problem 3: Reward Structure ⚠️ NOT YET ADDRESSED

**Current Reward** (all fixes use this):
```python
reward = -(energy_bought * buy_price - energy_sold * sell_price + degradation_cost)
```

**Why This Creates "Selling is Safe" Local Minimum**:

At timestep with 1 kWh PV surplus:

**Strategy A: SELL NOW**
- Immediate reward: +€0.04 (sell price)
- Risk: None (guaranteed)
- Learning difficulty: Easy (direct feedback)

**Strategy B: STORE FOR LATER**
- Immediate reward: -€0.001 (degradation cost only)
- Future reward: +€0.19 (saved purchase) 8 hours later
- Discounted value: €0.19 × 0.999^8 = €0.188
- Net benefit: €0.188 - €0.001 = €0.187 (4.7x better!)
- Risk: Must discharge at right time (credit assignment problem)
- Learning difficulty: HARD (8-hour delayed feedback)

**The Problem**: Even with gamma=0.999, the immediate reward from selling provides a clear, consistent training signal. Storing requires the agent to:
1. Learn to coordinate charging and discharging actions 8 hours apart
2. Trust that future rewards will materialize
3. Explore long enough to discover the delayed reward

This is a **sparse reward problem** - the agent must take many steps with no positive feedback before seeing the benefit.

---

## Proposed Fix #3: Comprehensive Solution

### Changes to Make

#### 1. Price Normalization (from Fix #1) ✓
```python
# Both prices normalized by SAME factor
max_buy_price = self.data['buy_price'].abs().max() + 1e-8  # ~0.21
norm_factors['buy_price'] = max_buy_price
norm_factors['sell_price'] = max_buy_price  # Same factor!
```

#### 2. Softmax Action Space (from Fix #2) ✓
```python
# action[0:3] = PV allocation logits [house, batt, grid]
pv_logits = action[0:3]
pv_fracs = softmax(pv_logits)  # Ensures sum = 1.0

pv_to_house_frac = pv_fracs[0]
pv_to_batt_frac = pv_fracs[1]
pv_to_grid_frac = pv_fracs[2]  # Explicit control!

# action[3] = battery discharge to house fraction
batt_to_house_frac = clip(action[3], 0, 1)
```

#### 3. Higher Gamma (from Fix #1) ✓
```python
gamma=0.999  # Up from 0.99
```

#### 4. NEW: Intelligent Reward Shaping

**Why Reward Shaping is Justified Here**:

You were right to be skeptical! Reward shaping should be used carefully. However, it's justified when:
1. The natural reward signal is too sparse
2. The shaped reward accelerates learning of the optimal policy
3. The shaped reward doesn't change the optimal policy itself

**Key Insight**: The problem isn't that the agent CAN'T learn the optimal policy - it's that the learning signal is too weak due to temporal sparsity. We need to make the immediate consequences more visible WITHOUT changing what's optimal.

**Proposed Reward Shaping**:

```python
# Base reward (unchanged)
base_reward = -(energy_bought * buy_price - energy_sold * sell_price + degradation_cost)

# Shaping #1: Opportunity cost penalty for selling
# This makes the immediate cost of selling more apparent
opportunity_cost = energy_sold * (buy_price - sell_price)
# Example: Selling 1 kWh at €0.04 when buy price is €0.19
#          opportunity_cost = 1 * (0.19 - 0.04) = €0.15
# This represents the "forgone savings" from not storing

# Shaping #2: SOC incentive (keep battery ready for peak hours)
# Encourage maintaining medium SOC (40-70%)
target_soc = 0.55  # 55% of capacity
soc_ratio = self.soc / self.battery_capacity
soc_penalty = 0.0
if soc_ratio < 0.4:
    soc_penalty = 0.02 * (0.4 - soc_ratio)  # Penalty for too low
elif soc_ratio > 0.7:
    soc_penalty = 0.01 * (soc_ratio - 0.7)  # Small penalty for too high

# Final shaped reward
shaped_reward = base_reward - 0.5 * opportunity_cost - soc_penalty
```

**Why This Works**:
1. **Opportunity cost penalty**: Makes selling immediately feel worse (closer to its true long-term cost)
2. **SOC incentive**: Encourages keeping battery ready without forcing specific charge/discharge timing
3. **Weights are tunable**: 0.5 factor on opportunity cost is conservative (doesn't fully eliminate immediate selling reward)
4. **Doesn't change optimal policy**: Storing is STILL better than selling (optimal policy unchanged)

**Alternative: Minimal Reward Shaping**

If you prefer a more conservative approach:

```python
# Only add opportunity cost, no SOC shaping
base_reward = -(energy_bought * buy_price - energy_sold * sell_price + degradation_cost)
opportunity_cost = energy_sold * (buy_price - sell_price)
shaped_reward = base_reward - 0.3 * opportunity_cost  # Even more conservative weight
```

This version:
- Only addresses the sparse reward problem
- Doesn't try to guide SOC management
- Uses a smaller weight (0.3 instead of 0.5)

---

## Implementation Plan

### Step 1: Modify gym_env.py

**Normalization** (lines 39-58):
```python
# Compute normalization factors
self.norm_factors = {}
max_buy_price = None

for col in self.state_cols:
    if col == 'buy_price':
        max_buy_price = self.data[col].abs().max() + 1e-8
        self.norm_factors[col] = max_buy_price
    elif col == 'sell_price':
        # Use same factor as buy_price to preserve ratio
        if max_buy_price is None:
            max_buy_price = self.data['buy_price'].abs().max() + 1e-8
        self.norm_factors[col] = max_buy_price
    else:
        # All other features: normalize independently
        self.norm_factors[col] = self.data[col].abs().max() + 1e-8

self.norm_factors["soc"] = battery_capacity
```

**Observation Space** (line 55):
```python
# Prices are normalized to ~0-1 range
self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32)
```

**Action Space** (lines 56-59 - keep as is from Fix #2):
```python
# Action space: 4 continuous actions
# action[0:3] = PV allocation logits (will be softmaxed to sum to 1.0)
# action[3] = battery discharge to house fraction (0-1)
self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)
```

**Step Function** (lines 92-159):
Keep the softmax action space from Fix #2, but modify reward:

```python
# Apply softmax to PV allocation logits (keep from Fix #2)
pv_logits = action[0:3]
pv_exp = np.exp(pv_logits - np.max(pv_logits))
pv_fracs = pv_exp / np.sum(pv_exp)

pv_to_house_frac = float(pv_fracs[0])
pv_to_batt_frac = float(pv_fracs[1])
pv_to_grid_frac = float(pv_fracs[2])

batt_to_house_frac = float(np.clip(action[3], 0, 1))

# ... (keep all energy flow logic from Fix #2)

# NEW REWARD CALCULATION:
gross_cost = energy_from_grid * price_buy - energy_to_grid * price_sell
degradation_penalty = self.degradation_cost * (batt_charge_energy + battery_used)
base_reward = -(gross_cost + degradation_penalty)

# Reward shaping: opportunity cost of selling
opportunity_cost = energy_to_grid * (price_buy - price_sell)

# Reward shaping: SOC management incentive
soc_ratio = self.soc / self.battery_capacity
soc_penalty = 0.0
if soc_ratio < 0.4:
    soc_penalty = 0.02 * (0.4 - soc_ratio)
elif soc_ratio > 0.7:
    soc_penalty = 0.01 * (soc_ratio - 0.7)

# Final shaped reward
reward = base_reward - 0.5 * opportunity_cost - soc_penalty
```

### Step 2: Keep train_1.py as is
- gamma=0.999 (already set)
- All other hyperparameters unchanged

### Step 3: Training & Evaluation
```bash
cd scripts/RL_training_&_testing
python train_1.py  # Train with Fix #3
python evaluate_new_agent.py  # Generate detailed metrics
python compare_old_vs_new.py  # Compare against Fix #1 baseline
```

---

## Expected Results

**Hypothesis**: Fix #3 should achieve:
- ✓ Better than Fix #1 (€512/year) due to explicit selling control + reward shaping
- ✓ Much better than Fix #2 (€623/year) due to proper normalization
- ✓ Lower selling percentage (target: <50% vs 72% in Fix #1)
- ✓ Higher battery utilization (target: 15-30% SOC vs 9% in Fix #1)
- ✓ Fewer grid purchases (target: <3,000 kWh vs 3,586 kWh in Fix #1)

**Target Performance**: €450-480/year (18-24% improvement over original)

---

## Alternative: Even More Aggressive Reward Shaping

If Fix #3 still sells too much, we can try Fix #3b with stronger shaping:

```python
# More aggressive opportunity cost weight
shaped_reward = base_reward - 0.8 * opportunity_cost - soc_penalty

# Or: Add explicit selling penalty
sell_penalty = 0.05 * energy_to_grid  # Flat penalty per kWh sold
shaped_reward = base_reward - 0.5 * opportunity_cost - soc_penalty - sell_penalty
```

---

## Questions for User

1. **Reward shaping preference**:
   - Option A: Conservative (0.3 × opportunity_cost only)
   - Option B: Moderate (0.5 × opportunity_cost + SOC shaping) [Recommended]
   - Option C: Aggressive (0.8 × opportunity_cost + SOC shaping)

2. **Should we train Fix #3 and see results before deciding on more changes?**

3. **If Fix #3 still underperforms, are you open to re-scoping the project** (adding controllable loads like EV charging)?

---

## Summary

Fix #3 combines:
- ✓ Fix #1's price normalization (preserves economic relationships)
- ✓ Fix #2's softmax action space (explicit selling control)
- ✓ Gamma=0.999 (values future rewards properly)
- ✓ NEW: Intelligent reward shaping (breaks "selling is safe" local minimum)

This addresses all three root causes:
1. Normalization bug → FIXED
2. Action space design → IMPROVED
3. Sparse reward signal → ADDRESSED with shaping

The reward shaping is theoretically justified because it doesn't change the optimal policy - it just makes the immediate feedback better reflect long-term consequences.
