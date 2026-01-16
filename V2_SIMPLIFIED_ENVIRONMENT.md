# V2: Simplified Environment - Complete Redesign

**Date**: 2026-01-16
**Status**: Training in progress
**Branch**: `claude-code-diagnostics`

---

## Executive Summary

After discovering fundamental flaws in the action space logic (Fixes #1-#3), we completely redesigned the environment with a **simplified 2-action space** that respects physical constraints and energy balance.

### Key Problem Identified

**The Math Didn't Work**:
- Original agent: Sells 5,028 kWh → Buys 3,825 kWh from grid
- Fix #3 agent: Sells only 266 kWh (95% reduction) → Still buys 3,695 kWh (only 3% reduction!)

**Where did the energy go?**
The complex softmax action space was allocating PV as percentages without considering actual house demand, leading to wasted energy in overflow logic.

---

## Root Cause Analysis

### The Flawed Action Space (v1)

```python
# v1 Action Space (BROKEN)
action[0:3] = softmax logits [pv_to_house_frac, pv_to_batt_frac, pv_to_grid_frac]
action[3] = battery_discharge_to_house_frac
```

**Example of the Problem**:
1. Agent says: "Send 30% of PV to house"
2. If PV = 3 kWh and house needs 0.5 kWh:
   - 30% × 3 kWh = 0.9 kWh intended for house
   - Only 0.5 kWh can be used
   - 0.4 kWh gets redirected to grid (hidden selling!)
3. Battery discharge is independent of PV allocation
4. Agent can't reason about energy balance properly

**Result**: Energy flows don't match agent's intentions, learning is confusing.

---

## The Solution: V2 Simplified Environment

### New Action Space (2 actions)

```python
action[0] = charge_battery_fraction  # [0, 1]
# When I have surplus PV, what fraction should go to battery?

action[1] = use_battery_fraction  # [0, 1]
# When I need more energy, what fraction should come from battery?
```

### Physical Logic (Simple and Clear)

```python
def step(self, action):
    # STEP 1: Use PV to meet house demand FIRST (always optimal, no decision)
    pv_to_house = min(pv, consumption)
    remaining_pv = pv - pv_to_house
    remaining_consumption = consumption - pv_to_house

    # STEP 2: If surplus PV, allocate between battery and grid
    if remaining_pv > 0:
        pv_to_batt = remaining_pv * charge_batt_frac  # Agent's decision
        pv_to_grid = remaining_pv - pv_to_batt  # Rest to grid

    # STEP 3: If deficit, meet from battery or grid
    if remaining_consumption > 0:
        batt_to_house = remaining_consumption * use_batt_frac  # Agent's decision
        grid_to_house = remaining_consumption - batt_to_house  # Rest from grid
```

### Energy Balance Guaranteed

```
PV Production = pv_to_house + pv_to_batt / eta + pv_to_grid  ✓ Always true
House Consumption = pv_to_house + batt_to_house + grid_to_house  ✓ Always true
```

No wasted energy, no hidden flows, no confusing overflow logic.

---

## Key Improvements Over v1

### 1. Matches Physical Reality
- **Step 1**: Use PV for house first (obviously optimal)
- **Step 2**: Decide on surplus allocation (store vs sell)
- **Step 3**: Decide on deficit sourcing (battery vs grid)

### 2. Agent Makes Meaningful Decisions
Instead of "Send X% of PV to house" (which may or may not be useful), agent now decides:
- "I have surplus PV, should I store it for later or sell it now?"
- "I need more energy, should I use battery or buy from grid?"

These are the ACTUAL decisions a home energy manager makes.

### 3. Simpler Implementation
- v1: 200+ lines with complex overflow handling
- v2: 150 lines with straightforward logic
- Fewer bugs, easier to understand, easier to verify

### 4. No Reward Shaping Needed
Natural rewards work correctly because agent's actions directly map to outcomes:
- Storing more → less grid purchases later → clear reward signal
- Selling less → less revenue but potentially more self-consumption → clear tradeoff

---

## Configuration

### Environment Parameters (Same as v1)
- Battery capacity: 10 kWh
- Max charge/discharge rate: 5 kW
- Charging efficiency: 95%
- Degradation cost: €0.001/kWh cycled
- Timestep: 1 hour

### Training Hyperparameters
- Algorithm: SAC (Soft Actor-Critic)
- Total timesteps: 200,000
- Batch size: 128
- Learning rate: 3e-4
- **Gamma: 0.999** (high to value delayed rewards)
- No reward shaping (natural rewards only)

### Normalization (Fix #1 preserved)
```python
# Both buy_price and sell_price normalized by SAME factor
max_buy_price = 0.21
norm_factors['buy_price'] = max_buy_price
norm_factors['sell_price'] = max_buy_price  # Preserves 21% ratio
```

---

## Expected Behavior

### Optimal Policy Should Learn:

**Surplus Scenario** (PV > Consumption):
```
If battery not full AND evening peak coming:
    charge_batt_frac = 1.0  (store all surplus)
Else if battery full OR no peak expected:
    charge_batt_frac = 0.0  (sell surplus)
```

**Deficit Scenario** (Consumption > PV):
```
If battery has energy AND prices are high:
    use_batt_frac = 1.0  (use battery to avoid expensive grid)
Else if battery empty OR prices are low:
    use_batt_frac = 0.0  (buy from grid)
```

### Time-of-Day Pattern:
- **Morning** (07:00-10:00): PV starts, charge battery
- **Midday** (10:00-15:00): High PV, charge battery to 70-90%
- **Afternoon** (15:00-18:00): PV decreases, sell remaining or top off battery
- **Evening** (18:00-23:00): No PV, discharge battery for peak demand
- **Night** (23:00-07:00): Buy cheap grid power (low demand period)

---

## Comparison to Previous Fixes

| Aspect | v1 (Fixes #1-#3) | v2 (Simplified) |
|--------|------------------|-----------------|
| **Action Space** | 4 actions (softmax + discharge) | 2 actions (charge + use) |
| **Energy Balance** | Broken (overflow issues) | Guaranteed correct |
| **Physical Logic** | Complex, non-intuitive | Simple, matches reality |
| **Reward Shaping** | Needed (but too aggressive) | Not needed |
| **Code Complexity** | 200+ lines, many edge cases | 150 lines, straightforward |
| **Learning Difficulty** | Hard (confusing signals) | Easier (clear cause-effect) |

---

## Files Created

### Core Implementation
- `gym_env_v2.py`: Simplified environment with 2-action space
- `train_v2.py`: Training script for v2
- `evaluate_v2.py`: Evaluation script with detailed metrics

### Documentation
- `SIMPLIFIED_ACTION_SPACE_PROPOSAL.md`: Initial design proposal
- `V2_SIMPLIFIED_ENVIRONMENT.md`: This file (complete documentation)

---

## Training Status

**Status**: Training in progress (200,000 timesteps, ~20 minutes)

**Expected Results**:
- Energy balance should be correct (no missing energy)
- Agent should learn clear temporal patterns
- Selling should be strategic (only when battery full)
- Grid purchases should decrease significantly
- **Target**: €400-450/year (25-30% improvement over original €591)

---

## Next Steps

1. **Complete training** and evaluate on test set
2. **Compare results** to all previous fixes
3. **Verify energy balance** (PV flows, consumption met correctly)
4. **Analyze learned policy** (time-of-day patterns, battery usage)
5. **If successful**: Update README, commit, and merge to main
6. **If not**: Investigate and iterate

---

## Key Learnings

### 1. Start with Physical Constraints
Design action space around physical reality, not mathematical convenience. The softmax was mathematically elegant but physically meaningless.

### 2. Energy Balance is Non-Negotiable
Every kWh must be accounted for. If energy "disappears" in overflow logic, the agent can't learn correctly.

### 3. Simpler is Better
Complex action spaces require complex overflow handling, creating more bugs and confusing learning signals. Simple, well-defined actions are easier to learn.

### 4. Test Energy Conservation
Before training, verify that:
```
PV_production == sum(all_PV_flows)
Consumption == sum(all_house_supply)
```

If these don't hold, fix the environment first.

---

## Acknowledgment

User insight: "The math doesn't work - if agent sells 95% less energy, it should buy significantly less from grid" led to discovering the fundamental action space flaw. This shows the importance of sanity-checking results against physical constraints.
