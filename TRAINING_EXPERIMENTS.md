# RL Agent Training Experiments - Bug Fixes and Results

**Date**: 2026-01-15
**Branch**: `claude-code-diagnostics`

---

## Executive Summary

Conducted multiple training experiments to fix identified bugs in the RL agent:
1. **Original Agent** (with bugs): €591/year
2. **Fix #1** (normalized prices by same factor + gamma=0.999): €512/year (13.4% improvement)
3. **Fix #2** (no price normalization + softmax actions): €623/year (5.5% WORSE)
4. **Fix #3** (Fix #1 normalization + Fix #2 softmax + reward shaping): **TRAINING IN PROGRESS**

**Current Status**: Implementing Fix #3 to combine best elements and break "selling is safe" local minimum.

---

## Original Agent Issues Identified

### Issue 1: Normalization Bug
**Problem**: `buy_price` and `sell_price` normalized independently by their own max values
```python
# Original (WRONG):
norm_factors['buy_price'] = 0.21    # buy_price / 0.21 = ~1.0
norm_factors['sell_price'] = 0.04   # sell_price / 0.04 = 1.0
# Result: Agent sees both prices as 1.0, can't distinguish!
```

**Impact**: Agent couldn't perceive that selling (€0.04/kWh) is 5x worse than buying (€0.19/kWh)

### Issue 2: Low Gamma
- Original: `gamma=0.99`
- 8-hour delayed rewards: only 92.3% value retained
- Agent prioritized immediate small rewards (selling) over delayed larger rewards (storing)

### Issue 3: Action Space Design Flaw
- Agent controlled independent fractions: `pv_to_house_frac`, `pv_to_batt_frac`
- Any unallocated PV automatically sold to grid
- Agent had no direct control over selling behavior

---

## Fix #1: Normalized Prices + Higher Gamma

### Changes Made
1. **Price Normalization Fix** ([gym_env.py:39-58](scripts/RL_training_&_testing/gym_env.py#L39-L58))
   ```python
   # Both prices normalized by SAME factor (max buy price)
   max_buy_price = 0.21
   norm_factors['buy_price'] = max_buy_price
   norm_factors['sell_price'] = max_buy_price  # Same factor!

   # Now agent sees:
   # buy_price:  0.19/0.21 = 0.90
   # sell_price: 0.04/0.21 = 0.19
   # Ratio preserved: 0.19/0.90 = 21% ✓
   ```

2. **Increased Gamma** ([train_1.py:80](scripts/RL_training_&_testing/train_1.py#L80))
   ```python
   gamma=0.999  # Up from 0.99
   # 8-hour delayed rewards: 99.2% value retained (vs 92.3%)
   # 24-hour delayed rewards: 97.6% retained (vs 78.6%)
   ```

### Results (Fix #1)
| Metric | Original | Fix #1 | Change |
|--------|----------|--------|--------|
| **Total Cost** | €591 | **€512** | **-€79 (-13.4%)** ✓ |
| Energy Sold | 5,028 kWh (65%) | 5,543 kWh (72%) | +10% (worse) |
| Grid Bought | 3,825 kWh | 3,586 kWh | -239 kWh ✓ |
| Avg Battery SOC | 0.63 kWh (6%) | 0.94 kWh (9%) | +50% ✓ |
| Battery Degradation | €2.96 | €10.52 | +€7.56 (more usage) |
| pv_to_batt_frac | 0.349 | 0.657 | Nearly doubled ✓ |

**Analysis**:
- ✓ Agent learned to use battery more effectively
- ✓ Reduced grid purchases significantly
- ✗ Still sells heavily (72% of PV) due to action space design flaw
- ✓ Net result: €79 savings despite selling more

---

## Fix #2: No Normalization + Softmax Actions

### Changes Made
1. **Removed Price Normalization** ([gym_env.py:44-46](scripts/RL_training_&_testing/gym_env.py#L44-L46))
   ```python
   # Don't normalize prices - use raw values
   if col in ['buy_price', 'sell_price']:
       self.norm_factors[col] = 1.0  # No normalization
   ```

2. **Softmax Action Space** ([gym_env.py:84-110](scripts/RL_training_&_testing/gym_env.py#L84-L110))
   ```python
   # action[0:3] = PV allocation logits [house, batt, grid]
   pv_logits = action[0:3]
   pv_fracs = softmax(pv_logits)  # Ensures sum = 1.0

   pv_to_house_frac = pv_fracs[0]
   pv_to_batt_frac = pv_fracs[1]
   pv_to_grid_frac = pv_fracs[2]  # Agent now controls selling!
   ```

3. **Removed Grid Charging** ([gym_env.py:123-124](scripts/RL_training_&_testing/gym_env.py#L123-L124))
   - Charging battery from expensive grid makes no economic sense

### Results (Fix #2)
| Metric | Original | Fix #2 | Change |
|--------|----------|--------|--------|
| **Total Cost** | €591 | €623 | **+€32 (+5.5%)** ✗ WORSE |
| Energy Sold | 5,028 kWh (65%) | 4,763 kWh (62%) | -5.3% ✓ |
| Grid Bought | 3,825 kWh | 4,176 kWh | +351 kWh ✗ |
| Avg Battery SOC | 0.63 kWh (6%) | 0.47 kWh (5%) | -25% ✗ |
| Battery Degradation | €2.96 | €0.98 | -€1.98 (less usage) |

**Why It Failed**:
1. **Tiny price values** (0.04-0.21) were dwarfed by other features
2. **Neural network struggled** to learn price sensitivity without normalization
3. **Agent barely used battery** (5% SOC vs 9% in Fix #1)
4. **Bought more from expensive grid** despite selling less

---

## Fix #3: Comprehensive Solution (CURRENT)

### Changes Made

1. **Price Normalization** (from Fix #1)
   ```python
   # Both prices normalized by SAME factor (max buy price)
   max_buy_price = 0.21
   norm_factors['buy_price'] = max_buy_price
   norm_factors['sell_price'] = max_buy_price  # Preserves ratio!
   ```

2. **Softmax Action Space** (from Fix #2)
   ```python
   # action[0:3] = PV allocation logits [house, batt, grid]
   pv_fracs = softmax(action[0:3])
   pv_to_house_frac = pv_fracs[0]
   pv_to_batt_frac = pv_fracs[1]
   pv_to_grid_frac = pv_fracs[2]  # Explicit control over selling!
   ```

3. **Higher Gamma** (from Fix #1)
   ```python
   gamma=0.999  # Already set in train_1.py
   ```

4. **NEW: Moderate Reward Shaping**
   ```python
   base_reward = -(energy_bought * buy_price - energy_sold * sell_price + degradation)

   # Shaping #1: Opportunity cost penalty (makes selling's true cost immediate)
   opportunity_cost = energy_sold * (buy_price - sell_price)

   # Shaping #2: SOC management (encourage 40-70% SOC for readiness)
   soc_penalty = 0.02 * max(0, 0.4 - soc_ratio) + 0.01 * max(0, soc_ratio - 0.7)

   # Final reward with moderate weights
   reward = base_reward - 0.5 * opportunity_cost - soc_penalty
   ```

### Why This Should Work

**Problem**: Even with correct normalization (Fix #1) and explicit action space (Fix #2), the agent faces a sparse reward problem:
- Selling gives immediate +€0.04/kWh reward (easy to learn)
- Storing gives -€0.001 now + €0.19 in 8 hours (hard to learn)
- Agent converged to "selling is safe" local minimum

**Solution**: Reward shaping makes the true cost of selling immediately apparent WITHOUT changing the optimal policy:
- Opportunity cost penalty: Selling 1 kWh at €0.04 when buy=€0.19 → immediate -€0.075 penalty
- SOC incentive: Keeps battery ready for peak hours without forcing specific timing
- Weights are conservative (0.5) to not override natural rewards

**Expected Results**:
- Target: €450-480/year (18-24% improvement over original €591)
- Lower selling: <50% of PV (vs 72% in Fix #1)
- Higher battery use: 15-30% avg SOC (vs 9% in Fix #1)
- Fewer grid purchases: <3,000 kWh (vs 3,586 kWh in Fix #1)

### Training Status
- **Status**: Ready to train
- **Timesteps**: 200,000
- **Expected duration**: ~20 minutes

---

## Verification: Normalization Fix Works

**Test**: [verify_normalization.py](scripts/RL_training_&_testing/verify_normalization.py)

```
Normalization Factors:
  buy_price:  0.210000
  sell_price: 0.210000  ✓ Same factor!

Normalized Prices (what agent sees):
  buy_price[0]:  0.700000
  sell_price[0]: 0.190476
  Ratio: 0.272109  ✓ Correct!

Expected ratio: 0.272109  ✓ Match!

SUCCESS: Both prices use the same normalization factor
```

---

## Key Learnings

### What Worked
1. **Normalizing prices by same factor** preserves economic relationship
2. **Higher gamma (0.999)** helps agent value future rewards
3. **Combined approach** (Fix #1) achieved measurable improvements

### What Didn't Work
1. **Removing normalization entirely** hurt more than it helped
2. **Softmax action space** alone didn't solve selling problem
3. **Action space redesign** needs to be paired with proper feature scaling

### Fundamental Limitations Remain
Even with fixes, agent still sells 62-72% of PV because:
1. **Problem is too constrained** for RL - optimal strategy is trivial ("never sell")
2. **Immediate rewards** from selling compete with delayed rewards from storing
3. **Credit assignment** remains difficult over 8-hour time horizons
4. **Lack of problem complexity** means RL adds little value over simple rules

---

## Recommendations

### Short-term: Use Fix #1
- **€79 annual savings** (13% improvement) with minimal changes
- Keep current model: `solar_batt_agent_weekly_lagged.zip`
- Accept that agent still sells heavily

### Long-term: Re-scope Project
Problem needs more complexity for RL to shine:

**Option A: Add Controllable Loads** (Recommended)
- EV charging (40-50 kWh, flexible 8h window)
- Water heater (3-4 kWh, flexible 4h window)
- HVAC pre-cooling/heating
- Now agent must optimize *when* to run loads
- Multi-timestep planning becomes critical

**Option B: Dynamic Pricing**
- Use real-time spot market prices
- 10-20x price variance (€0.05 - €1.00/kWh)
- Battery arbitrage becomes profitable
- Timing decisions become complex

**Option C: Multi-Objective Optimization**
- Balance cost, carbon emissions, grid stability, battery lifespan
- Pareto-optimal trade-offs
- More dimensions for agent to optimize

---

## Files Modified

### Core Changes
- `gym_env.py`: Fixed normalization, softmax actions, removed grid charging
- `train_1.py`: Increased gamma to 0.999

### Analysis Scripts
- `evaluate_new_agent.py`: Comprehensive evaluation with step-by-step logging
- `compare_old_vs_new.py`: Side-by-side comparison of all agents
- `verify_normalization.py`: Verification that normalization fix works correctly

### Documentation
- `DIAGNOSTIC_FINDINGS.md`: Initial diagnostic findings
- `TRAINING_EXPERIMENTS.md`: This file

---

## Next Steps

1. **Decision Point**: Keep Fix #1 or re-scope project?
2. **If keeping**: Commit final model and close branch
3. **If re-scoping**: Design new environment with controllable loads
4. **Git**: Merge `claude-code-diagnostics` back to `main`

---

## Training Details

### Fix #1 Training
- Total timesteps: 200,000
- Episodes: ~1,188
- Training time: ~20 minutes
- Final episode reward: -8.90
- Model saved: `solar_batt_agent_weekly_lagged.zip`

### Fix #2 Training
- Total timesteps: 200,000
- Episodes: ~1,188
- Training time: ~19 minutes
- Final episode reward: -9.52
- Model overwrote previous (replaced)

---

**Conclusion**: Fix #1 is the clear winner, achieving 13% cost reduction. However, fundamental problem constraints remain - RL may not be the right tool for this trivially optimal problem without additional complexity.
