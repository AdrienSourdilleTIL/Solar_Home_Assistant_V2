# Fix #3 Results and Analysis

**Date**: 2026-01-15
**Status**: Completed - Requires Adjustment

---

## Executive Summary

Fix #3 successfully taught the agent to avoid excessive selling (3.5% vs 65.5% in original), but at a **net cost of €171 MORE per year**. This reveals a fundamental insight about the French electricity market.

**Key Finding**: The reward shaping weight (0.5) was too aggressive, causing the agent to avoid selling even when it was economically beneficial.

---

## What We Fixed

### 1. Normalization (from Fix #1) ✓
Both buy_price and sell_price normalized by same factor (0.21), preserving their 21% ratio.

### 2. Softmax Action Space (from Fix #2) ✓
Agent now has explicit control over PV allocation to house, battery, and grid.

### 3. Higher Gamma (0.999) ✓
Better values future rewards over 8-hour horizons.

### 4. Reward Shaping (NEW) ⚠️
```python
opportunity_cost = energy_sold * (buy_price - sell_price)
reward = base_reward - 0.5 * opportunity_cost - soc_penalty
```

### 5. CRITICAL Bug Fix ✓
Fixed environment bug where excess PV to house was wasted instead of redirected to grid.

---

## Results

| Metric | Original | Fix #3 | Change |
|--------|----------|--------|--------|
| **Total Cost** | €543 | **€714** | **+€171 (+31%)** ✗ WORSE |
| Energy Sold | 5,028 kWh (65%) | 266 kWh (3.5%) | -95% ✓ |
| Grid Bought | 3,825 kWh | 3,695 kWh | -130 kWh ✓ |
| Avg Battery SOC | 0.63 kWh (6%) | 1.00 kWh (10%) | +59% ✓ |
| Battery Degradation | €2.96 | €12.93 | +€9.97 |

### Cost Breakdown

**Original Agent**:
- Revenue from selling: +€201
- Cost of grid purchases: -€741
- Battery degradation: -€3
- **Net cost: €543**

**Fix #3 Agent**:
- Revenue from selling: +€11 (lost €190 revenue!)
- Cost of grid purchases: -€712 (saved €29)
- Battery degradation: -€13 (lost €10)
- **Net cost: €714**

**Delta**: €29 saved - €190 lost - €10 degradation = **-€171 WORSE**

---

## What We Learned

### Critical Insight: Selling Has Net Value!

Even though selling at €0.04/kWh is terrible compared to avoiding purchases at €0.19/kWh, **it's still better than wasting PV energy**:

- Selling 1 kWh: +€0.04 revenue
- Wasting 1 kWh: €0.00 revenue
- **Selling > Wasting!**

The reward shaping penalty made the agent treat selling as if it were equivalent to wasting energy. This was too aggressive.

### The Optimal Strategy

The agent should:
1. **First priority**: Use PV to meet house demand directly (saves €0.19/kWh)
2. **Second priority**: Store excess in battery when:
   - Battery has capacity
   - Evening peak hours are coming (higher value)
3. **Third priority**: Sell to grid when:
   - Battery is full
   - House demand is fully met
   - Selling €0.04 > wasting €0.00

The original agent (despite bugs) was closer to optimal because it sold surplus PV and earned revenue, even if it sold too much.

---

## Why the Reward Shaping Backfired

**Opportunity Cost Penalty**:
```python
penalty = 0.5 * energy_sold * (buy_price - sell_price)
# Example: Selling 1 kWh at €0.04 when buy=€0.19
# penalty = 0.5 * 1 * (0.19 - 0.04) = €0.075
```

**The Problem**:
- Immediate selling reward: +€0.04
- Opportunity cost penalty: -€0.075
- Net immediate reward: -€0.035 (negative!)

This made selling IMMEDIATELY feel worse than the true long-term cost. The agent learned "never sell," which is suboptimal.

**What We Should Have Done**:
- Much smaller weight (0.1-0.2) to nudge behavior without reversing the sign
- Or: Only penalize selling when battery has capacity (context-aware shaping)

---

## Comparison to Fix #1

Remember Fix #1 (€512/year)?

| Metric | Fix #1 | Fix #3 | Better? |
|--------|--------|--------|---------|
| Total Cost | €512 | €714 | Fix #1 ✓ |
| Energy Sold | 72% | 3.5% | Fix #3 ✓ |
| Avg SOC | 9% | 10% | Fix #3 ✓ |

Fix #1 is still the best performer because:
- It learned to use battery more (9% SOC vs 6% original)
- It reduced grid purchases (3,586 vs 3,825 kWh)
- It didn't avoid selling entirely (still earned revenue)

---

## Next Steps: Fix #3b Proposal

### Option 1: Reduce Reward Shaping Weight
```python
# Try weight = 0.1 instead of 0.5
reward = base_reward - 0.1 * opportunity_cost - soc_penalty
```

Expected: Agent learns to store more but still sells when beneficial.

### Option 2: Context-Aware Reward Shaping
```python
# Only penalize selling when battery has capacity
if soc_ratio < 0.7:
    opportunity_cost = energy_sold * (buy_price - sell_price)
    penalty = 0.3 * opportunity_cost
else:
    penalty = 0.0  # Battery full, selling is OK

reward = base_reward - penalty - soc_penalty
```

Expected: Agent learns to prioritize battery storage but sells when battery is full.

### Option 3: Remove Reward Shaping Entirely
Use Fix #1 normalization + Fix #2 softmax action space, but NO reward shaping.

Expected: Agent explores naturally without biased signal.

### Option 4: Accept Fix #1 and Move On
Fix #1 achieved 13% improvement (€512 vs €591). Not perfect, but measurably better.

Focus efforts on re-scoping project (add controllable loads like EV charging).

---

## Recommendation

I recommend **Option 3** or **Option 4**:

**Option 3**: Test if the softmax action space alone (without reward shaping) allows the agent to learn better allocation. The explicit control over selling might be enough.

**Option 4**: Accept that Fix #1 is good enough given the fundamental problem constraints. The 13% improvement is real. Re-scope the project to add complexity (EV charging, dynamic pricing) where RL can truly shine.

---

## Technical Learnings

### 1. Reward Shaping is Dangerous
Even theoretically sound reward shaping can backfire if weights are wrong. Always validate that shaped rewards maintain the correct relative values.

### 2. Environment Bugs Have Major Impact
The PV overflow bug (wasting excess PV to house) would have made any agent perform poorly. Always verify physical constraints are correctly modeled.

### 3. Market Economics Matter
Understanding the actual economics is critical. We assumed selling was "bad" (opportunity cost), but forgot it has absolute value (revenue > zero).

### 4. Simpler is Often Better
Fix #1 (just normalization + gamma) outperformed Fix #3 (normalization + softmax + reward shaping). Adding complexity doesn't always help.

---

## Files Modified

### Core Changes
- `gym_env.py`:
  - Fixed price normalization (same factor for both prices)
  - Added softmax action space
  - Added reward shaping (too aggressive)
  - Fixed PV overflow bug (critical)

### Analysis Scripts
- `evaluate_new_agent.py`: Evaluation with step-by-step logging
- `compare_old_vs_new.py`: Side-by-side comparison

### Documentation
- `TRAINING_EXPERIMENTS.md`: Updated with Fix #3 results
- `FIX_3_PROPOSAL.md`: Original proposal
- `FIX_3_RESULTS.md`: This file

---

## Conclusion

Fix #3 successfully demonstrated that:
1. ✓ Agent CAN learn to avoid excessive selling
2. ✓ Softmax action space provides explicit control
3. ✓ Environment bugs were identified and fixed
4. ✗ Reward shaping weight was too aggressive
5. ✗ Net cost increased due to lost revenue

**Fix #1 remains the best performing solution at €512/year (13% improvement).**

Next decision: Try Fix #3b with reduced shaping, or accept Fix #1 and re-scope the project?
