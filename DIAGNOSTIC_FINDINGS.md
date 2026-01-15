# Diagnostic Analysis - RL Agent Performance Issues

**Date**: 2026-01-15
**Branch**: `claude-code-diagnostics`
**Analysis Script**: `scripts/RL_training_&_testing/simple_diagnostic.py`

---

## Executive Summary

The RL agent **significantly underperforms** due to a fundamental problem: it sells massive amounts of electricity to the grid at terrible prices (€0.04/kWh) while buying back from the grid at expensive prices (€0.19/kWh). The economics of the French residential solar market make the optimal strategy trivial, rendering advanced RL unnecessary.

---

## Key Findings

### 1. Catastrophic Selling Behavior
- **5,028 kWh sold to grid** (54% of total PV production)
- Sell price: **€0.04/kWh** (only 21% of buy price)
- Revenue from selling: **€201**
- **Could have stored 4,169 kWh** instead (battery had room)

### 2. Battery Severely Underutilized
- Average SOC: **0.63 kWh** (only 6% of 10 kWh capacity)
- Battery stays nearly empty all year
- The agent is NOT using the battery for arbitrage

### 3. Poor Grid Economics
- **3,825 kWh bought from grid** at €0.19/kWh
- Total grid purchase cost: **€723**
- Self-sufficiency: only **33%**
- Net loss from buy-sell arbitrage: **€522**

### 4. Temporal Patterns
**Worst selling hours** (hours 10-14):
- Peak solar production time
- Agent dumps 730-757 kWh to grid per hour
- Should be charging battery instead

**Worst days**:
- December 17-18, 2023: €4.85 and €4.54 daily costs
- Low PV production + high consumption + selling behavior

---

## Root Cause Analysis

### Why the Agent Fails

The problem is **economic constraints**, not RL algorithm failure:

1. **Sell/Buy price ratio is 21%** - selling is almost never rational
2. **Optimal strategy is trivial**:
   - Never sell to grid (price too low)
   - Store all excess PV
   - Use battery to avoid grid purchases
3. **RL adds no value** when the solution is a simple rule

### The Scope Problem

The current problem formulation has **insufficient complexity** for RL:
- Only 1 decision: how to route power flows
- Prices are relatively static (off-peak discount only)
- No scheduling flexibility
- No multi-timestep optimization needed

An `if/else` rule-based policy would outperform the trained agent.

---

## Comparison: Agent vs Optimal Simple Rule

### Agent Strategy (Current)
```
Total cost: €590.73
- Sells 5,028 kWh at €0.04/kWh → €201 revenue
- Buys 3,825 kWh at €0.19/kWh → €723 cost
- Net: -€522 from grid interaction
```

### Optimal Simple Strategy (Expected)
```
Never sell, maximize self-consumption:
- Store all excess PV in battery (4,169 kWh)
- Use stored energy to reduce grid purchases
- Estimated savings: ~€300-400/year vs current agent
```

---

## Visualizations Generated

See `outputs/diagnostics/agent_diagnostic.png`:

1. **Daily costs** - Shows cost volatility throughout 2023
2. **Hourly cost pattern** - Peak costs during evening hours
3. **Energy flows by hour** - PV vs consumption mismatch
4. **Selling behavior** - Massive selling during hours 10-14
5. **Battery SOC (Week 1)** - Battery barely charges
6. **Price context when selling** - Agent sells at ALL price points
7. **Monthly costs** - Winter months (Jan, Dec) are most expensive

---

## Recommendations for Re-scoping

To make RL valuable, the problem needs more complexity:

### Option A: Add Controllable Loads ⭐ (Recommended)
Add appliances with scheduling flexibility:
- **EV charging** (40-50 kWh, flexible 8-hour window)
- **Water heater** (3-4 kWh, flexible 4-hour window)
- **Dishwasher/washing machine** (2 kWh, flexible windows)
- **HVAC pre-cooling/heating** (temperature setpoint flexibility)

**Why this helps**:
- Now agent must optimize *when* to run loads
- Multi-timestep planning becomes critical
- Trade-offs between comfort, cost, battery management
- RL can learn non-obvious scheduling patterns

### Option B: Dynamic Pricing
Use real-time spot market prices:
- 10-20x price variance (€0.05 - €1.00/kWh)
- Negative prices during high renewable periods
- Battery arbitrage becomes profitable
- Timing decisions become complex

### Option C: Community Microgrid
Simulate 5-10 homes sharing resources:
- Peer-to-peer energy trading
- Shared battery management
- Local vs grid pricing
- Multi-agent coordination

### Option D: Multi-Objective Optimization
Balance multiple goals:
- Cost minimization
- Carbon emissions reduction
- Grid stability contribution
- Battery lifespan maximization
- Pareto-optimal trade-offs

---

## Technical Debt Identified

1. **Gym → Gymnasium migration needed** (warnings throughout)
2. **NumPy 2.0 compatibility issues**
3. **Column naming inconsistencies** (temperature vs temperature_C)
4. **No automated testing** of policies

---

## Next Steps

1. **Choose re-scoping direction** (A, B, C, or D above)
2. **Design new environment** with added complexity
3. **Update reward function** to reflect new objectives
4. **Retrain and benchmark** against new baselines
5. **Validate that RL outperforms rules** in new setting

---

## Files Generated by This Analysis

```
outputs/diagnostics/
├── agent_diagnostic.png          # Multi-panel visualization
├── full_analysis.csv              # Timestep-level data with pricing
├── daily_summary.csv              # Daily aggregated metrics
└── hourly_summary.csv             # Hourly patterns
```

---

## Conclusion

The diagnostic confirms the initial suspicion: **the problem is too constrained for RL to add value**. The agent's poor performance is a symptom of insufficient problem complexity, not a failure of the SAC algorithm or training process.

**Action**: Re-scope to add controllable loads (Option A) to create genuine scheduling complexity where RL can shine.
