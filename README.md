# Solar Battery Management: RL vs Rule-Based

Reinforcement learning agent for home solar battery management, trained using Soft Actor-Critic (SAC).

## Problem

Manage a home solar + battery system to minimize electricity costs. In France, the economics changed dramatically:

- **Before 2025:** Selling PV to grid at €0.35/kWh made selling everything optimal
- **Now:** Selling at €0.04/kWh while buying at €0.19/kWh makes self-consumption critical

The question: Can RL learn better battery management than simple rules?

## Results

Comparison of V2 RL agent vs simple rule-based policy on 2023 test data (8,760 hours):

| Method | Annual Cost | Grid Purchases | Battery Usage |
|--------|-------------|----------------|---------------|
| **Rule-Based** | **€152** | 1,660 kWh | 51% avg SOC |
| RL Agent (SAC) | €199 | 1,782 kWh | 64% avg SOC |

**Winner: Rule-Based Policy** (saves €47/year, 24% lower cost)

### The Simple Rules That Work

```python
# When PV > consumption (surplus):
charge_battery = 1.0 if battery_soc < 80% else 0.0

# When consumption > PV (deficit):
use_battery = 1.0 if battery_soc > 30% else 0.0
```

This simple threshold-based approach outperforms the learned RL policy.

## Why RL Doesn't Excel Here

The problem is **too simple** for RL to show advantages:

- **Static pricing:** Buy and sell prices don't vary by time of day, eliminating arbitrage opportunities
- **Trivial optimal strategy:** Simple thresholds capture the essential logic
- **No complex timing decisions:** When prices are constant, temporal optimization provides no benefit
- **No controllable loads:** Agent can't shift consumption to cheaper times

For RL to excel, you need:
- **Dynamic pricing** (time-of-use tariffs, spot market)
- **Controllable loads** (EV charging, water heater, HVAC)
- **Multiple conflicting objectives** (comfort vs cost, grid stability)
- **Uncertain forecasts** where learned adaptation matters

## Environment

The V2 environment uses a simplified action space that respects physical constraints:

**Actions (2D continuous):**
- `charge_battery_fraction` [0-1]: When surplus PV available, how much to store vs sell
- `use_battery_fraction` [0-1]: When deficit occurs, how much from battery vs grid

**Observations:**
- PV production and household consumption
- Battery state-of-charge (SOC)
- Energy prices (buy/sell)
- Lagged features (3 timesteps of P, consumption, prices)
- Cyclical time encodings (hour, day of week)

**Physical Logic:**
1. Use PV to meet house demand first (automatic, always optimal)
2. If surplus: allocate between battery and grid per agent decision
3. If deficit: source from battery and grid per agent decision

**Constraints:**
- Battery: 10 kWh capacity, 5 kW max charge rate, 95% efficiency
- Energy balance: 100% conservation guaranteed (no wasted energy)

## Training

**Algorithm:** Soft Actor-Critic (SAC)
- Off-policy RL for continuous control
- Entropy regularization for exploration

**Data:**
- Training: 2015-2022 (6 years of hourly data)
- Testing: 2023 (1 year)

**Reward:** Minimize `-(electricity_cost + degradation_penalty)`

## Repository Structure

```
scripts/RL_training_&_testing/
├── gym_env_v2.py                  # V2 simplified environment
├── train_v2.py                    # Train SAC agent
├── evaluate_v2.py                 # Evaluate on test set
├── compare_agent_vs_rules.py      # Compare RL vs rule-based
└── solar_batt_agent_v2.zip        # Trained model

data/main/processed/
├── train.csv                      # 2015-2022 training data
├── validation.csv                 # Validation split
└── test.csv                       # 2023 test data

outputs/
└── agent_step_data_v2.csv         # Detailed evaluation results
```

## Usage

**Train agent:**
```bash
cd scripts/RL_training_&_testing
python train_v2.py
```

**Evaluate agent:**
```bash
python evaluate_v2.py
```

**Compare RL vs rule-based:**
```bash
python compare_agent_vs_rules.py
```

## Lessons Learned

1. **Simple problems don't need RL:** When optimal strategy can be captured by thresholds, RL adds complexity without benefit
2. **Baseline comparison is critical:** Always compare against sensible heuristics before claiming RL success
3. **Problem complexity matters:** RL shines when there are complex temporal dependencies, multiple objectives, or uncertain dynamics
4. **Static pricing kills RL advantage:** Without time-varying prices, there's no arbitrage opportunity to learn

For this specific problem with static French residential tariffs, **simple rules win**.

## Future Work

To make RL valuable here, consider:
- **Dynamic pricing** (Tempo tariff, spot market)
- **Controllable loads** (EV, water heater scheduling)
- **Multiple batteries** or shared community storage
- **Grid services** (frequency regulation, demand response)
- **Forecast uncertainty** modeling for robust decisions

With these additions, the problem becomes complex enough for RL to demonstrate value over simple rules.
