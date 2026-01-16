# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a solar battery management project that compares **Reinforcement Learning (RL) vs Rule-Based policies** for optimizing home energy systems. The current results show that **simple rule-based policies outperform the RL agent** (€152/year vs €199/year) because the problem is too simple with static pricing.

**Key Finding**: RL doesn't excel here due to static electricity prices (buy €0.19/kWh, sell €0.04/kWh). For RL to provide value, the problem needs dynamic pricing, controllable loads, or multi-objective optimization.

## Development Commands

### Training and Evaluation

```bash
# Train RL agent (SAC algorithm)
cd scripts/RL_training_&_testing
python train_v2.py  # Trains on 2015-2022 data, saves to solar_batt_agent_v2.zip

# Evaluate trained agent on test set
python evaluate_v2.py  # Runs on 2023 data, outputs to outputs/agent_step_data_v2.csv

# Compare RL vs rule-based policy
python compare_agent_vs_rules.py  # Shows physical flows and cost comparison
```

### Data Processing

The data pipeline joins multiple sources into a single dataset:

```bash
cd scripts/data_processing

# Main join script (combines all data sources)
python join_script.py  # Creates data/main/raw/main_historic.csv

# Split into train/test
python test_vs_train.py  # Creates data/main/processed/{train.csv, test.csv}
```

## Architecture

### 1. Environment: `gym_env_v2.py`

The core RL environment (`SolarBatteryEnvV2`) implements a **simplified 2-action space**:

**Action Space** (both continuous [0, 1]):
- `action[0]`: `charge_battery_fraction` - when surplus PV, what fraction to store vs sell to grid
- `action[1]`: `use_battery_fraction` - when deficit, what fraction from battery vs grid

**Physical Logic** (guaranteed energy conservation):
1. Always use PV to meet house demand first (automatic, always optimal)
2. If surplus PV: allocate between battery charging and grid selling per agent decision
3. If deficit: allocate between battery discharge and grid purchase per agent decision

**Key Constraints**:
- Battery: 10 kWh capacity, 5 kW max charge rate, 95% efficiency
- 10% minimum SOC reserve (safety margin)
- All energy flows respect physical limits (no wasted energy)

**Observations** (normalized):
- PV production, household consumption
- Battery state-of-charge (SOC)
- Buy/sell prices (normalized by same factor to preserve ratio)
- Lagged features (3 timesteps of P, consumption, prices)
- Cyclical time encodings (hour_sin/cos, day_sin/cos)

**Reward**: `-(electricity_cost + degradation_penalty)`
- Electricity cost = grid purchases - grid sales
- Degradation penalty = 0.001 * (charge + discharge energy)

### 2. Data Pipeline

**Data Sources** (in `data/` directory):
- `consumption/`: Synthetic household consumption data
- `PV_production/`: Solar production from weather data (2015-2023)
- `electricity_consumer_prices/`: Residential buy/sell prices
- `day_ahead_retail_prices/`: Wholesale market prices (France_clean.csv) - **not currently used but available for V3**
- `forecast_pv/`: 12-hour ahead PV forecasts
- `forecast_load/`: 12-hour ahead consumption forecasts

**Processing Flow**:
1. `join_script.py`: Merges all sources on datetime, adds 12-hour forecast columns
2. `test_vs_train.py`: Splits into train (2015-2022) and test (2023)
3. Training scripts apply feature engineering:
   - Drop unused columns (Gb, Gd, Gr)
   - Cyclical encoding for hour and day_of_week
   - Create lag features (3 timesteps for P, consumption, prices)

**Final Dataset Columns**:
- Datetime and core: `datetime`, `P` (PV production), `consumption_kWh`
- Prices: `buy_price`, `sell_price`
- Weather: `T2M`, `PS`, `ALLSKY_SFC_SW_DWN`, `WS10M`
- Forecasts: `pv_forecast_1` through `pv_forecast_12`, `load_forecast_1` through `load_forecast_12`
- Engineered: `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `P_lag1-3`, `consumption_kWh_lag1-3`, etc.

### 3. Training

**Algorithm**: Soft Actor-Critic (SAC) from stable-baselines3
- Off-policy RL for continuous control
- Entropy regularization for exploration
- γ=0.999 (high discount to value future rewards)
- Learning rate: 3e-4, batch size: 128
- 200k timesteps (~4 passes through 6 years of hourly data)

**Model Storage**: Trained model saved as `solar_batt_agent_v2.zip` (3.6 MB)

### 4. Rule-Based Baseline

Simple threshold policy that beats RL:

```python
# Charge battery if SOC < 80%, else sell surplus
charge_battery_fraction = 1.0 if battery_soc < 0.8 else 0.0

# Use battery if SOC > 30%, else buy from grid
use_battery_fraction = 1.0 if battery_soc > 0.3 else 0.0
```

This captures the near-optimal strategy for static pricing scenarios.

## Important Notes

### Feature Engineering Must Match Training

When evaluating or extending the environment, **feature engineering must exactly match** what was used during training:

1. Drop columns: `Gb`, `Gd`, `Gr`
2. Cyclical encoding: `hour` → `hour_sin/cos`, `day_of_week` → `day_sin/cos`
3. Lag features: Create 3 lags for `P`, `consumption_kWh`, `buy_price`, `sell_price`
4. Drop NaN rows after creating lags

This preprocessing is duplicated in `train_v2.py`, `evaluate_v2.py`, and `compare_agent_vs_rules.py`. Any changes must be synchronized.

### Normalization Critical Detail

In `gym_env_v2.py`, buy_price and sell_price are **normalized by the same factor** (max of buy_price). This preserves their ratio, which is critical for the agent to learn the correct price relationship:

```python
max_buy_price = self.data['buy_price'].abs().max() + 1e-8
self.norm_factors['buy_price'] = max_buy_price
self.norm_factors['sell_price'] = max_buy_price  # Same factor!
```

### Path Conventions

Scripts use **absolute Windows paths** (e.g., `C:\Users\AdrienSourdille\...`). When running on different systems, update paths in:
- `train_v2.py` (line 12)
- `evaluate_v2.py` (line 14)
- `compare_agent_vs_rules.py` (line 21)
- All `scripts/data_processing/*.py` files

## Future V3 Direction

To make RL valuable, consider adding:

1. **Dynamic pricing**: Use `data/day_ahead_retail_prices/processed/France_clean.csv` for time-varying wholesale prices
2. **Industrial scenario**: Model a factory with flexible loads (bottling lines, cooling, CIP cleaning)
3. **Controllable loads**: Add EV charging, water heater, HVAC that can be scheduled
4. **Multi-objective**: Balance cost vs production deadlines vs equipment constraints

The wholesale price data is already available but not integrated. This would create arbitrage opportunities where RL can learn temporal optimization patterns that simple rules cannot capture.
