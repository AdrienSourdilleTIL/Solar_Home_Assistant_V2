import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class SolarBatteryEnv(gym.Env):
    """
    Solar-battery-grid environment.
    The agent decides:
        - PV allocation: house, battery, grid
        - Battery discharge: house, grid
        - Grid-to-battery charging
    The environment ensures home consumption is always met and prevents battery overflow.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, data: pd.DataFrame,
                 battery_capacity=10.0,
                 max_charge_rate=5.0,
                 timestep_h=1.0,
                 eta=0.95,
                 degradation_cost=0.001):
        super().__init__()

        self.data = data.reset_index(drop=True)
        self.battery_capacity = battery_capacity
        self.max_charge_rate = max_charge_rate
        self.timestep_h = timestep_h
        self.eta = eta  # charging efficiency
        self.degradation_cost = degradation_cost

        self.idx = 0
        self.soc = 0.5 * battery_capacity  # initial SOC

        # Determine observation columns dynamically
        self.state_cols = self._get_state_cols()

        # Compute normalization factors
        # FIX #3: Normalize buy_price and sell_price by SAME factor to preserve ratio
        # This is critical - agent must see that sell_price (€0.04) is ~21% of buy_price (€0.19)
        self.norm_factors = {}
        max_buy_price = None

        for col in self.state_cols:
            if col == 'buy_price':
                max_buy_price = self.data[col].abs().max() + 1e-8
                self.norm_factors[col] = max_buy_price
            elif col == 'sell_price':
                # Use same normalization factor as buy_price to preserve economic relationship
                if max_buy_price is None:
                    max_buy_price = self.data['buy_price'].abs().max() + 1e-8
                self.norm_factors[col] = max_buy_price  # Same factor!
            else:
                # All other features: normalize independently
                self.norm_factors[col] = self.data[col].abs().max() + 1e-8

        self.norm_factors["soc"] = battery_capacity

        obs_dim = len(self.state_cols) + 1  # SOC included
        # Observation space: normalized prices in ~0-1 range, other features also normalized
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32)
        # Action space: 4 continuous actions
        # action[0:3] = PV allocation logits (will be softmaxed to sum to 1.0)
        # action[3] = battery discharge to house fraction (0-1)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)

    # ----------------------
    # Helper: state columns
    # ----------------------
    def _get_state_cols(self):
        ignore_cols = ["Gb", "Gd", "Gr"]
        base_cols = [c for c in self.data.columns if c not in ignore_cols]

        forecast_cols = [c for c in base_cols if c.startswith("pv_forecast_") or c.startswith("load_forecast_")]
        lag_cols = [c for c in base_cols if "_lag" in c]
        cyclical_cols = [c for c in base_cols if c.endswith("_sin") or c.endswith("_cos")]
        binary_cols = [c for c in base_cols if c in ["is_weekend", "is_holiday"]]
        numeric_cols = [c for c in base_cols if np.issubdtype(self.data[c].dtype, np.number)
                        and c not in forecast_cols + lag_cols + cyclical_cols + binary_cols]

        return numeric_cols + cyclical_cols + binary_cols + forecast_cols + lag_cols

    # ----------------------
    # Observation builder
    # ----------------------
    def _get_obs(self):
        row = self.data.iloc[self.idx]
        obs_values = [
            row[col] / self.norm_factors[col] if np.issubdtype(self.data[col].dtype, np.number) else 0.0
            for col in self.state_cols
        ]
        obs_values.append(self.soc / self.norm_factors["soc"])
        return np.array(obs_values, dtype=np.float32)

    # ----------------------
    # Step logic
    # ----------------------
    def step(self, action):
        row = self.data.iloc[self.idx]

        consumption = row["consumption_kWh"]
        pv = max(row["P"], 0.0)
        price_buy = row["buy_price"]
        price_sell = row["sell_price"]

        # ACTION SPACE REDESIGN:
        # action[0:3] = PV allocation logits [house, batt, grid]
        # action[3] = battery discharge to house fraction

        # Apply softmax to PV allocation logits to get mutually exclusive fractions
        pv_logits = action[0:3]
        pv_exp = np.exp(pv_logits - np.max(pv_logits))  # numerical stability
        pv_fracs = pv_exp / np.sum(pv_exp)  # softmax

        pv_to_house_frac = float(pv_fracs[0])
        pv_to_batt_frac = float(pv_fracs[1])
        pv_to_grid_frac = float(pv_fracs[2])  # Agent now explicitly controls selling!

        batt_to_house_frac = float(np.clip(action[3], 0, 1))

        # PV allocation (fractions now sum to 1.0 by construction)
        pv_to_house_intended = pv * pv_to_house_frac
        pv_to_batt_intended = pv * pv_to_batt_frac
        pv_to_grid_intended = pv * pv_to_grid_frac

        # CRITICAL FIX: Handle excess PV to house (can't use more than consumption)
        # Any excess gets redirected to grid (can't be wasted)
        pv_to_house_actual = min(pv_to_house_intended, consumption)
        pv_house_overflow = max(0, pv_to_house_intended - consumption)

        # Handle battery overflow (if battery is full, excess goes to grid)
        max_batt_charge = self.battery_capacity - self.soc
        pv_to_batt_actual = min(pv_to_batt_intended * self.eta, max_batt_charge)
        pv_batt_overflow = max(0, (pv_to_batt_intended * self.eta) - max_batt_charge)

        # Final grid sale includes intended + overflow from full battery + overflow from house
        pv_to_grid = pv_to_grid_intended + pv_batt_overflow + pv_house_overflow

        # Use actual PV to house (not intended)
        pv_to_house = pv_to_house_actual

        # Battery discharge
        batt_discharge_power = self.max_charge_rate
        discharge_to_house = 0.0
        discharge_to_grid = 0.0
        battery_used = 0.0
        if self.soc > 0.1 * self.battery_capacity:
            available_energy = min(self.soc, batt_discharge_power * self.timestep_h)
            discharge_to_house = available_energy * batt_to_house_frac
            discharge_to_grid = available_energy * (1 - batt_to_house_frac)
            battery_used = available_energy

        # Grid charging removed - agent should not charge battery from expensive grid
        grid_to_batt = 0.0

        # Update SOC with charging efficiency
        batt_charge_energy = pv_to_batt_actual
        self.soc = np.clip(self.soc + batt_charge_energy - battery_used, 0, self.battery_capacity)

        # House demand
        energy_supplied = pv_to_house + discharge_to_house
        energy_deficit = max(consumption - energy_supplied, 0)
        grid_to_house = energy_deficit

        # Grid flows
        energy_from_grid = grid_to_house + grid_to_batt
        energy_to_grid = pv_to_grid + discharge_to_grid

        # Cost and reward (FIX #3: Add moderate reward shaping)
        gross_cost = energy_from_grid * price_buy - energy_to_grid * price_sell
        degradation_penalty = self.degradation_cost * (batt_charge_energy + battery_used)
        base_reward = -(gross_cost + degradation_penalty)

        # Reward Shaping #1: Opportunity cost penalty
        # This makes the immediate cost of selling more apparent to the agent
        # When selling 1 kWh at €0.04 when buy price is €0.19, the opportunity cost is €0.15
        # This represents the "forgone savings" from not storing for later use
        opportunity_cost = energy_to_grid * (price_buy - price_sell)

        # Reward Shaping #2: SOC management incentive
        # Encourage maintaining 40-70% SOC to keep battery ready for peak hours
        soc_ratio = self.soc / self.battery_capacity
        soc_penalty = 0.0
        if soc_ratio < 0.4:
            # Penalty for low SOC (battery not ready for evening peak)
            soc_penalty = 0.02 * (0.4 - soc_ratio)
        elif soc_ratio > 0.7:
            # Small penalty for too high SOC (lost charging opportunity)
            soc_penalty = 0.01 * (soc_ratio - 0.7)

        # Final shaped reward (moderate weights: 0.5 for opportunity cost)
        reward = base_reward - 0.5 * opportunity_cost - soc_penalty

        # Info dictionary
        info = dict(
            datetime=row["datetime"],
            step=self.idx,
            soc_kWh=self.soc,
            soc_ratio=self.soc / self.battery_capacity,
            pv_to_house_kWh=pv_to_house,
            pv_to_batt_kWh=pv_to_batt_actual,
            pv_to_grid_kWh=pv_to_grid,
            discharge_to_house_kWh=discharge_to_house,
            discharge_to_grid_kWh=discharge_to_grid,
            grid_to_batt_kWh=grid_to_batt,
            grid_to_house_kWh=grid_to_house,
            energy_from_grid_kWh=energy_from_grid,
            energy_to_grid_kWh=energy_to_grid,
            cost_eur=gross_cost,
            degradation_penalty=degradation_penalty
        )

        # Advance timestep
        self.idx += 1
        terminated = self.idx >= len(self.data) - 1
        truncated = False

        return self._get_obs(), reward, terminated, truncated, info

    # ----------------------
    # Reset
    # ----------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.soc = 0.5 * self.battery_capacity
        return self._get_obs(), {}

    # ----------------------
    # Render
    # ----------------------
    def render(self):
        print(f"Step {self.idx}: SOC={self.soc:.2f} kWh")
