"""
Simplified Solar Battery Environment (v2)
==========================================

Key improvements over v1:
1. Much simpler action space (2 actions instead of 4)
2. Respects physical energy balance
3. No wasted energy due to overflow logic
4. Agent makes meaningful decisions aligned with physical reality

Action Space:
    action[0] = charge_battery_fraction [0, 1]
        - When PV > consumption, what fraction of surplus to store?
    action[1] = use_battery_fraction [0, 1]
        - When consumption > PV, what fraction of deficit from battery?

Physical Logic:
    Step 1: Use PV to meet house demand first (always optimal)
    Step 2: If surplus PV, allocate between battery and grid
    Step 3: If deficit, allocate between battery and grid
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class SolarBatteryEnvV2(gym.Env):
    """
    Simplified solar battery management environment.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        battery_capacity: float = 10.0,
        max_charge_rate: float = 5.0,
        timestep_h: float = 1.0,
        eta: float = 0.95,
        degradation_cost: float = 0.001,
    ):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.battery_capacity = battery_capacity
        self.max_charge_rate = max_charge_rate
        self.timestep_h = timestep_h
        self.eta = eta
        self.degradation_cost = degradation_cost

        self.idx = 0
        self.soc = 0.5 * battery_capacity

        # Determine observation columns
        self.state_cols = self._get_state_cols()

        # Compute normalization factors
        # CRITICAL: buy_price and sell_price normalized by SAME factor
        self.norm_factors = {}
        max_buy_price = None

        for col in self.state_cols:
            if col == 'buy_price':
                max_buy_price = self.data[col].abs().max() + 1e-8
                self.norm_factors[col] = max_buy_price
            elif col == 'sell_price':
                if max_buy_price is None:
                    max_buy_price = self.data['buy_price'].abs().max() + 1e-8
                self.norm_factors[col] = max_buy_price  # Same factor preserves ratio
            else:
                self.norm_factors[col] = self.data[col].abs().max() + 1e-8

        self.norm_factors["soc"] = battery_capacity

        obs_dim = len(self.state_cols) + 1  # +1 for SOC
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32)

        # SIMPLIFIED ACTION SPACE: 2 continuous actions [0, 1]
        # action[0] = charge_battery_fraction (when PV > consumption)
        # action[1] = use_battery_fraction (when consumption > PV)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    def _get_state_cols(self):
        """Extract relevant state columns from data."""
        ignore_cols = ["Gb", "Gd", "Gr"]
        base_cols = [c for c in self.data.columns if c not in ignore_cols]

        forecast_cols = [c for c in base_cols if c.startswith("pv_forecast_") or c.startswith("load_forecast_")]
        lag_cols = [c for c in base_cols if "_lag" in c]
        cyclical_cols = [c for c in base_cols if c.endswith("_sin") or c.endswith("_cos")]
        binary_cols = [c for c in base_cols if c in ["is_weekend", "is_holiday"]]
        numeric_cols = [
            c for c in base_cols
            if np.issubdtype(self.data[c].dtype, np.number)
            and c not in forecast_cols + lag_cols + cyclical_cols + binary_cols
        ]

        return numeric_cols + cyclical_cols + binary_cols + forecast_cols + lag_cols

    def _get_obs(self):
        """Build observation vector."""
        row = self.data.iloc[self.idx]
        obs_values = [
            row[col] / self.norm_factors[col] if np.issubdtype(self.data[col].dtype, np.number) else 0.0
            for col in self.state_cols
        ]
        obs_values.append(self.soc / self.norm_factors["soc"])
        return np.array(obs_values, dtype=np.float32)

    def step(self, action):
        """
        Execute one timestep with simplified action space.

        Physical logic:
        1. Use PV to meet house demand first (always optimal)
        2. If PV > consumption: allocate surplus to battery or grid
        3. If consumption > PV: meet deficit from battery or grid
        """
        row = self.data.iloc[self.idx]

        consumption = row["consumption_kWh"]
        pv = max(row["P"], 0.0)
        price_buy = row["buy_price"]
        price_sell = row["sell_price"]

        charge_batt_frac = float(np.clip(action[0], 0, 1))
        use_batt_frac = float(np.clip(action[1], 0, 1))

        # =============================================================================
        # STEP 1: Use PV to meet house demand first (always optimal - no decision)
        # =============================================================================
        pv_to_house = min(pv, consumption)
        remaining_pv = pv - pv_to_house
        remaining_consumption = consumption - pv_to_house

        # Initialize energy flows
        pv_to_batt = 0.0
        pv_to_grid = 0.0
        batt_to_house = 0.0
        grid_to_house = 0.0

        # =============================================================================
        # STEP 2: Allocate remaining PV (if any) between battery and grid
        # =============================================================================
        if remaining_pv > 0:
            # We have surplus PV after meeting house demand

            # Calculate max battery charge capacity
            available_capacity = self.battery_capacity - self.soc
            max_charge_power = self.max_charge_rate * self.timestep_h
            max_batt_charge = min(available_capacity / self.eta, max_charge_power)

            # Agent decides how much surplus to store
            pv_to_batt_intended = remaining_pv * charge_batt_frac
            pv_to_batt_raw = min(pv_to_batt_intended, max_batt_charge)
            pv_to_batt = pv_to_batt_raw * self.eta  # Account for charging efficiency

            # Rest goes to grid (can't waste it)
            pv_to_grid = remaining_pv - pv_to_batt_raw

        # =============================================================================
        # STEP 3: Meet remaining house demand (if any) from battery or grid
        # =============================================================================
        if remaining_consumption > 0:
            # We have deficit after using all PV

            # Calculate available battery energy
            min_soc = 0.1 * self.battery_capacity  # Keep 10% reserve
            available_batt = max(0, self.soc - min_soc)
            max_discharge_power = self.max_charge_rate * self.timestep_h
            max_batt_discharge = min(available_batt, max_discharge_power)

            # Agent decides how much deficit to meet from battery
            batt_to_house_intended = remaining_consumption * use_batt_frac
            batt_to_house = min(batt_to_house_intended, max_batt_discharge)

            # Rest must come from grid (no choice)
            grid_to_house = remaining_consumption - batt_to_house

        # =============================================================================
        # Update battery SOC
        # =============================================================================
        self.soc = np.clip(
            self.soc + pv_to_batt - batt_to_house,
            0,
            self.battery_capacity
        )

        # =============================================================================
        # Calculate costs and reward
        # =============================================================================
        energy_bought = grid_to_house
        energy_sold = pv_to_grid

        gross_cost = energy_bought * price_buy - energy_sold * price_sell
        degradation_penalty = self.degradation_cost * (pv_to_batt + batt_to_house)

        # Natural reward (no shaping)
        reward = -(gross_cost + degradation_penalty)

        # =============================================================================
        # Build info dictionary
        # =============================================================================
        info = dict(
            datetime=row["datetime"],
            step=self.idx,
            soc_kWh=self.soc,
            soc_ratio=self.soc / self.battery_capacity,
            pv_production_kWh=pv,
            consumption_kWh=consumption,
            pv_to_house_kWh=pv_to_house,
            pv_to_batt_kWh=pv_to_batt,
            pv_to_grid_kWh=pv_to_grid,
            batt_to_house_kWh=batt_to_house,
            grid_to_house_kWh=grid_to_house,
            energy_from_grid_kWh=energy_bought,
            energy_to_grid_kWh=energy_sold,
            cost_eur=gross_cost,
            degradation_penalty=degradation_penalty,
            # Store actions for analysis
            charge_batt_frac=charge_batt_frac,
            use_batt_frac=use_batt_frac,
        )

        # Check if episode is done BEFORE advancing index
        terminated = (self.idx + 1) >= len(self.data)
        truncated = False

        # Advance timestep
        self.idx += 1

        # Get next observation (or zeros if terminated)
        if terminated:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            next_obs = self._get_obs()

        return next_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.idx = 0
        self.soc = 0.5 * self.battery_capacity
        return self._get_obs(), {}

    def render(self):
        """Render current state."""
        print(f"Step {self.idx}: SOC={self.soc:.2f} kWh")
