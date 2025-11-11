import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class SolarBatteryEnv(gym.Env):
    """
    Solar-battery-grid environment where the agent decides energy flows.
    The agent decides:
        - PV allocation: house, battery, grid
        - Battery discharge: house, grid
        - Grid-to-battery charging
    The environment ensures home consumption is always met.
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
        self.eta = eta
        self.degradation_cost = degradation_cost

        self.idx = 0
        self.soc = 0.5 * battery_capacity

        # Determine observation columns dynamically
        self.state_cols = self._get_state_cols()
        self.norm_factors = {col: self.data[col].abs().max() + 1e-8 for col in self.state_cols}
        self.norm_factors["soc"] = battery_capacity

        obs_dim = len(self.state_cols) + 1  # include SOC
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # 4 continuous actions: PV→house, PV→battery, battery→house, grid→battery
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

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

    def _get_obs(self):
        row = self.data.iloc[self.idx]
        obs_values = [row[col] / self.norm_factors[col] if np.issubdtype(self.data[col].dtype, np.number) else 0.0
                      for col in self.state_cols]
        obs_values.append(self.soc / self.norm_factors["soc"])
        return np.array(obs_values, dtype=np.float32)

    def step(self, action):
        row = self.data.iloc[self.idx]

        consumption = row["consumption_kWh"]
        pv = max(row["P"], 0.0)
        price_buy = row["buy_price"]
        price_sell = row["sell_price"]

        # Clip actions
        pv_to_house_frac = float(np.clip(action[0], 0, 1))
        pv_to_batt_frac = float(np.clip(action[1], 0, 1))
        batt_to_house_frac = float(np.clip(action[2], 0, 1))
        grid_to_batt_frac = float(np.clip(action[3], 0, 1))

        # PV allocation
        pv_to_house = pv * pv_to_house_frac
        pv_to_batt = pv * pv_to_batt_frac
        pv_to_grid = max(pv - pv_to_house - pv_to_batt, 0.0)

        # Battery discharge
        batt_discharge_power = self.max_charge_rate
        discharge_to_house = 0.0
        discharge_to_grid = 0.0
        battery_used = 0.0
        if self.soc > 0.1 * self.battery_capacity:
            available_energy = min(self.soc, batt_discharge_power * self.timestep_h)
            discharge_to_house = available_energy * batt_to_house_frac * self.eta
            discharge_to_grid = available_energy * (1 - batt_to_house_frac) * self.eta
            battery_used = available_energy

        # Grid charging
        grid_to_batt = 0.0
        if self.soc < self.battery_capacity:
            grid_to_batt = min(self.max_charge_rate * grid_to_batt_frac * self.timestep_h,
                               self.battery_capacity - self.soc)

        batt_charge_energy = (pv_to_batt + grid_to_batt) * self.eta
        self.soc = np.clip(self.soc + batt_charge_energy - battery_used, 0, self.battery_capacity)

        # House demand
        energy_supplied = pv_to_house + discharge_to_house
        energy_deficit = max(consumption - energy_supplied, 0)
        grid_to_house = energy_deficit

        energy_from_grid = grid_to_house + grid_to_batt
        energy_to_grid = pv_to_grid + discharge_to_grid

        gross_cost = energy_from_grid * price_buy - energy_to_grid * price_sell
        degradation_penalty = self.degradation_cost * (batt_charge_energy + battery_used)
        reward = -(gross_cost + degradation_penalty)

        info = dict(
            datetime=row["datetime"],  # Include datetime to fix the KeyError
            step=self.idx,
            soc_kWh=self.soc,
            soc_ratio=self.soc / self.battery_capacity,
            pv_to_house_kWh=pv_to_house,
            pv_to_batt_kWh=pv_to_batt,
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

        self.idx += 1
        terminated = self.idx >= len(self.data) - 1
        truncated = False
        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.soc = 0.5 * self.battery_capacity
        return self._get_obs(), {}

    def render(self):
        print(f"Step {self.idx}: SOC={self.soc:.2f} kWh")
