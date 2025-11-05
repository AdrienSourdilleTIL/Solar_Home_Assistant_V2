import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class SolarBatteryEnv(gym.Env):
    """
    Solar-battery-grid environment where the agent decides how to allocate energy flows.
    The agent decides:
        - how much PV goes to battery vs grid vs house
        - how much battery discharges to house vs grid
        - how much to import from grid to charge the battery
    The environment ensures home consumption is always met (grid compensates deficits).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, data: pd.DataFrame,
                 battery_capacity=10.0,
                 max_charge_rate=5.0,
                 timestep_h=1.0,
                 eta=0.95,
                 degradation_cost=0.001):
        super().__init__()

        # --- Parameters ---
        self.data = data.reset_index(drop=True)
        self.battery_capacity = battery_capacity
        self.max_charge_rate = max_charge_rate
        self.timestep_h = timestep_h
        self.eta = eta
        self.degradation_cost = degradation_cost  # penalty for cycling battery

        self.idx = 0
        self.soc = 0.5 * battery_capacity

        # --- Observation fields (use dataset names directly) ---
        self.state_cols = [
            "consumption_kWh", "P", "price", "temperature_C",
            "hour", "day_of_week", "is_weekend", "is_holiday",
            "Gb(i)", "Gd(i)", "Gr(i)", "H_sun", "T2m", "WS10m"
        ]

        # Add all forecast columns (pv_forecast_X and load_forecast_X)
        forecast_cols = [c for c in data.columns if c.startswith("pv_forecast_") or c.startswith("load_forecast_")]
        self.state_cols += forecast_cols

        # --- Normalization factors ---
        self.norm_factors = {col: self.data[col].abs().max() + 1e-8 for col in self.state_cols}
        self.norm_factors["soc"] = battery_capacity

        # --- Spaces ---
        obs_dim = len(self.state_cols) + 1  # include SOC
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # Action space:
        # action[0]: fraction of PV sent to house (0–1)
        # action[1]: fraction of PV sent to battery (0–1)
        # (remainder goes to grid)
        # action[2]: fraction of battery discharge sent to house (0–1)
        # (remainder goes to grid)
        # action[3]: fraction of grid energy used to charge battery (0–1)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

    # --- Helpers ---
    def _get_obs(self):
        row = self.data.iloc[self.idx]
        obs_values = [row[c] / self.norm_factors[c] for c in self.state_cols]
        obs_values.append(self.soc / self.norm_factors["soc"])
        return np.array(obs_values, dtype=np.float32)

    # --- Step logic ---
    def step(self, action):
        row = self.data.iloc[self.idx]

        # Current energy context
        consumption = row["consumption_kWh"]
        pv = max(row["P"], 0.0)
        price = row["price"]

        # Parse and clip actions
        pv_to_house_frac = float(np.clip(action[0], 0, 1))
        pv_to_batt_frac = float(np.clip(action[1], 0, 1))
        batt_to_house_frac = float(np.clip(action[2], 0, 1))
        grid_to_batt_frac = float(np.clip(action[3], 0, 1))

        # --- PV allocation ---
        pv_to_house = pv * pv_to_house_frac
        pv_to_batt = pv * pv_to_batt_frac
        pv_to_grid = max(pv - pv_to_house - pv_to_batt, 0)

        # --- Battery discharge ---
        batt_discharge_power = self.max_charge_rate  # max discharge capacity
        discharge_to_house = 0.0
        discharge_to_grid = 0.0
        battery_used = 0.0

        if self.soc > 0.1 * self.battery_capacity:
            available_energy = min(self.soc, batt_discharge_power * self.timestep_h)
            discharge_to_house = available_energy * batt_to_house_frac * self.eta
            discharge_to_grid = available_energy * (1 - batt_to_house_frac) * self.eta
            battery_used = available_energy

        # --- Grid and battery charging ---
        grid_to_batt = 0.0
        if self.soc < self.battery_capacity:
            grid_to_batt = self.max_charge_rate * grid_to_batt_frac * self.timestep_h
            grid_to_batt = min(grid_to_batt, (self.battery_capacity - self.soc))

        # Effective battery charging
        batt_charge_energy = (pv_to_batt + grid_to_batt) * self.eta
        self.soc = np.clip(self.soc + batt_charge_energy - battery_used, 0, self.battery_capacity)

        # --- Energy to meet house demand ---
        energy_supplied = pv_to_house + discharge_to_house
        energy_deficit = max(consumption - energy_supplied, 0)
        grid_to_house = energy_deficit

        # --- Grid flows ---
        energy_from_grid = grid_to_house + grid_to_batt
        energy_to_grid = pv_to_grid + discharge_to_grid

        # --- Costs and reward ---
        gross_cost = energy_from_grid * price - energy_to_grid * price
        degradation_penalty = self.degradation_cost * (batt_charge_energy + battery_used)
        reward = -(gross_cost + degradation_penalty)

        info = dict(
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

        # Advance time
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


