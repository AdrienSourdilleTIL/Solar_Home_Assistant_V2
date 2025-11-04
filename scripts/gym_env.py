import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class SolarBatteryEnvHybrid(gym.Env):
    """
    Gymnasium environment with:
    - Continuous actions (partial charging/discharging)
    - Explicit separation of PV vs grid charging and home vs grid discharging
    - Tracks all energy flows for clear cost calculations
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, data: pd.DataFrame, battery_capacity=10.0, max_charge_rate=5.0,
                 timestep_h=1.0, eta=0.95):
        super().__init__()

        self.data = data.reset_index(drop=True)
        self.battery_capacity = battery_capacity
        self.max_charge_rate = max_charge_rate
        self.timestep_h = timestep_h
        self.eta = eta  # battery round-trip efficiency

        self.idx = 0
        self.soc = 0.5 * battery_capacity

        # State columns
        forecast_cols = [c for c in data.columns if c.startswith("pv_forecast_") or c.startswith("load_forecast_")]
        base_cols = [
            "consumption_kWh", "P", "price",
            "temperature_C", "hour", "day_of_week",
            "is_weekend", "is_holiday"
        ]
        self.state_cols = base_cols + forecast_cols

        # Normalization
        self.norm_factors = {col: self.data[col].abs().max() for col in self.state_cols}
        self.norm_factors["soc"] = battery_capacity

        # Observation space
        obs_dim = len(self.state_cols) + 1  # SOC included
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # Action space (continuous)
        # action[0]: charge/discharge fraction (-1 discharge, +1 charge)
        # action[1]: fraction of charge from PV (0=all from grid, 1=all from PV)
        # action[2]: fraction of discharge to home (0=all to grid, 1=all to home)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

    def _get_obs(self):
        row = self.data.iloc[self.idx]
        obs_values = [row[c] / (self.norm_factors[c] + 1e-8) for c in self.state_cols]
        obs_values.append(self.soc / self.norm_factors["soc"])
        return np.array(obs_values, dtype=np.float32)

    def step(self, action):
        row = self.data.iloc[self.idx]
        consumption = row["consumption_kWh"]
        pv = max(row["P"], 0.0)
        price = row["price"]

        # --- parse action ---
        charge_frac = float(np.clip(action[0], -1.0, 1.0))
        pv_frac = float(np.clip(action[1], 0.0, 1.0))
        home_frac = float(np.clip(action[2], 0.0, 1.0))

        power = charge_frac * self.max_charge_rate
        energy = power * self.timestep_h

        # --- charging ---
        if energy >= 0:
            pv_charge = min(energy * pv_frac, pv)
            grid_charge = energy - pv_charge
            effective_charge = (pv_charge + grid_charge) * self.eta
            self.soc = np.clip(self.soc + effective_charge, 0, self.battery_capacity)
        else:
            # discharging
            energy = -energy  # make positive
            energy = min(energy, self.soc)  # cannot discharge more than SOC
            soc_discharge = energy / self.eta
            self.soc -= soc_discharge
            effective_discharge = soc_discharge

        # --- PV allocation to load ---
        pv_to_load = min(consumption, pv)
        remaining_pv = pv - pv_to_load

        # --- PV to battery (if any charge fraction wants it) ---
        pv_to_batt = min(remaining_pv, pv_charge if energy >= 0 else 0)
        pv_to_grid = remaining_pv - pv_to_batt

        # --- discharge allocation ---
        if energy < 0:
            discharge_to_home = effective_discharge * home_frac
            discharge_to_grid = effective_discharge * (1 - home_frac)
        else:
            discharge_to_home = 0
            discharge_to_grid = 0

        # --- net grid flows ---
        net_load = consumption - pv_to_load - discharge_to_home
        energy_from_grid = max(net_load + grid_charge, 0)
        energy_to_grid = discharge_to_grid + pv_to_grid

        # --- reward ---
        reward = -(energy_from_grid * price - energy_to_grid * price)

        info = dict(
            step=self.idx,
            soc_kWh=self.soc,
            soc_ratio=self.soc / self.battery_capacity,
            energy_from_grid_kWh=energy_from_grid,
            energy_to_grid_kWh=energy_to_grid,
            pv_to_load_kWh=pv_to_load,
            pv_to_battery_kWh=pv_to_batt,
            pv_to_grid_kWh=pv_to_grid,
            discharge_to_home_kWh=discharge_to_home,
            discharge_to_grid_kWh=discharge_to_grid,
            cost_eur=energy_from_grid * price - energy_to_grid * price,
        )

        # --- next step ---
        self.idx += 1
        terminated = self.idx >= len(self.data) - 1
        truncated = False
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.soc = 0.5 * self.battery_capacity
        return self._get_obs(), {}
