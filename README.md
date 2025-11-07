# Solar Battery Reinforcement Learning Agent

## Overview

Picture this: you just installed solar panels and a home battery. You want to **use as much of your solar energy as possible**, minimize electricity bills, and maybe even **sell back excess energy** to the grid. But making smart decisions about **when to charge, discharge, buy, or sell electricity** is tricky — solar production and consumption fluctuate hour by hour, and prices change throughout the day.

This project demonstrates how a **reinforcement learning (RL) agent** can manage a household battery automatically. Using historical data, environmental conditions, and short-term forecasts, the agent learns how to **maximize self-consumption and reduce energy costs**, outperforming both simple and context-aware rule-based strategies.

![Cumulative Reward Comparison](outputs/cumulative_rewards_comparison.png)

The graph above compares cumulative rewards over a test year (2023) for:

- **Trained SAC Agent**  
- **Simple Rule-Based Policy**  
- **Forecast-Aware / Context-Aware Rule-Based Policy**  

Even with up to 12-hour forecasts and intelligent rules, the RL agent consistently achieves the highest rewards by learning **non-linear, context-sensitive strategies** that fixed rules cannot capture.

---

## How It Works

The environment simulates a **single household with solar panels and a battery**, with one-hour timesteps. At each step, the agent observes:

- **PV production** (`P`) and **household consumption** (`consumption_kWh`)  
- **Battery state-of-charge (SOC)**  
- **Environmental variables:** temperature, pressure, solar irradiance, wind speed  
- **Energy prices:** `buy_price` and `sell_price`  
- **Forecasts:** PV and load for up to 12 hours ahead  

The agent chooses:

- How much PV energy to **use immediately**  
- How much to **store in the battery**  
- How much to **discharge from the battery to the house**  
- Whether to **buy from or sell to the grid**  

---

## Reward Function

The reward is designed to **minimize household energy costs** while accounting for battery degradation. At each timestep:

```python
gross_cost = energy_from_grid * price_buy - energy_to_grid * price_sell
degradation_penalty = self.degradation_cost * (batt_charge_energy + battery_used)
reward = -(gross_cost + degradation_penalty)
