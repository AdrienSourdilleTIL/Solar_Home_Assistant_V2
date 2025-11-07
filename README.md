# Solar Battery Reinforcement Learning Agent

## Overview

Imagine you’ve just installed solar panels and a home battery. Your goal is simple: **use your solar energy efficiently**, reduce electricity bills, and perhaps even sell excess energy back to the grid. But making smart decisions about **when to charge, discharge, buy, or sell electricity** is tricky — consumption fluctuates hourly, PV production is variable, and electricity prices can change as well.

In France, the situation has changed:

- **Before 2025:** households could sign fixed-price contracts to sell their PV electricity for up to **35 cents/kWh**, often more than the grid purchase price. Selling everything was highly incentivized.  
- **Now:** selling prices dropped to around **4 cents/kWh**, while purchasing electricity from the grid got more expensive. This makes **self-consumption and smart battery management more important than ever**, requiring strategies that consider forecasts, consumption patterns, and battery state.

A **reinforcement learning (RL) agent** can manage these decisions dynamically, outperforming rule-based approaches and maximizing household energy efficiency.

![Cumulative Reward Comparison](outputs/cumulative_rewards_comparison.png)

The graph above shows cumulative rewards over a test year (2023), comparing:  

- **Trained SAC Agent**  
- **Simple Rule-Based Policy**  
- **Forecast-Aware / Context-Aware Rule-Based Policy**  

Even with forecasts and sophisticated rules, the RL agent achieves the highest cumulative reward by learning **non-linear, context-aware strategies**.

---

## How It Works

The environment simulates a **single household with solar panels and a battery**, with hourly timesteps. Observations include:

- **PV production** (`P`) and **household consumption** (`consumption_kWh`)  
- **Battery state-of-charge (SOC)**  
- **Environmental variables:** temperature, pressure, solar irradiance, wind speed  
- **Energy prices:** `buy_price` and `sell_price`  
- **Forecasts:** PV and load for up to 12 hours ahead  

At each timestep, the agent decides:

- How much PV energy to **consume immediately**  
- How much to **store in the battery**  
- How much to **discharge from the battery to the house**  
- Whether to **buy from or sell to the grid**  

---

## Reward Function

The agent is trained to **minimize total household energy costs** while accounting for battery degradation.  

The reward considers:

- **Gross cost:** net expense of buying electricity minus revenue from selling PV  
- **Degradation penalty:** small cost for charging/discharging the battery to reflect wear  
- **Objective:** minimize total costs, encouraging **self-consumption, battery usage, and selective selling**  

This is critical given the current low selling prices in France.

---

## Rule-Based Policies for Comparison

To benchmark performance, we implemented two rule-based strategies:

1. **Simple Rule-Based Policy**  
   - Uses PV for household consumption first  
   - Charges the battery if there’s excess PV  
   - Discharges the battery when PV is insufficient  
   - Falls back to the grid if the battery is empty  

2. **Forecast-Aware / Context-Aware Rule-Based Policy**  
   - Uses up to **12-hour forecasts** of PV and load  
   - Adjusts battery reserves based on expected PV and time of day  
   - Considers electricity prices for buying and selling  

Even with forecasts and smarter rules, the RL agent **consistently outperforms both**, learning **complex strategies that balance immediate and future costs**.

---

## Training the Agent

The agent was trained using **Soft Actor-Critic (SAC)**, a state-of-the-art **off-policy RL algorithm** ideal for **continuous control problems** like battery management:

- **Why SAC:**  
  - Handles **continuous action spaces** naturally (charging/discharging rates)  
  - Balances **exploration and exploitation** using entropy regularization  
  - Stable and **sample-efficient**, suitable for long historical energy datasets  

- **Training data:** 2015–2022 (~6 years of hourly data)  
- **Testing data:** 2023 (1 year)  
- **Observations included:**  
  - Environmental variables (temperature, pressure, solar irradiance, wind)  
  - PV production and household load  
  - PV and load forecasts for up to 12 hours  
  - Energy prices (`buy_price`, `sell_price`)  

- **Objective:** maximize cumulative reward, i.e., **minimize total costs including battery degradation**  

By training on several years of realistic data, the agent learned **how consumption patterns, PV variability, and price changes interact**, enabling context-aware decisions that fixed rule-based policies cannot replicate.

---
