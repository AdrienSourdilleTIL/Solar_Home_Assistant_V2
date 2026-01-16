# Simplified Action Space Proposal

**Problem Identified**: Current action space is too complex and doesn't match physical reality.

---

## Current Action Space (FLAWED)

```python
action[0:3] = softmax logits [pv_to_house_frac, pv_to_batt_frac, pv_to_grid_frac]
action[3] = battery_discharge_to_house_frac
```

**Why It's Broken**:
1. Agent allocates PV as percentages without knowing house demand
2. Example: Agent says "send 30% of PV to house"
   - But if house needs 0.5 kWh and PV=3 kWh, that's 0.9 kWh
   - 0.4 kWh wasted (gets redirected to grid)
3. Battery discharge is independent of PV allocation
4. No clear decision about "should I use battery or grid to meet remaining demand?"

**Result**: Agent can't reason about energy balance properly

---

## Simplified Action Space (PROPOSED)

### The Physical Reality

At each timestep, agent faces this situation:
```
INPUTS:
- PV production: X kWh
- House consumption: Y kWh
- Battery SOC: Z kWh
- Prices: buy_price, sell_price

DECISIONS:
1. How much PV to store in battery? (rest goes to house or grid)
2. If house needs more energy, use battery or grid?
```

### Proposed Actions (2 continuous values)

```python
action[0] = charge_battery_fraction  # [0, 1]
# What fraction of available PV (after meeting house demand) should go to battery?

action[1] = use_battery_fraction  # [0, 1]
# When house needs more energy, what fraction should come from battery vs grid?
```

### Step Logic (MUCH SIMPLER)

```python
def step(self, action):
    # Get current state
    consumption = row["consumption_kWh"]
    pv = max(row["P"], 0.0)
    price_buy = row["buy_price"]
    price_sell = row["sell_price"]

    charge_batt_frac = float(np.clip(action[0], 0, 1))
    use_batt_frac = float(np.clip(action[1], 0, 1))

    # STEP 1: Use PV to meet house demand first (always optimal)
    pv_to_house = min(pv, consumption)
    remaining_pv = pv - pv_to_house
    remaining_consumption = consumption - pv_to_house

    # STEP 2: Allocate remaining PV between battery and grid
    if remaining_pv > 0:
        # We have surplus PV
        max_batt_charge = (self.battery_capacity - self.soc) / self.eta
        pv_to_batt_intended = remaining_pv * charge_batt_frac
        pv_to_batt_actual = min(pv_to_batt_intended, max_batt_charge) * self.eta

        # Rest goes to grid (can't waste it)
        pv_to_grid = remaining_pv - pv_to_batt_intended

        # If battery is full, overflow goes to grid
        if pv_to_batt_intended > max_batt_charge:
            pv_to_grid += (pv_to_batt_intended - max_batt_charge)
    else:
        pv_to_batt_actual = 0.0
        pv_to_grid = 0.0

    # STEP 3: Meet remaining house demand from battery or grid
    if remaining_consumption > 0:
        # We have deficit
        available_batt = min(self.soc - 0.1*self.battery_capacity,
                            self.max_charge_rate * self.timestep_h)
        available_batt = max(0, available_batt)

        batt_to_house_intended = remaining_consumption * use_batt_frac
        batt_to_house = min(batt_to_house_intended, available_batt)

        # Rest must come from grid (no choice)
        grid_to_house = remaining_consumption - batt_to_house
    else:
        batt_to_house = 0.0
        grid_to_house = 0.0

    # Update battery SOC
    self.soc = np.clip(self.soc + pv_to_batt_actual - batt_to_house,
                      0, self.battery_capacity)

    # Calculate costs
    energy_bought = grid_to_house
    energy_sold = pv_to_grid
    gross_cost = energy_bought * price_buy - energy_sold * price_sell
    degradation = self.degradation_cost * (pv_to_batt_actual + batt_to_house)
    reward = -(gross_cost + degradation)

    return obs, reward, terminated, truncated, info
```

---

## Why This Is Better

### 1. Matches Physical Reality
- **Step 1**: Use PV for house first (always optimal, no decision needed)
- **Step 2**: Decide how to allocate SURPLUS PV (store vs sell)
- **Step 3**: Decide how to meet DEFICIT (battery vs grid)

### 2. Agent Makes Meaningful Decisions
- **action[0]**: "I have surplus PV, should I store it or sell it?"
- **action[1]**: "I need more energy, should I use battery or buy from grid?"

### 3. Energy Balance is Guaranteed
```
House consumption = pv_to_house + batt_to_house + grid_to_house  (always true)
PV production = pv_to_house + pv_to_batt + pv_to_grid  (always true)
```

### 4. No Complex Overflow Logic
- Surplus PV naturally goes to battery first, then grid
- Deficit naturally comes from battery first, then grid
- No wasted energy

### 5. Simpler Observation Space
Agent needs to see:
- Current PV production
- Current consumption
- Battery SOC
- Prices
- Time features (for anticipating peaks)
- Forecasts (optional)

---

## Expected Behavior

### Optimal Policy Should Be:

**When PV > Consumption** (surplus):
- If battery not full AND evening peak coming → charge_batt_frac = 1.0 (store)
- If battery full OR no peak expected → charge_batt_frac = 0.0 (sell)

**When Consumption > PV** (deficit):
- If battery has energy AND cheap to discharge → use_batt_frac = 1.0
- If battery empty OR expensive to discharge → use_batt_frac = 0.0 (buy from grid)

### Agent Can Learn:
- Time-of-day patterns (charge battery during midday, discharge during evening)
- Price sensitivity (buy from grid when cheap, use battery when expensive)
- Battery management (maintain SOC for peaks)

---

## Migration Plan

1. Create new simplified `gym_env_v2.py` with this logic
2. Keep normalization fix from Fix #1 (both prices by same factor)
3. Remove all reward shaping (let agent learn naturally)
4. Use gamma=0.999 (from Fix #1)
5. Train and compare results

**Expected**: Should outperform all previous fixes because:
- Clearer action space → faster learning
- Guaranteed energy balance → no wasted PV
- Simpler logic → fewer bugs
- Natural rewards → no shaping artifacts

---

## What do you think?

This matches your intuition: agent sees production and consumption, then decides how to allocate resources. Much cleaner than the current complex softmax approach.

Should I implement this?
