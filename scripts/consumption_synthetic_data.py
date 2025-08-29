import pandas as pd
import numpy as np
import os
import holidays

# --- Paths ---
input_path = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\consumption\raw\2015-2023_hourly.csv"
output_path = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\consumption\processed\synthetic_consumption.csv"

# --- Load weather data ---
df = pd.read_csv(input_path)

# Strip column names to avoid whitespace issues
df.columns = df.columns.str.strip()

# Ensure timestamp column exists
df["DATE"] = pd.to_datetime(df["Date"])
df = df.sort_values("DATE")

# Extract useful fields
df["temperature_C"] = df["TMP"].astype(float)  # already in °C
df["hour"] = df["DATE"].dt.hour
df["day_of_week"] = df["DATE"].dt.dayofweek
df["month"] = df["DATE"].dt.month

# Mark weekends
df["is_weekend"] = df["day_of_week"].isin([5, 6])

# Mark holidays (France)
years = range(df["DATE"].dt.year.min(), df["DATE"].dt.year.max() + 1)
fr_holidays = holidays.France(years=years)
df["is_holiday"] = df["DATE"].dt.date.isin(fr_holidays)

# --- Baseline hourly profile (kWh per hour, normalized to ~14 kWh/day) ---
base_profile = {
    0: 0.2, 1: 0.15, 2: 0.15, 3: 0.15, 4: 0.2, 5: 0.3, 6: 0.7, 7: 0.9,
    8: 0.6, 9: 0.4, 10: 0.35, 11: 0.35, 12: 0.7, 13: 0.5, 14: 0.4, 15: 0.45,
    16: 0.6, 17: 1.0, 18: 1.2, 19: 1.3, 20: 1.0, 21: 0.8, 22: 0.5, 23: 0.3
}
df["base_load"] = df["hour"].map(base_profile)

# --- Modifiers ---
# Heating demand (below 12°C) - reduced for efficiency
df["heating_factor"] = np.where(df["temperature_C"] < 12, (12 - df["temperature_C"]) * 0.05, 0)

# Cooling demand (above 24°C) - reduced for efficiency
df["cooling_factor"] = np.where(df["temperature_C"] > 24, (df["temperature_C"] - 24) * 0.02, 0)

# Weekend boost
df["weekend_factor"] = np.where(df["is_weekend"], 1.1, 1.0)

# Holiday boost
df["holiday_factor"] = np.where(df["is_holiday"], 1.2, 1.0)

# Random noise
np.random.seed(42)
df["noise"] = np.random.normal(1.0, 0.05, len(df))

# --- Final consumption model ---
df["consumption_kWh"] = (
    df["base_load"] * df["weekend_factor"] * df["holiday_factor"] * df["noise"]
    + df["heating_factor"] + df["cooling_factor"]
)

# --- Keep useful columns ---
result = df[[
    "DATE", "consumption_kWh", "temperature_C", "hour",
    "day_of_week", "is_weekend", "is_holiday"
]]

# --- Save output ---
os.makedirs(os.path.dirname(output_path), exist_ok=True)
result.to_csv(output_path, index=False)

print(f"✅ Synthetic consumption dataset saved to: {output_path}")
