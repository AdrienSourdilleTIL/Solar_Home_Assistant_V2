import pandas as pd
import numpy as np
import os
import holidays

# --- Paths ---
input_path = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\consumption\raw\2015_2023_hourly.csv"
output_path = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\consumption\processed\synthetic_consumption.csv"

# --- Load weather data ---
df = pd.read_csv(input_path)

# Ensure timestamp column exists
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE")

# Extract useful fields
df["temperature_C"] = df["TMP"].str.split(",").str[0].astype(float) / 10.0  # TMP is tenths of °C
df["hour"] = df["DATE"].dt.hour
df["day_of_week"] = df["DATE"].dt.dayofweek
df["month"] = df["DATE"].dt.month

# Mark weekends
df["is_weekend"] = df["day_of_week"].isin([5, 6])

# Mark holidays (France)
fr_holidays = holidays.France(years=range(df["DATE"].dt.year.min(), df["DATE"].dt.year.max() + 1))
df["is_holiday"] = df["DATE"].dt.date.isin(fr_holidays)

# --- Baseline hourly profile (kWh per hour, normalized to ~20 kWh/day) ---
base_profile = {
    0: 0.4, 1: 0.3, 2: 0.3, 3: 0.3, 4: 0.4,
    5: 0.6, 6: 1.2, 7: 1.6, 8: 1.0, 9: 0.7,
    10: 0.6, 11: 0.6, 12: 1.2, 13: 0.8, 14: 0.7,
    15: 0.8, 16: 1.0, 17: 1.8, 18: 2.0, 19: 2.2,
    20: 1.8, 21: 1.4, 22: 0.9, 23: 0.6
}
df["base_load"] = df["hour"].map(base_profile)

# --- Modifiers ---
# Heating demand (below 12°C)
df["heating_factor"] = np.where(df["temperature_C"] < 12,
                                (12 - df["temperature_C"]) * 0.1, 0)

# Cooling demand (above 24°C)
df["cooling_factor"] = np.where(df["temperature_C"] > 24,
                                (df["temperature_C"] - 24) * 0.05, 0)

# Weekend boost
df["weekend_factor"] = np.where(df["is_weekend"], 1.1, 1.0)

# Holiday boost
df["holiday_factor"] = np.where(df["is_holiday"], 1.2, 1.0)

# Random noise
np.random.seed(42)
df["noise"] = np.random.normal(1.0, 0.05, len(df))

# --- Final consumption model ---
df["consumption_kWh"] = (
    df["base_load"] *
    df["weekend_factor"] *
    df["holiday_factor"] *
    df["noise"]
    + df["heating_factor"]
    + df["cooling_factor"]
)

# --- Keep useful columns ---
result = df[[
    "DATE", "consumption_kWh", "temperature_C",
    "hour", "day_of_week", "is_weekend", "is_holiday"
]]

# --- Save output ---
os.makedirs(os.path.dirname(output_path), exist_ok=True)
result.to_csv(output_path, index=False)

print(f"✅ Synthetic consumption dataset saved to: {output_path}")
