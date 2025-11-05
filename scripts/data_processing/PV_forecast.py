import pandas as pd
import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------
pv_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\PV_production\processed\PV_production_2015_2023.csv"
load_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\consumption\processed\synthetic_consumption.csv"

output_pv_fcst_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\forecast_pv\processed\pv_forecast.csv"
output_load_fcst_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\forecast_load\processed\load_forecast.csv"

pv_noise_std = 0.10   # 10% noise
load_noise_std = 0.20 # 20% noise

# -----------------------------
# LOAD DATA
# -----------------------------
pv_df = pd.read_csv(pv_file, parse_dates=["DateTime"])
pv_df["P"] = pd.to_numeric(pv_df["P"], errors="coerce")
pv_df = pv_df.dropna(subset=["P"])

# Load consumption data without parse_dates
load_df = pd.read_csv(load_file)
# Detect the column containing datetime
datetime_col = None
for col in load_df.columns:
    if "date" in col.lower():
        datetime_col = col
        break
if datetime_col is None:
    raise ValueError("No datetime-like column found in load CSV.")

load_df[datetime_col] = pd.to_datetime(load_df[datetime_col])

# -----------------------------
# SORT AND RESET INDEX
# -----------------------------
pv_df = pv_df.sort_values("DateTime").reset_index(drop=True)
load_df = load_df.sort_values(datetime_col).reset_index(drop=True)

# -----------------------------
# CREATE NOISY FORECAST
# -----------------------------
np.random.seed(42)

pv_df["pv_forecast_kwh"] = np.maximum(
    pv_df["P"] + np.random.normal(0, pv_noise_std * pv_df["P"]), 0
)

load_df["load_forecast_kwh"] = np.maximum(
    load_df["consumption_kWh"] + np.random.normal(0, load_noise_std * load_df["consumption_kWh"]), 0
)

# -----------------------------
# KEEP ONLY RELEVANT FIELDS AND RENAME
# -----------------------------
pv_df = pv_df[["DateTime", "pv_forecast_kwh"]].rename(columns={"DateTime": "datetime"})
load_df = load_df[[datetime_col, "load_forecast_kwh"]].rename(columns={datetime_col: "datetime"})

# -----------------------------
# SAVE OUTPUT
# -----------------------------
pv_df.to_csv(output_pv_fcst_file, index=False)
load_df.to_csv(output_load_fcst_file, index=False)

print(f"PV forecast saved to {output_pv_fcst_file}")
print(f"Load forecast saved to {output_load_fcst_file}")
