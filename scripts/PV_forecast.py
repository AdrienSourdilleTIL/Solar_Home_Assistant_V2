import pandas as pd
import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------
pv_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\PV_production\processed\PV_production_2015_2023.csv"
load_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\consumption\processed\synthetic_consumption.csv"

output_pv_fcst_file = "data/forecast_PV/processed/pv_forecast.csv"
output_load_fcst_file = "data/forecast_load/processed/load_forecast.csv"

pv_noise_std = 0.10   # 10% noise
load_noise_std = 0.20 # 20% noise

# -----------------------------
# LOAD DATA
# -----------------------------
pv_df = pd.read_csv(pv_file, parse_dates=["DateTime"])
load_df = pd.read_csv(load_file, parse_dates=["Date"])

pv_df = pv_df.sort_values("Date").reset_index(drop=True)
load_df = load_df.sort_values("DateTime").reset_index(drop=True)

# -----------------------------
# CREATE NOISY FORECAST
# -----------------------------
pv_df["pv_forecast_kwh"] = np.maximum(
    pv_df["P"] + np.random.normal(0, pv_noise_std * pv_df["P"]), 0
)

load_df["load_forecast_kwh"] = np.maximum(
    load_df["consumption_kWh"] + np.random.normal(0, load_noise_std * load_df["consumption_kWh"]), 0
)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
pv_df.to_csv(output_pv_fcst_file, index=False)
load_df.to_csv(output_load_fcst_file, index=False)

print(f"PV forecast saved to {output_pv_fcst_file}")
print(f"Load forecast saved to {output_load_fcst_file}")
