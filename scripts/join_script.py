import pandas as pd

# -----------------------------
# CONFIGURATION
# -----------------------------
# Actuals
consumption_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\consumption\processed\synthetic_consumption.csv"     # columns: DATE, load_actual_kwh
pv_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\PV_production\processed\PV_production_2015_2023.csv"       # columns: DateTime, pv_actual_kwh
prices_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\day_ahead_prices\processed\France_clean.csv"           # columns: datetime, price_eur_mwh

# Forecasts
pv_fcst_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\forecast_pv\processed\pv_forecast.csv"       # column: pv_forecast_kwh
load_fcst_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\forecast_load\processed\load_forecast.csv" # column: load_forecast_kwh

forecast_horizon = 12

output_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\processed\main_historic.csv"

# -----------------------------
# LOAD DATA
# -----------------------------
consumption_df = pd.read_csv(consumption_file, parse_dates=["DATE"])
consumption_df.rename(columns={"DATE": "datetime"}, inplace=True)

pv_df = pd.read_csv(pv_file, parse_dates=["DateTime"])
pv_df.rename(columns={"DateTime": "datetime"}, inplace=True)

prices_df = pd.read_csv(prices_file, parse_dates=["datetime"])  # already named 'datetime'

pv_fcst_df = pd.read_csv(pv_fcst_file, parse_dates=["DateTime"])
pv_fcst_df.rename(columns={"DateTime": "datetime"}, inplace=True)

load_fcst_df = pd.read_csv(load_fcst_file, parse_dates=["DATE"])
load_fcst_df.rename(columns={"DATE": "datetime"}, inplace=True)

# Sort by datetime
for df in [consumption_df, pv_df, prices_df, pv_fcst_df, load_fcst_df]:
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

# -----------------------------
# CREATE BASE DATASET
# -----------------------------
main_df = consumption_df.merge(pv_df, on="datetime", how="left")
main_df = main_df.merge(prices_df, on="datetime", how="left")

# -----------------------------
# ADD FORECAST COLUMNS
# -----------------------------
for h in range(1, forecast_horizon + 1):
    # PV forecast
    main_df[f"pv_forecast_{h}"] = pv_fcst_df["pv_forecast_kwh"].shift(-h + 1).reindex(main_df.index)
    
    # Load forecast
    main_df[f"load_forecast_{h}"] = load_fcst_df["load_forecast_kwh"].shift(-h + 1).reindex(main_df.index)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
main_df.to_csv(output_file, index=False)
print(f"RL-ready dataset saved to {output_file}")
