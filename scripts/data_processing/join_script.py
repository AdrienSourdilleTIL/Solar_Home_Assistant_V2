import pandas as pd

# -----------------------------
# CONFIGURATION
# -----------------------------
# Actuals
consumption_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\consumption\processed\synthetic_consumption.csv"     # columns: DATE, load_actual_kwh
pv_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\PV_production\processed\PV_production_2015_2023.csv"       # columns: DateTime, pv_actual_kwh
prices_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\electricity_consumer_prices\processed\residential_prices_2015_2023.csv"  # columns: datetime, buy_price, sell_price

# Forecasts
pv_fcst_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\forecast_pv\processed\pv_forecast.csv"       # column: pv_forecast_kwh
load_fcst_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\forecast_load\processed\load_forecast.csv" # column: load_forecast_kwh

forecast_horizon = 12
output_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\processed\main_historic.csv"

# -----------------------------
# LOAD DATA AND ENSURE DATETIME
# -----------------------------
consumption_df = pd.read_csv(consumption_file)
consumption_df.rename(columns={"DATE": "datetime"}, inplace=True)
consumption_df["datetime"] = pd.to_datetime(consumption_df["datetime"], errors="coerce")

pv_df = pd.read_csv(pv_file)
pv_df.rename(columns={"DateTime": "datetime"}, inplace=True)
pv_df["datetime"] = pd.to_datetime(pv_df["datetime"], errors="coerce")

prices_df = pd.read_csv(prices_file)
prices_df["datetime"] = pd.to_datetime(prices_df["datetime"], errors="coerce")

pv_fcst_df = pd.read_csv(pv_fcst_file)
pv_fcst_df.rename(columns={"DateTime": "datetime"}, inplace=True)
pv_fcst_df["datetime"] = pd.to_datetime(pv_fcst_df["datetime"], errors="coerce")

load_fcst_df = pd.read_csv(load_fcst_file)
load_fcst_df.rename(columns={"DATE": "datetime"}, inplace=True)
load_fcst_df["datetime"] = pd.to_datetime(load_fcst_df["datetime"], errors="coerce")

# -----------------------------
# SORT DATAFRAMES
# -----------------------------
for df in [consumption_df, pv_df, prices_df, pv_fcst_df, load_fcst_df]:
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

# -----------------------------
# CREATE BASE DATASET
# -----------------------------
main_df = consumption_df.merge(pv_df, on="datetime", how="left")
main_df = main_df.merge(prices_df, on="datetime", how="left")  # adds buy_price and sell_price

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

# -----------------------------
# SAMPLE OUTPUT
# -----------------------------
print(main_df.head(10))
