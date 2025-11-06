import pandas as pd

# -----------------------------
# CONFIGURATION
# -----------------------------
# Actuals
consumption_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\consumption\processed\synthetic_consumption.csv"
pv_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\PV_production\processed\PV_production_2015_2023.csv"
prices_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\electricity_consumer_prices\processed\residential_prices_2015_2023.csv"

# Forecasts
pv_fcst_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\forecast_pv\processed\pv_forecast.csv"
load_fcst_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\forecast_load\processed\load_forecast.csv"

forecast_horizon = 12
output_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\raw\main_historic.csv"

# -----------------------------
# LOAD DATA
# -----------------------------
consumption_df = pd.read_csv(consumption_file)
pv_df = pd.read_csv(pv_file)
prices_df = pd.read_csv(prices_file)
pv_fcst_df = pd.read_csv(pv_fcst_file)
load_fcst_df = pd.read_csv(load_fcst_file)

# -----------------------------
# RENAME DATETIME COLUMNS
# -----------------------------
consumption_df.rename(columns={"DATE": "datetime"}, inplace=True)
pv_df.rename(columns={"DateTime": "datetime"}, inplace=True)
pv_fcst_df.rename(columns={"DateTime": "datetime"}, inplace=True)
load_fcst_df.rename(columns={"DATE": "datetime"}, inplace=True)

# -----------------------------
# CONVERT TO NAIVE UTC
# -----------------------------
for df in [consumption_df, pv_df, prices_df, pv_fcst_df, load_fcst_df]:
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)  # parse as UTC
    df["datetime"] = df["datetime"].dt.tz_localize(None)  # remove timezone info

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
    main_df[f"pv_forecast_{h}"] = pv_fcst_df["pv_forecast_kwh"].shift(-h + 1).reindex(main_df.index)
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
