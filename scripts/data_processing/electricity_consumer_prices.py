import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
start_date = "2015-01-01 00:00"
end_date = "2023-12-31 23:00"

# Modern realistic surplus sale price
fixed_sell_price = 0.04  # €/kWh

# Heures creuses hours (off-peak): 22:00–06:00
offpeak_hours = list(range(22, 24)) + list(range(0, 6))
offpeak_discount = 0.7  # 30% cheaper

# Base buying prices per year (€/kWh)
buy_prices_by_year = {
    2015: 0.17,
    2016: 0.173,
    2017: 0.176,
    2018: 0.18,
    2019: 0.183,
    2020: 0.187,
    2021: 0.19,
    2022: 0.205,
    2023: 0.21
}

# -----------------------------
# CREATE DATETIME INDEX
# -----------------------------
datetime_index = pd.date_range(start=start_date, end=end_date, freq="H", tz="Europe/Paris")
df_prices = pd.DataFrame({"datetime": datetime_index})

# Convert to UTC for consistent merging
df_prices["datetime"] = df_prices["datetime"].dt.tz_convert("UTC")

df_prices["hour"] = df_prices["datetime"].dt.hour
df_prices["year"] = df_prices["datetime"].dt.year

# -----------------------------
# ASSIGN BUYING PRICES
# -----------------------------
def compute_buy_price(row):
    base_price = buy_prices_by_year.get(row["year"], 0.21)  # default to 0.21 if missing
    if row["hour"] in offpeak_hours:
        return base_price * offpeak_discount
    else:
        return base_price

df_prices["buy_price"] = df_prices.apply(compute_buy_price, axis=1)

# -----------------------------
# ASSIGN SELLING PRICE
# -----------------------------
df_prices["sell_price"] = fixed_sell_price

# -----------------------------
# CLEAN AND SAVE
# -----------------------------
df_prices = df_prices.drop(columns=["hour", "year"])
output_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\electricity_consumer_prices\processed\residential_prices_2015_2023.csv"
df_prices.to_csv(output_file, index=False)
print(f"Hourly residential price dataset saved to {output_file}")

# -----------------------------
# SAMPLE OUTPUT
# -----------------------------
print(df_prices.head(10))
