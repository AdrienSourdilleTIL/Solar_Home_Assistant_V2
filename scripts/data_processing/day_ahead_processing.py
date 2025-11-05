import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
input_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\day_ahead_prices\raw\France.csv"
output_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\day_ahead_prices\processed\France_clean.csv"

# Parameters
fixed_resell_price = 0.32  # €/kWh
retail_multiplier = 3.5    # roughly 3–4x markup from wholesale to retail
monthly_subscription = 12  # €/month (typical residential subscription)
hours_per_month = 30 * 24
fixed_hourly_fee = monthly_subscription / hours_per_month  # €/hour

# Columns to drop
cols_to_drop = ["Country", "ISO3 Code", "Datetime (UTC)"]

# -----------------------------
# LOAD AND CLEAN
# -----------------------------
df = pd.read_csv(input_file)

# Drop unnecessary columns
df_clean = df.drop(columns=cols_to_drop)

# Rename columns for convenience
df_clean.rename(columns={"Datetime (Local)": "datetime", "Price (EUR/MWhe)": "price"}, inplace=True)

# Ensure datetime column is datetime64[ns, UTC]
df_clean["datetime"] = pd.to_datetime(df_clean["datetime"], errors="coerce", utc=True)

# Convert price from EUR/MWh to EUR/kWh
df_clean["price"] = df_clean["price"] / 1000.0

# Convert wholesale to retail by applying multiplier and adding hourly fixed cost
df_clean["retail_price"] = df_clean["price"] * retail_multiplier + fixed_hourly_fee

# Add fixed resell price column
df_clean["resell_price"] = fixed_resell_price

# -----------------------------
# SAVE CLEANED FILE
# -----------------------------
df_clean.to_csv(output_file, index=False)
print(f"Cleaned file saved to {output_file}")
