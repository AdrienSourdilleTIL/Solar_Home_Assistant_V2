import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
input_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\day_ahead_prices\raw\France.csv"
output_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\day_ahead_prices\processed\France_clean.csv"

# Columns to drop
cols_to_drop = ["Country", "ISO3 Code", "Datetime (UTC)"]

# -----------------------------
# LOAD AND CLEAN
# -----------------------------
df = pd.read_csv(input_file)

# Drop columns
df_clean = df.drop(columns=cols_to_drop)

# Optional: rename columns for convenience
df_clean.rename(columns={"Datetime (Local)": "datetime", "Price (EUR/MWhe)": "price"}, inplace=True)

# -----------------------------
# SAVE CLEANED FILE
# -----------------------------
df_clean.to_csv(output_file, index=False)
print(f"Cleaned file saved to {output_file}")
