import pandas as pd

input_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\PV_production\raw\PV_production_2015_2023.csv"
output_file = r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\PV_production\processed\PV_production_2015_2023.csv"

# Load CSV
pv_df = pd.read_csv(input_file, parse_dates=["DateTime"], low_memory=False)

# Make sure the 'P' column is numeric
pv_df["P"] = pd.to_numeric(pv_df["P"], errors="coerce")

# Optional: check for bad values
bad_rows = pv_df[pd.isna(pv_df["P"])]
if not bad_rows.empty:
    print(f"Found {len(bad_rows)} rows with invalid P values. They will become NaN in kWh.")

# Convert Wh to kWh
pv_df["P"] = pv_df["P"] / 1000

# Rename datetime column for consistency
pv_df.rename(columns={"DateTime": "datetime"}, inplace=True)

# Save to new file
pv_df.to_csv(output_file, index=False)
print(f"Processed PV production saved to {output_file}")
