import pandas as pd

# Load data
df = pd.read_csv(
    r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\raw\main_historic.csv",
    parse_dates=["datetime"]
)

# Sort chronologically and handle missing values
df = df.sort_values("datetime").fillna(method="ffill").fillna(0)

# Select features for the state space (can be tuned later)
state_cols = [
    "consumption_kWh", "temperature_C", "hour", "day_of_week",
    "is_weekend", "is_holiday", "P", "Gb(i)", "Gd(i)", "Gr(i)",
    "price", "pv_forecast_1", "load_forecast_1"
]

# --- Skip normalization --- 
# All columns remain in their original physical units

# Split into train/test sets (e.g., last 1 year for testing)
train_df = df[df["datetime"] < "2023-01-01"]
test_df = df[df["datetime"] >= "2023-01-01"]

# Reset indices
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Save processed data
train_df.to_csv(
    r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\processed\train.csv",
    index=False
)
test_df.to_csv(
    r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\processed\test.csv",
    index=False
)

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
