import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\raw\main_historic.csv", parse_dates=["datetime"])

# Sort chronologically and handle missing values
df = df.sort_values("datetime").fillna(method="ffill").fillna(0)

# Select features for the state space (can be tuned later)
state_cols = [
    "consumption_kWh", "temperature_C", "hour", "day_of_week",
    "is_weekend", "is_holiday", "P", "Gb(i)", "Gd(i)", "Gr(i)",
    "price", "pv_forecast_1", "load_forecast_1"
]

# Normalize continuous variables
scaler = StandardScaler()
df[state_cols] = scaler.fit_transform(df[state_cols])

# Split into train/test sets (e.g., last 1 year for testing)
train_df = df[df["datetime"] < "2023-01-01"]
test_df = df[df["datetime"] >= "2023-01-01"]

# Save processed data
train_df.to_csv(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\processed\train.csv", index=False)
test_df.to_csv(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\data\main\processed\test.csv", index=False)

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
