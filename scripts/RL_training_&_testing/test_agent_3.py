import pandas as pd
from pathlib import Path

# Load your resulting dataset
file_path = Path(r"C:\Users\AdrienSourdille\Solar_Home_Assistant_V2\outputs\agent_step_data.csv")
df = pd.read_csv(file_path)

# Compute total PV allocation
df["pv_total_allocated"] = df["pv_to_house_kWh"] + df["pv_to_batt_kWh"] + df["pv_to_grid_kWh"]

# Compare to actual PV production
df["pv_diff"] = df["pv_total_allocated"] - df["pv_production_kWh"]

# Check rows where allocation exceeds production
problem_rows = df[df["pv_diff"] > 1e-6]  # tiny tolerance for float errors

print("Rows with PV allocation mismatch:")
print(problem_rows[["datetime", "pv_total_allocated", "pv_production_kWh", "pv_diff"]])

# Optionally, check max deviation
print("Maximum deviation:", df["pv_diff"].abs().max())
